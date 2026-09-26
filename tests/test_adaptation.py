"""Adaptation, evaluation and artifact tests on a stub UNet/MoVQ (torch + diffusers required, no weights):
the training loop, epoch selection, the transactional guarantee and the artifact round trip with its refusals."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from conftest import synthetic_records  # noqa: E402
from kandinsky_inpainting_pipeline import LORA_TENSORS, KandinskyInpaintPipeline, lora_parameter_names  # noqa: E402
from kandinsky_inpainting_pipeline import pipeline as pl  # noqa: E402

RANK, DIM, EMB = 2, 6, 8


class _StubUNet(torch.nn.Module):
    """Names follow peft's layout; the output depends on the LoRA tensors so training moves them."""

    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        for block in range(2):
            for target in ("to_q", "to_k"):
                stem = f"down_blocks.{block}.attentions.0.{target}".replace(".", "__")
                self.register_parameter(f"{stem}__lora_A__default__weight", torch.nn.Parameter(torch.randn(RANK, DIM) * 0.1))
                self.register_parameter(f"{stem}__lora_B__default__weight", torch.nn.Parameter(torch.zeros(DIM, RANK)))

    def named_parameters(self, *args, **kwargs):  # type: ignore[override]
        for name, param in super().named_parameters(*args, **kwargs):
            yield name.replace("__", "."), param

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return {k.replace("__", "."): v for k, v in super().state_dict(*args, **kwargs).items()}

    def load_state_dict(self, state, strict=True):  # type: ignore[override]
        return super().load_state_dict({k.replace(".", "__"): v for k, v in state.items()}, strict=strict)

    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs, return_dict):
        gain = 1.0
        for name, param in self.named_parameters():
            if ".lora_B." in name:
                a = dict(self.named_parameters())[name.replace("lora_B", "lora_A")]
                gain = gain + (param @ a).mean()
        image_embeds = added_cond_kwargs.get("image_embeds", torch.zeros(1))
        pred = sample * gain + 0.01 * image_embeds.mean()
        # Kandinsky 2.2 returns 8 channels (4 mean + 4 var)
        return (torch.cat([pred, pred], dim=1),)


class _StubMoVQ:
    config = SimpleNamespace(scaling_factor=0.18215)

    def encode(self, pixels):
        pooled = torch.nn.functional.adaptive_avg_pool2d(pixels, 4)[:, :pl.LATENT_CHANNELS]
        if pooled.shape[1] < pl.LATENT_CHANNELS:
            pooled = pooled.repeat(1, pl.LATENT_CHANNELS // pooled.shape[1] + 1, 1, 1)[:, :pl.LATENT_CHANNELS]
        return SimpleNamespace(latents=pooled)

    def decode(self, latents, force_not_quantize=True):
        up = torch.nn.functional.interpolate(latents[:, :3], size=(32, 32))
        return SimpleNamespace(sample=up)


def _pipeline(monkeypatch) -> KandinskyInpaintPipeline:
    unet = _StubUNet()
    names = lora_parameter_names(unet)
    monkeypatch.setattr(pl, "LORA_PARAMETERS", sum(dict(unet.named_parameters())[n].numel() for n in names))
    pipe = KandinskyInpaintPipeline(
        unet=unet,
        movq=_StubMoVQ(),
        scheduler_config={
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "prediction_type": "epsilon",
        },
        prior_dir=Path("unused"),
        device="cpu",
        dtype=torch.float32,
        weights_dir=Path("unused"),
        source="stub",
        use_lora=True,
    )
    generator = torch.Generator().manual_seed(0)
    for prompt in (pl.NEGATIVE_PROMPT, "a photo of a red bird", "a photo of a blue bird"):
        img_emb = torch.randn(EMB, generator=generator)
        neg_emb = torch.randn(EMB, generator=generator)
        pipe._prompt_cache[prompt] = {"image_embeds": img_emb, "negative_image_embeds": neg_emb}
    return pipe


def test_stub_matches_the_contract_shape():
    names = lora_parameter_names(_StubUNet())
    assert len(names) == 8 and all(".lora_" in n for n in names) and LORA_TENSORS == 176


def test_evaluate_is_paired_and_seeded(monkeypatch):
    pipe = _pipeline(monkeypatch)
    records = synthetic_records(4)
    run1 = pipe.evaluate(records, seed=42)
    run2 = pipe.evaluate(records, seed=42)
    assert run1["denoising_mse"] == run2["denoising_mse"]
    assert run1["per_record"] == run2["per_record"]
    assert set(run1["by_timestep"]) == {str(t) for t in pl.EVAL_TIMESTEPS}


def test_adapt_minimises_loss_and_records_history(monkeypatch):
    pipe = _pipeline(monkeypatch)
    records = synthetic_records(4)
    result = pipe.adapt(train=records, val=records[:2], epochs=2, lr=1e-3, seed=0)
    assert len(result["history"]) == 3  # epoch 0 (frozen) + 2 training epochs
    assert result["history"][0]["train_loss"] is None
    assert result["best_epoch"] in (0, 1, 2)
    assert pipe.adapter is not None
    assert pipe.adapter["epochs"] == 2


def test_adapt_transactional_guarantee_on_error(monkeypatch):
    pipe = _pipeline(monkeypatch)
    records = synthetic_records(4)
    initial_weights = {k: v.clone() for k, v in pipe.unet.state_dict().items()}

    def exploding_scheduler():
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(pipe, "_noise_scheduler", exploding_scheduler)
    with pytest.raises(RuntimeError, match="simulated training failure"):
        pipe.adapt(train=records, epochs=1)

    assert pipe.adapter is None
    for k, v in pipe.unet.state_dict().items():
        assert torch.equal(v, initial_weights[k]), f"weight {k} mutated after failed adapt()"


def test_artifact_save_and_reload_roundtrip(tmp_path, monkeypatch):
    pipe = _pipeline(monkeypatch)
    records = synthetic_records(4)
    pipe.adapt(train=records, epochs=1, seed=0)

    artifact_dir = tmp_path / "kandinsky_adapter"
    manifest = pipe.save_artifact(artifact_dir, metadata={"run_by": "test"})
    assert (artifact_dir / pl.ARTIFACT_MANIFEST_NAME).is_file()
    assert (artifact_dir / pl.ARTIFACT_WEIGHTS_NAME).is_file()
    assert manifest["format"] == pl.ARTIFACT_FORMAT
    assert manifest["metadata"]["run_by"] == "test"

    # Reload into fresh stub pipeline
    fresh = _pipeline(monkeypatch)
    loaded_manifest = fresh.load_adapter(artifact_dir)
    assert loaded_manifest["weights"]["sha256"] == manifest["weights"]["sha256"]
    assert fresh.adapter == manifest["adapter"]


def test_artifact_refuses_tampered_weights(tmp_path, monkeypatch):
    pipe = _pipeline(monkeypatch)
    records = synthetic_records(4)
    pipe.adapt(train=records, epochs=1, seed=0)

    artifact_dir = tmp_path / "kandinsky_adapter_tampered"
    pipe.save_artifact(artifact_dir)

    # Tamper with the weights file
    weights_path = artifact_dir / pl.ARTIFACT_WEIGHTS_NAME
    content = bytearray(weights_path.read_bytes())
    content[-1] ^= 0xFF
    weights_path.write_bytes(content)

    fresh = _pipeline(monkeypatch)
    with pytest.raises(ValueError, match="weights digest mismatch"):
        fresh.load_adapter(artifact_dir)


def test_artifact_refuses_mismatched_model_identity(tmp_path, monkeypatch):
    pipe = _pipeline(monkeypatch)
    records = synthetic_records(4)
    pipe.adapt(train=records, epochs=1, seed=0)

    artifact_dir = tmp_path / "kandinsky_adapter_wrong_id"
    pipe.save_artifact(artifact_dir)

    # Alter manifest modelId
    manifest_path = artifact_dir / pl.ARTIFACT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["base_model"]["id"] = "different/model"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    fresh = _pipeline(monkeypatch)
    with pytest.raises(ValueError, match="does not match"):
        fresh.load_adapter(artifact_dir)
