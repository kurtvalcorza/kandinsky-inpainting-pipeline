"""DIMER-oriented pipeline for Kandinsky 2.2 inpainting (decoder inpaint + shared prior).

Exposes:
- Snapshot verification and staging at immutable revisions
- Inpaint UNet construction with rank-8 LoRA adapter injection
- Lazy prior pipeline execution with prompt-embedding cache
- Inpainting generation with unmasked preservation guarantees
- Held-out evaluation reporting denoising MSE, PSNR and SSIM on unmasked regions
- Bounded LoRA fine-tuning and portable safetensors artifact serialization
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import compute_psnr, compute_ssim

MODEL_ID = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
MODEL_REVISION = "db790ad5cbcabed886f069ef2710774657621702"
MODEL_KEY = "kandinsky-2-2-decoder-inpaint"

PRIOR_ID = "kandinsky-community/kandinsky-2-2-prior"
PRIOR_REVISION = "9fc51ad5732afc5d031724219d22e6c42179c5a8"
PRIOR_KEY = "kandinsky-2-2-prior"

SCORER_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
SCORER_REVISION = "1a25a446712ba5ee05982a381eed697ef9b435cf"
SCORER_KEY = "clip-vit-b-32-laion2b"

MODEL_LICENSE = "apache-2.0"
ARTIFACT_FORMAT = "org.valcorza.kandinsky-inpainting.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
MANIFEST_NAME = "dimer-base-manifest.json"

_WEIGHTS_ROOT = Path(__file__).resolve().parents[2] / "weights"
DEFAULT_WEIGHTS_DIR = _WEIGHTS_ROOT / MODEL_KEY
DEFAULT_PRIOR_DIR = _WEIGHTS_ROOT / PRIOR_KEY
DEFAULT_SCORER_DIR = _WEIGHTS_ROOT / SCORER_KEY

UNET_PARAMETERS = 1253074568
UNET_TENSORS = 724
LORA_RANK = 8
LORA_ALPHA = 8
LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
LORA_TENSORS = 176
LORA_PARAMETERS = 1646592

DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 4.0
MAX_STEPS = 100
MAX_GUIDANCE = 20.0
RESOLUTION = 512
LATENT_CHANNELS = 4
MAX_CAPTION_CHARS = 1_000
MIN_IMAGE_SIDE = 256
MAX_IMAGE_SIDE = 4_096
MIN_TRAIN_RECORDS = 4
MAX_RECORDS = 2_000
NUM_TRAIN_TIMESTEPS = 1_000
EVAL_TIMESTEPS = (100, 300, 500, 700, 900)
NEGATIVE_PROMPT = ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_seed(prompt: str) -> int:
    """Seed for the prior's sampler: the first 4 bytes of SHA-256(prompt), masked to 31 bits.

    Stable across processes, unlike ``hash(prompt)``, which Python salts per interpreter.
    """
    return int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF


def _read_manifest(dir_path: Path) -> dict[str, Any]:
    manifest_path = dir_path / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest = _read_manifest(root)
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    for entry in manifest.get("files", []):
        relative_path = entry["path"]
        target = root / relative_path
        if not target.is_file():
            raise FileNotFoundError(f"snapshot file missing: {target}")
        if target.stat().st_size != entry["bytes"]:
            raise ValueError(f"{relative_path}: size {target.stat().st_size} != manifest {entry['bytes']}")
        digest = _sha256_file(target)
        if digest != entry["sha256"]:
            raise ValueError(f"{relative_path}: sha256 {digest} != manifest {entry['sha256']}")
        if not relative_path.endswith((".safetensors", ".json", ".model", ".txt", ".md")):
            raise ValueError(f"{relative_path}: unexpected file type in a code-free snapshot")
    return manifest


def verify_snapshot(weights_dir: str | Path = DEFAULT_WEIGHTS_DIR) -> dict[str, Any]:
    """Verify that all files in the decoder inpaint snapshot exist and match recorded hashes."""
    return _verify_manifest(Path(weights_dir), MODEL_ID, MODEL_REVISION)


def verify_prior_snapshot(prior_dir: str | Path = DEFAULT_PRIOR_DIR) -> dict[str, Any]:
    """Verify that all files in the prior snapshot exist and match recorded hashes."""
    return _verify_manifest(Path(prior_dir), PRIOR_ID, PRIOR_REVISION)


def verify_scorer_snapshot(scorer_dir: str | Path = DEFAULT_SCORER_DIR) -> dict[str, Any]:
    """Verify that all files in the CLIP scorer snapshot exist and match recorded hashes."""
    return _verify_manifest(Path(scorer_dir), SCORER_ID, SCORER_REVISION)


def _hub_download(relative_path: str, root: Path, model_id: str, revision: str) -> None:
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(model_id, relative_path, revision=revision)
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, target)


def _stage_missing(
    root: Path,
    model_id: str,
    revision: str,
    allow_download: bool,
    downloader: Callable[[str, Path], None] | None,
) -> list[str]:
    manifest = _read_manifest(root)
    if manifest.get("modelId") != model_id or manifest.get("revision") != revision:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {model_id}@{revision}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest.get("files", []) if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them")
    fetch = downloader or (lambda relative_path, destination: _hub_download(relative_path, destination, model_id, revision))
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def stage_missing_files(
    weights_dir: str | Path = DEFAULT_WEIGHTS_DIR,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Stage missing files for the inpaint decoder snapshot."""
    return _stage_missing(Path(weights_dir), MODEL_ID, MODEL_REVISION, allow_download, downloader)


def stage_missing_prior_files(
    prior_dir: str | Path = DEFAULT_PRIOR_DIR,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Stage missing files for the prior snapshot."""
    return _stage_missing(Path(prior_dir), PRIOR_ID, PRIOR_REVISION, allow_download, downloader)


def stage_missing_scorer_files(
    scorer_dir: str | Path = DEFAULT_SCORER_DIR,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Stage missing files for the CLIP scorer snapshot."""
    return _stage_missing(Path(scorer_dir), SCORER_ID, SCORER_REVISION, allow_download, downloader)


INPUT_SCHEMA: dict[str, Any] = {
    "record": "{id, image, mask_image?, caption}: an RGB image, optional inpainting mask, and prompt",
    "image_side": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "resolution": RESOLUTION,
    "preprocessing": (
        f"image and mask are resized together so the shorter side is {RESOLUTION} px and centre-cropped "
        f"to {RESOLUTION}x{RESOLUTION}"
    ),
    "caption_chars": [1, MAX_CAPTION_CHARS],
    "records": [MIN_TRAIN_RECORDS, MAX_RECORDS],
    "generation": {"steps": [1, MAX_STEPS], "guidance_scale": [1.0, MAX_GUIDANCE], "size": RESOLUTION},
    "validation": (
        "record shape, image decodability and side limits, optional mask shape, caption length and duplicate ids only. "
        "Nothing checks semantic alignment: any RGB image with any string is accepted"
    ),
}


def _load_image(value: Any, *, mode: str, label: str) -> Any:
    from PIL import Image

    image = value
    if isinstance(value, str | Path):
        path = Path(value)
        if not path.is_file():
            raise ValueError(f"{label} file not found: {path}")
        image = Image.open(path)
        image.load()
    if not isinstance(image, Image.Image):
        raise ValueError(f"{label} must be a PIL.Image.Image or a file path")
    return image.convert(mode)


def create_center_mask(image_or_size: Any) -> Any:
    """Create a binary center-box mask covering the middle half of an image."""
    from PIL import Image, ImageDraw

    size = image_or_size if isinstance(image_or_size, tuple) else image_or_size.size
    width, height = size
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rectangle((width // 4, height // 4, 3 * width // 4, 3 * height // 4), fill=255)
    return mask


def unet_keep_mask(inpaint_mask: Any) -> Any:
    """Convert an inpaint mask (1 = region to repaint, the repository convention) into the Kandinsky 2.2 inpainting
    UNet's mask channel (1 = region to keep).

    The repository convention (255 / 1 = repaint) is what records, ``create_center_mask`` and ``generate`` accept.
    The UNet expects the opposite convention: diffusers v0.40.0
    ``pipelines/kandinsky2_2/pipeline_kandinsky2_2_inpainting.py`` inverts the user mask (``mask = 1 - mask``,
    L238), multiplies the image latents by it (L452) and concatenates it as the last UNet input channel (L479).
    Works on NumPy arrays and torch tensors alike.
    """
    return 1.0 - inpaint_mask


def _check_record(record: Any, index: int) -> dict[str, Any]:
    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a mapping with id/image/caption")
    for key in ("id", "image", "caption"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    record_id = record["id"]
    if not isinstance(record_id, str) or not record_id or len(record_id) > 64:
        raise ValueError(f"{label}: id must be a non-empty string of at most 64 characters")
    image = _load_image(record["image"], mode="RGB", label=f"{label}: image")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE or max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"{label}: image sides must be within {MIN_IMAGE_SIDE}..{MAX_IMAGE_SIDE} px, got {image.size}")
    caption = record["caption"]
    if not isinstance(caption, str) or not caption.strip() or len(caption) > MAX_CAPTION_CHARS:
        raise ValueError(f"{label}: caption must be a non-empty string of at most {MAX_CAPTION_CHARS} characters")
    mask = (
        _load_image(record["mask_image"], mode="L", label=f"{label}: mask_image")
        if "mask_image" in record
        else create_center_mask(image)
    )
    if mask.size != image.size:
        raise ValueError(f"{label}: mask_image size {mask.size} must match image size {image.size}")
    item = {"id": record_id, "image": image, "mask_image": mask, "caption": caption.strip()}
    for key in ("label", "common_name", "scientific_name", "observer", "inat_photo_id", "inat_observation_url", "source_id"):
        if key in record:
            item[key] = record[key]
    return item


def image_digest(image: Any) -> str:
    """Return a stable SHA-256 over decoded RGB pixels and dimensions."""
    rgb = image.convert("RGB")
    return hashlib.sha256(f"{rgb.size[0]}x{rgb.size[1]}:".encode() + rgb.tobytes()).hexdigest()


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = []
    for record in records:
        mask = record.get("mask_image")
        mask_digest = hashlib.sha256(mask.convert("L").tobytes()).hexdigest() if mask is not None else None
        payload.append([record["id"], image_digest(record["image"]), mask_digest, record["caption"]])
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_TRAIN_RECORDS,
    max_records: int = MAX_RECORDS,
) -> dict[str, Any]:
    """Validate inpainting records before importing any model dependency."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, image, caption} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    crops = 0
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        crops += item["image"].size[0] != item["image"].size[1]
        checked.append(item)
    sides = [min(item["image"].size) for item in checked]
    return {
        "records": checked,
        "n_records": len(checked),
        "n_captions": len({item["caption"] for item in checked}),
        "shorter_side": {"min": min(sides), "max": max(sides)},
        "centre_cropped": crops,
        "resolution": RESOLUTION,
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def validate_inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record and report its deterministic preprocessing shape."""
    item = _check_record(record, 0)
    width, height = item["image"].size
    scale = RESOLUTION / min(width, height)
    return {
        "id": item["id"],
        "size": (width, height),
        "resized_to": (round(width * scale), round(height * scale)),
        "centre_crop": (RESOLUTION, RESOLUTION),
        "caption_chars": len(item["caption"]),
    }


def validate_prompts(prompts: Sequence[str]) -> list[str]:
    """Validate and normalize generation prompts."""
    if isinstance(prompts, str) or not isinstance(prompts, Sequence) or not prompts:
        raise ValueError("prompts must be a non-empty list of strings")
    normalized = []
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_CAPTION_CHARS:
            raise ValueError(f"prompts[{index}] must be a non-empty string of at most {MAX_CAPTION_CHARS} characters")
        normalized.append(prompt.strip())
    return normalized


def _resize_and_crop(image: Any, *, target_size: int, resample: Any) -> Any:
    width, height = image.size
    scale = target_size / min(width, height)
    resized_size = (max(target_size, round(width * scale)), max(target_size, round(height * scale)))
    resized = image.resize(resized_size, resample)
    left = (resized_size[0] - target_size) // 2
    top = (resized_size[1] - target_size) // 2
    return resized.crop((left, top, left + target_size, top + target_size))


def preprocess_image(image: Any, *, target_size: int = RESOLUTION) -> Any:
    """Resize the shorter side and center-crop an image to a square RGB image."""
    from PIL import Image

    return _resize_and_crop(image.convert("RGB"), target_size=target_size, resample=Image.Resampling.BICUBIC)


def preprocess_image_and_mask(image: Any, mask_image: Any, *, target_size: int = RESOLUTION) -> tuple[Any, Any]:
    """Apply identical resize/crop geometry to an RGB image and its binary mask."""
    from PIL import Image

    rgb = image.convert("RGB")
    mask = mask_image.convert("L")
    if rgb.size != mask.size:
        raise ValueError(f"mask_image size {mask.size} must match image size {rgb.size}")
    return (
        _resize_and_crop(rgb, target_size=target_size, resample=Image.Resampling.BICUBIC),
        _resize_and_crop(mask, target_size=target_size, resample=Image.Resampling.NEAREST),
    )


def synthetic_image(*, width: int = 640, height: int = 480, seed: int = 0) -> Any:
    """Create a deterministic RGB image for examples and smoke tests."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def synthetic_records(n: int = 8, *, seed: int = 0) -> list[dict[str, Any]]:
    """Create deterministic inpainting records for local examples."""
    records = []
    for index in range(n):
        image = synthetic_image(seed=seed + index)
        records.append(
            {
                "id": f"rec-{index:03d}",
                "image": image,
                "mask_image": create_center_mask(image),
                "caption": "a photo of a red bird" if index % 2 == 0 else "a photo of a blue bird",
            }
        )
    return records


def _lora_config() -> Any:
    from peft import LoraConfig

    return LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, init_lora_weights="gaussian", target_modules=list(LORA_TARGETS))


def lora_parameter_names(unet: Any) -> list[str]:
    """Return sorted list of attached LoRA parameter names."""
    return sorted(name for name, _p in unet.named_parameters() if ".lora_A." in name or ".lora_B." in name)


def build_unet(weights_dir: Path, *, dtype: Any, use_lora: bool) -> Any:
    """Build UNet2DConditionModel from weights_dir, optionally attaching LoRA."""
    from diffusers import UNet2DConditionModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = UNet2DConditionModel.from_pretrained(str(weights_dir), subfolder="unet", torch_dtype=dtype)

    n_params = sum(p.numel() for p in model.parameters())
    if n_params != UNET_PARAMETERS or len(model.state_dict()) != UNET_TENSORS:
        raise ValueError(
            f"unet parameter mismatch: got {n_params} parameters in {len(model.state_dict())} tensors, "
            f"expected {UNET_PARAMETERS} / {UNET_TENSORS}"
        )
    for p in model.parameters():
        p.requires_grad_(False)

    if use_lora:
        from peft import inject_adapter_in_model

        inject_adapter_in_model(_lora_config(), model, adapter_name="default")
        names = lora_parameter_names(model)
        if len(names) != LORA_TENSORS:
            raise ValueError(f"adapter attached {len(names)} LoRA tensors, expected {LORA_TENSORS}")
        import torch

        name_set = set(names)
        for name, param in model.named_parameters():
            if name in name_set:
                # LoRA tensors are kept in float32 so AdamW state and updates do not underflow in float16.
                param.data = param.data.to(torch.float32)
                param.requires_grad_(True)
                if ".lora_B." in name:
                    param.data.zero_()
            else:
                param.requires_grad_(False)
    return model


class KandinskyInpaintPipeline:
    """DIMER pipeline for Kandinsky 2.2 Inpainting with optional LoRA adaptation."""

    def __init__(
        self,
        unet: Any,
        movq: Any,
        scheduler_config: dict[str, Any],
        *,
        device: str,
        dtype: Any,
        prior_dir: Path,
        weights_dir: Path,
        source: str = "local-snapshot (verified safetensors)",
        use_lora: bool = False,
    ):
        self.unet = unet
        self.movq = movq
        self.scheduler_config = scheduler_config
        self.device = device
        self.dtype = dtype
        self.prior_dir = prior_dir
        self.weights_dir = weights_dir
        self.source = source
        self.use_lora = use_lora
        self.adapter: dict[str, Any] | None = None
        self._prior_pipeline = None
        self._prompt_cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_pretrained(
        cls,
        weights_dir: str | Path = DEFAULT_WEIGHTS_DIR,
        prior_dir: str | Path = DEFAULT_PRIOR_DIR,
        *,
        device: str | None = None,
        use_lora: bool = False,
        allow_download: bool = False,
    ) -> KandinskyInpaintPipeline:
        """Instantiate pipeline from verified snapshot directories."""
        import torch
        from diffusers import VQModel

        w_path = Path(weights_dir)
        p_path = Path(prior_dir)

        if allow_download:
            stage_missing_files(w_path, allow_download=True)
            stage_missing_prior_files(p_path, allow_download=True)

        verify_snapshot(w_path)
        verify_prior_snapshot(p_path)

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if dev == "cuda" else torch.float32

        unet = build_unet(w_path, dtype=dtype, use_lora=use_lora).to(dev)
        movq = VQModel.from_pretrained(str(w_path), subfolder="movq", torch_dtype=dtype).to(dev).eval()
        for p in movq.parameters():
            p.requires_grad_(False)

        sched_path = w_path / "scheduler" / "scheduler_config.json"
        scheduler_config = json.loads(sched_path.read_text(encoding="utf-8"))

        return cls(
            unet=unet,
            movq=movq,
            scheduler_config=scheduler_config,
            device=dev,
            dtype=dtype,
            prior_dir=p_path,
            weights_dir=w_path,
            use_lora=use_lora,
        )

    def _get_prior(self) -> Any:
        if self._prior_pipeline is not None:
            return self._prior_pipeline
        import warnings

        from diffusers import KandinskyV22PriorPipeline

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._prior_pipeline = KandinskyV22PriorPipeline.from_pretrained(
                str(self.prior_dir),
                torch_dtype=self.dtype,
            ).to(self.device)
        return self._prior_pipeline

    def encode_prompts(self, prompts: Sequence[str]) -> dict[str, Any]:
        """Encode prompts with the prior into CLIP image embeddings.

        Each prompt's prior sampler is seeded with ``prompt_seed(prompt)``, so a prompt's embedding does not depend
        on the other prompts encoded with it or on Python's per-process string-hash salt.
        """
        distinct = sorted(set([NEGATIVE_PROMPT, *validate_prompts(prompts)]))
        missing = [p for p in distinct if p not in self._prompt_cache]
        if not missing:
            return {"n_cached": len(self._prompt_cache), "n_new": 0}
        import torch

        prior = self._get_prior()
        prior.set_progress_bar_config(disable=True)
        for prompt in missing:
            generator = torch.Generator(device=self.device).manual_seed(prompt_seed(prompt))
            out = prior(prompt=prompt, num_inference_steps=25, generator=generator)
            self._prompt_cache[prompt] = {
                "image_embeds": out.image_embeds[0].detach().to("cpu", self.dtype),
                "negative_image_embeds": out.negative_image_embeds[0].detach().to("cpu", self.dtype),
            }
        return {"n_cached": len(self._prompt_cache), "n_new": len(missing)}

    def release_prior(self) -> bool:
        """Release the prior pipeline to reclaim GPU VRAM."""
        import torch

        if self._prior_pipeline is None:
            return False
        del self._prior_pipeline
        self._prior_pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    def export_prompt_cache(self) -> dict[str, Any]:
        return {
            prompt: {
                "image_embeds": entry["image_embeds"].clone(),
                "negative_image_embeds": entry["negative_image_embeds"].clone(),
            }
            for prompt, entry in self._prompt_cache.items()
        }

    def import_prompt_cache(self, cache: Mapping[str, Mapping[str, Any]]) -> int:
        for prompt, entry in cache.items():
            if "image_embeds" not in entry or "negative_image_embeds" not in entry:
                raise ValueError(f"prompt cache entry for {prompt[:40]!r} missing required embedding keys")
            self._prompt_cache[prompt] = {
                "image_embeds": entry["image_embeds"],
                "negative_image_embeds": entry["negative_image_embeds"],
            }
        return len(self._prompt_cache)

    def _embeds(self, prompt: str) -> tuple[Any, Any]:
        if prompt not in self._prompt_cache:
            raise ValueError(f"prompt not encoded; call encode_prompts([...]) first: {prompt[:60]!r}")
        entry = self._prompt_cache[prompt]
        return (
            entry["image_embeds"].to(self.device, self.dtype),
            entry["negative_image_embeds"].to(self.device, self.dtype),
        )

    def _diffusers_pipeline(self) -> Any:
        from diffusers import DDPMScheduler
        from diffusers import KandinskyV22InpaintPipeline as _Upstream

        return _Upstream(
            unet=self.unet,
            scheduler=DDPMScheduler.from_config(self.scheduler_config),
            movq=self.movq,
        )

    def generate(
        self,
        records: Sequence[dict[str, Any]],
        *,
        seed: int = 0,
        steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
    ) -> dict[str, Any]:
        """Run inpainting on `{image, mask_image, caption}` records."""
        import torch

        prompts = [r["caption"] for r in records]
        if any(p not in self._prompt_cache for p in [NEGATIVE_PROMPT, *prompts]):
            self.encode_prompts(prompts)

        pipe = self._diffusers_pipeline()
        pipe.set_progress_bar_config(disable=True)
        started = time.perf_counter()
        results = []

        for idx, rec in enumerate(records):
            img, mask = preprocess_image_and_mask(rec["image"], rec["mask_image"], target_size=RESOLUTION)
            prompt = rec["caption"]
            img_emb, neg_emb = self._embeds(prompt)
            generator = torch.Generator(device=self.device).manual_seed(seed + idx)

            with torch.inference_mode():
                out = pipe(
                    prompt=prompt,
                    image=img,
                    mask_image=mask,
                    image_embeds=img_emb[None],
                    negative_image_embeds=neg_emb[None],
                    height=RESOLUTION,
                    width=RESOLUTION,
                    num_inference_steps=steps,
                    guidance_scale=float(guidance_scale),
                    generator=generator,
                    output_type="pil",
                )
            gen_img = out.images[0]
            unmasked = (np.asarray(mask) == 0)
            psnr = compute_psnr(img, gen_img, mask=unmasked)
            ssim = compute_ssim(img, gen_img, mask=unmasked)

            results.append({
                "id": rec.get("id", f"sample-{idx}"),
                "prompt": prompt,
                "image": gen_img,
                "unmasked_psnr_db": round(psnr, 2),
                "unmasked_ssim": round(ssim, 4),
            })

        return {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "adapted": self.adapter is not None},
            "steps": steps,
            "guidance_scale": float(guidance_scale),
            "size": (RESOLUTION, RESOLUTION),
            "results": results,
            "seconds": round(time.perf_counter() - started, 2),
        }

    def _conditioning(self, records: Sequence[Mapping[str, Any]]) -> tuple[Any, Any, Any]:
        """Encode images and masks into latents, masked latents and the latent-resolution inpaint mask.

        Returns ``(latents, masked_latents, inpaint_mask_latents)``: ``masked_latents`` keeps the latents outside the
        repaint region (zero inside it), and ``inpaint_mask_latents`` is 1 inside the repaint region (repository
        convention). ``_predict_noise`` converts it to the UNet's keep-mask channel with ``unet_keep_mask``.
        """
        import torch
        import torch.nn.functional as F

        images = []
        masks = []
        for record in records:
            image, mask = preprocess_image_and_mask(record["image"], record["mask_image"], target_size=RESOLUTION)
            images.append(np.asarray(image, dtype=np.float32) / 127.5 - 1.0)
            masks.append(np.asarray(mask, dtype=np.float32) / 255.0)
        pixels = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(self.device, self.dtype)
        mask_pixels = torch.from_numpy(np.stack(masks))[:, None].to(self.device, self.dtype)
        with torch.no_grad():
            encoded = self.movq.encode(pixels)
            latents = encoded.latents if hasattr(encoded, "latents") else encoded["latents"]
        mask_latents = F.interpolate(mask_pixels, size=latents.shape[-2:], mode="nearest")
        masked_latents = latents * (1.0 - mask_latents)
        return latents.to(self.dtype), masked_latents.to(self.dtype), mask_latents.to(self.dtype)

    def _noise_scheduler(self) -> Any:
        from diffusers import DDPMScheduler

        return DDPMScheduler.from_config(self.scheduler_config)

    def _predict_noise(
        self,
        noisy: Any,
        timesteps: Any,
        image_embeds: Any,
        masked_latents: Any,
        inpaint_mask_latents: Any,
    ) -> Any:
        """Predict noise from the nine-channel input ``[noisy latents, masked latents, keep mask]``: the channel order
        and keep-mask convention of diffusers v0.40.0 ``KandinskyV22InpaintPipeline`` (L452, L479)."""
        import torch

        keep = unet_keep_mask(inpaint_mask_latents).to(noisy.dtype)
        model_input = torch.cat([noisy, masked_latents.to(noisy.dtype), keep], dim=1)
        output = self.unet(
            sample=model_input,
            timestep=timesteps,
            encoder_hidden_states=None,
            added_cond_kwargs={"image_embeds": image_embeds},
            return_dict=False,
        )
        prediction = output[0] if isinstance(output, tuple) else output.sample
        return prediction[:, :LATENT_CHANNELS]

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        seed: int = 0,
        batch_size: int = 4,
    ) -> dict[str, Any]:
        """Evaluate seeded, paired denoising MSE across the fixed timestep schedule."""
        import torch

        checked = validate_dataset(records, min_records=1)["records"]
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an int in 1..32")
        self.encode_prompts([record["caption"] for record in checked])
        scheduler = self._noise_scheduler()
        per_timestep: dict[int, list[float]] = {timestep: [] for timestep in EVAL_TIMESTEPS}
        per_record: dict[str, float] = {}
        self.unet.eval()
        for start in range(0, len(checked), batch_size):
            batch = checked[start : start + batch_size]
            latents, masked_latents, mask_latents = self._conditioning(batch)
            image_embeds = torch.stack([self._embeds(record["caption"])[0] for record in batch])
            record_losses = [0.0] * len(batch)
            for timestep in EVAL_TIMESTEPS:
                generator = torch.Generator(device="cpu").manual_seed(seed * 1_000 + timestep + start)
                noise = torch.randn(latents.shape, generator=generator).to(self.device, self.dtype)
                timesteps = torch.full((len(batch),), timestep, device=self.device, dtype=torch.long)
                noisy = scheduler.add_noise(latents.float(), noise.float(), timesteps).to(self.dtype)
                use_amp = self.dtype == torch.float16
                autocast = torch.autocast(device_type=self.device.split(":")[0], dtype=torch.float16, enabled=use_amp)
                with torch.inference_mode(), autocast:
                    prediction = self._predict_noise(noisy, timesteps, image_embeds, masked_latents, mask_latents)
                losses = ((prediction.float() - noise.float()) ** 2).mean(dim=(1, 2, 3))
                for index, value in enumerate(losses.tolist()):
                    per_timestep[timestep].append(value)
                    record_losses[index] += value / len(EVAL_TIMESTEPS)
            for record, value in zip(batch, record_losses, strict=True):
                per_record[record["id"]] = round(value, 6)
        by_timestep = {
            str(timestep): round(sum(values) / len(values), 6) for timestep, values in per_timestep.items()
        }
        return {
            "metric": "denoising_mse",
            "n_records": len(checked),
            "timesteps": list(EVAL_TIMESTEPS),
            "seed": seed,
            "denoising_mse": round(sum(per_record.values()) / len(per_record), 6),
            "by_timestep": by_timestep,
            "per_record": per_record,
            "adapted": self.adapter is not None,
        }

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 4,
        lr: float = 1e-4,
        batch_size: int = 1,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Train only the bounded LoRA tensor set on the inpainting noise-prediction objective.

        AdamW on the float32 LoRA tensors, float16 autocast with a ``GradScaler`` on CUDA (float32 on CPU), loss in
        float32, gradient-norm clipping at 1.0. Epoch 0 records the frozen model; the lowest validation loss is kept.
        """
        if not self.use_lora:
            raise ValueError("adapt() needs a pipeline built with use_lora=True")
        if not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not 0.0 < lr <= 1e-2:
            raise ValueError("lr must be in (0, 1e-2]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 8:
            raise ValueError("batch_size must be an int in 1..8")
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None

        import torch

        prompts = [record["caption"] for record in train_checked]
        if val_checked:
            prompts.extend(record["caption"] for record in val_checked)
        self.encode_prompts(prompts)
        torch.manual_seed(seed)
        started = time.perf_counter()
        model = self.unet
        names = lora_parameter_names(model)
        name_set = set(names)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in name_set)
        parameters = [parameter for name, parameter in model.named_parameters() if name in name_set]
        n_trainable = sum(parameter.numel() for parameter in parameters)
        if n_trainable != LORA_PARAMETERS:
            raise ValueError(f"{n_trainable} trainable parameters, expected {LORA_PARAMETERS}")
        initial_state = {key: value.detach().clone() for key, value in model.state_dict().items() if key in name_set}
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
        use_amp = self.dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        scheduler = self._noise_scheduler()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        conditioning_by_id = {}
        for start in range(0, len(train_checked), 4):
            batch = train_checked[start : start + 4]
            latents, masked_latents, mask_latents = self._conditioning(batch)
            for index, record in enumerate(batch):
                conditioning_by_id[record["id"]] = (
                    latents[index],
                    masked_latents[index],
                    mask_latents[index],
                )

        try:
            history: list[dict[str, Any]] = []
            baseline: dict[str, Any] = {
                "epoch": 0,
                "train_loss": None,
                "note": "frozen model (LoRA at initialization: B = 0)",
                "val_loss": self.evaluate(val_checked, seed=seed)["denoising_mse"] if val_checked else None,
            }
            history.append(baseline)
            if progress:
                progress(baseline)
            best_val = baseline["val_loss"] if baseline["val_loss"] is not None else math.inf
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items() if key in name_set}
            best_epoch = 0
            n_steps = 0
            for epoch in range(1, epochs + 1):
                model.train()
                order = torch.randperm(len(train_checked), generator=generator).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    batch = [train_checked[index] for index in order[start : start + batch_size]]
                    latents = torch.stack([conditioning_by_id[record["id"]][0] for record in batch])
                    masked_latents = torch.stack([conditioning_by_id[record["id"]][1] for record in batch])
                    mask_latents = torch.stack([conditioning_by_id[record["id"]][2] for record in batch])
                    image_embeds = torch.stack([self._embeds(record["caption"])[0] for record in batch])
                    noise = torch.randn(latents.shape, generator=generator).to(self.device, self.dtype)
                    timesteps = torch.randint(0, NUM_TRAIN_TIMESTEPS, (len(batch),), generator=generator).to(self.device)
                    noisy = scheduler.add_noise(latents.float(), noise.float(), timesteps).to(self.dtype)
                    with torch.autocast(device_type=self.device.split(":")[0], dtype=torch.float16, enabled=use_amp):
                        prediction = self._predict_noise(noisy, timesteps, image_embeds, masked_latents, mask_latents)
                    loss = torch.nn.functional.mse_loss(prediction.float(), noise.float())
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    losses.append(float(loss.detach()))
                    n_steps += 1
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses)}
                entry["val_loss"] = self.evaluate(val_checked, seed=seed)["denoising_mse"] if val_checked else None
                history.append(entry)
                if progress:
                    progress(entry)
                if entry["val_loss"] is None or entry["val_loss"] < best_val:
                    best_val = entry["val_loss"] if entry["val_loss"] is not None else best_val
                    best_state = {key: value.detach().clone() for key, value in model.state_dict().items() if key in name_set}
                    best_epoch = epoch
        except BaseException:
            restored = dict(model.state_dict())
            restored.update(initial_state)
            model.load_state_dict(restored, strict=True)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.adapter = None
            raise

        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.adapter = {
            "method": "LoRA (peft)",
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "targets": list(LORA_TARGETS),
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(parameter.numel() for parameter in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "lr": lr,
            "batch_size": batch_size,
            "optimizer": "AdamW (weight_decay 0, grad-norm clip 1.0)",
            "precision": "float16 autocast + GradScaler" if use_amp else "float32",
        }
        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val if best_val != math.inf else None,
            "adapter": self.adapter,
            "steps": n_steps,
            "seconds": round(time.perf_counter() - started, 2),
        }

    def save_artifact(self, output_dir: str | Path, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Save the trained LoRA tensor set as safetensors plus a digest-bound manifest."""
        import safetensors.torch

        if self.adapter is None:
            raise ValueError("no adapter attached; call adapt() or load_adapter() first")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {key: value.detach().cpu() for key, value in self.unet.state_dict().items() if key in names}
        weights_file = output / ARTIFACT_WEIGHTS_NAME
        safetensors.torch.save_file(tensors, weights_file)
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY},
            "adapter": self.adapter,
            "weights": {
                "file": ARTIFACT_WEIGHTS_NAME,
                "bytes": weights_file.stat().st_size,
                "sha256": _sha256_file(weights_file),
                "n_tensors": len(tensors),
                "n_parameters": sum(tensor.numel() for tensor in tensors.values()),
            },
            "metadata": dict(metadata) if metadata else {},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (output / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    save_adapter = save_artifact

    def load_adapter(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Load and verify an adapter produced by `save_artifact`."""
        import safetensors.torch

        if not self.use_lora:
            raise ValueError("load_adapter() needs a pipeline built with use_lora=True")
        source = Path(artifact_dir)
        manifest_file = source / ARTIFACT_MANIFEST_NAME
        weights_file = source / ARTIFACT_WEIGHTS_NAME
        if not manifest_file.is_file() or not weights_file.is_file():
            raise FileNotFoundError(f"not an adapter directory: {source}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"unexpected artifact format: {manifest.get('format')!r}")
        base_model = manifest.get("base_model", {})
        if base_model.get("id") != MODEL_ID or base_model.get("revision") != MODEL_REVISION:
            raise ValueError(f"adapter base model {base_model} does not match {MODEL_ID}@{MODEL_REVISION}")
        digest = _sha256_file(weights_file)
        if digest != manifest["weights"]["sha256"]:
            raise ValueError(f"weights digest mismatch: {digest} != manifest {manifest['weights']['sha256']}")
        tensors = safetensors.torch.load_file(weights_file)
        current = dict(self.unet.state_dict())
        current.update({key: value.to(self.device, current[key].dtype) for key, value in tensors.items()})
        self.unet.load_state_dict(current, strict=True)
        self.adapter = manifest["adapter"]
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        weights_dir: str | Path = DEFAULT_WEIGHTS_DIR,
        prior_dir: str | Path = DEFAULT_PRIOR_DIR,
        device: str | None = None,
    ) -> KandinskyInpaintPipeline:
        """Instantiate a verified pipeline and load an adapter artifact."""
        pipe = cls.from_pretrained(weights_dir=weights_dir, prior_dir=prior_dir, device=device, use_lora=True)
        pipe.load_adapter(artifact_dir)
        return pipe
