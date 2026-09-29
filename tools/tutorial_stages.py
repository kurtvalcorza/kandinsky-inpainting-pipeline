"""Stage runner for the standalone Kandinsky 2.2 inpainting tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated-environment pattern).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --weights WEIGHTS_DIR --stage prepare [--byod PATH]

Nothing is installed into the notebook kernel. Each stage is a separate process, so a stage starts from files only:
the verified snapshots under ``--weights``, the dataset recorded by ``prepare``, the prompt-embedding cache written by
``encode``, the adapter artifact written by ``adapt`` and the JSON records of earlier stages. Learner-facing exports go
to ``RUN_DIR/outputs``; hand-off state goes to ``RUN_DIR/state``. On failure a stage writes
``RUN_DIR/state/<stage>.error.json`` with the exception type and message, which the notebook re-raises in the kernel.

Stages: weights → prepare → encode → frozen → adapt → evaluate → reload, plus the optional ``activity``.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

STEM = "kandinsky_inpainting"
EVAL_SEED = 0
GENERATION_SEED = 1000
NEW_PROMPT_SEED = 2000
PARITY_SEED = 3000
NEW_PROMPT = "a photo of a House Finch (Haemorhous mexicanus) perched on a snow-covered branch in winter"
SNAPSHOT_KEYS = ("kandinsky-2-2-decoder-inpaint", "kandinsky-2-2-prior", "clip-vit-b-32-laion2b")
PRIOR_LICENSE = "apache-2.0"
SCORER_LICENSE = "mit"
SAMPLE_CACHE = "inat-birds"
PROMPT_CACHE = "prompt_cache.safetensors"
PROMPT_INDEX = "prompt_cache.json"
MASK_SIDES = {"small": 0.25, "large": 0.75}
CENTRE_REPAINT_FRACTION = 0.25


# --------------------------------------------------------------------------------------------------
# run context and small helpers
# --------------------------------------------------------------------------------------------------


class Run:
    """Paths of one run: carried sources and state under ``root``, snapshots under ``weights``."""

    def __init__(self, root: Path, weights: Path, options: argparse.Namespace) -> None:
        self.root = root
        self.weights = weights
        self.options = options
        self.out = root / "outputs"
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def snapshot(self, key: str) -> Path:
        return self.weights / key

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from the top)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path


def gpu_memory_gb() -> float | None:
    import torch

    return round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None


def prepared(records: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    """The 512 × 512 photographs and masks exactly as the pipeline sees them."""
    from kandinsky_inpainting_pipeline import preprocess_image_and_mask

    pairs = [preprocess_image_and_mask(r["image"], r["mask_image"]) for r in records]
    return [image for image, _ in pairs], [mask for _, mask in pairs]


def mean_fill(image: Any, mask: Any) -> Any:
    """The mean-fill floor: the repaint region filled with the mean colour of the kept region (no model at all)."""
    import numpy as np
    from PIL import Image

    pixels = np.asarray(image).copy()
    repaint = np.asarray(mask) > 0
    pixels[repaint] = pixels[~repaint].mean(axis=0).astype(np.uint8)
    return Image.fromarray(pixels)


def masked_view(image: Any, mask: Any) -> Any:
    import numpy as np
    from PIL import Image

    pixels = np.asarray(image).copy()
    pixels[np.asarray(mask) > 0] = 128
    return Image.fromarray(pixels)


def triptych_grid(images: list[Any], masks: list[Any], outputs: list[Any], path: Path, rows: int = 6) -> Path:
    """Photograph, masked view and output side by side, one row per record."""
    from PIL import Image

    sheet = Image.new("RGB", (256 * 3, 256 * min(rows, len(images))), "white")
    for i, (image, mask, output) in enumerate(list(zip(images, masks, outputs, strict=True))[:rows]):
        for j, tile in enumerate((image, masked_view(image, mask), output)):
            sheet.paste(tile.resize((256, 256)), (256 * j, 256 * i))
    sheet.save(path)
    return path


def box_mask(image: Any, side_fraction: float) -> Any:
    """A centred box mask whose side is ``side_fraction`` of each image side (white = repaint)."""
    from PIL import Image

    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    margin_x, margin_y = round(width * (1 - side_fraction) / 2), round(height * (1 - side_fraction) / 2)
    mask.paste(255, (margin_x, margin_y, width - margin_x, height - margin_y))
    return mask


def clip_mean(scores: dict[str, Any]) -> float:
    return round(scores["mean_prompt_similarity"], 3)


def without_images(generation: dict[str, Any]) -> dict[str, Any]:
    """A generation result without its PIL images (JSON-serialisable)."""
    return {**generation, "results": [{k: v for k, v in g.items() if k != "image"} for g in generation["results"]]}


def runtime_versions() -> dict[str, Any]:
    import diffusers
    import peft
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# --------------------------------------------------------------------------------------------------
# model and data factories (the CPU pre-flight test replaces these with stubs)
# --------------------------------------------------------------------------------------------------


def load_pipeline(run: Run) -> Any:
    import torch

    from kandinsky_inpainting_pipeline import KandinskyInpaintPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return KandinskyInpaintPipeline.from_pretrained(
        weights_dir=run.snapshot(SNAPSHOT_KEYS[0]), prior_dir=run.snapshot(SNAPSHOT_KEYS[1]), device=device, use_lora=True
    )


def load_from_artifact(run: Run, artifact_dir: Path) -> Any:
    import torch

    from kandinsky_inpainting_pipeline import KandinskyInpaintPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return KandinskyInpaintPipeline.from_artifact(
        artifact_dir, weights_dir=run.snapshot(SNAPSHOT_KEYS[0]), prior_dir=run.snapshot(SNAPSHOT_KEYS[1]), device=device
    )


def load_scorer(run: Run, device: str) -> Any:
    from kandinsky_inpainting_pipeline import ClipScorer, verify_scorer_snapshot

    verify_scorer_snapshot(run.snapshot(SNAPSHOT_KEYS[2]))
    return ClipScorer(device=device, weights_dir=str(run.snapshot(SNAPSHOT_KEYS[2])))


def load_sample_splits(run: Run) -> dict[str, list[dict[str, Any]]]:
    from kandinsky_inpainting_pipeline import fetch_sample_dataset

    return fetch_sample_dataset(cache_dir=run.weights / SAMPLE_CACHE)


def stage_snapshots(run: Run) -> list[dict[str, Any]]:
    """Install the carried manifests, fetch the absent files at the pinned revisions and verify every file."""
    from kandinsky_inpainting_pipeline import (
        MODEL_ID,
        MODEL_LICENSE,
        MODEL_REVISION,
        PRIOR_ID,
        PRIOR_REVISION,
        SCORER_ID,
        SCORER_REVISION,
        stage_missing_files,
        stage_missing_prior_files,
        stage_missing_scorer_files,
        verify_prior_snapshot,
        verify_scorer_snapshot,
        verify_snapshot,
    )
    from kandinsky_inpainting_pipeline.pipeline import MANIFEST_NAME

    plan = (
        (SNAPSHOT_KEYS[0], MODEL_ID, MODEL_REVISION, MODEL_LICENSE, stage_missing_files, verify_snapshot),
        (SNAPSHOT_KEYS[1], PRIOR_ID, PRIOR_REVISION, PRIOR_LICENSE, stage_missing_prior_files, verify_prior_snapshot),
        (SNAPSHOT_KEYS[2], SCORER_ID, SCORER_REVISION, SCORER_LICENSE, stage_missing_scorer_files, verify_scorer_snapshot),
    )
    records = []
    for key, model_id, revision, license_name, stage, verify in plan:
        carried = run.root / "weights" / key / MANIFEST_NAME
        manifest = json.loads(carried.read_text(encoding="utf-8"))
        if (manifest["modelId"], manifest["revision"]) != (model_id, revision):
            raise RuntimeError(f"carried {key} manifest does not name the identity pinned by the package; regenerate the notebook")
        target = run.snapshot(key)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(carried, target / MANIFEST_NAME)
        print({"model_id": model_id, "revision": revision, "license": license_name, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"]}, flush=True)
        fetched = stage(target, allow_download=True)
        print({"weights_dir": str(target), "fetched": fetched}, flush=True)
        verified = verify(target)
        print({"verified_files": len(verified["files"]), "revision": verified["revision"]}, flush=True)
        records.append({"key": key, "id": model_id, "revision": revision, "license": license_name, "files": len(verified["files"]), "fetched": fetched})
    return records


def validated_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Validate every split; validate_dataset attaches the mask: a record's own mask, or the deterministic centre mask."""
    from kandinsky_inpainting_pipeline import validate_dataset

    return {name: validate_dataset(splits[name], min_records=1)["records"] for name in ("train", "validation", "test")}


def load_data(run: Run, stage: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Rebuild the splits recorded by `prepare` from their source and refuse if they changed."""
    from kandinsky_inpainting_pipeline import dataset_manifest, load_byod_dataset, split_dataset

    data = run.read_state("data.json", stage)
    splits = split_dataset(load_byod_dataset(data["byod_path"]), seed=0) if data["source"] == "byod" else load_sample_splits(run)
    splits = validated_splits(splits)
    digest = dataset_manifest(splits)["digest"]
    if digest != data["dataset_digest"]:
        raise RuntimeError(f"the dataset changed since 'prepare' (digest {digest[:16]}… != {data['dataset_digest'][:16]}…); re-run from Section 4")
    return splits, data


def save_prompt_cache(run: Run, cache: dict[str, Any]) -> None:
    import safetensors.torch

    prompts = list(cache)
    tensors = {}
    for i, prompt in enumerate(prompts):
        tensors[f"{i}.image_embeds"] = cache[prompt]["image_embeds"].contiguous()
        tensors[f"{i}.negative_image_embeds"] = cache[prompt]["negative_image_embeds"].contiguous()
    safetensors.torch.save_file(tensors, run.state / PROMPT_CACHE)
    run.write_state(PROMPT_INDEX, {"prompts": prompts})


def load_prompt_cache(run: Run, pipe: Any, stage: str) -> int:
    import safetensors.torch

    prompts = run.read_state(PROMPT_INDEX, stage)["prompts"]
    tensors = safetensors.torch.load_file(run.state / PROMPT_CACHE)
    cache = {p: {"image_embeds": tensors[f"{i}.image_embeds"], "negative_image_embeds": tensors[f"{i}.negative_image_embeds"]} for i, p in enumerate(prompts)}
    return pipe.import_prompt_cache(cache)


# --------------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    """Section 3: install the carried manifests, fetch the absent files at the pinned revisions, verify every file."""
    snapshots = stage_snapshots(run)
    run.write_output("weights.json", {"snapshots": snapshots})


def stage_prepare(run: Run) -> None:
    """Section 4: fetch the sample (or read the BYOD zip), attach the masks, validate, split, and probe the refusals."""
    import numpy as np
    from PIL import Image

    from kandinsky_inpainting_pipeline import (
        INPUT_SCHEMA,
        dataset_manifest,
        load_byod_dataset,
        preprocess_image_and_mask,
        sample_prompts,
        split_dataset,
        validate_dataset,
        validate_inputs,
        write_dataset_csv,
    )
    from kandinsky_inpainting_pipeline.samples import SAMPLE_LABEL_SOURCE

    byod = run.options.byod
    if byod:
        byod_path = Path(byod).resolve()
        byod_records = load_byod_dataset(byod_path)
        own_masks = sum("mask_image" in r for r in byod_records)
        print({"byod_records": len(byod_records), "own_masks": own_masks, "centre_mask_fallback": len(byod_records) - own_masks})
        splits = split_dataset(byod_records, seed=0)
        data_source = "BYOD (" + byod_path.name + ")"
    else:
        byod_path = None
        splits = load_sample_splits(run)
        data_source = SAMPLE_LABEL_SOURCE
    splits = validated_splits(splits)
    train_records, val_records, test_records = splits["train"], splits["validation"], splits["test"]

    dataset_report = dataset_manifest({"train": train_records, "validation": val_records, "test": test_records})
    print({"data_source": data_source, "splits": {k: v["n_records"] for k, v in dataset_report["splits"].items()}, "captions": dataset_report["splits"]["train"]["n_captions"], "disjoint": dataset_report["disjoint"]})
    print({"shorter_side": dataset_report["splits"]["train"]["shorter_side"], "centre_cropped": dataset_report["splits"]["train"]["centre_cropped"], "digest": dataset_report["digest"][:16] + "..."})
    _, first_mask = preprocess_image_and_mask(test_records[0]["image"], test_records[0]["mask_image"])
    repaint_fraction = round(float((np.asarray(first_mask) > 0).mean()), 4)
    print({"first_test_record": validate_inputs(test_records[0]), "caption": test_records[0]["caption"], "repaint_fraction": repaint_fraction})
    prompts = sample_prompts(train_records)
    print({"prompts": prompts})
    sample_csv = write_dataset_csv(test_records, run.out / f"{STEM}_sample_captions.csv")
    print({"sample_csv": str(sample_csv)})

    print({"validation": INPUT_SCHEMA["validation"]})
    probes = {
        "missing caption": [{"id": r["id"], "image": r["image"]} for r in train_records[:4]],
        "image too small": [{**train_records[0], "image": Image.new("RGB", (200, 200))}, *train_records[1:4]],
        "mask size differs from image": [{**train_records[0], "mask_image": Image.new("L", (300, 300))}, *train_records[1:4]],
        "duplicate id": [train_records[0], *train_records[:4]],
    }
    probe_results = []
    for name, records in probes.items():
        try:
            validate_dataset(records)
            probe_results.append({"probe": name, "verdict": "accepted"})
        except (TypeError, ValueError) as exc:
            probe_results.append({"probe": name, "rejected": str(exc)[:110]})
        print(probe_results[-1])

    run.write_state(
        "data.json",
        {
            "source": "byod" if byod_path else "sample",
            "byod_path": str(byod_path) if byod_path else None,
            "data_source": data_source,
            "dataset_digest": dataset_report["digest"],
            "ids": {name: [r["id"] for r in records] for name, records in splits.items()},
        },
    )
    run.write_output("prepare.json", {"data_source": data_source, "dataset": dataset_report, "prompts": prompts, "repaint_fraction": repaint_fraction, "probes": probe_results})


def stage_encode(run: Run) -> None:
    """Section 5: encode every caption with the prior, release it, and write the prompt-embedding cache."""
    from kandinsky_inpainting_pipeline import prompt_seed, sample_prompts

    splits, _data = load_data(run, "encode")
    pipe = load_pipeline(run)
    all_prompts = sample_prompts(splits["train"] + splits["validation"] + splits["test"]) + [NEW_PROMPT]
    encode_report = pipe.encode_prompts(all_prompts)
    print({**encode_report, "gpu_memory_gb_with_prior": gpu_memory_gb()})
    print({"prompt_seed": {p[:48]: prompt_seed(p) for p in all_prompts[:2]}})
    released = pipe.release_prior()
    print({"prior_released": released, "gpu_memory_gb_after_release": gpu_memory_gb(), "device": pipe.device, "precision": str(pipe.dtype).replace("torch.", "")})
    cache = pipe.export_prompt_cache()
    save_prompt_cache(run, cache)
    print({"prompt_cache": str(run.state / PROMPT_CACHE), "prompts": len(cache)})
    run.write_output("encode.json", {**encode_report, "prior_released": released, "cached_prompts": list(cache), "device": pipe.device})


def stage_frozen(run: Run) -> None:
    """Section 6: the frozen model's held-out denoising loss, preservation and CLIP similarity between the mean-fill
    floor and the original-photograph ceiling."""
    from kandinsky_inpainting_pipeline import score_generations, score_inpainting_preservation

    opts = run.options
    splits, data = load_data(run, "frozen")
    val_records, test_records = splits["validation"], splits["test"]
    pipe = load_pipeline(run)
    load_prompt_cache(run, pipe, "frozen")
    scorer = load_scorer(run, pipe.device)
    t0 = time.perf_counter()
    frozen_val = pipe.evaluate(val_records, seed=EVAL_SEED)
    frozen_test = pipe.evaluate(test_records, seed=EVAL_SEED)
    print({"frozen_denoising_mse": {"validation": frozen_val["denoising_mse"], "test": frozen_test["denoising_mse"]}, "by_timestep_test": frozen_test["by_timestep"], "seconds": round(time.perf_counter() - t0, 1)})

    test_images, test_masks = prepared(test_records)
    test_prompts = [r["caption"] for r in test_records]
    frozen_generation = pipe.generate(test_records, seed=GENERATION_SEED, steps=opts.steps, guidance_scale=opts.guidance)
    frozen_images = [g["image"] for g in frozen_generation["results"]]
    print({"inpainted": len(frozen_images), "steps": frozen_generation["steps"], "guidance_scale": frozen_generation["guidance_scale"], "seconds": frozen_generation["seconds"], "adapted": frozen_generation["model"]["adapted"]})
    frozen_preservation = score_inpainting_preservation(test_images, frozen_images, test_masks)
    frozen_clip = score_generations(scorer, frozen_images, test_prompts)
    fill_clip = score_generations(scorer, [mean_fill(i, m) for i, m in zip(test_images, test_masks, strict=True)], test_prompts)
    real_clip = score_generations(scorer, test_images, test_prompts)
    print({"clip_prompt_similarity": {"mean_fill_floor": clip_mean(fill_clip), "frozen": clip_mean(frozen_clip), "original_photo_ceiling": clip_mean(real_clip)}})
    print({"frozen_preservation": {"mean_unmasked_psnr_db": round(frozen_preservation["mean_unmasked_psnr_db"], 2), "mean_unmasked_ssim": round(frozen_preservation["mean_unmasked_ssim"], 4)}})
    for result, clip in list(zip(frozen_generation["results"], frozen_clip["prompt_similarities"], strict=True))[:6]:
        print({"id": result["id"], "caption": result["prompt"][:40], "unmasked_psnr_db": result["unmasked_psnr_db"], "unmasked_ssim": result["unmasked_ssim"], "clip": round(clip, 2)})
    grid_path = triptych_grid(test_images, test_masks, frozen_images, run.out / f"{STEM}_frozen_grid.jpg")
    print({"grid": str(grid_path)})
    record = {
        "data_source": data["data_source"],
        "generation": {"steps": opts.steps, "guidance_scale": opts.guidance, "seed": GENERATION_SEED},
        "validation": frozen_val,
        "test": frozen_test,
        "generation_result": without_images(frozen_generation),
        "preservation": frozen_preservation,
        "clip": frozen_clip,
        "mean_fill_floor_clip": fill_clip,
        "original_photo_ceiling_clip": real_clip,
    }
    run.write_state("frozen.json", record)
    run.write_output("frozen.json", record)


def stage_adapt(run: Run) -> None:
    """Section 7: bounded LoRA fine-tuning; record the in-memory model's reference values; export the adapter."""
    opts = run.options
    splits, data = load_data(run, "adapt")
    frozen = run.read_state("frozen.json", "adapt")
    settings = frozen["generation"]
    pipe = load_pipeline(run)
    load_prompt_cache(run, pipe, "adapt")

    def report(entry: dict[str, Any]) -> None:
        row = {"epoch": entry["epoch"], "train_loss": None if entry["train_loss"] is None else round(entry["train_loss"], 4), "val_denoising_mse": entry["val_loss"]}
        if "note" in entry:
            row["note"] = entry["note"]
        print(row, flush=True)

    t0 = time.perf_counter()
    adapt_result = pipe.adapt(splits["train"], splits["validation"], epochs=opts.epochs, lr=opts.lr, batch_size=opts.batch_size, seed=EVAL_SEED, progress=report)
    adapt_seconds = round(time.perf_counter() - t0, 1)
    print({"trainable_parameters": adapt_result["adapter"]["n_trainable"], "total_parameters": adapt_result["adapter"]["n_total"], "steps": adapt_result["steps"], "best_epoch": adapt_result["best_epoch"], "optimizer": adapt_result["adapter"]["optimizer"], "precision": adapt_result["adapter"]["precision"], "seconds": adapt_seconds, "gpu_memory_gb": gpu_memory_gb()})

    # Reference values of the trained, in-memory model, for the fresh-process reload check in Section 9 (VER2/VER4).
    in_memory_test = pipe.evaluate(splits["test"], seed=EVAL_SEED)
    parity_image = pipe.generate(splits["test"][:1], seed=PARITY_SEED, steps=settings["steps"], guidance_scale=settings["guidance_scale"])["results"][0]["image"]
    parity_image.save(run.state / "parity_in_memory.png")

    artifact_dir = run.out / f"{STEM}_adapter"
    shutil.rmtree(artifact_dir, ignore_errors=True)
    manifest = pipe.save_artifact(artifact_dir, metadata={"tutorial": STEM, "data_source": data["data_source"], "dataset_digest": data["dataset_digest"]})
    print({"artifact": str(artifact_dir), "format": manifest["format"], "tensors": manifest["weights"]["n_tensors"], "bytes": manifest["weights"]["bytes"], "sha256": manifest["weights"]["sha256"][:16] + "..."})
    record = {
        "history": adapt_result["history"],
        "best_epoch": adapt_result["best_epoch"],
        "steps": adapt_result["steps"],
        "adaptation": {k: v for k, v in adapt_result["adapter"].items() if k != "trainable_names"},
        "adaptation_seconds": adapt_seconds,
        "hyperparameters": {"epochs": opts.epochs, "lr": opts.lr, "batch_size": opts.batch_size, "seed": EVAL_SEED},
        "in_memory": {"test_denoising_mse": in_memory_test["denoising_mse"], "parity_record": splits["test"][0]["id"], "parity_seed": PARITY_SEED, "parity_image": "state/parity_in_memory.png"},
        "artifact": {"dir": str(artifact_dir), "sha256": manifest["weights"]["sha256"], "bytes": manifest["weights"]["bytes"], "tensors": manifest["weights"]["n_tensors"]},
    }
    run.write_state("adapt.json", record)
    run.write_output("adapt.json", record)


def stage_evaluate(run: Run) -> None:
    """Section 8: a fresh process loads the exported adapter and measures it exactly as the frozen model was measured."""
    from kandinsky_inpainting_pipeline import (
        MODEL_ID,
        MODEL_KEY,
        MODEL_REVISION,
        PRIOR_ID,
        PRIOR_REVISION,
        SCORER_ID,
        SCORER_REVISION,
        score_generations,
        score_inpainting_preservation,
    )

    splits, data = load_data(run, "evaluate")
    val_records, test_records = splits["validation"], splits["test"]
    frozen = run.read_state("frozen.json", "evaluate")
    adapted_state = run.read_state("adapt.json", "evaluate")
    settings = frozen["generation"]
    pipe = load_from_artifact(run, Path(adapted_state["artifact"]["dir"]))
    load_prompt_cache(run, pipe, "evaluate")
    scorer = load_scorer(run, pipe.device)
    adapted_val = pipe.evaluate(val_records, seed=EVAL_SEED)
    adapted_test = pipe.evaluate(test_records, seed=EVAL_SEED)
    test_images, test_masks = prepared(test_records)
    test_prompts = [r["caption"] for r in test_records]
    adapted_generation = pipe.generate(test_records, seed=settings["seed"], steps=settings["steps"], guidance_scale=settings["guidance_scale"])
    adapted_images = [g["image"] for g in adapted_generation["results"]]
    adapted_preservation = score_inpainting_preservation(test_images, adapted_images, test_masks)
    adapted_clip = score_generations(scorer, adapted_images, test_prompts)
    frozen_val, frozen_test = frozen["validation"], frozen["test"]
    frozen_preservation, frozen_clip = frozen["preservation"], frozen["clip"]
    fill_clip, real_clip = frozen["mean_fill_floor_clip"], frozen["original_photo_ceiling_clip"]
    comparison = {
        "denoising_mse_validation": {"frozen": frozen_val["denoising_mse"], "adapted": adapted_val["denoising_mse"]},
        "denoising_mse_test": {"frozen": frozen_test["denoising_mse"], "adapted": adapted_test["denoising_mse"]},
        "denoising_mse_test_by_timestep": {t: {"frozen": frozen_test["by_timestep"][t], "adapted": adapted_test["by_timestep"][t]} for t in adapted_test["by_timestep"]},
        "mean_unmasked_psnr_db": {"frozen": round(frozen_preservation["mean_unmasked_psnr_db"], 2), "adapted": round(adapted_preservation["mean_unmasked_psnr_db"], 2)},
        "mean_unmasked_ssim": {"frozen": round(frozen_preservation["mean_unmasked_ssim"], 4), "adapted": round(adapted_preservation["mean_unmasked_ssim"], 4)},
        "clip_prompt_similarity": {"mean_fill_floor": clip_mean(fill_clip), "frozen": clip_mean(frozen_clip), "adapted": clip_mean(adapted_clip), "original_photo_ceiling": clip_mean(real_clip)},
    }
    for name, row in comparison.items():
        print({name: row})
    frozen_results = frozen["generation_result"]["results"]
    for before, after, clip_before, clip_after in list(zip(frozen_results, adapted_generation["results"], frozen_clip["prompt_similarities"], adapted_clip["prompt_similarities"], strict=True))[:6]:
        print({"id": before["id"], "caption": before["prompt"][:40], "clip": {"frozen": round(clip_before, 2), "adapted": round(clip_after, 2)}, "unmasked_psnr_db": {"frozen": before["unmasked_psnr_db"], "adapted": after["unmasked_psnr_db"]}})
    grid_path = triptych_grid(test_images, test_masks, adapted_images, run.out / f"{STEM}_adapted_grid.jpg")
    print({"grid": str(grid_path)})
    evaluation_report = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY},
        "components": {"prior": {"id": PRIOR_ID, "revision": PRIOR_REVISION}, "scorer": {"id": SCORER_ID, "revision": SCORER_REVISION}},
        "data_source": data["data_source"],
        "dataset": json.loads((run.out / "prepare.json").read_text(encoding="utf-8"))["dataset"],
        "mask": "record mask, or the deterministic centre mask (middle half of each side) when a record has none",
        "generation": dict(settings),
        "frozen": {"validation": frozen_val, "test": frozen_test, "preservation": frozen_preservation, "clip": frozen_clip},
        "adapted": {"validation": adapted_val, "test": adapted_test, "preservation": adapted_preservation, "clip": adapted_clip, "loaded_from": "exported artifact, fresh process"},
        "references": {"mean_fill_floor_clip": fill_clip, "original_photo_ceiling_clip": real_clip},
        "comparison": comparison,
        "adaptation": adapted_state["adaptation"],
        "history": adapted_state["history"],
        "adaptation_seconds": adapted_state["adaptation_seconds"],
    }
    report_path = run.write_output(f"{STEM}_evaluation_report.json", evaluation_report)
    history = adapted_state["history"]
    best = history[adapted_state["best_epoch"]]
    # Guaranteed by the procedure: epoch 0 (the frozen model) is a candidate, so the kept validation loss is no worse.
    if not best["val_loss"] <= history[0]["val_loss"]:
        raise AssertionError(f"kept epoch validation loss {best['val_loss']} exceeds the frozen model's {history[0]['val_loss']}")
    # The exported adapter, loaded in this fresh process, reproduces the kept epoch's validation loss.
    if not abs(adapted_val["denoising_mse"] - best["val_loss"]) < 1e-4:
        raise AssertionError(f"the exported adapter gives validation loss {adapted_val['denoising_mse']}, the kept epoch recorded {best['val_loss']}")
    print({"test_denoising_mse_change": round(adapted_test["denoising_mse"] - frozen_test["denoising_mse"], 6), "note": "held-out observation, not asserted"})
    print({"report": str(report_path)})
    run.write_state("evaluate.json", {"comparison": comparison, "adapted_test_denoising_mse": adapted_test["denoising_mse"], "adapted_results": without_images(adapted_generation)["results"], "adapted_clip": adapted_clip})


def stage_reload(run: Run) -> None:
    """Section 9: another fresh process rebuilds the pipeline from the artifact, checks parity with the trained
    in-memory model recorded by `adapt`, inpaints the new caption, and writes the result record."""
    import numpy as np
    from PIL import Image

    from kandinsky_inpainting_pipeline import CORPUS_BASE_URL, CORPUS_LICENSE, MODEL_LICENSE, score_generations

    splits, data = load_data(run, "reload")
    frozen = run.read_state("frozen.json", "reload")
    adapted_state = run.read_state("adapt.json", "reload")
    evaluated = run.read_state("evaluate.json", "reload")
    settings = frozen["generation"]
    test_records = splits["test"]
    artifact_dir = Path(adapted_state["artifact"]["dir"])
    artifact_manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    print({"artifact": str(artifact_dir), "format": artifact_manifest["format"], "tensors": artifact_manifest["weights"]["n_tensors"], "bytes": artifact_manifest["weights"]["bytes"], "sha256": artifact_manifest["weights"]["sha256"][:16] + "..."})

    reloaded = load_from_artifact(run, artifact_dir)
    load_prompt_cache(run, reloaded, "reload")
    reloaded_test = reloaded.evaluate(test_records, seed=EVAL_SEED)
    in_memory = adapted_state["in_memory"]
    if test_records[0]["id"] != in_memory["parity_record"]:
        raise RuntimeError("the parity record differs from the one 'adapt' used; re-run from Section 4")
    after = reloaded.generate(test_records[:1], seed=in_memory["parity_seed"], steps=settings["steps"], guidance_scale=settings["guidance_scale"])["results"][0]["image"]
    with Image.open(run.root / in_memory["parity_image"]) as saved:
        before = np.asarray(saved.convert("RGB"), dtype=np.float32)
    parity = {
        "denoising_mse_diff": round(abs(reloaded_test["denoising_mse"] - in_memory["test_denoising_mse"]), 8),
        "mean_abs_pixel_diff": round(float(np.abs(before - np.asarray(after, dtype=np.float32)).mean()), 4),
        "compared_with": "the trained in-memory model of the 'adapt' process",
    }
    print({"reload_parity": parity, "reloaded_best_epoch": reloaded.adapter["best_epoch"]})
    if not (parity["denoising_mse_diff"] < 1e-6 and parity["mean_abs_pixel_diff"] < 1.0):
        raise AssertionError(f"reload parity failed: {parity}")

    scorer = load_scorer(run, reloaded.device)
    new_records = [{**r, "caption": NEW_PROMPT} for r in test_records[:2]]
    new_generation = reloaded.generate(new_records, seed=NEW_PROMPT_SEED, steps=settings["steps"], guidance_scale=settings["guidance_scale"])
    new_clip = score_generations(scorer, [g["image"] for g in new_generation["results"]], [NEW_PROMPT] * len(new_records))
    new_predictions = []
    for i, (result, clip) in enumerate(zip(new_generation["results"], new_clip["prompt_similarities"], strict=True)):
        path = run.out / f"{STEM}_new_prompt_{i}.png"
        result["image"].save(path)
        new_predictions.append({"id": result["id"], "prompt": NEW_PROMPT, "image": path.relative_to(run.root).as_posix(), "unmasked_psnr_db": result["unmasked_psnr_db"], "unmasked_ssim": result["unmasked_ssim"], "clip_prompt_similarity": round(clip, 3)})
        print({**new_predictions[-1], "note": "sanity check, not an evaluation"})

    source_path = run.root / "source.json"
    notebook_source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else None
    weights = {s["key"]: s["files"] for s in json.loads((run.out / "weights.json").read_text(encoding="utf-8"))["snapshots"]}
    report = json.loads((run.out / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    result_payload = {
        "notebook_source": notebook_source,
        "repository_revision": notebook_source["revision"] if notebook_source else None,
        "model": {**report["model"], "model_license": MODEL_LICENSE, "device": reloaded.device, "precision": str(reloaded.dtype).replace("torch.", ""), "source": reloaded.source},
        "components": {**report["components"], "licenses": {"prior": PRIOR_LICENSE, "scorer": SCORER_LICENSE}},
        "provenance": {
            "snapshots": {"decoder": weights[SNAPSHOT_KEYS[0]], "prior": weights[SNAPSHOT_KEYS[1]], "scorer": weights[SNAPSHOT_KEYS[2]]},
            "safetensors_only": True,
            "remote_code_executed": False,
            "prior_released_before_training": json.loads((run.out / "encode.json").read_text(encoding="utf-8"))["prior_released"],
            "prompt_seed": "SHA-256 of the caption, first 4 bytes, 31 bits",
            "data_base_url": CORPUS_BASE_URL,
            "data_license": CORPUS_LICENSE,
        },
        "runtime": {**runtime_versions(), "environment": "isolated hash-locked environment (one process per stage)"},
        "data_source": data["data_source"],
        "comparison": evaluated["comparison"],
        "new_prompt_predictions": new_predictions,
        "artifact": {"dir": str(artifact_dir), "sha256": artifact_manifest["weights"]["sha256"], "bytes": artifact_manifest["weights"]["bytes"]},
        "reload_parity": parity,
    }
    run.write_output(f"{STEM}_result.json", result_payload)
    run.write_output("reload.json", {"reload_parity": parity, "new_prompt_predictions": new_predictions})
    print("outputs/:")
    for path in sorted(run.out.rglob("*")):
        if path.is_file():
            print(f"  - {path.relative_to(run.root).as_posix()} ({path.stat().st_size / 1024:.1f} KB)")


def stage_activity(run: Run) -> None:
    """Section 10 (optional): inpaint the first two test photographs again with a small or a large centred mask."""
    from kandinsky_inpainting_pipeline import score_generations

    splits, _data = load_data(run, "activity")
    frozen = run.read_state("frozen.json", "activity")
    adapted_state = run.read_state("adapt.json", "activity")
    evaluated = run.read_state("evaluate.json", "activity")
    settings = frozen["generation"]
    choice = run.options.mask
    side = MASK_SIDES[choice]
    pipe = load_from_artifact(run, Path(adapted_state["artifact"]["dir"]))
    load_prompt_cache(run, pipe, "activity")
    scorer = load_scorer(run, pipe.device)
    changed = [{**r, "mask_image": box_mask(r["image"], side)} for r in splits["test"][:2]]
    activity_generation = pipe.generate(changed, seed=settings["seed"], steps=settings["steps"], guidance_scale=settings["guidance_scale"])
    activity_clip = score_generations(scorer, [g["image"] for g in activity_generation["results"]], [r["caption"] for r in changed])
    rows = []
    for i, (centre, new) in enumerate(zip(evaluated["adapted_results"][:2], activity_generation["results"], strict=True)):
        rows.append({"id": centre["id"], "repaint_fraction": {"centre": CENTRE_REPAINT_FRACTION, choice: round(side * side, 4)}, "unmasked_psnr_db": {"centre": centre["unmasked_psnr_db"], choice: new["unmasked_psnr_db"]}, "clip": {"centre": round(evaluated["adapted_clip"]["prompt_similarities"][i], 2), choice: round(activity_clip["prompt_similarities"][i], 2)}})
        print(rows[-1])
    activity_images, activity_masks = prepared(changed)
    grid_path = triptych_grid(activity_images, activity_masks, [g["image"] for g in activity_generation["results"]], run.out / f"{STEM}_activity_{choice}_grid.jpg")
    print({"grid": str(grid_path)})
    run.write_output("activity.json", {"mask": choice, "side_fraction": side, "rows": rows, "clip": activity_clip, "results": without_images(activity_generation)["results"]})


STAGES = {
    "weights": stage_weights,
    "prepare": stage_prepare,
    "encode": stage_encode,
    "frozen": stage_frozen,
    "adapt": stage_adapt,
    "evaluate": stage_evaluate,
    "reload": stage_reload,
    "activity": stage_activity,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="run directory holding the carried sources")
    parser.add_argument("--weights", type=Path, required=True, help="directory holding the three snapshots and the photo cache")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--byod", default="", help="prepare: a BYOD zip or directory instead of the sample")
    parser.add_argument("--steps", type=int, default=20, help="frozen: inpainting steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="frozen: classifier-free guidance scale")
    parser.add_argument("--epochs", type=int, default=4, help="adapt: training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="adapt: AdamW learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="adapt: records per optimiser step")
    parser.add_argument("--mask", choices=sorted(MASK_SIDES), default="large", help="activity: the centred box mask size")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    root = options.root.resolve()
    carried_src = root / "src"
    if carried_src.is_dir() and str(carried_src) not in sys.path:
        sys.path.insert(0, str(carried_src))
    run = Run(root, options.weights.resolve(), options)
    error_file = run.state / f"{options.stage}.error.json"
    error_file.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        STAGES[options.stage](run)
    except Exception as exc:  # the notebook re-raises this message in the kernel
        traceback.print_exc()
        message = str(exc) or repr(exc)
        error_file.write_text(json.dumps({"stage": options.stage, "type": type(exc).__name__, "message": message}), encoding="utf-8")
        print(f"STAGE FAILED ({options.stage}): {type(exc).__name__}: {message}", flush=True)
        return 2
    print({"stage": options.stage, "status": "ok", "seconds": round(time.perf_counter() - started, 1)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
