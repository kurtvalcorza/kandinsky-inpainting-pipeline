"""Offline tests for the three snapshot manifests, staging, the dataset contract and the artifact-manifest
rejections. No model library is imported."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from conftest import synthetic_records
from kandinsky_inpainting_pipeline import (
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    PRIOR_ID,
    PRIOR_REVISION,
    RESOLUTION,
    SCORER_REVISION,
    dataset_digest,
    preprocess_image,
    stage_missing_files,
    validate_dataset,
    validate_prompts,
    verify_prior_snapshot,
    verify_scorer_snapshot,
    verify_snapshot,
)
from kandinsky_inpainting_pipeline import pipeline as pl

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_FILE = "unet/diffusion_pytorch_model.safetensors"


def _write_snapshot(root: Path, model_id: str, revision: str, files: dict[str, bytes]) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
        entries.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": "k",
        "modelId": model_id,
        "revision": revision,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
    }
    (root / pl.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


# --- identity and committed manifests -------------------------------------------------------------------


def test_identity_is_immutable_and_committed_manifests_agree():
    assert len(MODEL_REVISION) == len(PRIOR_REVISION) == len(SCORER_REVISION) == 40
    for key, model_id, revision in (
        (pl.MODEL_KEY, MODEL_ID, MODEL_REVISION),
        (pl.PRIOR_KEY, PRIOR_ID, PRIOR_REVISION),
        (pl.SCORER_KEY, pl.SCORER_ID, SCORER_REVISION),
    ):
        manifest = json.loads((ROOT / "weights" / key / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert (manifest["modelId"], manifest["revision"], manifest["modelKey"]) == (model_id, revision, key)
        assert manifest["totalBytes"] == sum(e["bytes"] for e in manifest["files"])
        assert all(len(e["sha256"]) == 64 for e in manifest["files"])
        assert not any(e["path"].endswith((".bin", ".pt", ".pth", ".ckpt", ".pickle", ".py")) for e in manifest["files"])
    decoder = json.loads((ROOT / "weights" / pl.MODEL_KEY / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
    expected = {
        "README.md",
        "model_index.json",
        "movq/config.json",
        "movq/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
    }
    assert {e["path"] for e in decoder["files"]} == expected


# --- snapshot verification and staging --------------------------------------------------------------------


def test_verify_snapshot_refuses_mismatches(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    manifest = _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"unet/config.json": b"{}", WEIGHTS_FILE: b"tensors"})
    assert verify_snapshot(root)["files"] == manifest["files"]
    (root / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"tensorz")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(root)
    (root / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"tensors-longer")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(root)
    (root / "unet" / "diffusion_pytorch_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_snapshot(root)
    _write_snapshot(root, "someone/else", MODEL_REVISION, {"a.json": b"{}"})
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(root)
    _write_snapshot(root, MODEL_ID, "0" * 40, {"a.json": b"{}"})
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(root)
    with pytest.raises(FileNotFoundError, match="manifest"):
        verify_snapshot(tmp_path / "nowhere")


def test_verify_snapshot_refuses_executable_file_types(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"unet/diffusion_pytorch_model.bin": b"pickle"})
    with pytest.raises(ValueError, match="unexpected file type"):
        verify_snapshot(root)


def test_each_snapshot_has_its_own_identity(tmp_path, forbid_model_imports):
    prior = tmp_path / "prior"
    _write_snapshot(prior, PRIOR_ID, PRIOR_REVISION, {"prior/config.json": b"{}"})
    assert verify_prior_snapshot(prior)["modelId"] == PRIOR_ID
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(prior)
    scorer = tmp_path / "scorer"
    _write_snapshot(scorer, pl.SCORER_ID, SCORER_REVISION, {"config.json": b"{}"})
    assert verify_scorer_snapshot(scorer)["modelId"] == pl.SCORER_ID
    with pytest.raises(ValueError, match="modelId"):
        verify_prior_snapshot(scorer)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"unet/config.json": b"{}", WEIGHTS_FILE: b"tensors"})
    (root / "unet" / "diffusion_pytorch_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    calls = []

    def downloader(rel, dst):
        calls.append(rel)
        (dst / rel).write_bytes(b"tensors")

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == [WEIGHTS_FILE]
    assert calls == ["unet/diffusion_pytorch_model.safetensors"]
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []
    assert verify_snapshot(root)["revision"] == MODEL_REVISION


# --- dataset validation -----------------------------------------------------------------------------------


def test_validate_dataset_enforces_limits_and_shapes(forbid_model_imports):
    records = synthetic_records(6)
    report = validate_dataset(records)
    assert report["n_records"] == 6 and report["resolution"] == RESOLUTION
    with pytest.raises(ValueError, match="records must be a list"):
        validate_dataset("not a list")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="4..2000 are required"):
        validate_dataset(records[:2])
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset(records + [records[0]])
    bad_id = [{**records[0], "id": ""}, *records[1:]]
    with pytest.raises(ValueError, match="id must be a non-empty string"):
        validate_dataset(bad_id)
    bad_cap = [{**records[0], "caption": "   "}, *records[1:]]
    with pytest.raises(ValueError, match="caption must be a non-empty string"):
        validate_dataset(bad_cap)
    too_small = [{**records[0], "image": Image.new("RGB", (MIN_IMAGE_SIDE - 1, 512))}, *records[1:]]
    with pytest.raises(ValueError, match="image sides must be within"):
        validate_dataset(too_small)
    too_large = [{**records[0], "image": Image.new("RGB", (MAX_IMAGE_SIDE + 1, 512))}, *records[1:]]
    with pytest.raises(ValueError, match="image sides must be within"):
        validate_dataset(too_large)


def test_validate_prompts_enforces_strings_and_length(forbid_model_imports):
    assert validate_prompts([" a prompt ", "another"]) == ["a prompt", "another"]
    with pytest.raises(ValueError, match="non-empty list of strings"):
        validate_prompts([])
    with pytest.raises(ValueError, match="non-empty string"):
        validate_prompts([""])


def test_preprocess_image_crops_and_scales(forbid_model_imports):
    wide = Image.new("RGB", (800, 600), color=(10, 20, 30))
    tall = Image.new("RGB", (600, 800), color=(40, 50, 60))
    for img in (wide, tall):
        out = preprocess_image(img)
        assert out.size == (RESOLUTION, RESOLUTION) and out.mode == "RGB"


def test_dataset_digest_is_deterministic(forbid_model_imports):
    r1 = synthetic_records(4)
    r2 = synthetic_records(4)
    assert dataset_digest(r1) == dataset_digest(r2)
    assert len(dataset_digest(r1)) == 64
