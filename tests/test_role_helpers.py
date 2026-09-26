"""Offline tests for the public validation-stage helpers and the package surface."""

from __future__ import annotations

import kandinsky_inpainting_pipeline as pkg
from conftest import synthetic_records
from kandinsky_inpainting_pipeline import INPUT_SCHEMA, LORA_TENSORS, RESOLUTION, validate_inputs


def test_input_schema_names_the_contract():
    assert INPUT_SCHEMA["resolution"] == RESOLUTION == 512
    assert "any RGB image with any string is accepted" in INPUT_SCHEMA["validation"]
    assert LORA_TENSORS == 176


def test_validate_inputs_reports_the_record():
    report = validate_inputs(synthetic_records(1)[0])
    assert report["id"] == "rec-000" and report["size"] == (640, 480) and report["centre_crop"] == (512, 512)


def test_public_surface_is_exported():
    for name in pkg.__all__:
        assert hasattr(pkg, name), name
    assert "KandinskyInpaintPipeline" in pkg.__all__ and "ClipScorer" in pkg.__all__


def test_scorer_unwraps_transformers_model_outputs():
    """`transformers` 5 returns a model output whose `pooler_output` is the projection; older releases a tensor."""
    import numpy as np

    from kandinsky_inpainting_pipeline.metrics import _projected

    tensor = np.zeros((2, 512), dtype=np.float32)
    assert _projected(tensor) is tensor

    class _Output:
        pooler_output = tensor

    assert _projected(_Output()) is tensor
