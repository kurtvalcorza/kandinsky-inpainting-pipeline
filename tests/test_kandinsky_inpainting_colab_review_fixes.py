"""Regression tests for the 2026-10-02 notebook review of `kandinsky_inpainting_colab.ipynb` (KIP-m1..m4).

CI installs numpy, pillow, pytest and ruff only (no torch): the metric, split, notebook-text and validator tests run there
with a stand-in scorer. The stage-runner test reuses the CPU pre-flight of `test_tutorial_stages.py` (stub models) and
skips without torch + diffusers + safetensors; it proves the plumbing, not the model.
"""
# ruff: noqa: E501

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import itertools
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import synthetic_image
from kandinsky_inpainting_pipeline import (
    MIN_TRAIN_RECORDS,
    REAL_PHOTO_REFERENCE_KIND,
    count_above_reference,
    load_byod_dataset,
    near_duplicate_pairs,
    real_photo_baseline,
    real_photo_reference,
    split_dataset,
)

# Windows DLL load-order trap (eo-notebook-test): import torch before any NumPy work in this process, if it exists.
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "kandinsky_inpainting_colab.ipynb"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _runner() -> str:
    return (ROOT / "tools" / "tutorial_stages.py").read_text(encoding="utf-8")


# --- KIP-m1: the original photograph is a reference line, not a ceiling -------------------------------------------


class _Scorer:
    """CLIP stand-in: a fixed score per (image key, caption)."""

    def __init__(self, table: dict[tuple[str, str], float]) -> None:
        self.table = table

    def score(self, images, prompts):
        return [self.table[(image, prompt)] for image, prompt in zip(images, prompts, strict=True)]


def test_real_photo_reference_scores_each_photo_against_its_own_caption_only() -> None:
    scorer = _Scorer({("a", "A"): 30.0, ("b", "A"): 28.0, ("c", "C"): 26.0})
    report = real_photo_reference(scorer, ["a", "b", "c"], ["A", "A", "C"])
    assert report["prompt_similarities"] == [30.0, 28.0, 26.0]
    assert report["mean_prompt_similarity"] == pytest.approx(28.0)
    assert report["n_images"] == 3
    assert report["reference_kind"] == REAL_PHOTO_REFERENCE_KIND and "not a ceiling" in REAL_PHOTO_REFERENCE_KIND
    assert set(report["reading"]) == {"mean_prompt_similarity"} and "not an upper bound" in report["reading"]["mean_prompt_similarity"]
    assert "not a ceiling" in report["note"] and "no self-comparison" in report["note"]
    assert real_photo_baseline is real_photo_reference  # the generation siblings' name gives this measure


def test_count_above_reference_reproduces_the_recorded_default_run() -> None:
    """Per image, paired with each output's own original photograph (synthetic values shaped like the recorded run: the
    frozen model above its original on 7 of 12, the adapted model on 5 of 12, the mean-fill floor on 0 of 12)."""
    original = [30.0] * 12
    assert count_above_reference([31.0] * 7 + [29.0] * 5, original) == {"n_above": 7, "n": 12, "text": "7 of 12"}
    assert count_above_reference([31.0] * 5 + [29.0] * 7, original)["text"] == "5 of 12"
    assert count_above_reference([24.0] * 12, original)["n_above"] == 0
    with pytest.raises(ValueError):
        count_above_reference([1.0], [1.0, 2.0])


def test_no_learner_text_calls_the_original_photograph_a_ceiling(nb: dict) -> None:
    md = _markdown(nb)
    loose = [m.start() for m in re.finditer(r"\bceiling", md, re.I) if not md[: m.start()].endswith("not a ")]
    assert not loose
    assert md.count("not a ceiling") >= 3 and "original-photograph reference" in md
    for stale in ("between the floor and the ceiling", "little room for CLIP to rise", "lands between the two", "places the outputs between"):
        assert stale not in md
    assert "7 of the 12 frozen outputs scored above their own original" in md
    assert "`clip_above_own_original_photograph`" in md
    assert "original-photograph reference of [reference]" in md and "[ceiling]" not in md
    code = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "code")
    assert "real_clip = real_photo_reference(scorer, test_images, test_prompts)" in code
    assert "original_photo_ceiling" not in code
    for name in ("README.md", "MODEL_CARD.md", "tutorials/README.md", "STATUS.md", "docs/release-verification.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "original-photograph ceiling" not in text and "the original photograph the ceiling" not in text, name


def test_validator_refuses_a_ceiling_or_a_removed_phrase_coming_back(nb: dict) -> None:
    va = _load("validate_release_assets")
    build = _load("build_notebook")
    code_cells, markdown = va._validate_notebook_structure(NOTEBOOK, nb)
    carrier, files = va._validate_carrier(NOTEBOOK, nb, code_cells, build)
    va._validate_notebook_content(NOTEBOOK, nb, code_cells, markdown, carrier, files)  # the committed text passes
    with pytest.raises(va.ValidationError, match="ceiling"):
        va._validate_notebook_content(NOTEBOOK, nb, code_cells, markdown + "\nCLIP sits near the ceiling.", carrier, files)
    for phrase in ("Your records are split by caption", "you need at least four photographs", "No clean-runtime timing of this version has been recorded yet"):
        with pytest.raises(va.ValidationError, match="removed is back"):
            va._validate_notebook_content(NOTEBOOK, nb, code_cells, markdown + "\n" + phrase, carrier, files)


# --- KIP-m2: the independence assumption, the stratified split and near-duplicates --------------------------------


def test_split_assumption_is_stated_before_the_split_runs(nb: dict) -> None:
    md = _markdown(nb)
    assert "**The split assumes independent photographs.**" in md and "near-duplicates" in md
    assert "stratified within each caption" in md and "new photographs of seen captions" in md
    for stale in ("split by caption", "grouped by caption", "Hold out by caption"):
        assert stale not in md
    cells = [_src(c) for c in nb["cells"] if c["cell_type"] == "markdown"]
    section_4 = next(i for i, text in enumerate(cells) if text.startswith("## 4."))
    assert "**The split assumes independent photographs.**" in cells[section_4]
    assert "STRATIFIED WITHIN EACH CAPTION" in split_dataset.__doc__ and "independent" in split_dataset.__doc__
    assert "grouped by caption" not in split_dataset.__doc__
    assert "split is by caption" not in (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")


def _records(sizes: tuple[int, ...], *, width: int = 256, height: int = 256) -> list[dict]:
    out, seed = [], 0
    for caption, n in enumerate(sizes):
        for _ in range(n):
            out.append({"id": f"r{seed}", "image": synthetic_image(width=width, height=height, seed=seed), "caption": f"caption {caption}"})
            seed += 1
    return out


def test_every_caption_appears_in_every_split() -> None:
    splits = split_dataset(_records((5, 5)), seed=0)
    for part in splits.values():
        assert {r["caption"] for r in part} == {"caption 0", "caption 1"}


def test_one_pixel_near_duplicate_across_splits_is_reported() -> None:
    """The review's probe: a copy with one pixel changed survives the pixel-identical de-duplication; with seed 0 it can
    land in a different split from its original. `near_duplicate_pairs` reports such a pair."""
    records = _records((6,), width=320, height=320)
    pixels = np.asarray(records[0]["image"]).copy()
    pixels[0, 0] = 255 - pixels[0, 0]
    records.append({"id": "copy-of-r0", "image": Image.fromarray(pixels), "caption": "caption 0"})
    splits = split_dataset(records, seed=0)
    assert sum(len(v) for v in splits.values()) == 7  # the near-copy is kept (not pixel-identical)
    where = {r["id"]: name for name, part in splits.items() for r in part}
    pairs = near_duplicate_pairs(splits)
    crossing = [p for p in pairs if {p["a"], p["b"]} == {"r0", "copy-of-r0"}]
    if where["r0"] != where["copy-of-r0"]:
        assert crossing and crossing[0]["correlation"] > 0.99
    else:
        assert not crossing  # same split: nothing to report
    # Forced across splits: always reported; distinct synthetic photographs are not.
    forced = {"train": [records[0], *records[1:4]], "test": [records[6]], "validation": [records[4]]}
    pairs = near_duplicate_pairs(forced)
    assert [(p["a"], p["b"]) for p in pairs] == [("r0", "copy-of-r0")]
    assert pairs[0]["a_split"] == "train" and pairs[0]["b_split"] == "test"


def test_prepare_reports_near_duplicates() -> None:
    runner = _runner()
    assert 'near_duplicates = near_duplicate_pairs({"train": train_records, "validation": val_records, "test": test_records})' in runner
    assert '"near_duplicates_across_splits": near_duplicates' in runner


# --- KIP-m3: the stated BYOD minimum is exactly what the code accepts ----------------------------------------------


def _stated_rule_accepts(sizes: tuple[int, ...]) -> bool:
    """The notebook's stated contract for captions of at most 7 photographs: a caption with 3..7 distinct photographs
    gives one test and one validation photograph, a caption with 1..2 goes to training only, and at least 4 training
    photographs remain."""
    train = sum(n - 2 if n >= 3 else n for n in sizes)
    return any(n >= 3 for n in sizes) and train >= MIN_TRAIN_RECORDS


@pytest.mark.parametrize(
    "sizes",
    sorted({tuple(sorted(c, reverse=True)) for k in (1, 2, 3) for c in itertools.product(range(1, 8), repeat=k) if sum(c) <= 9}),
)
def test_split_accepts_exactly_the_layouts_the_stated_rule_accepts(sizes) -> None:
    if _stated_rule_accepts(sizes):
        splits = split_dataset(_records(sizes), seed=0)
        assert len(splits["train"]) >= MIN_TRAIN_RECORDS and splits["test"] and splits["validation"]
    else:
        with pytest.raises(ValueError, match=r"split leaves|records; 4..2000 are required"):
            split_dataset(_records(sizes), seed=0)


def test_the_stated_examples_hold() -> None:
    assert _stated_rule_accepts((6,)) and _stated_rule_accepts((4, 4))
    assert not _stated_rule_accepts((5,)) and not _stated_rule_accepts((3, 3)) and not _stated_rule_accepts((4,))
    assert not _stated_rule_accepts((4, 3))  # 2 + 1 training photographs: "one photograph fewer" than four of each


def _zip(path: Path, sizes: tuple[int, ...]) -> Path:
    rows = io.StringIO()
    writer = csv.writer(rows)
    writer.writerow(("id", "file", "caption"))
    with zipfile.ZipFile(path, "w") as archive:
        i = 0
        for caption, n in enumerate(sizes):
            for _ in range(n):
                buffer = io.BytesIO()
                synthetic_image(width=320, height=320, seed=50 + i).save(buffer, format="PNG")
                archive.writestr(f"bird{i}.png", buffer.getvalue())
                writer.writerow((f"r{i}", f"bird{i}.png", f"a photo of bird {caption}"))
                i += 1
        archive.writestr("captions.csv", rows.getvalue())
    return path


def test_stated_smallest_byod_zips_are_accepted_and_one_less_is_refused_with_the_rule(tmp_path: Path) -> None:
    six = split_dataset(load_byod_dataset(_zip(tmp_path / "six.zip", (6,))), seed=0)
    assert {k: len(v) for k, v in six.items()} == {"test": 1, "validation": 1, "train": 4}
    four_four = split_dataset(load_byod_dataset(_zip(tmp_path / "four_four.zip", (4, 4))), seed=0)
    assert {k: len(v) for k, v in four_four.items()} == {"test": 2, "validation": 2, "train": 4}
    rule = r"at least 6 distinct images are needed, for example six of one caption or four of each of two captions"
    with pytest.raises(ValueError, match=r"split leaves 3 training records; at least 4 are required\. .*" + rule):
        split_dataset(load_byod_dataset(_zip(tmp_path / "five.zip", (5,))), seed=0)
    with pytest.raises(ValueError, match=r"split leaves 3 training records; .*" + rule):
        split_dataset(load_byod_dataset(_zip(tmp_path / "four_three.zip", (4, 3))), seed=0)


def test_opening_and_troubleshooting_state_the_same_minimum(nb: dict) -> None:
    md = _markdown(nb)
    assert md.count("at least **6 distinct photographs**") >= 2  # the BYOD paragraph and the troubleshooting row
    assert md.count("for example six of one caption or four of each of two captions") + md.count("for example six of one caption, or four of each of two captions") >= 2
    assert "two captions of three photographs each are refused" in md
    assert "`split leaves N training records`" in md
    assert "at least four photographs" not in md and "at least four training photographs" not in md


def test_reload_gives_two_new_caption_outputs_from_a_one_photo_test_set() -> None:
    """Section 9 shows `_new_prompt_0.png` and `_new_prompt_1.png`; a 6-photo BYOD set has one test photograph."""
    runner = _runner()
    assert 'new_records = [{**test_records[i % len(test_records)], "caption": NEW_PROMPT} for i in range(2)]' in runner
    assert "test_records[:2]]" not in runner.split("def stage_reload", 1)[1].split("def stage_activity", 1)[0]


# --- KIP-m4: the measured run time is stated ------------------------------------------------------------------------


def test_run_time_is_the_recorded_measurement(nb: dict) -> None:
    md = _markdown(nb)
    assert "904.7 s" in md and "about 15 minutes on a Kaggle T4" in md and "has been recorded yet" not in md
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    assert "904.7 s on a strict Kaggle T4" in registry and "not yet recorded" not in registry
    assert "904.7 s" in (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")


def test_status_is_candidate_with_the_previous_revision_labelled() -> None:
    assert "Current status: **Candidate**" in (ROOT / "STATUS.md").read_text(encoding="utf-8")
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "record the previous revision of the notebook (blob `447723996da8`)" in record
    assert "**Candidate.** The notebook was regenerated after the 2026-10-02 notebook review" in record


# --- stage runner, CPU pre-flight (stub models; skips without torch + diffusers) -----------------------------------

from test_tutorial_stages import _run  # noqa: E402
from test_tutorial_stages import preflight as _preflight_fixture  # noqa: E402,F401  (registered here as `_preflight_fixture`)


@pytest.fixture
def preflight(request):
    """The CPU pre-flight of test_tutorial_stages.py (stub models, monkeypatched with teardown); skips without torch."""
    return request.getfixturevalue("_preflight_fixture")


def test_cpu_preflight_smallest_byod_dataset_runs_every_stage(preflight, tmp_path: Path, capsys) -> None:
    """KIP-m3 end to end: six photographs of one caption split 4 / 1 / 1 and reach the end of Section 10 (one test
    photograph, used twice for the new caption); the reference and the per-image count are exported."""
    stages, run_root, weights = preflight
    archive = _zip(tmp_path / "byod.zip", (6,))
    for stage, options in (
        ("weights", ()),
        ("prepare", ("--byod", str(archive))),
        ("encode", ()),
        ("frozen", ("--steps", "2")),
        ("adapt", ("--epochs", "1", "--lr", "1e-3")),
        ("evaluate", ()),
        ("reload", ()),
        ("activity", ("--mask", "small")),
    ):
        _run(stages, run_root, weights, stage, *options)
    out = run_root / "outputs"
    prepare = json.loads((out / "prepare.json").read_text(encoding="utf-8"))
    assert {k: v["n_records"] for k, v in prepare["dataset"]["splits"].items()} == {"train": 4, "validation": 1, "test": 1}
    assert prepare["near_duplicates_across_splits"] == []
    frozen = json.loads((out / "frozen.json").read_text(encoding="utf-8"))
    assert frozen["original_photo_reference_clip"]["reference_kind"] == REAL_PHOTO_REFERENCE_KIND
    report = json.loads((out / f"{stages.STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    assert set(report["clip_above_own_original_photograph"]) == {"mean_fill_floor", "frozen", "adapted"}
    assert all(v.endswith(" of 1") for v in report["clip_above_own_original_photograph"].values())
    assert "original_photo_reference_clip" in report["references"]
    for i in (0, 1):
        assert (out / f"{stages.STEM}_new_prompt_{i}.png").is_file()
    result = json.loads((out / f"{stages.STEM}_result.json").read_text(encoding="utf-8"))
    assert result["data_source"] == "BYOD (byod.zip)" and len(result["new_prompt_predictions"]) == 2
    printed = capsys.readouterr().out
    assert "'near_duplicates_across_splits': 0" in printed and "'original_photo_reference'" in printed
    assert "clip_above_own_original_photograph" in printed
