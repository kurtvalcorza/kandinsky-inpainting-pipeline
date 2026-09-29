# Release verification

`tutorials/kandinsky_inpainting_colab.ipynb` (`E2E`, `GUIDED`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `ruff check src tests tools`, the offline unit suite (`pytest`), `tools/validate_release_assets.py` and
`tools/build_notebook.py --check`. The validator checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, the `GUIDED` mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`metrics.py`, `pipeline.py`, `samples.py`), each equal to its source after the
  generator's documented rewrite; the inline `MANIFEST`, `PRIOR_MANIFEST` and `SCORER_MANIFEST` equal to the three
  committed snapshot manifests and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook
  byte-identical (on LF) to `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with
  its restart-on-stale-import guard;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest), the
  revision a 40-hex immutable commit, and the same identity in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md`
  with no stray revisions;
- the stage calls the notebook must make against the real package API: staging and verification of the three
  snapshots, `KandinskyInpaintPipeline.from_pretrained(..., prior_dir=DEFAULT_PRIOR_DIR, use_lora=True)`,
  `fetch_sample_dataset`, `load_byod_dataset` behind `USE_BYOD = False` with a `BYOD_PATH` location field,
  `validate_dataset` with the refusal probes, `dataset_manifest`, `write_dataset_csv`, `encode_prompts` and
  `release_prior`, frozen and adapted `evaluate` / `generate` / `score_inpainting_preservation` /
  `score_generations` with the mean-fill floor and the original-photograph ceiling, `adapt` with explicit
  hyperparameters and the two guaranteed assertions, the new-caption inference, `save_artifact`, `from_artifact` +
  `import_prompt_cache` with the reload-parity assertion, the provenance fields, the six expected `outputs/` paths,
  and the optional `RUN_ACTIVITY` mask-size activity;
- forbidden patterns: credential-in-URL, any `git clone` / `github.com` / repository import in code, a mutable
  `revision='main'`, direct `huggingface_hub` / `safetensors` / `urllib` / `diffusers` / `transformers` / `peft` /
  `CLIPModel` use, `torch.load(` / `pickle.load`, `torch.no_grad(` / `torch.inference_mode(` / `.backward(` outside
  the carried module cells, `trust_remote_code=True`;
- the §3.5 guided layer: **How to use this notebook**, audience, roadmap, **Input → Model/System → Output**,
  observable learning objectives, the glossary, at least five **Predict before running** prompts, **Question tested**,
  **Expected result** / **What to notice** notes, at least five **Check your reasoning** boxes, the
  **Predict → Change one thing → Run → Observe → Explain** activity, the section tags, **Troubleshooting**, the
  conclusion scaffold, no use of "workshop" for the notebook, and every generated cell collapsed with the install
  and model cells titled `# @title Infrastructure: …`;
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release claim;
- `MODEL_CARD.md` front matter (`model_card_spec`, `pipeline_tag`, `base_model` equal to `MODEL_ID`,
  `date_published` in a valid form), no non-public or maintenance vocabulary, a single H1, the 19 required
  headings in order, and the immutable provenance section.

## Manual verification coverage

CI has no GPU and cannot download the 16.5 GB of pinned weights, so the notebook's model execution is verified
manually (REL9). A supported executor is a Google Colab GPU runtime (T4 or better, at least 15 GB) or an equivalent
fresh GPU container running the committed notebook verbatim with **no repository checkout**.

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact commit under review and confirm static CI is green;
2. open that exact notebook revision in a new GPU runtime with no repository checkout, an empty Hugging Face cache,
   and no pre-staged files under `weights/`; the runtime needs about 20 GB of free disk and a GPU of at least 15 GB;
3. run the notebook top-to-bottom (`Run all`) with every form field at its default: `USE_BYOD = False`,
   `BYOD_PATH = ''`, `STEPS = 20`, `GUIDANCE_SCALE = 4.0`, `EPOCHS = 4`, `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 1`,
   `RUN_ACTIVITY = True`, `ACTIVITY_MASK = 'large'`;
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to `metadata.dimer.generated_from.revision`
   and installed versions equal to the inline `PINS`;
5. verify every default-path stage completes, both assertions in Section 8 and the parity assertion in Section 9
   pass, and no training loss is `nan`;
6. verify the exports exist (`outputs/kandinsky_inpainting_sample_captions.csv`, `…_frozen_grid.jpg`,
   `…_adapted_grid.jpg`, `…_evaluation_report.json`, `…_adapter/`, `…_result.json`) and that the interpretation
   section matches the observed values;
7. BYOD (REL12): with `USE_BYOD = True` and `BYOD_PATH` pointing at a small zip of photographs, masks and
   `captions.csv`, re-run from Section 3 and confirm the branch reaches export; then confirm that a zip whose mask
   size differs from its photograph is refused with the row, id and file named;
8. record the notebook Git blob id, commit, runtime, outcome and observed metrics below as a verification record;
9. record no access tokens or other secrets.

## Recorded executions

The first two rows are clean-runtime executions at `6fd3ab4`: the default path and the REL12 BYOD journey. The CPU
row below them is a pre-flight run, which is not promotion evidence.

| Date (UTC) | Subject | Runtime | Procedure | Observed result | Caveats |
|---|---|---|---|---|---|
| 2026-09-29 | `tutorials/kandinsky_inpainting_colab.ipynb` at commit `6fd3ab42084891c95ccfd6373e35343dc004c6d7`, blob `c8930286cd797f972dd49542d24d7ee9722066ca` | Kaggle batch kernel, Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0` | Notebook fetched at the commit and blob-verified, then `Run all` in a fresh interpreter with `nbclient`, an empty Hugging Face cache and no repository checkout; form fields at their defaults (`USE_BYOD = False`). The install cell's restart guard fired once because the kernel had preloaded older `numpy`, `protobuf` and `cuda-bindings`; the kernel was restarted and run again from the top | **PASS** in 906.0 s: 12/12 code cells, 0 errors. Held-out test `denoising_mse` 0.028245 (frozen) → 0.027997 (adapted). Kept-region PSNR 25.30 → 25.59 dB and SSIM 0.9379 → 0.9420. `clip_prompt_similarity` 31.078 → 30.298, against 24.026 for the mean-fill floor and 30.274 for the original photographs. Reload parity: `denoising_mse_diff` 0.0 and `mean_abs_pixel_diff` 0.068 on the 0–255 scale, inside the asserted tolerance of 1.0. Adapter `adapter.safetensors` 6,607,664 bytes, SHA-256 `a2c8a575…` | One run on one seeded split; sample-sanity evidence, not a benchmark. The BYOD branch was not exercised. The non-zero pixel difference on reload, with an identical denoising loss, is attributed to non-deterministic GPU kernels during sampling; that attribution is inferred, not isolated |
| 2026-09-29 | `tutorials/kandinsky_inpainting_colab.ipynb` at commit `6fd3ab42084891c95ccfd6373e35343dc004c6d7`, blob `c8930286cd797f972dd49542d24d7ee9722066ca`: REL12 BYOD journey | Kaggle batch kernel, Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0` | Notebook fetched at the commit and blob-verified; in the executed copy only (not committed), the BYOD form was set to `USE_BYOD = True` and `BYOD_PATH` = a zip built in the kernel from 12 CC0 research-grade iNaturalist photographs (6 Northern Cardinal, 6 Blue Jay, 12 observers, not in the sample corpus), each checked against a pinned SHA-256, with a `mask` column naming a rectangular repaint mask for 6 of them; the committed cell source was checked by SHA-256 before the edit. `Run all` in a fresh interpreter with an empty Hugging Face cache, then one appended harness cell re-ran the committed BYOD cell against three incompatible zips | **PASS** in 639.7 s: 13/13 code cells, 0 errors. Positive: 12 records (6 own masks, 6 centre-mask fallbacks) split 8 / 2 / 2 by caption and carried through validation, prompt encoding, frozen evaluation, LoRA fine-tuning, adapted evaluation, inpainting, export and reload. Negative: a `captions.csv` without `caption` was refused with `captions.csv is missing columns ['caption']; required: ['id', 'file', 'caption'], optional: mask`; a 200 × 200 image with `records[0]: image sides must be within 256..4096 px, got (200, 200)`; and a 300 × 300 mask on a 500 × 333 photograph with `captions.csv row 2 (id 'northern_cardinal-00'): mask 'mask_wrong_size.png' is 300x300 px but its image is 500x333 px; they must match`, each before any model ran on it | With one test photograph per caption, the BYOD numbers show that the path runs, not how well the model adapts to such data |
| 2026-09-29 | Stage cells (Sections 4–10) of `tutorials/kandinsky_inpainting_colab.ipynb` generated at commit `28940a7` | CPU only, CPython 3.12.12, `torch 2.14.0+cpu`; no `diffusers` sampling, no pinned weights | The carried module cells were executed, then every learner-facing stage cell in order, once on the sample path and once on the BYOD path with `BYOD_PATH` set (8 photographs, 4 with their own masks). The UNet and MoVQ were the stub models from `tests/test_adaptation.py`; the prior, the CLIP scorer, the corpus fetch and `generate` were fakes | All stage cells completed on both paths; both Section 8 assertions and the Section 9 parity assertion passed; the Section 6 frozen-model guard refused an already-adapted pipeline; all six expected `outputs/` paths were written; BYOD reported 4 own masks and 4 centre-mask fallbacks | Exercises the notebook's code paths and exports only; says nothing about the model, its numbers, GPU memory or run time; the install and model-loading cells were not run |

## Current status

**Candidate.** The notebook stops at its install cell on hosted runtimes that preload `numpy`, `protobuf` and `cuda-bindings` (Google Colab and Kaggle), because the pinned install replaces those loaded packages and the cell then asks for a manual runtime restart. Notebook Specification 2.2 RUN1 and RUN10 forbid a manual restart on the `Run all` path, so the tutorial is not release-ready. The Kaggle runs recorded in `docs/release-verification.md` completed only because the executor restarted the kernel automatically; they remain valid evidence for everything after the install cell. Found in a Colab `Run all` on 2026-09-29; the fix (an isolated, hash-locked environment for the tutorial stages) is in progress.
