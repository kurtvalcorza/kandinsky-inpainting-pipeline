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
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version,
  the standalone carrier and the isolated environment; `metadata.dimer` declares that profile, spec `2.2`, the
  `GUIDED` mode, `standalone: true`, `requires_dimer_worker: false` and `generated_from` (repository, revision,
  package-module SHA-256, per-file hashes, generator);
- the carrier (ST1–ST8, SRC4, PAR1–PAR4): exactly one carrier cell whose `CARRIED_FILES` equal the repository files
  named by the template (the four package modules, `tools/tutorial_stages.py`, `tutorials/requirements-colab.lock.txt`,
  the three snapshot manifests, `LICENSE`) plus the generated `source.json`; every `CARRIED_HASHES` entry is the
  SHA-256 of its text and is recorded in the cell and notebook metadata; the cell writes each file and raises on a
  hash mismatch; the notebook is byte-identical (on LF) to `tools/build_notebook.py` output for its recorded revision;
- the lock (ENV1, ENV2): it pins every `pyproject.toml` runtime pin at the same version, every entry carries
  `--hash=sha256:`, and its header records `--generate-hashes --only-binary :all:` and the manylinux x86_64 target;
- the isolated install (RUN10, ENV6, §25.13): no kernel cell runs `pip` except `uv pip install --python <isolated
  env> --require-hashes`; no `sys.executable`, `importlib`, `pip install` or `-m pip`; kernel imports limited to the
  standard library, `IPython.display` and `google.colab`; downloads (`urllib`) only in the install cell; `UV_URL`,
  `UV_BYTES` and a 64-hex `UV_SHA256` equal to the template's pinned `uv` wheel; `--managed-python` CPython 3.12.12,
  `--only-binary :all:`, the Hugging Face token and `PYTHONPATH` removal, `MPLBACKEND='Agg'`, the CUDA check in the
  isolated environment, and a `run_stage` that re-raises the stage's own error type and message;
- the learner path: the stages called in order (`weights`, `prepare`, `encode`, `frozen`, `adapt`, `evaluate`,
  `reload`, then the gated `activity`); both BYOD form fields exactly as `USE_BYOD = False  # @param
  {type:"boolean"}` and `BYOD_PATH = ''  # @param {type:"string"}` in the cell that runs `prepare` (EXE1/EXE2);
  `USE_BYOD` and `RUN_ACTIVITY` each assigned once; no learner prose that asks for a runtime restart;
- the carried stage runner's required calls (`KandinskyInpaintPipeline.from_pretrained(..., use_lora=True)`, staging
  and verification of the three snapshots, `fetch_sample_dataset`, `load_byod_dataset` and `split_dataset`,
  `validate_dataset` attaching the masks and the four refusal probes (including a mask whose size differs from its
  photograph), `dataset_manifest`, `write_dataset_csv`, the dataset-digest re-check, `encode_prompts`,
  `release_prior` and the prompt cache, the frozen `evaluate`, `generate`, `score_inpainting_preservation` and
  `score_generations` with the mean-fill floor and `real_photo_reference` (the original-photograph reference, not a
  ceiling) with the per-image count of outputs above it, `near_duplicate_pairs`, `pipe.adapt` with its explicit
  hyperparameters, the in-memory reference values, `save_artifact`, the fresh-process `from_artifact` evaluation
  with the two guaranteed checks, the second fresh-process reload with the parity check, the new-caption inference,
  the mask-size activity, the provenance fields `safetensors_only: True`, `remote_code_executed: False` and the data
  base URL, the error record), and the expected exports;
- `MODEL_ID`/`MODEL_REVISION` never rebound in a kernel cell, the revision absent from every kernel cell (it lives in
  the carried package and manifests), a 40-hex immutable commit, and the same identity string in `README.md`,
  `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions;
- forbidden patterns in the kernel and in every carried file: credential-in-URL, `git clone` / `github.com`, an
  editable install, a mutable `revision='main'`, `trust_remote_code=True`, unsafe deserialization, `extractall`,
  magics, `--no-binary` / `--no-build-isolation` / `--trusted-host` / `--extra-index-url`; and in the kernel only, a
  repository-package import;
- the §3.5 guided layer (GDL1–GDL15): **How to use this notebook**, **Where the code runs**, audience, roadmap,
  **Input → Model/System → Output**, observable learning objectives, the glossary, at least five **Predict before
  running** prompts, **Question tested**, **Expected result** / **What to notice** notes, at least five **Check your
  reasoning** boxes, the **Predict → Change one thing → Run → Observe → Explain** activity after the reload stage and
  gated by `if RUN_ACTIVITY:`, every numbered section tagged, **Troubleshooting**, the conclusion scaffold, no use of
  "workshop" for the notebook, the four infrastructure cells titled `# @title Infrastructure: …` and collapsed, and
  the learner cells not collapsed;
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release claim;
- `MODEL_CARD.md` front matter (`model_card_spec`, `pipeline_tag`, `base_model` equal to `MODEL_ID`,
  `date_published` in a valid form), no non-public or maintenance vocabulary, a single H1, the 19 required
  headings in order, and the immutable provenance section.

In CI, which has no `torch`, `tests/test_tutorial_stages.py` runs its kernel-side tests (the generated carrier writes
and verifies every file, and `run_stage` re-raises an invalid BYOD zip's refusal text) and skips the CPU pre-flight.

## CPU pre-flight of the stage runner (not runtime evidence)

On 2026-09-30, on Windows (CPython 3.12.12, `torch 2.14.0+cpu`, `diffusers 0.40.0`, in the repository `.venv`, which
has no `transformers` or `peft`, so the pre-flight stubs the runtime-version record), `tests/test_tutorial_stages.py`
passed 7 of 7:

- the generated carrier cell wrote the 11 carried files and verified each against `CARRIED_HASHES`;
- the generated `run_stage`, driving the carried `tutorial_stages.py` in a subprocess, raised
  `RuntimeError: Stage 'prepare' failed (exit 2): ValueError: …` with the validator's own text for three invalid BYOD
  zips: `captions.csv is missing columns ['caption']; required: ['id', 'file', 'caption'], optional: mask` for a zip
  without a `caption` column, `records[0]: image sides must be within 256..4096 px, got (200, 200)` for 200 × 200
  images, and `captions.csv row 2 (id 'r0'): mask 'mask0.png' is 300x300 px but its image is 500x333 px; they must
  match` for a mask of the wrong size;
- every stage (`weights`, `prepare`, `encode`, `frozen`, `adapt`, `evaluate`, `reload`, `activity`) ran in order on
  CPU against a stub UNet/MoVQ (from `tests/test_adaptation.py`), a stub prior, a stub inpainting generator (the
  kept region copied, the mask region filled), a stub CLIP scorer and 12 synthetic records with masks, each stage
  building its own pipeline so that everything crossed over through the run directory; all expected exports were
  written, the four refusal probes were rejected, the reload parity was `denoising_mse_diff` 0.0 and
  `mean_abs_pixel_diff` 0.0, and a stage run before `prepare` was refused with `data.json is missing`.

This proves the stage plumbing and the file hand-offs only. The real snapshot staging, the real prior, UNet, MoVQ,
diffusers inpainting and CLIP scorer, the `uv` bootstrap and the locked install were not exercised: they need a Linux
x86_64 GPU runtime.

## Manual verification coverage

CI has no GPU and cannot download the 16.5 GB of pinned weights, so the notebook's model execution is verified
manually (REL9). A supported executor is a Google Colab GPU runtime (T4 or better, at least 15 GB) or an equivalent
fresh Linux x86_64 GPU container running the committed notebook verbatim with **no repository checkout**.

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact commit under review and confirm static CI is green;
2. open that exact notebook revision in a new Linux x86_64 GPU runtime with no repository checkout, an empty Hugging
   Face cache, and no pre-staged files under `weights/`; the runtime needs about 30 GB of free disk and a GPU of at
   least 15 GB;
3. run the notebook top-to-bottom with `Run all` and **no runtime restart**, with every form field at its default:
   `USE_BYOD = False`, `BYOD_PATH = ''`, `STEPS = 20`, `GUIDANCE_SCALE = 4.0`, `EPOCHS = 4`,
   `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 1`, `RUN_ACTIVITY = True`, `ACTIVITY_MASK = 'large'`;
4. verify that the Section 2 carrier reports the revision recorded in `metadata.dimer.generated_from`, that the
   isolated environment reports CPython 3.12.12 and the locked versions (`torch 2.14.0`, `diffusers 0.40.0`,
   `transformers 5.17.0`, `peft 0.21.0`) with `'cuda': True`, and that the kernel's own packages were not changed;
5. verify every default-path stage completes, both guaranteed checks in the `evaluate` stage and the parity check in
   the `reload` stage pass, and no training loss is `nan`;
6. verify the exports exist in the run directory's `outputs/` (`kandinsky_inpainting_sample_captions.csv`,
   `…_frozen_grid.jpg`, `…_adapted_grid.jpg`, `…_evaluation_report.json`, `…_adapter/`, `…_result.json`,
   `…_new_prompt_0.png`, `…_new_prompt_1.png`, `…_activity_large_grid.jpg`) and that the interpretation section
   matches the observed values;
7. BYOD (REL12): with `USE_BYOD = True` and `BYOD_PATH` pointing at a small zip of photographs, masks and
   `captions.csv`, re-run from Section 4 and confirm the branch reaches export and reload; then confirm that a zip
   whose mask size differs from its photograph stops Section 4 with a `RuntimeError` naming the row, id and file;
8. record the notebook Git blob id, commit, runtime, outcome and observed metrics below as a verification record;
9. record no access tokens or other secrets.

## Recorded executions

The first three rows record the previous revision of the notebook (blob `447723996da8`), which ran the same stages in an isolated hash-locked environment, at `507e06d`; the current revision was regenerated after the 2026-10-02 review and has no hosted record yet. The rows after them record the previous notebook, which pip-installed its pins into the kernel and needed a manual restart on hosted runtimes; they remain evidence for the stage logic both revisions share. The CPU row is a pre-flight run, which is not promotion evidence.

| Date (UTC) | Subject | Runtime | Procedure | Observed result | Caveats |
|---|---|---|---|---|---|
| 2026-09-30 | `tutorials/kandinsky_inpainting_colab.ipynb` at commit `507e06d7fef2f322b48325a3afc7e4e1ff98feb4`, blob `447723996da8868d563af5edb9587238cfdcac10` | Google Colab, Tesla T4 (15,360 MiB); kernel Python 3.13.15; isolated environment: Python 3.12.12, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`, 68 hash-locked packages; environment set up in 79 s | `Run all` from a fresh runtime, form fields at their defaults | **PASS**: 11/11 code cells in one pass (execution counts 1–11), 0 errors, no restart. Held-out test `denoising_mse` 0.028245 (frozen) → 0.027998 (adapted); kept-region PSNR 25.30 → 25.61 dB and SSIM 0.9379 → 0.9421. Fresh-process reload: `denoising_mse_diff` 0.0, `mean_abs_pixel_diff` 0.0766 (0–255 scale, inside the asserted tolerance of 1.0). The executed notebook's cell sources are identical to the committed notebook | One run on one seeded split; sample-sanity evidence, not a benchmark |
| 2026-09-29 | `tutorials/kandinsky_inpainting_colab.ipynb` at commit `507e06d7fef2f322b48325a3afc7e4e1ff98feb4`, blob `447723996da8868d563af5edb9587238cfdcac10` | Kaggle batch kernel, Tesla T4 (15,360 MiB); isolated environment: Python 3.12.12, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`, 68 hash-locked packages; notebook fetched at the commit and blob-verified, Hugging Face cache empty at start | `Run all` with `nbclient` in **strict single-pass mode** (a restart request fails the run) | **PASS** in 904.7 s: 11/11 code cells in one pass, 0 errors. Held-out test `denoising_mse` 0.028245 → 0.028008; PSNR 25.30 → 25.62 dB; SSIM 0.9379 → 0.9422; reload `denoising_mse_diff` 0.0, `mean_abs_pixel_diff` 0.0723 | As above |
| 2026-09-29 | `tutorials/kandinsky_inpainting_colab.ipynb` at commit `507e06d7fef2f322b48325a3afc7e4e1ff98feb4`, blob `447723996da8868d563af5edb9587238cfdcac10`: REL12 BYOD journey | as above, strict single pass | In the executed copy only, `USE_BYOD = True` and `BYOD_PATH` = a zip built in the kernel from 12 pinned CC0 iNaturalist photographs (6 Northern Cardinal, 6 Blue Jay), 6 with their own rectangular mask; then the committed BYOD cell was re-run against three incompatible zips | **PASS** in 702.9 s: 12/12 code cells in one pass, 0 errors. Positive: 12 records carried through every stage to a fresh-process reload. Negative, each raised in the kernel before any model ran: `Stage 'prepare' failed (exit 2): ValueError: captions.csv is missing columns ['caption']; …`, `… image sides must be within 256..4096 px, got (200, 200)`, and `… mask 'mask_wrong_size.png' is 300x300 px but its image is 500x333 px; they must match` | With one test photograph per caption, these numbers show the path runs, not how well it performs |
| 2026-09-29 | `tutorials/kandinsky_inpainting_colab.ipynb` at commit `6fd3ab42084891c95ccfd6373e35343dc004c6d7`, blob `c8930286cd797f972dd49542d24d7ee9722066ca` | Kaggle batch kernel, Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0` | Notebook fetched at the commit and blob-verified, then `Run all` in a fresh interpreter with `nbclient`, an empty Hugging Face cache and no repository checkout; form fields at their defaults (`USE_BYOD = False`). The install cell's restart guard fired once because the kernel had preloaded older `numpy`, `protobuf` and `cuda-bindings`; the kernel was restarted and run again from the top | **PASS** in 906.0 s: 12/12 code cells, 0 errors. Held-out test `denoising_mse` 0.028245 (frozen) → 0.027997 (adapted). Kept-region PSNR 25.30 → 25.59 dB and SSIM 0.9379 → 0.9420. `clip_prompt_similarity` 31.078 → 30.298, against 24.026 for the mean-fill floor and 30.274 for the original photographs. Reload parity: `denoising_mse_diff` 0.0 and `mean_abs_pixel_diff` 0.068 on the 0–255 scale, inside the asserted tolerance of 1.0. Adapter `adapter.safetensors` 6,607,664 bytes, SHA-256 `a2c8a575…` | One run on one seeded split; sample-sanity evidence, not a benchmark. The BYOD branch was not exercised. The non-zero pixel difference on reload, with an identical denoising loss, is attributed to non-deterministic GPU kernels during sampling; that attribution is inferred, not isolated |
| 2026-09-29 | `tutorials/kandinsky_inpainting_colab.ipynb` at commit `6fd3ab42084891c95ccfd6373e35343dc004c6d7`, blob `c8930286cd797f972dd49542d24d7ee9722066ca`: REL12 BYOD journey | Kaggle batch kernel, Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0` | Notebook fetched at the commit and blob-verified; in the executed copy only (not committed), the BYOD form was set to `USE_BYOD = True` and `BYOD_PATH` = a zip built in the kernel from 12 CC0 research-grade iNaturalist photographs (6 Northern Cardinal, 6 Blue Jay, 12 observers, not in the sample corpus), each checked against a pinned SHA-256, with a `mask` column naming a rectangular repaint mask for 6 of them; the committed cell source was checked by SHA-256 before the edit. `Run all` in a fresh interpreter with an empty Hugging Face cache, then one appended harness cell re-ran the committed BYOD cell against three incompatible zips | **PASS** in 639.7 s: 13/13 code cells, 0 errors. Positive: 12 records (6 own masks, 6 centre-mask fallbacks) split 8 / 2 / 2 by caption and carried through validation, prompt encoding, frozen evaluation, LoRA fine-tuning, adapted evaluation, inpainting, export and reload. Negative: a `captions.csv` without `caption` was refused with `captions.csv is missing columns ['caption']; required: ['id', 'file', 'caption'], optional: mask`; a 200 × 200 image with `records[0]: image sides must be within 256..4096 px, got (200, 200)`; and a 300 × 300 mask on a 500 × 333 photograph with `captions.csv row 2 (id 'northern_cardinal-00'): mask 'mask_wrong_size.png' is 300x300 px but its image is 500x333 px; they must match`, each before any model ran on it | With one test photograph per caption, the BYOD numbers show that the path runs, not how well the model adapts to such data |
| 2026-09-29 | Stage cells (Sections 4–10) of `tutorials/kandinsky_inpainting_colab.ipynb` generated at commit `28940a7` | CPU only, CPython 3.12.12, `torch 2.14.0+cpu`; no `diffusers` sampling, no pinned weights | The carried module cells were executed, then every learner-facing stage cell in order, once on the sample path and once on the BYOD path with `BYOD_PATH` set (8 photographs, 4 with their own masks). The UNet and MoVQ were the stub models from `tests/test_adaptation.py`; the prior, the CLIP scorer, the corpus fetch and `generate` were fakes | All stage cells completed on both paths; both Section 8 assertions and the Section 9 parity assertion passed; the Section 6 frozen-model guard refused an already-adapted pipeline; all six expected `outputs/` paths were written; BYOD reported 4 own masks and 4 centre-mask fallbacks | Exercises the notebook's code paths and exports only; says nothing about the model, its numbers, GPU memory or run time; the install and model-loading cells were not run |

## Current status

**Candidate.** The notebook was regenerated after the 2026-10-02 notebook review (findings KIP-m1..m4: the original-photograph reference is no longer called a ceiling, the split's independence assumption and near-duplicate report, the stated BYOD minimum, the recorded run time, and small printed outputs), so its blob no longer matches the records above, and a hosted `Run all` of the current revision is required before it can be promoted again. The previous revision was release-grade: at `507e06d` (notebook blob `447723996da8`) the notebook runs every stage in an isolated hash-locked environment and installs nothing into the kernel. Its default `Run all` path passed in one pass on Google Colab (T4) and on a clean Kaggle T4 in strict single-pass mode, and its BYOD branch passed the REL12 journey (representative photographs with and without their own masks accepted and carried through adaptation, evaluation, export and a fresh-process reload; three incompatible inputs refused with the validator's message).
