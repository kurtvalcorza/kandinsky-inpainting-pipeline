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

No clean-runtime execution of the notebook has been recorded. The table below holds pre-flight runs, which are not
promotion evidence.

| Date (UTC) | Subject | Runtime | Procedure | Observed result | Caveats |
|---|---|---|---|---|---|
| 2026-09-29 | Stage cells (Sections 4–10) of `tutorials/kandinsky_inpainting_colab.ipynb` generated at commit `28940a7` | CPU only, CPython 3.12.12, `torch 2.14.0+cpu`; no `diffusers` sampling, no pinned weights | The carried module cells were executed, then every learner-facing stage cell in order, once on the sample path and once on the BYOD path with `BYOD_PATH` set (8 photographs, 4 with their own masks). The UNet and MoVQ were the stub models from `tests/test_adaptation.py`; the prior, the CLIP scorer, the corpus fetch and `generate` were fakes | All stage cells completed on both paths; both Section 8 assertions and the Section 9 parity assertion passed; the Section 6 frozen-model guard refused an already-adapted pipeline; all six expected `outputs/` paths were written; BYOD reported 4 own masks and 4 centre-mask fallbacks | Exercises the notebook's code paths and exports only; says nothing about the model, its numbers, GPU memory or run time; the install and model-loading cells were not run |

## Current status

**Candidate.** Promotion to **Release-grade** requires a clean-runtime record produced by the procedure above.
