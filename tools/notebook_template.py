"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone, §25.13 isolated environment).

The generator writes the infrastructure cells (runtime check, carrier, isolated install + stage runner, snapshot
staging) from repository files; this template holds the learner-facing prose, the list of carried files and the
learner cells. Every learner cell calls ``run_stage(...)``: the carried ``tutorial_stages.py`` (``tools/`` in the
repository) runs one stage per process in an isolated, hash-locked environment, so nothing is installed into the
notebook kernel.

This template configures an E2E, GUIDED inpainting fine-tuning workflow: the pinned Kandinsky 2.2 inpainting
decoder, its shared diffusion prior and a CLIP scorer are staged and digest-verified; 60 pinned CC0 iNaturalist bird
photographs are fetched, given the deterministic centre mask, validated and split; every prompt is encoded once with
the prior and the prior is released; the frozen model is measured (held-out denoising loss, preservation of the kept
region, CLIP prompt similarity beside a mean-colour-fill floor and an original-photograph reference); a bounded LoRA
fine-tuning runs and the adapter is exported; the exported adapter is measured in a fresh process on identical
inputs; a second fresh process reloads it, checks parity with the trained in-memory model and inpaints a new caption
into held-out photographs; and one optional Predict -> Change one thing -> Run -> Observe -> Explain activity changes
the mask size.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "kandinsky-inpainting-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/kandinsky_inpainting_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-kandinsky--community%2Fkandinsky--2--2--decoder--inpaint-ffcc4d?style=flat",
        "https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-ai--forever%2FKandinsky--2-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/ai-forever/Kandinsky-2",
    ),
    ("License", "https://img.shields.io/badge/License-Apache--2.0-blue.svg", "https://www.apache.org/licenses/LICENSE-2.0"),
]

ORIENTATION = [
    (
        "## How to use this notebook\n\n"
        "**Who this notebook is for.** Learners who can open a hosted notebook, run cells in order and read short Python, "
        "and who want to see how a diffusion inpainting model is measured and adapted. No prior diffusion-model experience is "
        "assumed: each term is explained where it is first used, and the glossary below collects them. A Linux **GPU** runtime "
        "is required (Colab: Runtime → Change runtime type → T4 GPU), with about 30 GB of free disk.\n\n"
        "- **Run all** from a fresh GPU runtime (Runtime → Run all). Every cell runs in order without editing; nothing asks for "
        "input on the default path, and no runtime restart is needed. Sections 1–3 build an isolated environment from "
        "hash-locked packages and download about 16.5 GB of verified weights, so they take the longest before any model runs.\n"
        "- **Infrastructure** cells (Sections 1–3) are collapsed and titled *Infrastructure*. They check the runtime, carry the "
        "repository's code, build the isolated environment and verify the model files. You may run them without studying "
        "their implementation.\n"
        "- **Form fields** (`# @param`) such as `USE_BYOD`, `STEPS` or `EPOCHS` can be changed in the form view. The defaults "
        "are the canonical path; change them only after one complete run.\n"
        "- Each section is tagged by kind: **[Concept]** explains an idea, **[Evaluation practice]** concerns fair "
        "measurement, and **[Engineering]** concerns reproducibility.\n"
        "- Before each principal result you are asked to **predict**. Write your prediction down first. After the result, a "
        "**What to notice** note describes normal output, and a **Check your reasoning** box holds a sample answer: open it "
        "only after you have answered.\n\n"
        "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell calls "
        "`run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated environment's "
        "Python, streams what it prints, and stops the notebook with the stage's own error message if it fails. Stages hand "
        "results to each other only through files in the run directory — the verified snapshots, the caption embeddings, the "
        "adapter artifact and JSON records — and GPU memory is released when each stage ends.\n\n"
        "**Roadmap.** 1–3 check the runtime, carry the code, build the isolated environment and verify the model "
        "(Infrastructure) → 4 photographs, masks, validation and splits → 5 prompt encoding → 6 the frozen baseline → "
        "7 LoRA fine-tuning and export → 8 the paired held-out comparison → 9 fresh reload and a new caption → 10 change one "
        "thing: the mask size → interpretation, conclusion and troubleshooting. Optional: re-run from Section 4 with your own "
        "photographs and masks (Bring Your Own Data).\n\n"
        "**Input → Model/System → Output.** A photograph, a mask marking the region to repaint (white = repaint, black = keep) "
        "and a caption → prior (caption → CLIP image embedding) → inpainting UNet (denoises a latent, conditioned on the "
        "embedding, the kept latents and the keep mask) → MoVQ decoder → a 512 × 512 photograph whose masked region is "
        "repainted to match the caption and whose kept region should be preserved."
    ),
    (
        "<details>\n"
        "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
        "| Term | Meaning in this notebook |\n"
        "|---|---|\n"
        "| **Inpainting** | Repainting a masked region of a photograph so that it matches a caption and blends with the rest. |\n"
        "| **Mask** | A greyscale image the size of the photograph. In this repository white (255) marks the region to repaint and black (0) the region to keep. |\n"
        "| **Keep mask** | The inverse mask (1 = keep) that the Kandinsky 2.2 inpainting UNet reads as its ninth input channel; `unet_keep_mask` converts one into the other. |\n"
        "| **Latent** | A compressed 64 × 64 × 4 representation of a 512 × 512 image, produced by the MoVQ encoder; diffusion runs on latents, not pixels. |\n"
        "| **Diffusion / denoising** | Generation starts from random noise and removes it step by step; the UNet predicts the noise to remove. |\n"
        "| **Prior** | A second diffusion model that turns a caption into a CLIP image embedding, the condition the UNet reads. |\n"
        "| **Classifier-free guidance** | Mixing a conditioned and an unconditioned prediction; `GUIDANCE_SCALE` sets how strongly the caption steers the result. |\n"
        "| **Denoising loss** | The mean squared error between predicted and true noise: the training objective, measured here on held-out photographs. |\n"
        "| **Preservation (PSNR, SSIM)** | How closely the kept region of the output matches the input photograph. Higher is closer. |\n"
        "| **CLIP prompt similarity** | A frozen CLIP model's similarity score between an image and its caption: an automated proxy, not a human judgement. |\n"
        "| **LoRA** | Low-rank adapter matrices added to the UNet's attention projections; only they are trained, the base model stays frozen. |\n"
        "| **Held-out** | Photographs never used for training or for choosing the kept epoch (the test split). |\n"
        "| **Paired comparison** | Measuring the frozen and the adapted model on identical inputs, noise and seeds, so the difference is the adaptation. |\n"
        "| **Hash-locked environment** | A separate Python environment built from a requirements file that pins every package to one version and one set of SHA-256 digests; the installer refuses anything else. |\n"
        "| **Stage** | One step of the workflow run as its own process by `run_stage`; it reads the files earlier stages wrote and writes its own. |\n\n"
        "</details>"
    ),
]

CHECK = "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n{body}\n\n</details>"

TEMPLATE = {
    "package": "kandinsky_inpainting_pipeline",
    "repo_name": REPO,
    "stem": "kandinsky_inpainting",
    "notebook_name": "kandinsky_inpainting_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh Linux **GPU** runtime (a 16 GB T4 is enough; see the Prerequisites) builds an isolated "
        "Python environment from the carried hash-locked requirements (torch, diffusers, transformers, peft, accelerate, "
        "sentencepiece, safetensors, huggingface-hub, numpy, pillow and their dependencies) without touching the notebook kernel's "
        "own packages, then runs each stage below in its own process: it stages and digest-verifies three pinned snapshots from "
        "the Hub — the 5.28 GB Kandinsky 2.2 inpainting decoder (UNet + MoVQ), the 10.57 GB Kandinsky 2.2 diffusion prior and a "
        "0.61 GB CLIP scorer — fetches 60 CC0 iNaturalist bird photographs as digest-verified JPEGs (about 6 MB, no credential), "
        "gives each the deterministic centre mask, validates them and splits them 36 / 12 / 12 by seed, encodes every prompt "
        "with the prior and releases the prior, measures the frozen model (held-out denoising loss, preservation of the kept "
        "region, and CLIP prompt similarity beside a mean-colour-fill floor and an original-photograph reference), runs a "
        "bounded LoRA fine-tuning (4 epochs over 36 photographs) and exports the adapter as safetensors with a manifest, "
        "measures the exported adapter in a fresh process on identical inputs, reloads it in a second fresh process to verify "
        "parity with the trained in-memory model before inpainting a new caption into held-out photographs, and finally runs "
        "one small optional activity that changes the mask size. The default path needs no repository clone, no DIMER worker "
        "or service, no credential, no upload dialog, no configuration edit and no runtime restart (NOTEBOOK_SPEC 2.2 §5). The "
        "whole run took about 15 minutes on a Kaggle T4 (904.7 s, measured 2026-09-29 in strict single-pass mode with an empty "
        "model cache, so the 16.5 GB of downloads are included) at the previous revision of this notebook, which ran the same "
        "stages; this revision changes explanations and small printed outputs only. The BYOD journey on the same runtime took "
        "702.9 s."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell: every stage is a "
        "fresh process that rebuilds an unadapted pipeline from the verified snapshots (already staged), so the baseline you "
        "measure is the frozen model again. Supply your own photographs, masks and captions as a zip holding `captions.csv` "
        "beside the files. The table needs the columns `id`, `file` and `caption`, and may carry a `mask` column naming a mask "
        "image of the same size (white = repaint, black = keep); a row without a mask gets the deterministic centre mask, and "
        "the cell reports how many records used it. Photographs must have a shorter side of at least 256 px and a longer side "
        "of at most 4096 px. The split is stratified within each caption: a caption with 3..7 distinct photographs gives one "
        "test and one validation photograph (about 20 % each from 8 photographs on), a caption with one or two photographs "
        "goes to training only, and at least 4 training photographs must remain — so you need at least **6 distinct "
        "photographs**, for example six of one caption or four of each of two captions; two captions of three photographs "
        "each are refused (2 training photographs). The split assumes your photographs are independent: remove near-duplicates "
        "(bursts, crops or edits of one scene), because only pixel-identical copies are removed; Section 4 reports any "
        "near-duplicate pair that crosses splits. Set `BYOD_PATH` to a zip or directory already in the runtime to skip the "
        "upload dialog. Every caption appears in training, validation and test, and your records flow through the same contract — "
        "validation, prompt encoding, frozen baseline, adaptation, held-out evaluation, inference, artifact export and reload "
        "parity. An invalid zip stops Section 4 with a `RuntimeError` that repeats the validator's message. Uploaded files "
        "stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "weights_key": "kandinsky-2-2-decoder-inpaint",
    # Carried byte for byte (UTF-8 text, LF newlines) into the run directory and verified against CARRIED_HASHES.
    "carried": {
        "src/kandinsky_inpainting_pipeline/__init__.py": "src/kandinsky_inpainting_pipeline/__init__.py",
        "src/kandinsky_inpainting_pipeline/pipeline.py": "src/kandinsky_inpainting_pipeline/pipeline.py",
        "src/kandinsky_inpainting_pipeline/samples.py": "src/kandinsky_inpainting_pipeline/samples.py",
        "src/kandinsky_inpainting_pipeline/metrics.py": "src/kandinsky_inpainting_pipeline/metrics.py",
        "tutorial_stages.py": "tools/tutorial_stages.py",
        "requirements.txt": "tutorials/requirements-colab.lock.txt",
        "weights/kandinsky-2-2-decoder-inpaint/dimer-base-manifest.json": "weights/kandinsky-2-2-decoder-inpaint/dimer-base-manifest.json",
        "weights/kandinsky-2-2-prior/dimer-base-manifest.json": "weights/kandinsky-2-2-prior/dimer-base-manifest.json",
        "weights/clip-vit-b-32-laion2b/dimer-base-manifest.json": "weights/clip-vit-b-32-laion2b/dimer-base-manifest.json",
        "LICENSE": "LICENSE",
    },
    "stage_runner": "tutorial_stages.py",
    "lock": "requirements.txt",
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "disk_gib": {"weights": 17.5, "environment": 12},
    "runtime_modules": ["torch", "diffusers", "transformers", "peft"],
    "title": "Kandinsky 2.2 Inpainting — DIMER guided end-to-end notebook (standalone)",
    "badges": BADGES,
    "capability": "masked image inpainting with a 1.25 B-parameter UNet diffusion model, held-out denoising-loss, preservation and CLIP-scored evaluation, and bounded LoRA fine-tuning to a set of captioned photographs",
    "intro": (
        "Kandinsky 2.2 is a two-stage latent diffusion model from ai-forever (Razzhigaev et al., 2023). A diffusion prior with "
        "CLIP text and image encoders maps a caption to a CLIP image embedding. The inpainting decoder's UNet "
        "(1,253,074,568 parameters) then denoises a 64 × 64 × 4 latent. Its input has nine channels: the four noisy latent "
        "channels, the four latent channels of the photograph with the repaint region zeroed, and one **keep mask** channel "
        "(1 = keep). A MoVQ autoencoder encodes the photograph to latents and decodes the result to 512 × 512 RGB pixels. "
        "Three pinned snapshots make one inpainting system: the decoder (5.28 GB safetensors), the shared diffusion prior "
        "(10.57 GB safetensors) and, for evaluation only, a CLIP ViT-B/32 scorer.\n\n"
        "Two properties are handled in the open. **The prior pipeline can be released before training**: Section 5 encodes "
        "every caption the notebook uses into image embeddings, each prior run seeded from a SHA-256 of its caption, and "
        "releases the prior so that the UNet, the MoVQ, the scorer and a training graph fit on a 16 GB GPU. **Inpainting has "
        "no single ground truth** for the repainted region, so the notebook reads three kinds of number: the held-out "
        "*denoising loss* (the training objective on photographs the model never trained on), the *preservation* of the kept "
        "region, and CLIP scores of the outputs beside two references: a mean-colour-fill floor and the original-photograph "
        "reference (a reference line, not a ceiling — an output can score above its original). None of these is a human "
        "judgement of image quality."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to **explain** what the mask, the keep-mask channel, the prior and the "
        "latent are, and how an inpainting UNet uses them; **inspect** a validated, digest-pinned photograph dataset and "
        "**identify** what validation refuses; **predict**, then **compare**, the frozen and the adapted model on identical "
        "held-out inputs; **interpret** a denoising loss, a preservation score and a CLIP similarity, and **explain** why none "
        "of them alone measures image quality; **apply** a bounded LoRA fine-tuning with explicit hyperparameters; "
        "**verify** that an exported safetensors adapter reloads against the pinned base with the same outputs; and "
        "**diagnose** how the mask size changes what the model repaints and preserves."
    ),
    "exclusions": (
        "text-to-image generation without a source photograph, depth or ControlNet conditioning, outpainting, full fine-tuning, "
        "DreamBooth identifiers, safety filtering of prompts or images, human preference studies, FID, prompt engineering, "
        "and any claim that a CLIP score, a preservation score or a denoising loss measures image quality."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported **Linux x86_64 GPU** runtime (Google Colab T4 or better, Kaggle T4, or a Linux Jupyter kernel with a CUDA GPU of at least 15 GB). The kernel's own Python version does not matter: the notebook installs nothing into it, and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (torch 2.14.0, whose Linux wheel is the CUDA 13.0 build). The prior runs in float16 while it encodes captions and is then released; the UNet and MoVQ run in float16; the LoRA tensors are kept in float32 and training uses float16 autocast with loss scaling. CPU-only runtimes are not supported for this notebook. About 18 GB of disk is needed for the snapshots and about 12 GB for the isolated environment.",
        "- **Knowledge:** basic Python and notebooks. Helpful but not required: what a diffusion model does at inference, what classifier-free guidance is, and why a training loss is not a quality score. The glossary above covers the terms.",
        "- **Weights:** the decoder, the diffusion prior and the CLIP scorer are all safetensors; nothing is unpickled and no Hub-hosted code is executed — the model classes come from `diffusers`, `transformers` and `peft` on PyPI. The Kandinsky weights are released under Apache-2.0; the scorer is MIT.",
        "- **Data contract:** a record is `{{id, image, caption}}` plus an optional `mask_image` — an RGB photograph with shorter side at least 256 px and longer side at most 4096 px, a caption of 1..1000 characters, and a greyscale mask of the same size (255 = repaint). A record without a mask gets the deterministic centre mask (the middle half of each side). Photograph and mask are resized together so the shorter side is 512 px and centre-cropped to 512 × 512. Validation is structural: nothing checks that a caption describes its photograph or that a mask covers anything sensible.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of identifiable people, licensed stock images or client material are exactly that. The default path uploads nothing, and uploaded BYOD files are processed only inside this runtime.",
        "- **External access (data):** besides the Hub, the default path fetches 60 pinned photographs (about 6 MB) from the public iNaturalist open-data bucket `inaturalist-open-data.s3.amazonaws.com` over HTTPS, digest-verified before decoding; every photo is CC0 and its observation page is recorded.",
    ],
    "guided": {"opening": ORIENTATION},
    "setup": [
        {
            "cell": "check",
            "md": (
                "## 1. Check the runtime · [Engineering]\n\n"
                "> **Infrastructure.** The code cells in Sections 1–3 are collapsed. You may run them without studying their "
                "implementation; they exist for reproducibility and provenance. The learning activities start in Section 4.\n\n"
                "**Input:** a fresh hosted runtime. **System:** checks that it is Linux x86_64 with a CUDA GPU and enough free "
                "disk, and creates a new run directory. **Output:** the GPU name and the directories this run will use. Each run "
                "writes to a new directory under `outputs/{stem}/`, so an earlier export cannot be mistaken for a current result. "
                "The verified snapshots are kept in `weights/` and reused by a later run."
            ),
            "after": (
                "**Expected result:** one dictionary naming the GPU (for example `Tesla T4, 15360 MiB`), the kernel's Python "
                "version, the run directory, the weights directory, the isolated environment's directory and the free disk. "
                "If the cell stops with a GPU or disk message, see **Troubleshooting**."
            ),
        },
        {
            "cell": "carrier",
            "md": (
                "## 2. Carry the code and install the locked runtime · [Engineering]\n\n"
                "> **Infrastructure.** The next two code cells are collapsed. The first **is** the repository's code, carried so "
                "that this notebook works on its own; the second builds the environment every stage runs in.\n\n"
                "The first cell holds, as text, the files the workflow needs: the package's four modules under "
                "`src/kandinsky_inpainting_pipeline/` (identity constants, snapshot verification and staging, validation and masks, "
                "the pipeline class, the sample corpus and the preservation and CLIP metrics), the stage runner "
                "`tutorial_stages.py`, the hash-locked `requirements.txt` ({n_locked} packages), the three snapshot manifests and "
                "the licence. It writes each file into the run directory and checks its SHA-256 against `CARRIED_HASHES`, stopping "
                "on any mismatch. The text is the repository's files byte for byte; the repository's parity test "
                "(`tests/test_notebook_parity.py`) fails whenever the two diverge, so what runs here is what the repository tests. "
                "Nothing in this cell runs a model."
            ),
            "after": (
                "**Expected result:** `carried_files`, `verified: True`, and the repository revision the notebook was generated "
                "from.\n\n"
                "The next cell installs nothing into this notebook's kernel. It downloads one pinned file — the `uv` installer "
                "wheel, refused unless its size and SHA-256 match — creates a separate virtual environment with its own CPython "
                "3.12.12, and installs `requirements.txt` into it with `--require-hashes --only-binary :all:`: every package must "
                "be the locked version, a prebuilt wheel, and match a locked digest. The hosted runtime's own packages are never "
                "replaced, which is why no restart is needed. The cell also defines `run_stage`, `load_record` and `show_image`, "
                "the three helpers the learner cells use."
            ),
        },
        {
            "cell": "install",
            "md": (
                "**Infrastructure: the isolated environment.** Installation messages from `uv` are normal and can take a few "
                "minutes. A failed download or a hash mismatch stops the cell; never remove a pin or a hash to get past one."
            ),
            "after": (
                "**Expected result:** one dictionary with the generating revision, the isolated environment's Python (3.12.12), "
                "`torch`, `diffusers`, `transformers` and `peft` versions, `'cuda': True`, the number of locked packages and the "
                "setup time. If `'cuda'` is `False` the cell stops: this notebook is not supported on a CPU-only runtime; see "
                "**Troubleshooting**."
            ),
        },
        {
            "cell": "weights",
            "md": (
                "## 3. Pin, stage and verify the model · [Engineering]\n\n"
                "> **Infrastructure.** The next code cell is collapsed. It downloads about 16.5 GB of pinned weights and checks "
                "every file's size and SHA-256; you may run it without studying its implementation.\n\n"
                "The model identity is carried twice — `MODEL_ID`/`MODEL_REVISION` in the carried `pipeline.py` and the manifest "
                "of each snapshot (paths, byte sizes, SHA-256) — and the `weights` stage first checks that they agree. It installs "
                "each carried manifest into `weights/`, fetches exactly the files that are absent from the Hugging Face Hub **at "
                "the pinned revisions** (never `main`), and re-hashes every file, raising on the first size or digest mismatch. "
                "Three snapshots are staged: the inpainting decoder (`{MODEL_ID}` at `{MODEL_REVISION}`), the shared diffusion "
                "prior and the CLIP scorer. There is no fallback to a different download, and no remote model code is executed. "
                "Every later stage verifies the snapshots it loads again before loading them."
            ),
            "after": (
                "**What to notice:** for each of the three snapshots, its id, revision, licence and file count; a `fetched` list of "
                "the files downloaded on this run (empty on a rerun, because staging only fetches files that are absent); and a "
                "count of verified files. A size or SHA-256 mismatch stops the cell with a `ValueError` naming the file — see "
                "**Troubleshooting**, and never edit a manifest to get past one."
            ),
        },
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample photographs, masks, validation and splits · [Concept]\n\n"
                "From here on, every code cell runs one stage of the carried runner with `run_stage`; its printed dictionaries "
                "appear under the cell. This cell runs the `prepare` stage. The default dataset is 60 research-grade iNaturalist "
                "photographs of six common North American birds — 10 per species, one per observer per species, every one CC0 — "
                "fetched by photo id from the open-data bucket and refused on any byte-size or SHA-256 mismatch (`fetch_corpus`). "
                "Each caption is generated from the species by one template. `build_sample_dataset` draws a seeded stratified "
                "split — 6 / 2 / 2 per species for training, validation and test. The sample has no hand-drawn masks, so "
                "`validate_dataset` gives every record the deterministic **centre mask**: a white box over the middle half of each "
                "side, which covers a quarter of the photograph and usually most of the bird. `dataset_manifest` validates every "
                "split, checks that no photograph appears twice and records a digest. The stage records the split so that every "
                "later stage rebuilds exactly these records and refuses to run if they changed.\n\n"
                "**The split assumes independent photographs.** A seeded random split, stratified within each species, gives an "
                "honest held-out test only if no photograph has a near-copy on the other side; the sample takes one photograph per "
                "observer per species for that reason. Every species appears in all three sets, so the test measures *new "
                "photographs of seen captions*, not unseen captions. Only pixel-identical copies are removed, so the stage also "
                "reports `near_duplicates_across_splits`: pairs in different splits whose 16 × 16 greyscale thumbnails correlate at "
                "0.95 or more. With BYOD, remove near-duplicates (bursts, crops or edits of one scene) before uploading.\n\n"
                "**Predict before running:** the centre mask covers the middle half of each side. What fraction of each "
                "512 × 512 photograph will be repainted? Which of the four probes below do you expect validation to refuse?"
            ),
            "code": (
                'from pathlib import Path\n\n'
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                'BYOD_PATH = \'\'  # @param {{type:"string"}}\n\n'
                'prepare_options = []\n'
                'if USE_BYOD:\n'
                '    if BYOD_PATH:\n'
                '        byod_path = Path(BYOD_PATH)\n'
                '    else:\n'
                '        from google.colab import files\n'
                '        uploaded = files.upload()\n'
                '        file_name, payload = next(iter(uploaded.items()))\n'
                "        byod_path = ROOT / 'byod' / Path(file_name).name\n"
                '        byod_path.parent.mkdir(parents=True, exist_ok=True)\n'
                '        byod_path.write_bytes(payload)\n'
                "    prepare_options = ['--byod', byod_path.resolve()]\n"
                "run_stage('prepare', *prepare_options)"
            ),
        },
        {
            "md": (
                "**What to notice:** 36 / 12 / 12 records and six distinct captions on the sample path, `'disjoint'` listing the "
                "same counts, a shorter side of a few hundred pixels (every photograph is resized and centre-cropped to 512²), "
                "a `repaint_fraction` between 0.25 (a square photograph) and about 0.33 (a 4:3 photograph), a written `outputs/{stem}_sample_captions.csv` in the run directory in the shape "
                "BYOD expects (its `mask` column is empty because the sample uses the centre mask), and every probe rejected before "
                "any model runs, each message naming the failed rule. `near_duplicates_across_splits` should be 0 on the sample. "
                "With `USE_BYOD = True` the stage first reports how many of "
                "your records brought their own mask, and an invalid zip stops this cell with a `RuntimeError` that repeats the "
                "validator's message.\n\n"
                + CHECK.format(
                    body=(
                        "The mask box spans the middle half of the photograph's width and height, 0.5 × 0.5 = 0.25 of its "
                        "pixels. The centre crop to 512² then trims the longer side but keeps the whole box, so inside the "
                        "model's square input the box covers more: 0.25 for a square photograph, 0.5 × 0.667 ≈ 0.333 for a 4:3 "
                        "one. That is why `repaint_fraction` varies between photographs. All four probes are refused: a missing caption and a duplicate id "
                        "break the record schema, a 200 px side is below the 256 px minimum, and a mask of a different size "
                        "cannot be aligned with its photograph. Validation never looks at *content*: a mask over the sky and a "
                        "caption that describes another bird would both pass."
                    )
                )
            ),
        },
        {
            "md": (
                "## 5. Encode every caption with the prior, then release it · [Concept]\n\n"
                "The `encode` stage loads the pipeline and calls `pipe.encode_prompts`, which loads the Kandinsky 2.2 diffusion "
                "prior from the verified snapshot (float16 on CUDA), runs the prior's own diffusion process to produce a CLIP "
                "image embedding for each distinct caption, and keeps the embeddings on the CPU. Each caption's prior run is seeded "
                "with `prompt_seed(caption)`, a 31-bit number taken from the caption's SHA-256, so the same caption gets the same "
                "seed in every process. Encoded here: the six captions (training, validation and test share them), one new caption "
                "for Section 9, and the empty negative caption that classifier-free guidance needs. `release_prior` then frees the "
                "prior's GPU memory. The embeddings are written to a safetensors cache in the run directory, and every later stage "
                "reads them from there instead of loading the prior again.\n\n"
                "**Expected result:** eight captions encoded (seven on the sample path plus the empty one), GPU memory falling "
                "after the release, and the path of the prompt cache."
            ),
            "code": "run_stage('encode')",
        },
        {
            "md": (
                "## 6. The frozen baseline: denoising loss, preservation and CLIP · [Evaluation practice]\n\n"
                "**Question tested:** before any training, how well does the pretrained model predict noise on held-out "
                "photographs, how well does it preserve the kept region, and how well do its repainted photographs match their "
                "captions? The `frozen` stage builds the pipeline with `use_lora=True`, but the adapter's B matrices start at "
                "zero, so this is the pretrained model. These numbers are recorded as the **baseline** for Section 8.\n\n"
                "- **Held-out denoising loss** (`pipe.evaluate`): each held-out photograph is MoVQ-encoded, noised at five fixed "
                "timesteps (100, 300, 500, 700, 900) with seeded noise, and the UNet — given the masked latents and the keep "
                "mask — predicts that noise; the score is the mean squared error. The same seed gives the same latents, noise "
                "and timesteps later, so the adapted number is a paired comparison, not a re-draw.\n"
                "- **Preservation** (`score_inpainting_preservation`): PSNR in dB and SSIM between the input and the output, "
                "computed only over the kept (black) region. The UNet never repaints that region, but the output is decoded by "
                "the MoVQ, so preservation measures how faithfully the kept region survives the encode-decode round trip. The "
                "SSIM here is one global statistic over the kept pixels, not the windowed SSIM of image-quality papers.\n"
                "- **CLIP prompt similarity** (`score_generations`): the pinned CLIP ViT-B/32 scores each whole output against "
                "its caption. Two references sit beside it: a **mean-fill floor**, where the repaint region is filled with the "
                "mean colour of the kept region (no model at all), and the **original-photograph reference**, each unmasked "
                "photograph scored against its own caption the same way (`real_photo_reference`). The reference is a reference "
                "line, not a ceiling: a model that repaints the masked region to match the caption can match the caption more "
                "closely than a field photograph does, which only has to contain the bird, so an output can score above its own "
                "original.\n\n"
                "**Predict before running:** will the frozen model's CLIP similarity sit nearer the mean-fill floor, nearer the "
                "original-photograph reference, or above the reference? Will preservation be perfect (PSNR above 60 dB) or visibly "
                "imperfect?"
            ),
            "code": (
                'STEPS = 20  # @param {{type:"integer"}}\n'
                'GUIDANCE_SCALE = 4.0  # @param {{type:"number"}}\n\n'
                "run_stage('frozen', '--steps', STEPS, '--guidance', GUIDANCE_SCALE)\n"
                "show_image('{stem}_frozen_grid.jpg', 'Frozen model: photograph, masked view and output, one row per test photograph')"
            ),
        },
        {
            "md": (
                "**What to notice:** the denoising loss by timestep — noise is harder to predict at low timesteps (little noise) "
                "than the mean suggests, so compare timestep by timestep later. The mean-fill floor should be the lowest CLIP "
                "number; the frozen model may sit below or above the original-photograph reference; the grid under the output (`outputs/{stem}_frozen_grid.jpg` in the run directory) shows each "
                "photograph, its masked view and the frozen output side by side. Preservation is high but not perfect.\n\n"
                + CHECK.format(
                    body=(
                        "A mean-colour fill leaves most of the bird out, so CLIP usually scores it well below the original "
                        "photograph. A working inpainting model scores well above the floor, and it can score **above** the "
                        "original photograph: it repaints the masked region to match the caption, while the original only has to "
                        "contain the bird. In the recorded Kaggle T4 run of the previous revision the floor was 24.03, the "
                        "frozen model 31.08 and the original photographs 30.27, and 7 of the 12 frozen outputs scored above their own "
                        "original; on a BYOD run both models sat about 7 points above the reference. So the original photograph is a "
                        "reference line, not a ceiling. Preservation is not perfect because the kept region is not copied "
                        "pixel for pixel: it is carried through the MoVQ latents and decoded, which blurs fine texture slightly. "
                        "A PSNR in the high 20s or 30s dB is a round-trip loss, not the model repainting the kept region. None of "
                        "these numbers says whether a person would find the repainted bird convincing."
                    )
                )
            ),
        },
        {
            "md": (
                "## 7. Bounded LoRA fine-tuning · [Concept]\n\n"
                "The `adapt` stage calls `pipe.adapt`, which trains the 176 LoRA tensors (rank 8, 1,646,592 parameters — 0.13 % of "
                "the UNet) that `peft` attached to the UNet's attention projections `to_q`, `to_k`, `to_v` and `to_out.0`, and "
                "nothing else; the base UNet, the MoVQ and the prior are frozen. This is gradient-based parameter-efficient "
                "fine-tuning, not in-context conditioning. Each step takes one training photograph's latents and keep mask "
                "(MoVQ-encoded once), draws a timestep uniformly from the 1,000-step schedule and a noise tensor (both seeded), adds "
                "the noise, and minimises the mean squared error between the predicted and the true noise. The optimiser is AdamW "
                "at a fixed learning rate, with gradient-norm clipping at 1.0; on CUDA the LoRA tensors stay in float32 while the "
                "forward pass uses float16 autocast with a gradient scaler. Epoch 0 records the frozen model's validation loss, and "
                "the epoch with the lowest validation denoising loss is kept.\n\n"
                "Before its process ends, the stage records two reference values of the trained model still in memory — its test "
                "denoising loss and one inpainted photograph at a fixed seed — and exports the adapter with `pipe.save_artifact` "
                "as `outputs/{stem}_adapter/`: the 176 trained tensors as `adapter.safetensors` with a `manifest.json` recording the "
                "artifact format, the decoder's id and revision, the LoRA configuration, the tensor names, the optimiser and "
                "precision, the file size and SHA-256, and the metadata passed in. The base model is not in the artifact: it must "
                "be reloaded from the pinned revision.\n\n"
                "**Predict before running:** will the training loss fall steadily from epoch to epoch? Will the validation loss "
                "fall by more or less than the training loss?"
            ),
            "code": (
                'EPOCHS = 4  # @param {{type:"integer"}}\n'
                'LEARNING_RATE = 1e-4  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 1  # @param {{type:"integer"}}\n\n'
                "run_stage('adapt', '--epochs', EPOCHS, '--lr', LEARNING_RATE, '--batch-size', BATCH_SIZE)"
            ),
        },
        {
            "md": (
                "**What to notice:** one row per epoch, epoch 0 with no training loss; a training loss that jumps between epochs "
                "because each step draws a random timestep; a `best_epoch` that may be 0 if no epoch improved the validation "
                "loss — in that case the kept adapter is the untrained one; and the exported artifact: 176 tensors of about "
                "6.6 MB with its SHA-256.\n\n"
                + CHECK.format(
                    body=(
                        "The training loss is noisy, not a smooth curve: with one photograph per step, a step at a low timestep "
                        "and a step at a high timestep have very different losses. The validation loss is measured at five fixed "
                        "timesteps with fixed noise, so it moves more smoothly and usually by a smaller amount. A falling "
                        "validation loss shows the adapter predicts noise better on unseen photographs of the same birds; it does "
                        "not show the repainted birds look better."
                    )
                )
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation: the paired comparison · [Evaluation practice]\n\n"
                "**Question tested:** on photographs never used for training or for choosing the epoch, what changed? The "
                "`evaluate` stage is a fresh process: it loads the adapter from the exported files — the artifact you would ship, "
                "not the object that was trained — and measures it exactly as the frozen model was measured in Section 6: the same "
                "seed, so the same latents, noise and timesteps for the denoising loss, and the same photographs, masks and "
                "generation seeds for inpainting. The stage asserts only what the procedure guarantees — the kept epoch's "
                "validation loss is no higher than the frozen model's (epoch 0) — and that the exported adapter reproduces the "
                "validation loss recorded for the kept epoch. The test-split change is printed, not asserted.\n\n"
                "**Predict before running:** will the test denoising loss fall? Will CLIP prompt similarity rise or fall, and "
                "will the adapted outputs sit above or below the original-photograph reference? Will preservation change, and "
                "why or why not?"
            ),
            "code": (
                "run_stage('evaluate')\n"
                "show_image('{stem}_frozen_grid.jpg', 'Frozen model (Section 6)')\n"
                "show_image('{stem}_adapted_grid.jpg', 'Adapted model: same photographs, masks and seeds')"
            ),
        },
        {
            "md": (
                "**What to notice:** read the rows side by side. The denoising loss rows compare identical inputs; the CLIP row "
                "shows both models beside the floor and the original-photograph reference, and `clip_above_own_original_photograph` "
                "counts, per image, how many outputs scored above their own original (the reference is not a ceiling); the "
                "preservation rows should barely move. Compare the two "
                "grids under the output — same photographs, masks and seeds, so any difference comes from the adapter. The full "
                "record is `outputs/{stem}_evaluation_report.json` in the run directory.\n\n"
                + CHECK.format(
                    body=(
                        "Preservation barely moves because the kept region is carried through the latents in both models: LoRA "
                        "changes how the UNet repaints, not how the MoVQ decodes. A lower test denoising loss is the most direct "
                        "evidence that the adapter learned something that transfers to unseen photographs of these birds. A CLIP "
                        "change of a few tenths on twelve photographs from one seeded run is within the noise you would see from a "
                        "different generation seed; treat it as an observation, not an improvement. The original photograph does "
                        "not cap CLIP: outputs can score above it, so a model sitting near or above the reference has not reached "
                        "a limit, and a fall in CLIP towards the reference is not a loss of quality either."
                    )
                )
            ),
        },
        {
            "md": (
                "## 9. Fresh reload and a new caption · [Engineering]\n\n"
                "**Fresh reload.** The `reload` stage is a second fresh process. `KandinskyInpaintPipeline.from_artifact` "
                "re-verifies the decoder and prior snapshots, checks the artifact format, the base-model identity and the weights "
                "digest **before** loading the tensors, and builds a new UNet with the adapter attached — a new object from files; "
                "nothing of the trained model survives in memory between processes. It reads the caption embeddings from the cache "
                "and checks parity against the two reference values the `adapt` process recorded from the trained model while it "
                "was still in memory: the same held-out denoising loss and the same inpainted photograph for the same caption and "
                "seed, within a stated tolerance.\n\n"
                "**New-data inference.** Only after parity holds does the reloaded model inpaint `NEW_PROMPT` — a caption that "
                "appears in no training record — into the first two held-out test photographs with their masks (the same "
                "photograph twice if a BYOD test set holds one). The CLIP similarity and "
                "preservation are printed as a sanity check, not an evaluation. Finally the stage writes "
                "`outputs/{stem}_result.json` with the provenance, the runtime versions and the comparison, and lists every file in "
                "the run's `outputs/`.\n\n"
                "**Expected result:** the artifact of about 6.6 MB (1,646,592 float32 values) with its SHA-256, a reload parity with "
                "a denoising-loss difference below `1e-6` and a mean absolute pixel difference below 1.0, two new-caption "
                "predictions, the listing of `outputs/`, and the two saved PNGs."
            ),
            "code": (
                "run_stage('reload')\n"
                "show_image('{stem}_new_prompt_0.png', 'New caption, first held-out photograph (reloaded adapter)')\n"
                "show_image('{stem}_new_prompt_1.png', 'New caption, second held-out photograph (reloaded adapter)')"
            ),
        },
        {
            "md": (
                "**What to notice:** the reloaded pipeline was built in a process that never saw the trained model — only "
                "`adapter.safetensors`, `manifest.json` and the verified base snapshots — so matching numbers show that the files "
                "alone carry the adaptation. That is what the artifact boundary guarantees; it does not show that the outputs are "
                "good. `outputs/{stem}_result.json` gathers identity, provenance, runtime versions, the comparison, the new-caption "
                "predictions and the parity in one machine-readable file."
            ),
        },
        {
            "md": (
                "## 10. Change one thing: the mask size · [Evaluation practice]\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.** This optional activity runs the `activity` stage, which "
                "loads the exported adapter and inpaints the first two test photographs again with the same captions and the same "
                "generation seeds as Section 8. Only the mask changes: a centred box whose side is a quarter (`small`) or three "
                "quarters (`large`) of each side, instead of the centre mask's half. It runs after every required output has been "
                "written and changes nothing that earlier sections produced; set `RUN_ACTIVITY = False` to skip it.\n\n"
                "**Predict before running:** with a `large` mask, will the kept-region PSNR rise or fall? Will CLIP prompt "
                "similarity rise or fall? What do you expect with `small`?"
            ),
            "code": (
                'RUN_ACTIVITY = True  # @param {{type:"boolean"}}\n'
                'ACTIVITY_MASK = \'large\'  # @param ["small", "large"]\n\n'
                'if RUN_ACTIVITY:\n'
                "    run_stage('activity', '--mask', ACTIVITY_MASK)\n"
                "    show_image('{stem}_activity_' + ACTIVITY_MASK + '_grid.jpg', 'Adapted model with the ' + ACTIVITY_MASK + ' mask')\n"
                'else:\n'
                "    print({{'activity': 'skipped (optional)', 'to_run': 'set RUN_ACTIVITY = True and choose ACTIVITY_MASK, then run this cell'}})"
            ),
        },
        {
            "md": (
                "**Observe and explain:** compare each row's `centre` and changed values, then the activity grid under the output. "
                "Did the result match your prediction? Explain it in terms of what the model is allowed to repaint and what the "
                "metrics measure. Then switch `ACTIVITY_MASK` and run this cell again.\n\n"
                + CHECK.format(
                    body=(
                        "A larger mask gives the model more of the photograph to repaint, so more of the bird is generated rather "
                        "than kept. CLIP compares the whole photograph with the caption, so it can move either way: a well-repainted "
                        "large region can still score high, while a poor one pulls the score down. Kept-region PSNR is computed over "
                        "fewer pixels with a large mask, and those pixels sit near the edges of the photograph; its change reflects "
                        "the MoVQ round trip on different content, not better or worse inpainting. The lesson: preservation and "
                        "prompt similarity answer different questions, and both depend on the mask you chose."
                    )
                )
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits · [Evaluation practice]\n\n"
        "A LoRA of 1.6 million parameters trained for a few minutes on 36 photographs is compared with the frozen model on twelve "
        "held-out photographs, on identical inputs. The denoising loss shows whether the adapter predicts noise better on unseen "
        "photographs of the same six birds; preservation shows how much of the kept region survives the MoVQ round trip; CLIP "
        "reads the outputs beside a mean-fill floor and the original-photograph reference, which they can exceed. That is the "
        "claim, and it is sample-sanity evidence only.\n\n"
        "A denoising loss is the training objective, not a quality score. CLIP similarity is a frozen model's opinion, not a human "
        "judgement, and CLIP has its own biases about what a species name looks like. Twelve outputs per model from one seeded run "
        "give no dispersion estimate. Nothing here measures how natural the boundary between repainted and kept regions looks, "
        "the diversity of repaintings, or artefacts. The split assumes independent photographs; it is stratified within each "
        "caption, so the test measures new photographs of seen birds, not unseen ones. The upstream model may have seen photographs like these during pretraining, "
        "which cannot be ruled out. Fine-tuning on a narrow domain can also erode the model elsewhere — the new caption in "
        "Section 9 is one sanity check, not a test of generality.\n\n"
        "Three things to carry to real data. **Masks are part of the contract:** the model learns to repaint the region you mask, "
        "given the rest; a centre mask teaches centred subjects, so use masks shaped like the edits you intend. **Captions "
        "describe the whole photograph:** the adapter learns the association between caption text and pixels. **Stratify by "
        "caption, and keep near-duplicates together:** every caption has held-out photographs, so the test measures new "
        "photographs of seen captions; a random split assumes independent photographs, so bursts, crops or edits of one scene "
        "must not straddle it. **Licences travel "
        "with the outputs:** the Kandinsky weights are Apache-2.0 and the photographs here are CC0 — with your own data, the "
        "rights to the photographs and to what the adapter produces are yours to establish.\n\n"
        "## Conclude with evidence · [Evaluation practice]\n\n"
        "Complete this in your own words:\n\n"
        "> On [n] held-out photographs of [subject] with [mask], the adapted model's test denoising loss changed from [frozen] to "
        "[adapted], its kept-region PSNR from [frozen] to [adapted] dB, and its CLIP prompt similarity from [frozen] to [adapted], "
        "beside a mean-fill floor of [floor] and an original-photograph reference of [reference] ([k] of [n] adapted outputs "
        "above their own original). Changing the mask size showed "
        "[observation]. These are one seeded run on [n] photographs, scored by automated proxies; they do not establish "
        "[limitation]. Next I would [specific next step].\n\n"
        + CHECK.format(
            body=(
                "A strong conclusion names the photographs, the mask and the split; reports the frozen, adapted, floor and "
                "reference numbers side by side, including any that did not improve; separates what was measured (noise "
                "prediction, kept-region fidelity, CLIP agreement) from what was not (how convincing the repainting looks to a "
                "person, behaviour on other subjects); and ends with a concrete next step, such as repeating the comparison over "
                "several generation seeds or asking people to rate the repainted regions. A weak conclusion says only that "
                "\"fine-tuning improved inpainting\"."
            )
        )
        + "\n\n**Transfer:** repeat the notebook with `USE_BYOD = True` on your own photographs, with masks drawn over the regions "
        "you actually want to edit, and compare the same four numbers.\n\n"
        "## Troubleshooting\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `No GPU driver was found` or `CUDA was not detected`, or Section 2 stops with `The isolated "
        "environment cannot see a CUDA GPU` | the runtime has no GPU | *Runtime → Change runtime type → T4 GPU*, then run all "
        "again from the top. This notebook is not supported on a CPU-only runtime. If Colab offers no GPU, your GPU quota may "
        "be exhausted; try again later. |\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 GPU runtime` | a local Windows or macOS kernel, or an ARM "
        "machine | Use Google Colab, Kaggle, or a Linux x86_64 machine with a CUDA GPU: the locked environment is built for "
        "manylinux x86_64 wheels. |\n"
        "| Section 1 stops with `Not enough free disk` | the snapshots need about 16.5 GB and the isolated environment about "
        "12 GB | Start a fresh runtime with about 30 GB free; a `weights/` directory from an earlier run is reused and counted. |\n"
        "| `Carried file integrity failure` in Section 2 | a carried file was edited in the notebook | Do not edit the "
        "infrastructure cells; open a fresh copy of the notebook from the repository. |\n"
        "| `uv 0.12.15 wheel size/hash mismatch`, or a `URLError` / timeout while downloading it | a network failure or an "
        "unexpected response from PyPI | Re-run the Section 2 install cell. Never replace the pinned URL or digest. |\n"
        "| `CalledProcessError` from `uv venv` or `uv pip install` (for example a hash mismatch, `Failed to download` or HTTP "
        "5xx) | a transient PyPI or network failure, or a package that no longer matches the lock | Re-run the Section 2 install "
        "cell: `uv` reuses what it already downloaded. If a hash mismatch repeats, stop and report it — never remove "
        "`--require-hashes`, a pin or a hash to get past it. |\n"
        "| `RuntimeError: Stage '…' failed (exit 2): …` | the stage raised an error; the message after the colon is the stage's "
        "own error, and the stage's full log (with the traceback) is printed above it and kept in the run directory's `logs/` | "
        "Find the message in the rows below. A stage reads only files, so after fixing the cause you can re-run that cell and "
        "the cells after it. |\n"
        "| `… is missing: run the stage that writes it before …` | a learner cell was run before an earlier stage | Run the "
        "notebook from the top, or re-run the earlier cells in order. |\n"
        "| `the dataset changed since 'prepare'` | the BYOD zip or the photo cache changed after Section 4 | Re-run from "
        "Section 4. |\n"
        "| A download error (timeout, HTTP 429/5xx) in Section 3 | a transient Hugging Face Hub failure | Re-run the Section 3 "
        "cell: staging only fetches the files that are still absent. |\n"
        "| `ValueError: ...: size ... != manifest ...` or `sha256 ... != manifest ...` | a partial or corrupted download "
        "(staging checks that a file exists, verification checks its bytes) | Delete the named file under `weights/` and re-run "
        "the Section 3 cell. Never edit a manifest to get past a mismatch. |\n"
        "| `No space left on device` during a stage | the disk filled after the Section 1 check | Start a fresh runtime with "
        "about 30 GB free. |\n"
        "| A photograph fetch fails in Section 4 (`URLError`, or a size or SHA-256 mismatch) | a network failure or an unexpected "
        "response from the iNaturalist bucket | Re-run the Section 4 cell; photographs already verified are kept in "
        "`weights/inat-birds/` and reused. |\n"
        "| `CUDA out of memory` | a form field was raised (`BATCH_SIZE`), or another program in the runtime holds GPU memory | "
        "Set the fields back to their defaults and re-run that cell: each stage is its own process, so the failed stage's "
        "memory was released when it stopped. |\n"
        "| A training loss is `nan` | the learning rate is too high | Keep `LEARNING_RATE` at or below `1e-4` and re-run from "
        "Section 7; report the run if it persists. |\n"
        "| `ModuleNotFoundError: No module named 'google.colab'` with `USE_BYOD = True` | the upload dialog needs Google Colab | "
        "Set `BYOD_PATH` to a zip already in the runtime, use Colab for the upload, or keep `USE_BYOD = False`. |\n"
        "| BYOD: `captions.csv is missing columns [...]` or `file ... is not in the dataset` | the zip layout differs from the "
        "contract | Fix the named row or file in your zip: `captions.csv` needs `id`, `file`, `caption` (and optionally `mask`), "
        "and every named file must be in the zip; the run directory's `outputs/{stem}_sample_captions.csv` shows the shape. |\n"
        "| BYOD: `mask ... is WxH px but its image is ... px; they must match` | a mask saved at another size | Save each mask at "
        "exactly its photograph's size. |\n"
        "| BYOD: `image sides must be within 256..4096 px`, `split leaves no test record` or `split leaves N training records` "
        "| a photograph is too small or too large, or there are too few photographs per caption | Resize the photograph. Give "
        "at least one caption three or more photographs, and provide at least **6 distinct photographs** so that 4 remain for "
        "training after each caption with 3..7 photographs gives one test and one validation photograph — for example six of "
        "one caption, or four of each of two captions. |\n"
        "| `near_duplicates_across_splits` is above 0 in Section 4 | near-copies of one scene (bursts, crops, edits) landed in "
        "different splits | Remove all but one photograph of each scene from your zip and re-run from Section 4; otherwise the "
        "held-out numbers are optimistic. |\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules and stage runner, carried in this "
        "standalone notebook and run in an isolated hash-locked environment, can stage and digest-verify three pinned "
        "safetensors snapshots, fetch and validate digest-pinned real photographs with their masks, encode captions and release "
        "the prior, execute bounded LoRA fine-tuning, evaluate the frozen and the adapted model on identical held-out inputs "
        "beside a mean-fill floor and an original-photograph reference, reload the exported adapter in a fresh process with parity "
        "to the trained model, and emit the shown machine-readable artifacts — without the repository being reachable. It does "
        "**not** establish benchmark superiority, production fitness, or image quality beyond the checks shown.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/kandinsky-inpainting-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/kandinsky-inpainting-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights notes: https://github.com/kurtvalcorza/kandinsky-inpainting-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face decoder repository: https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint (revision `{MODEL_REVISION}`)\n"
        "- Hugging Face prior repository: https://huggingface.co/kandinsky-community/kandinsky-2-2-prior (revision 9fc51ad5732afc5d031724219d22e6c42179c5a8)\n"
        "- Razzhigaev, A., et al. (2023). Kandinsky: An improved text-to-image synthesis with image prior and latent diffusion. arXiv:2310.03502: https://arxiv.org/abs/2310.03502\n"
        "- uv (the installer that builds the isolated environment): https://docs.astral.sh/uv/\n"
        "- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR: https://arxiv.org/abs/2106.09685\n"
        "- Cherti, M., et al. (2023). Reproducible scaling laws for contrastive language-image learning. CVPR (the LAION CLIP scorer): https://arxiv.org/abs/2212.07143\n"
        "- Wang, Z., et al. (2004). Image quality assessment: From error visibility to structural similarity. IEEE TIP (SSIM): https://doi.org/10.1109/TIP.2003.819861\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.2 (in the ml-worker repository)\n\n"
        "**AI assistance disclosure:** generative AI assisted this notebook's code and technical writing under maintainer "
        "direction. Tests, pinned identities, manifests and source files remain the authoritative evidence.\n"
    ),
}
