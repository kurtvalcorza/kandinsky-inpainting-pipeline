"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier, §3.5 guided layer).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (metrics.py, pipeline.py, samples.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E, GUIDED inpainting fine-tuning workflow: the pinned Kandinsky 2.2 inpainting
decoder, its shared diffusion prior and a CLIP scorer are staged and digest-verified; 60 pinned CC0 iNaturalist bird
photographs are fetched, given the deterministic centre mask, validated and split; every prompt is encoded once with
the prior and the prior is released; the frozen model is measured (held-out denoising loss, preservation of the kept
region, CLIP prompt similarity between a mean-colour-fill floor and the original-photograph ceiling); a bounded LoRA
fine-tuning runs in the kernel; the same measurements are repeated on identical inputs; a new prompt is inpainted into
held-out photographs; the adapter is exported and reloaded; and one optional Predict -> Change one thing -> Run ->
Observe -> Explain activity changes the mask size.
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
        "assumed: each term is explained where it is first used, and the glossary below collects them. A **GPU** runtime is "
        "required (Colab: Runtime → Change runtime type → T4 GPU).\n\n"
        "- **Run all** from a fresh GPU runtime (Runtime → Run all). Every cell runs in order without editing; nothing asks for "
        "input on the default path, and no manual restart is needed.\n"
        "- **Infrastructure** cells (Sections 1–3) are collapsed and titled *Infrastructure*. They install pinned packages, "
        "carry the repository's code and verify the model files. You may run them without studying their implementation.\n"
        "- **Form fields** (`# @param`) such as `USE_BYOD`, `STEPS` or `EPOCHS` can be changed in the form view. The defaults "
        "are the canonical path; change them only after one complete run.\n"
        "- Each section is tagged by kind: **[Concept]** explains an idea, **[Evaluation practice]** concerns fair "
        "measurement, and **[Engineering]** concerns reproducibility.\n"
        "- Before each principal result you are asked to **predict**. Write your prediction down first. After the result, a "
        "**What to notice** note describes normal output, and a **Check your reasoning** box holds a sample answer: open it "
        "only after you have answered.\n\n"
        "**Roadmap.** 1–3 set up and verify the model (Infrastructure) → 4 photographs, masks, validation and splits → "
        "5 prompt encoding → 6 the frozen baseline → 7 LoRA fine-tuning → 8 the paired held-out comparison → 9 a new prompt, "
        "export and fresh reload → 10 change one thing: the mask size → interpretation, conclusion and troubleshooting. "
        "Optional: re-run from Section 4 with your own photographs and masks (Bring Your Own Data).\n\n"
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
        "| **Paired comparison** | Measuring the frozen and the adapted model on identical inputs, noise and seeds, so the difference is the adaptation. |\n\n"
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
        "Selecting **Run all** in a fresh **GPU** runtime (a 16 GB T4 is enough; see the Prerequisites) installs the pinned "
        "dependencies, stages and digest-verifies three pinned snapshots from the Hub — the 5.28 GB Kandinsky 2.2 inpainting "
        "decoder (UNet + MoVQ), the 10.57 GB Kandinsky 2.2 diffusion prior and a 0.61 GB CLIP scorer — loads the UNet in float16 "
        "with an untrained LoRA adapter attached, fetches 60 CC0 iNaturalist bird photographs as digest-verified JPEGs (about "
        "6 MB, no credential), gives each the deterministic centre mask, validates them and splits them 36 / 12 / 12 by seed, "
        "encodes every prompt with the prior and releases the prior, measures the frozen model (held-out denoising loss, "
        "preservation of the kept region, and CLIP prompt similarity between a mean-colour-fill floor and the original-photograph "
        "ceiling), runs a bounded LoRA fine-tuning (4 epochs over 36 photographs), repeats the measurements on identical inputs, "
        "inpaints a new prompt into held-out photographs, exports the adapter as safetensors with a manifest, reloads it into a "
        "fresh pipeline to verify parity, and finally runs one small optional activity that changes the mask size. The default "
        "path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.2 §5). No clean-runtime timing has been recorded yet; the downloads alone are about 16.5 GB."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4, then select the Section 3 cell and choose "
        "Runtime → Run cell and below: Section 3 builds a fresh, unadapted pipeline (the snapshots are already staged), so the "
        "baseline you measure is the frozen model again. Supply your own photographs, masks and captions as a zip holding `captions.csv` beside the files. The table needs the columns `id`, "
        "`file` and `caption`, and may carry a `mask` column naming a mask image of the same size (white = repaint, black = keep); "
        "a row without a mask gets the deterministic centre mask, and the cell reports how many records used it. Photographs "
        "must have a shorter side of at least 256 px and a longer side of at most 4096 px; you need at least four photographs, "
        "and at least one caption with three or more photographs so that a held-out record exists. Set `BYOD_PATH` to a zip "
        "or directory already in the runtime to skip the upload dialog. Your records are split by caption into training, "
        "validation and test sets and flow through the same contract — validation, prompt encoding, frozen baseline, adaptation, "
        "held-out evaluation, inference, artifact export and reload parity. Uploaded files stay inside this runtime. BYOD is "
        "optional and never part of the default path."
    ),
    "pipeline_class": "KandinskyInpaintPipeline",
    "model_load": "KandinskyInpaintPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, prior_dir=DEFAULT_PRIOR_DIR, device=('cuda' if torch.cuda.is_available() else 'cpu'), use_lora=True)",
    "weights_key": "kandinsky-2-2-decoder-inpaint",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "rewrites": [
        [
            r"^_WEIGHTS_ROOT = Path\(__file__\)[^\n]*$",
            '_WEIGHTS_ROOT = Path.cwd() / "weights"  # standalone rewrite (build_notebook.py): working-directory-relative',
        ]
    ],
    "extra_weights": [
        {
            "key": "kandinsky-2-2-prior",
            "var": "PRIOR_MANIFEST",
            "dir": "DEFAULT_PRIOR_DIR",
            "identity": ["PRIOR_ID", "PRIOR_REVISION"],
            "stage": "stage_missing_prior_files",
            "verify": "verify_prior_snapshot",
        },
        {
            "key": "clip-vit-b-32-laion2b",
            "var": "SCORER_MANIFEST",
            "dir": "DEFAULT_SCORER_DIR",
            "identity": ["SCORER_ID", "SCORER_REVISION"],
            "stage": "stage_missing_scorer_files",
            "verify": "verify_scorer_snapshot",
        },
    ],
    "runtime_imports": ["torch", "diffusers", "transformers", "peft"],
    "title": "Kandinsky 2.2 Inpainting — DIMER guided end-to-end notebook (standalone)",
    "badges": BADGES,
    "orientation": ORIENTATION,
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
        "region, and CLIP scores of the outputs between a mean-colour-fill floor and the original-photograph ceiling. None of "
        "these is a human judgement of image quality."
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
        "- **Runtime:** a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU of at least 15 GB and Python 3.12). The prior runs in float16 while it encodes captions and is then released; the UNet and MoVQ run in float16; the LoRA tensors are kept in float32 and training uses float16 autocast with loss scaling. CPU-only runtimes are not supported for this notebook. About 20 GB of free disk is needed for the snapshots.",
        "- **Knowledge:** basic Python and notebooks. Helpful but not required: what a diffusion model does at inference, what classifier-free guidance is, and why a training loss is not a quality score. The glossary above covers the terms.",
        "- **Weights:** the decoder, the diffusion prior and the CLIP scorer are all safetensors; nothing is unpickled and no Hub-hosted code is executed — the model classes come from `diffusers`, `transformers` and `peft` on PyPI. The Kandinsky weights are released under Apache-2.0; the scorer is MIT.",
        "- **Data contract:** a record is `{{id, image, caption}}` plus an optional `mask_image` — an RGB photograph with shorter side at least 256 px and longer side at most 4096 px, a caption of 1..1000 characters, and a greyscale mask of the same size (255 = repaint). A record without a mask gets the deterministic centre mask (the middle half of each side). Photograph and mask are resized together so the shorter side is 512 px and centre-cropped to 512 × 512. Validation is structural: nothing checks that a caption describes its photograph or that a mask covers anything sensible.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of identifiable people, licensed stock images or client material are exactly that. The default path uploads nothing, and uploaded BYOD files are processed only inside this runtime.",
        "- **External access (data):** besides the Hub, the default path fetches 60 pinned photographs (about 6 MB) from the public iNaturalist open-data bucket `inaturalist-open-data.s3.amazonaws.com` over HTTPS, digest-verified before decoding; every photo is CC0 and its observation page is recorded.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample photographs, masks, validation and splits · [Concept]\n\n"
                "The default dataset is 60 research-grade iNaturalist photographs of six common North American birds — 10 per "
                "species, one per observer per species, every one CC0 — fetched by photo id from the open-data bucket and "
                "refused on any byte-size or SHA-256 mismatch (`fetch_corpus`). Each caption is generated from the species by one "
                "template. `build_sample_dataset` draws a seeded stratified split — 6 / 2 / 2 per species for training, "
                "validation and test. The sample has no hand-drawn masks, so `validate_dataset` gives every record the "
                "deterministic **centre mask**: a white box over the middle half of each side, which covers a quarter of the "
                "photograph and usually most of the bird. `dataset_manifest` validates every split, checks that no photograph "
                "appears twice and records a digest.\n\n"
                "**Predict before running:** the centre mask covers the middle half of each side. What fraction of each "
                "512 × 512 photograph will be repainted? Which of the three probes below do you expect validation to refuse?"
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH:\n"
                "        byod_path = Path(BYOD_PATH)\n"
                "    else:\n"
                "        from google.colab import files\n"
                "        uploaded = files.upload()\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        byod_path = Path('work') / file_name\n"
                "        byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_path.write_bytes(payload)\n"
                "    byod_records = load_byod_dataset(byod_path)\n"
                "    own_masks = sum('mask_image' in r for r in byod_records)\n"
                "    print({{'byod_records': len(byod_records), 'own_masks': own_masks, 'centre_mask_fallback': len(byod_records) - own_masks}})\n"
                "    splits = split_dataset(byod_records, seed=0)\n"
                "    data_source = 'BYOD (' + byod_path.name + ')'\n"
                "else:\n"
                "    splits = fetch_sample_dataset(cache_dir='weights/inat-birds')\n"
                "    data_source = SAMPLE_LABEL_SOURCE\n"
                "# validate_dataset attaches the mask: a record's own mask, or the deterministic centre mask when it has none\n"
                "train_records, val_records, test_records = (validate_dataset(splits[name], min_records=1)['records'] for name in ('train', 'validation', 'test'))\n\n"
                "dataset_report = dataset_manifest({{'train': train_records, 'validation': val_records, 'test': test_records}})\n"
                "print({{'data_source': data_source, 'splits': {{k: v['n_records'] for k, v in dataset_report['splits'].items()}}, 'captions': dataset_report['splits']['train']['n_captions'], 'disjoint': dataset_report['disjoint']}})\n"
                "print({{'shorter_side': dataset_report['splits']['train']['shorter_side'], 'centre_cropped': dataset_report['splits']['train']['centre_cropped'], 'digest': dataset_report['digest'][:16] + '...'}})\n"
                "_, first_mask = preprocess_image_and_mask(test_records[0]['image'], test_records[0]['mask_image'])\n"
                "print({{'first_test_record': validate_inputs(test_records[0]), 'caption': test_records[0]['caption'], 'repaint_fraction': round(float((np.asarray(first_mask) > 0).mean()), 4)}})\n"
                "prompts = sample_prompts(train_records)\n"
                "print({{'prompts': prompts}})\n"
                "sample_csv = write_dataset_csv(test_records, 'outputs/{stem}_sample_captions.csv')\n"
                "print({{'sample_csv': str(sample_csv)}})\n\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                "probes = {{\n"
                "    'missing caption': [{{'id': r['id'], 'image': r['image']}} for r in train_records[:4]],\n"
                "    'image too small': [{{**train_records[0], 'image': Image.new('RGB', (200, 200))}}, *train_records[1:4]],\n"
                "    'mask size differs from image': [{{**train_records[0], 'mask_image': Image.new('L', (300, 300))}}, *train_records[1:4]],\n"
                "    'duplicate id': [train_records[0], *train_records[:4]],\n"
                "}}\n"
                "for name, records in probes.items():\n"
                "    try:\n"
                "        validate_dataset(records)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
            "after": (
                "**What to notice:** 36 / 12 / 12 records and six distinct captions on the sample path, `'disjoint'` listing the "
                "same counts, a shorter side of a few hundred pixels (every photograph is resized and centre-cropped to 512²), "
                "a `repaint_fraction` of 0.25, a written `outputs/{stem}_sample_captions.csv` in the shape BYOD expects "
                "(its `mask` column is empty because the sample uses the centre mask), and every probe rejected before any model "
                "runs, each message naming the failed rule.\n\n"
                + CHECK.format(
                    body=(
                        "The mask box spans the middle half of the width and the middle half of the height, so it covers "
                        "0.5 × 0.5 = 0.25 of the pixels. All four probes are refused: a missing caption and a duplicate id "
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
                "`pipe.encode_prompts` loads the Kandinsky 2.2 diffusion prior from the verified snapshot (float16 on CUDA), "
                "runs the prior's own diffusion process to produce a CLIP image embedding for each distinct caption, and keeps "
                "the embeddings on the CPU. Each caption's prior run is seeded with `prompt_seed(caption)`, a 31-bit number taken "
                "from the caption's SHA-256, so the same caption gets the same seed in every process. Encoded here: the six "
                "captions (training, validation and test share them), one new caption for Section 9, and the empty negative "
                "caption that classifier-free guidance needs. `release_prior` then frees the prior's GPU memory.\n\n"
                "**Expected result:** eight captions encoded (seven on the sample path plus the empty one), and GPU memory "
                "falling after the release."
            ),
            "code": (
                "import time\n\n"
                "NEW_PROMPT = 'a photo of a House Finch (Haemorhous mexicanus) perched on a snow-covered branch in winter'\n\n"
                "def gpu_memory_gb():\n"
                "    return round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None\n\n"
                "all_prompts = sample_prompts(train_records + val_records + test_records) + [NEW_PROMPT]\n"
                "encode_report = pipe.encode_prompts(all_prompts)\n"
                "print({{**encode_report, 'gpu_memory_gb_with_prior': gpu_memory_gb()}})\n"
                "print({{'prompt_seed': {{p[:48]: prompt_seed(p) for p in all_prompts[:2]}}}})\n"
                "released = pipe.release_prior()\n"
                "print({{'prior_released': released, 'gpu_memory_gb_after_release': gpu_memory_gb(), 'device': pipe.device, 'precision': str(pipe.dtype).replace('torch.', '')}})"
            ),
        },
        {
            "md": (
                "## 6. The frozen baseline: denoising loss, preservation and CLIP · [Evaluation practice]\n\n"
                "**Question tested:** before any training, how well does the pretrained model predict noise on held-out "
                "photographs, how well does it preserve the kept region, and how well do its repainted photographs match their "
                "captions? The pipeline was built with `use_lora=True`, but the adapter's B matrices start at zero, so until "
                "Section 7 this is the pretrained model.\n\n"
                "- **Held-out denoising loss** (`pipe.evaluate`): each held-out photograph is MoVQ-encoded, noised at five fixed "
                "timesteps (100, 300, 500, 700, 900) with seeded noise, and the UNet — given the masked latents and the keep "
                "mask — predicts that noise; the score is the mean squared error. The same seed gives the same latents, noise "
                "and timesteps later, so the adapted number is a paired comparison, not a re-draw.\n"
                "- **Preservation** (`score_inpainting_preservation`): PSNR in dB and SSIM between the input and the output, "
                "computed only over the kept (black) region. The UNet never repaints that region, but the output is decoded by "
                "the MoVQ, so preservation measures how faithfully the kept region survives the encode-decode round trip. The "
                "SSIM here is one global statistic over the kept pixels, not the windowed SSIM of image-quality papers.\n"
                "- **CLIP prompt similarity** (`score_generations`): the pinned CLIP ViT-B/32 scores each whole output against "
                "its caption. Two references frame it: a **mean-fill floor**, where the repaint region is filled with the mean "
                "colour of the kept region (no model at all), and the **original-photograph ceiling**, the unmasked photograph "
                "scored the same way.\n\n"
                "**Predict before running:** will the frozen model's CLIP similarity sit closer to the mean-fill floor or to "
                "the original-photograph ceiling? Will preservation be perfect (PSNR above 60 dB) or visibly imperfect?"
            ),
            "code": (
                "STEPS = 20  # @param {{type:\"integer\"}}\n"
                "GUIDANCE_SCALE = 4.0  # @param {{type:\"number\"}}\n"
                "EVAL_SEED = 0\n"
                "GENERATION_SEED = 1000\n\n"
                "def prepared(records):\n"
                "    pairs = [preprocess_image_and_mask(r['image'], r['mask_image']) for r in records]\n"
                "    return [image for image, _ in pairs], [mask for _, mask in pairs]\n\n"
                "def mean_fill(image, mask):\n"
                "    pixels = np.asarray(image).copy()\n"
                "    repaint = np.asarray(mask) > 0\n"
                "    pixels[repaint] = pixels[~repaint].mean(axis=0).astype(np.uint8)\n"
                "    return Image.fromarray(pixels)\n\n"
                "def masked_view(image, mask):\n"
                "    pixels = np.asarray(image).copy()\n"
                "    pixels[np.asarray(mask) > 0] = 128\n"
                "    return Image.fromarray(pixels)\n\n"
                "def triptych_grid(images, masks, outputs, path, rows=6):\n"
                "    sheet = Image.new('RGB', (256 * 3, 256 * min(rows, len(images))), 'white')\n"
                "    for i, (image, mask, output) in enumerate(list(zip(images, masks, outputs))[:rows]):\n"
                "        for j, tile in enumerate((image, masked_view(image, mask), output)):\n"
                "            sheet.paste(tile.resize((256, 256)), (256 * j, 256 * i))\n"
                "    sheet.save(path)\n"
                "    return path\n\n"
                "def clip_mean(scores):\n"
                "    return round(scores['mean_prompt_similarity'], 3)\n\n"
                "if pipe.adapter is not None:\n"
                "    raise RuntimeError('this pipeline is already adapted, so Section 6 would not measure the frozen model: re-run from Section 3 (Runtime → Run cell and below)')\n"
                "scorer = ClipScorer(device=pipe.device, weights_dir=str(DEFAULT_SCORER_DIR))\n"
                "t0 = time.perf_counter()\n"
                "frozen_val = pipe.evaluate(val_records, seed=EVAL_SEED)\n"
                "frozen_test = pipe.evaluate(test_records, seed=EVAL_SEED)\n"
                "print({{'frozen_denoising_mse': {{'validation': frozen_val['denoising_mse'], 'test': frozen_test['denoising_mse']}}, 'by_timestep_test': frozen_test['by_timestep'], 'seconds': round(time.perf_counter() - t0, 1)}})\n\n"
                "test_images, test_masks = prepared(test_records)\n"
                "test_prompts = [r['caption'] for r in test_records]\n"
                "frozen_generation = pipe.generate(test_records, seed=GENERATION_SEED, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "frozen_images = [g['image'] for g in frozen_generation['results']]\n"
                "print({{'inpainted': len(frozen_images), 'steps': frozen_generation['steps'], 'guidance_scale': frozen_generation['guidance_scale'], 'seconds': frozen_generation['seconds'], 'adapted': frozen_generation['model']['adapted']}})\n"
                "frozen_preservation = score_inpainting_preservation(test_images, frozen_images, test_masks)\n"
                "frozen_clip = score_generations(scorer, frozen_images, test_prompts)\n"
                "fill_clip = score_generations(scorer, [mean_fill(i, m) for i, m in zip(test_images, test_masks)], test_prompts)\n"
                "real_clip = score_generations(scorer, test_images, test_prompts)\n"
                "print({{'clip_prompt_similarity': {{'mean_fill_floor': clip_mean(fill_clip), 'frozen': clip_mean(frozen_clip), 'original_photo_ceiling': clip_mean(real_clip)}}}})\n"
                "print({{'frozen_preservation': {{'mean_unmasked_psnr_db': round(frozen_preservation['mean_unmasked_psnr_db'], 2), 'mean_unmasked_ssim': round(frozen_preservation['mean_unmasked_ssim'], 4)}}}})\n"
                "for result, clip in list(zip(frozen_generation['results'], frozen_clip['prompt_similarities']))[:6]:\n"
                "    print({{'id': result['id'], 'caption': result['prompt'][:40], 'unmasked_psnr_db': result['unmasked_psnr_db'], 'unmasked_ssim': result['unmasked_ssim'], 'clip': round(clip, 2)}})\n"
                "print({{'grid': str(triptych_grid(test_images, test_masks, frozen_images, 'outputs/{stem}_frozen_grid.jpg'))}})"
            ),
            "after": (
                "**What to notice:** the denoising loss by timestep — noise is harder to predict at low timesteps (little noise) "
                "than the mean suggests, so compare timestep by timestep later. The three CLIP numbers should be ordered, with the "
                "floor lowest; the grid shows each photograph, its masked view and the frozen output side by side. Preservation "
                "is high but not perfect.\n\n"
                + CHECK.format(
                    body=(
                        "A mean-colour fill leaves most of the bird out, so CLIP usually scores it well below the original "
                        "photograph; a working inpainting model lands between the two, often close to the ceiling because three "
                        "quarters of the photograph are kept. Preservation is not perfect because the kept region is not copied "
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
                "`pipe.adapt` trains the 176 LoRA tensors (rank 8, 1,646,592 parameters — 0.13 % of the UNet) that `peft` "
                "attached to the UNet's attention projections `to_q`, `to_k`, `to_v` and `to_out.0`, and nothing else; the "
                "base UNet, the MoVQ and the prior are frozen. This is gradient-based parameter-efficient fine-tuning, not "
                "in-context conditioning. Each step takes one training photograph's latents and keep mask (MoVQ-encoded once), "
                "draws a timestep uniformly from the 1,000-step schedule and a noise tensor (both seeded), adds the noise, and "
                "minimises the mean squared error between the predicted and the true noise. The optimiser is AdamW at a fixed "
                "learning rate, with gradient-norm clipping at 1.0; on CUDA the LoRA tensors stay in float32 while the forward "
                "pass uses float16 autocast with a gradient scaler. Epoch 0 records the frozen model's validation loss, and the "
                "epoch with the lowest validation denoising loss is kept.\n\n"
                "**Predict before running:** will the training loss fall steadily from epoch to epoch? Will the validation loss "
                "fall by more or less than the training loss?"
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 1  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4), 'val_denoising_mse': entry['val_loss']}}\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, seed=EVAL_SEED, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['adapter']['n_trainable'], 'total_parameters': adapt_result['adapter']['n_total'], 'steps': adapt_result['steps'], 'best_epoch': adapt_result['best_epoch'], 'optimizer': adapt_result['adapter']['optimizer'], 'precision': adapt_result['adapter']['precision'], 'seconds': adapt_seconds, 'gpu_memory_gb': gpu_memory_gb()}})"
            ),
            "after": (
                "**What to notice:** one row per epoch, epoch 0 with no training loss; a training loss that jumps between epochs "
                "because each step draws a random timestep; and a `best_epoch` that may be 0 if no epoch improved the validation "
                "loss — in that case the kept adapter is the untrained one.\n\n"
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
                "**Question tested:** on photographs never used for training or for choosing the epoch, what changed? The adapted "
                "model is measured exactly as the frozen model was in Section 6 — the same seed, so the same latents, noise and "
                "timesteps for the denoising loss, and the same photographs, masks and generation seeds for inpainting. The cell "
                "asserts only what the procedure guarantees: the kept epoch's validation loss is no higher than the frozen "
                "model's (epoch 0), and re-measuring it reproduces the history. The test-split change is printed, not asserted.\n\n"
                "**Predict before running:** will the test denoising loss fall? Will CLIP prompt similarity move towards the "
                "ceiling? Will preservation change, and why or why not?"
            ),
            "code": (
                "adapted_val = pipe.evaluate(val_records, seed=EVAL_SEED)\n"
                "adapted_test = pipe.evaluate(test_records, seed=EVAL_SEED)\n"
                "adapted_generation = pipe.generate(test_records, seed=GENERATION_SEED, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "adapted_images = [g['image'] for g in adapted_generation['results']]\n"
                "adapted_preservation = score_inpainting_preservation(test_images, adapted_images, test_masks)\n"
                "adapted_clip = score_generations(scorer, adapted_images, test_prompts)\n"
                "comparison = {{\n"
                "    'denoising_mse_validation': {{'frozen': frozen_val['denoising_mse'], 'adapted': adapted_val['denoising_mse']}},\n"
                "    'denoising_mse_test': {{'frozen': frozen_test['denoising_mse'], 'adapted': adapted_test['denoising_mse']}},\n"
                "    'denoising_mse_test_by_timestep': {{t: {{'frozen': frozen_test['by_timestep'][t], 'adapted': adapted_test['by_timestep'][t]}} for t in adapted_test['by_timestep']}},\n"
                "    'mean_unmasked_psnr_db': {{'frozen': round(frozen_preservation['mean_unmasked_psnr_db'], 2), 'adapted': round(adapted_preservation['mean_unmasked_psnr_db'], 2)}},\n"
                "    'mean_unmasked_ssim': {{'frozen': round(frozen_preservation['mean_unmasked_ssim'], 4), 'adapted': round(adapted_preservation['mean_unmasked_ssim'], 4)}},\n"
                "    'clip_prompt_similarity': {{'mean_fill_floor': clip_mean(fill_clip), 'frozen': clip_mean(frozen_clip), 'adapted': clip_mean(adapted_clip), 'original_photo_ceiling': clip_mean(real_clip)}},\n"
                "}}\n"
                "for name, row in comparison.items():\n"
                "    print({{name: row}})\n"
                "for before, after, clip_before, clip_after in list(zip(frozen_generation['results'], adapted_generation['results'], frozen_clip['prompt_similarities'], adapted_clip['prompt_similarities']))[:6]:\n"
                "    print({{'id': before['id'], 'caption': before['prompt'][:40], 'clip': {{'frozen': round(clip_before, 2), 'adapted': round(clip_after, 2)}}, 'unmasked_psnr_db': {{'frozen': before['unmasked_psnr_db'], 'adapted': after['unmasked_psnr_db']}}}})\n"
                "print({{'grid': str(triptych_grid(test_images, test_masks, adapted_images, 'outputs/{stem}_adapted_grid.jpg'))}})\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'components': {{'prior': {{'id': PRIOR_ID, 'revision': PRIOR_REVISION}}, 'scorer': {{'id': SCORER_ID, 'revision': SCORER_REVISION}}}},\n"
                "    'data_source': data_source,\n"
                "    'dataset': dataset_report,\n"
                "    'mask': 'record mask, or the deterministic centre mask (middle half of each side) when a record has none',\n"
                "    'generation': {{'steps': STEPS, 'guidance_scale': GUIDANCE_SCALE, 'seed': GENERATION_SEED}},\n"
                "    'frozen': {{'validation': frozen_val, 'test': frozen_test, 'preservation': frozen_preservation, 'clip': frozen_clip}},\n"
                "    'adapted': {{'validation': adapted_val, 'test': adapted_test, 'preservation': adapted_preservation, 'clip': adapted_clip}},\n"
                "    'references': {{'mean_fill_floor_clip': fill_clip, 'original_photo_ceiling_clip': real_clip}},\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result['adapter'].items() if k != 'trainable_names'}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "best = adapt_result['history'][adapt_result['best_epoch']]\n"
                "assert best['val_loss'] <= adapt_result['history'][0]['val_loss']\n"
                "assert abs(adapted_val['denoising_mse'] - best['val_loss']) < 1e-4\n"
                "print({{'test_denoising_mse_change': round(adapted_test['denoising_mse'] - frozen_test['denoising_mse'], 6), 'note': 'held-out observation, not asserted'}})\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
            "after": (
                "**What to notice:** read the rows side by side. The denoising loss rows compare identical inputs; the CLIP row "
                "places both models between the floor and the ceiling; the preservation rows should barely move. Open the two "
                "grids in `outputs/` and compare the same photograph in each.\n\n"
                + CHECK.format(
                    body=(
                        "Preservation barely moves because the kept region is carried through the latents in both models: LoRA "
                        "changes how the UNet repaints, not how the MoVQ decodes. A lower test denoising loss is the most direct "
                        "evidence that the adapter learned something that transfers to unseen photographs of these birds. A CLIP "
                        "change of a few tenths on twelve photographs from one seeded run is within the noise you would see from a "
                        "different generation seed; treat it as an observation, not an improvement. If the frozen model already "
                        "sits near the ceiling, there is little room for CLIP to rise at all."
                    )
                )
            ),
        },
        {
            "md": (
                "## 9. A new caption, artifact export and fresh reload · [Engineering]\n\n"
                "**New-data inference.** The adapted model inpaints `NEW_PROMPT` — a caption that appears in no training record — "
                "into two held-out test photographs with their centre masks. The CLIP similarity and preservation are printed "
                "as a sanity check, not an evaluation.\n\n"
                "**Export.** `pipe.save_artifact` writes the 176 trained tensors as `adapter.safetensors` with a `manifest.json` "
                "recording the artifact format, the decoder's id and revision, the LoRA configuration, the tensor names, the "
                "optimiser and precision, the file size and SHA-256, and the metadata passed in. The base model is not in the "
                "artifact: it must be reloaded from the pinned revision.\n\n"
                "**Fresh reload.** `KandinskyInpaintPipeline.from_artifact` re-verifies the decoder and prior snapshots, checks the "
                "artifact format, the base-model identity and the weights digest **before** loading the tensors, and builds a new "
                "UNet with the adapter attached — a new object from files, not the in-memory model. The fresh pipeline adopts the "
                "caption embeddings already encoded, and the cell asserts that it reproduces the held-out denoising loss and the "
                "same inpainted photograph for the same caption and seed within a stated tolerance.\n\n"
                "**Expected result:** two saved PNGs, an artifact of about 6.6 MB (1,646,592 float32 values), and a reload parity "
                "with a denoising-loss difference below `1e-6` and a mean absolute pixel difference below 1.0."
            ),
            "code": (
                "import platform\n"
                "import shutil\n\n"
                "new_records = [{{**r, 'caption': NEW_PROMPT}} for r in test_records[:2]]\n"
                "new_generation = pipe.generate(new_records, seed=2000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "new_clip = score_generations(scorer, [g['image'] for g in new_generation['results']], [NEW_PROMPT] * len(new_records))\n"
                "new_predictions = []\n"
                "for i, (result, clip) in enumerate(zip(new_generation['results'], new_clip['prompt_similarities'])):\n"
                "    path = f'outputs/{stem}_new_prompt_{{i}}.png'\n"
                "    result['image'].save(path)\n"
                "    new_predictions.append({{'id': result['id'], 'prompt': NEW_PROMPT, 'image': path, 'unmasked_psnr_db': result['unmasked_psnr_db'], 'unmasked_ssim': result['unmasked_ssim'], 'clip_prompt_similarity': round(clip, 3)}})\n"
                "    print({{**new_predictions[-1], 'note': 'sanity check, not an evaluation'}})\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source, 'dataset_digest': dataset_report['digest']}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': artifact_manifest['weights']['n_tensors'], 'bytes': artifact_manifest['weights']['bytes'], 'sha256': artifact_manifest['weights']['sha256'][:16] + '...'}})\n\n"
                "reloaded = KandinskyInpaintPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, prior_dir=DEFAULT_PRIOR_DIR, device=pipe.device)\n"
                "reloaded.import_prompt_cache(pipe.export_prompt_cache())\n"
                "reloaded_test = reloaded.evaluate(test_records, seed=EVAL_SEED)\n"
                "before = pipe.generate(test_records[:1], seed=3000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)['results'][0]['image']\n"
                "after = reloaded.generate(test_records[:1], seed=3000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)['results'][0]['image']\n"
                "parity = {{'denoising_mse_diff': round(abs(reloaded_test['denoising_mse'] - adapted_test['denoising_mse']), 8), 'mean_abs_pixel_diff': round(float(np.abs(np.asarray(before, dtype=np.float32) - np.asarray(after, dtype=np.float32)).mean()), 4)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['denoising_mse_diff'] < 1e-6 and parity['mean_abs_pixel_diff'] < 1.0\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'precision': str(pipe.dtype).replace('torch.', ''), 'source': pipe.source}},\n"
                "    'components': {{**evaluation_report['components'], 'licenses': {{'prior': 'apache-2.0', 'scorer': 'mit'}}}},\n"
                "    'provenance': {{\n"
                "        'snapshots': {{'decoder': len(MANIFEST['files']), 'prior': len(PRIOR_MANIFEST['files']), 'scorer': len(SCORER_MANIFEST['files'])}},\n"
                "        'safetensors_only': True,\n"
                "        'remote_code_executed': False,\n"
                "        'prior_released_before_training': released,\n"
                "        'prompt_seed': 'SHA-256 of the caption, first 4 bytes, 31 bits',\n"
                "        'data_base_url': CORPUS_BASE_URL,\n"
                "        'data_license': CORPUS_LICENSE,\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'diffusers': diffusers.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__, 'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}},\n"
                "    'data_source': data_source,\n"
                "    'comparison': comparison,\n"
                "    'new_prompt_predictions': new_predictions,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['weights']['sha256'], 'bytes': artifact_manifest['weights']['bytes']}},\n"
                "    'reload_parity': parity,\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
            "after": (
                "**What to notice:** the reload parity is what the artifact boundary guarantees — the same tensors, loaded into a "
                "freshly verified base model, reproduce the same outputs. It does not show that the outputs are good. "
                "`outputs/{stem}_result.json` gathers identity, provenance, runtime versions, the comparison, the new-caption "
                "predictions and the parity in one machine-readable file."
            ),
        },
        {
            "md": (
                "## 10. Change one thing: the mask size · [Evaluation practice]\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.** This optional activity inpaints the first two test "
                "photographs again with the adapted model, the same captions and the same generation seeds as Section 8. Only the "
                "mask changes: a centred box whose side is a quarter (`small`) or three quarters (`large`) of each side, instead "
                "of the centre mask's half. It runs after every required output has been written and changes nothing that "
                "earlier sections produced; set `RUN_ACTIVITY = False` to skip it.\n\n"
                "**Predict before running:** with a `large` mask, will the kept-region PSNR rise or fall? Will CLIP prompt "
                "similarity rise or fall? What do you expect with `small`?"
            ),
            "code": (
                "RUN_ACTIVITY = True  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_MASK = 'large'  # @param [\"small\", \"large\"]\n\n"
                "def box_mask(image, side_fraction):\n"
                "    width, height = image.size\n"
                "    mask = Image.new('L', (width, height), 0)\n"
                "    margin_x, margin_y = round(width * (1 - side_fraction) / 2), round(height * (1 - side_fraction) / 2)\n"
                "    mask.paste(255, (margin_x, margin_y, width - margin_x, height - margin_y))\n"
                "    return mask\n\n"
                "if RUN_ACTIVITY:\n"
                "    side = {{'small': 0.25, 'large': 0.75}}[ACTIVITY_MASK]\n"
                "    changed = [{{**r, 'mask_image': box_mask(r['image'], side)}} for r in test_records[:2]]\n"
                "    activity_generation = pipe.generate(changed, seed=GENERATION_SEED, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "    activity_clip = score_generations(scorer, [g['image'] for g in activity_generation['results']], test_prompts[:2])\n"
                "    for i, (centre, new) in enumerate(zip(adapted_generation['results'][:2], activity_generation['results'])):\n"
                "        print({{'id': centre['id'], 'repaint_fraction': {{'centre': 0.25, ACTIVITY_MASK: round(side * side, 4)}}, 'unmasked_psnr_db': {{'centre': centre['unmasked_psnr_db'], ACTIVITY_MASK: new['unmasked_psnr_db']}}, 'clip': {{'centre': round(adapted_clip['prompt_similarities'][i], 2), ACTIVITY_MASK: round(activity_clip['prompt_similarities'][i], 2)}}}})\n"
                "    activity_images, activity_masks = prepared(changed)\n"
                "    print({{'grid': str(triptych_grid(activity_images, activity_masks, [g['image'] for g in activity_generation['results']], 'outputs/{stem}_activity_' + ACTIVITY_MASK + '_grid.jpg'))}})"
            ),
            "after": (
                "**Observe and explain:** compare each row's `centre` and changed values, then open the activity grid. Did the "
                "result match your prediction? Explain it in terms of what the model is allowed to repaint and what the metrics "
                "measure. Then switch `ACTIVITY_MASK` and run this cell again.\n\n"
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
        "places the outputs between a mean-fill floor and the original photographs. That is the claim, and it is sample-sanity "
        "evidence only.\n\n"
        "A denoising loss is the training objective, not a quality score. CLIP similarity is a frozen model's opinion, not a human "
        "judgement, and CLIP has its own biases about what a species name looks like. Twelve outputs per model from one seeded run "
        "give no dispersion estimate. Nothing here measures how natural the boundary between repainted and kept regions looks, "
        "the diversity of repaintings, or artefacts. The upstream model may have seen photographs like these during pretraining, "
        "which cannot be ruled out. Fine-tuning on a narrow domain can also erode the model elsewhere — the new caption in "
        "Section 9 is one sanity check, not a test of generality.\n\n"
        "Three things to carry to real data. **Masks are part of the contract:** the model learns to repaint the region you mask, "
        "given the rest; a centre mask teaches centred subjects, so use masks shaped like the edits you intend. **Captions "
        "describe the whole photograph:** the adapter learns the association between caption text and pixels. **Licences travel "
        "with the outputs:** the Kandinsky weights are Apache-2.0 and the photographs here are CC0 — with your own data, the "
        "rights to the photographs and to what the adapter produces are yours to establish.\n\n"
        "## Conclude with evidence · [Evaluation practice]\n\n"
        "Complete this in your own words:\n\n"
        "> On [n] held-out photographs of [subject] with [mask], the adapted model's test denoising loss changed from [frozen] to "
        "[adapted], its kept-region PSNR from [frozen] to [adapted] dB, and its CLIP prompt similarity from [frozen] to [adapted], "
        "between a mean-fill floor of [floor] and an original-photograph ceiling of [ceiling]. Changing the mask size showed "
        "[observation]. These are one seeded run on [n] photographs, scored by automated proxies; they do not establish "
        "[limitation]. Next I would [specific next step].\n\n"
        + CHECK.format(
            body=(
                "A strong conclusion names the photographs, the mask and the split; reports the frozen, adapted, floor and "
                "ceiling numbers side by side, including any that did not improve; separates what was measured (noise "
                "prediction, kept-region fidelity, CLIP agreement) from what was not (how convincing the repainting looks to a "
                "person, behaviour on other subjects); and ends with a concrete next step, such as repeating the comparison over "
                "several generation seeds or asking people to rate the repainted regions. A weak conclusion says only that "
                "\"fine-tuning improved inpainting\"."
            )
        )
        + "\n\n**Transfer:** repeat the notebook with `USE_BYOD = True` on your own photographs, with masks drawn over the regions "
        "you actually want to edit, and compare the same four numbers.\n\n"
        "## Troubleshooting\n\n"
        "| Observation | Appropriate response |\n"
        "|---|---|\n"
        "| `'cuda': False` in Section 1 | Runtime → Change runtime type → T4 GPU, then Runtime → Run all. CPU-only runtimes are not supported. |\n"
        "| The install cell stops with \"Core dependencies changed\" | Runtime → Restart session, then Run all once more; the pinned versions are then already installed. |\n"
        "| A download fails, or a size or SHA-256 mismatch is raised | Re-run the cell once for a transient network error. A persistent mismatch means the pinned file changed upstream: stop and report it; do not remove the check. |\n"
        "| Out of disk | Start a fresh runtime; about 20 GB of free disk is needed for the three snapshots. |\n"
        "| CUDA out of memory | Use a fresh T4 runtime with the default `BATCH_SIZE = 1`, and make sure Section 5 released the prior. |\n"
        "| A training loss is `nan` | Keep `LEARNING_RATE` at or below `1e-4` and re-run from Section 3; report the run if it persists. |\n"
        "| BYOD: \"captions.csv is missing columns\" or \"file ... is not in the dataset\" | Fix the named row or file in your zip: `captions.csv` needs `id`, `file`, `caption` (and optionally `mask`), and every named file must be in the zip. |\n"
        "| BYOD: \"mask ... is WxH px but its image is ...\" | Save each mask at exactly its photograph's size. |\n"
        "| BYOD: \"split leaves no test record\" | Give at least one caption three or more photographs. |\n"
        "| \"this pipeline is already adapted\" in Section 6 | Re-run from Section 3 (select it, then Runtime → Run cell and below) so the baseline is the frozen model. |\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can stage and digest-verify three pinned safetensors snapshots, fetch and validate digest-pinned real "
        "photographs with their masks, encode captions and release the prior, execute bounded LoRA fine-tuning, evaluate the "
        "frozen and the adapted model on identical held-out inputs between a mean-fill floor and an original-photograph ceiling, "
        "and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** establish "
        "benchmark superiority, production fitness, or image quality beyond the checks shown.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/kandinsky-inpainting-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/kandinsky-inpainting-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights notes: https://github.com/kurtvalcorza/kandinsky-inpainting-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face decoder repository: https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint (revision `{MODEL_REVISION}`)\n"
        "- Hugging Face prior repository: https://huggingface.co/kandinsky-community/kandinsky-2-2-prior (revision 9fc51ad5732afc5d031724219d22e6c42179c5a8)\n"
        "- Razzhigaev, A., et al. (2023). Kandinsky: An improved text-to-image synthesis with image prior and latent diffusion. arXiv:2310.03502: https://arxiv.org/abs/2310.03502\n"
        "- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR: https://arxiv.org/abs/2106.09685\n"
        "- Cherti, M., et al. (2023). Reproducible scaling laws for contrastive language-image learning. CVPR (the LAION CLIP scorer): https://arxiv.org/abs/2212.07143\n"
        "- Wang, Z., et al. (2004). Image quality assessment: From error visibility to structural similarity. IEEE TIP (SSIM): https://doi.org/10.1109/TIP.2003.819861\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.2 (in the ml-worker repository)\n\n"
        "**AI assistance disclosure:** generative AI assisted this notebook's code and technical writing under maintainer "
        "direction. Tests, pinned identities, manifests and source files remain the authoritative evidence.\n"
    ),
}
