"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E text-to-image fine-tuning workflow: the pinned Kandinsky 2.2 decoder,
its shared diffusion prior and the CLIP scorer are staged and digest-verified, 60 pinned CC0
iNaturalist bird photographs are fetched, validated and split, every prompt is encoded once into
image embeddings with the prior and the prior is released, the frozen model is scored (held-out denoising
loss, CLIP-scored generations) against the real-photo ceiling, a bounded LoRA fine-tuning runs in the kernel,
the held-out scores are read again in a paired comparison, a new prompt is rendered, and the adapter is
exported and reloaded.
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
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-kandinsky--community%2Fkandinsky--2--2--decoder-ffcc4d?style=flat",
        "https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-ai--forever%2FKandinsky--2-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/ai-forever/Kandinsky-2",
    ),
    ("License", "https://img.shields.io/badge/License-Apache--2.0-blue.svg", "https://www.apache.org/licenses/LICENSE-2.0"),
]

TEMPLATE = {
    "package": "kandinsky_inpainting_pipeline",
    "repo_name": REPO,
    "stem": "kandinsky_inpainting",
    "notebook_name": "kandinsky_inpainting_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh **GPU** runtime (a 16 GB T4 is enough; see the Prerequisites) installs the pinned "
        "dependencies (torch, diffusers, transformers, peft, accelerate, sentencepiece, safetensors, huggingface-hub, numpy, "
        "pillow), stages and digest-verifies three pinned snapshots from the Hub — the 5.28 GB Kandinsky 2.2 decoder (UNet + MoVQ), "
        "the 10.57 GB Kandinsky 2.2 diffusion prior, and a 0.6 GB CLIP scorer — loads the UNet in float16 with an untrained LoRA "
        "adapter attached, fetches 60 CC0 iNaturalist bird photographs as digest-verified JPEGs (6 MB, no credential), validates "
        "them and splits them 36 / 12 / 12 by seed, encodes every prompt with the prior into image embeddings and releases the prior, "
        "scores the frozen model — the held-out denoising loss on the validation and test photographs, and twelve generated images "
        "scored by CLIP against their prompts, the held-out photographs and the real-photo ceiling — runs a bounded LoRA fine-tuning "
        "(4 epochs over 36 images), scores the adapted model on identical inputs, renders a new prompt, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify generation parity. The default path "
        "needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5). On a T4 the whole path takes about fifteen minutes of model time after the 16.5 GB of downloads."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "captioned photographs as a zip holding `captions.csv` (columns `id`, `file`, `caption`) beside the image files (JPEG or "
        "PNG, shorter side 256..4096 px; at least four images, and at least one caption with three or more images so a held-out "
        "record exists). Your records are split by caption into training, validation and test sets and flow through the same "
        "contract — validation, prompt encoding, frozen baseline, adaptation, held-out evaluation, generation, artifact export "
        "and reload parity. The expected schema, the ceilings and the privacy guidance are stated in the Prerequisites and in "
        "Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "KandinskyInpaintPipeline",
    "model_load": "KandinskyInpaintPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, prior_dir=PRIOR_WEIGHTS_DIR, device=('cuda' if torch.cuda.is_available() else 'cpu'), use_lora=True)",
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
            "dir": "PRIOR_WEIGHTS_DIR",
            "identity": ["PRIOR_ID", "PRIOR_REVISION"],
            "stage": "stage_missing_prior_files",
            "verify": "verify_prior_snapshot",
        },
        {
            "key": "clip-vit-b-32-laion2b",
            "var": "SCORER_MANIFEST",
            "dir": "SCORER_WEIGHTS_DIR",
            "identity": ["SCORER_ID", "SCORER_REVISION"],
            "stage": "stage_missing_scorer_files",
            "verify": "verify_scorer_snapshot",
        },
    ],
    "runtime_imports": ["torch", "diffusers", "transformers", "peft"],
    "title": "Kandinsky 2.2 — DIMER E2E text-to-image fine-tuning tutorial (standalone)",
    "badges": BADGES,
    "capability": "text-to-image generation with a 1.25 B-parameter UNet diffusion model, held-out denoising-loss and CLIP-scored evaluation, and bounded LoRA fine-tuning to a set of captioned photographs",
    "intro": (
        "Kandinsky 2.2 (Shakhmatov et al., 2023) is a two-stage latent diffusion architecture: a frozen diffusion prior with "
        "CLIP ViT-G/14 encoders maps text prompts to image embeddings, a 1.25 B-parameter UNet denoises a 4-channel latent "
        "conditioned on those image embeddings with classifier-free guidance, and a MoVQ decoder (67.8 M parameters) decodes "
        "the latent to 3-channel RGB pixels. Three pinned snapshots make one generator: the decoder (5.28 GB safetensors), the "
        "shared diffusion prior (10.57 GB safetensors) and, for evaluation only, a CLIP ViT-B/32 scorer.\n\n"
        "Two architectural properties are handled in the open. **The prior pipeline can be released before training**: "
        "Section 5 encodes every prompt the notebook will ever use — the 60 captions, the generation prompts, the empty "
        "negative prompt — into image embeddings on the CPU, and releases the prior so the UNet, the MoVQ, the scorer and a "
        "training graph fit comfortably on a 16 GB GPU. **Generation has no ground truth**, so the notebook reads three kinds of "
        "number: the held-out *denoising loss* (the training objective, measured on photographs the model never trained on), "
        "CLIP scores of generated images (prompt alignment and similarity to real photographs), and the same CLIP scores on "
        "the real photographs themselves — the ceiling. None of these is a human judgement of image quality."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, dataset and scorer modules; stage and digest-verify three "
        "pinned snapshots; fetch, validate and split a small real captioned-photograph dataset; encode prompts with the "
        "diffusion prior and release it; read a held-out denoising loss and CLIP-scored generations against a real-photo "
        "ceiling; run a bounded LoRA fine-tuning of the UNet with explicit hyperparameters; compare the adapted and frozen "
        "models on identical held-out inputs; render a new prompt; and export a safetensors adapter that reloads against "
        "the pinned base with verified generation parity."
    ),
    "exclusions": (
        "Kandinsky 2.1, inpainting, ControlNet depth conditioning (each in their own repository), full fine-tuning, "
        "DreamBooth identifiers, safety filtering of prompts or images, human preference benchmarks, prompt engineering, "
        "and any claim that a CLIP score or a denoising loss measures image quality. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU of at least 15 GB and Python 3.12). The prior runs in float16 while it encodes prompts and is then released; the UNet runs in float16 (2.5 GB) with the LoRA parameters in float32; the MoVQ decoder runs in float16 (0.13 GB). CPU-only runtimes are not supported for this notebook. About 20 GB of disk is needed for the snapshots.",
        "- **Knowledge:** what a two-stage latent diffusion model does at inference (prompt → image embedding via prior → latent via UNet → pixels via MoVQ), what classifier-free guidance is, what a LoRA adapter changes and what it does not, and why a training loss is not a quality score.",
        "- **Weights:** the decoder, the diffusion prior and the CLIP scorer are all safetensors; nothing is unpickled and no Hub-hosted code is executed — the model classes come from `diffusers`, `transformers` and `peft` on PyPI. Upstream weights are released under Apache-2.0; the scorer is MIT.",
        "- **Data contract:** a record is `{{id, image, caption}}` — an RGB image with shorter side 256..4096 px (resized so the shorter side is 512 px and centre-cropped to 512 × 512; the crop is reported) and a caption of 1..1000 characters. Validation is structural: nothing checks that a caption describes its image or that the model can render it.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of identifiable people, licensed stock images or client material are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches 60 pinned photographs (about 6 MB) from the public iNaturalist open-data bucket `inaturalist-open-data.s3.amazonaws.com` over HTTPS, digest-verified before decoding; every photo is CC0 and its observation page is recorded.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample photographs, validation and splits\n\n"
                "The default dataset is 60 research-grade iNaturalist photographs of six common North American birds — 10 per "
                "species, one per observer per species, every one CC0 — fetched by photo id from the open-data bucket and "
                "refused on any byte-size or SHA-256 mismatch (`fetch_corpus`). Each photo's caption is generated from its "
                "species by one template, so the adaptation teaches the generator what six names look like in this kind of "
                "photograph. `build_sample_dataset` draws a seeded stratified split — 6 / 2 / 2 per species for training, "
                "validation and test — and `dataset_manifest` validates every split, checks that no image appears twice and "
                "records a digest.\n\n"
                "Look for: 36 / 12 / 12 records, six distinct captions, a shorter side around 300..500 px (every photo is "
                "centre-cropped to 512²), a written `outputs/{stem}_sample_captions.csv` in the shape BYOD expects, and "
                "three refusal probes — a missing caption, a 200 px image, a duplicate id — each rejected before the model runs."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    splits = split_dataset(load_byod_dataset(byod_path), seed=0)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    splits = fetch_sample_dataset(cache_dir='weights/inat-birds')\n"
                "    data_source = SAMPLE_LABEL_SOURCE\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n\n"
                "dataset_report = dataset_manifest({{'train': train_records, 'validation': val_records, 'test': test_records}})\n"
                "print({{'data_source': data_source, 'splits': {{k: v['n_records'] for k, v in dataset_report['splits'].items()}}, 'captions': dataset_report['splits']['train']['n_captions'], 'disjoint': dataset_report['disjoint']}})\n"
                "print({{'shorter_side': dataset_report['splits']['train']['shorter_side'], 'centre_cropped': dataset_report['splits']['train']['centre_cropped'], 'digest': dataset_report['digest'][:16] + '...'}})\n"
                "print({{'first_test_record': validate_inputs(test_records[0]), 'caption': test_records[0]['caption']}})\n"
                "prompts = sample_prompts(train_records)\n"
                "print({{'prompts': prompts}})\n"
                "sample_csv = write_dataset_csv(test_records, 'outputs/{stem}_sample_captions.csv')\n"
                "print({{'sample_csv': str(sample_csv)}})\n\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                "probes = {{\n"
                "    'missing caption': [{{'id': r['id'], 'image': r['image']}} for r in train_records[:4]],\n"
                "    'image too small': [{{**train_records[0], 'image': Image.new('RGB', (200, 200))}}, *train_records[1:4]],\n"
                "    'duplicate id': [train_records[0], *train_records[:4]],\n"
                "}}\n"
                "for name, records in probes.items():\n"
                "    try:\n"
                "        validate_dataset(records)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Encode every prompt with the Prior, then release it\n\n"
                "`pipe.encode_prompts` loads the Kandinsky 2.2 diffusion prior from the verified snapshot (float16 on CUDA), "
                "runs the prior diffusion process to generate predicted CLIP image embeddings for each distinct prompt, and "
                "keeps the embeddings on the CPU. Encoded here: the six training captions (which are also the validation and "
                "test captions and the generation prompts), one new prompt for Section 9, and the empty negative prompt that "
                "classifier-free guidance needs. `release_prior` then drops the 1.03 B parameter prior so the UNet, the MoVQ, "
                "the scorer and a training graph fit on a 16 GB GPU.\n\n"
                "Look for: the prior loading from the verified safetensors, seven or eight prompts encoded in seconds, "
                "and GPU memory falling back after the release."
            ),
            "code": (
                "import time\n\n"
                "NEW_PROMPT = 'a photo of a House Finch (Haemorhous mexicanus) perched on a snow-covered branch in winter'\n\n"
                "def gpu_memory_gb():\n"
                "    return round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None\n\n"
                "all_prompts = sample_prompts(train_records + val_records + test_records) + [NEW_PROMPT]\n"
                "encode_report = pipe.encode_prompts(all_prompts)\n"
                "print({{**encode_report, 'gpu_memory_gb_with_prior': gpu_memory_gb()}})\n"
                "released = pipe.release_prior()\n"
                "print({{'prior_released': released, 'gpu_memory_gb_after_release': gpu_memory_gb(), 'device': pipe.device, 'precision': str(pipe.dtype).replace('torch.', '')}})"
            ),
        },
        {
            "md": (
                "## 6. The frozen model: held-out denoising loss and CLIP-scored generations\n\n"
                "The pipeline was built with `use_lora=True`: the adapter's B matrices start at zero, so until Section 7 this is "
                "the pretrained model. Two kinds of number are read here and kept for the comparison.\n\n"
                "**Held-out denoising loss** (`pipe.evaluate`): each held-out photograph is MoVQ-encoded, noised at five fixed "
                "timesteps (100, 300, 500, 700, 900) with a seeded noise tensor, and the UNet's noise prediction is scored "
                "against that noise (MSE over the latent). It is the training objective measured on photographs the model never "
                "trains on; the same seed gives the same latents, noise and timesteps later, so the adapted number is a paired "
                "comparison, not a re-draw.\n\n"
                "**CLIP-scored generations** (`pipe.generate` + `score_generations`): two images per training caption at fixed "
                "seeds (20 steps, guidance 4.0), scored by the frozen CLIP ViT-B/32 on prompt alignment (cosine × 100), "
                "zero-shot label accuracy (which of the six captions is nearest) and similarity to the mean embedding of the "
                "held-out real photographs. `real_photo_baseline` scores the real test photographs the same way: the ceiling."
            ),
            "code": (
                "STEPS = 20  # @param {{type:\"integer\"}}\n"
                "GUIDANCE_SCALE = 4.0  # @param {{type:\"number\"}}\n"
                "IMAGES_PER_PROMPT = 2  # @param {{type:\"integer\"}}\n"
                "EVAL_SEED = 0\n\n"
                "def grid(images, path, columns=6):\n"
                "    tiles = [im.resize((256, 256)) for im in images]\n"
                "    rows = (len(tiles) + columns - 1) // columns\n"
                "    sheet = Image.new('RGB', (256 * columns, 256 * rows), 'white')\n"
                "    for i, tile in enumerate(tiles):\n"
                "        sheet.paste(tile, (256 * (i % columns), 256 * (i // columns)))\n"
                "    sheet.save(path)\n"
                "    return path\n\n"
                "scorer = ClipScorer(weights_dir=SCORER_WEIGHTS_DIR, device=pipe.device)\n"
                "t0 = time.perf_counter()\n"
                "frozen_val = pipe.evaluate(val_records, seed=EVAL_SEED)\n"
                "frozen_test = pipe.evaluate(test_records, seed=EVAL_SEED)\n"
                "print({{'frozen_denoising_mse': {{'validation': frozen_val['denoising_mse'], 'test': frozen_test['denoising_mse']}}, 'by_timestep_test': frozen_test['by_timestep'], 'seconds': round(time.perf_counter() - t0, 1)}})\n\n"
                "generation_prompts = [p for p in prompts for _ in range(IMAGES_PER_PROMPT)]\n"
                "t0 = time.perf_counter()\n"
                "frozen_generation = pipe.generate(generation_prompts, seed=1000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "print({{'generated': len(frozen_generation['images']), 'steps': frozen_generation['steps'], 'guidance_scale': frozen_generation['guidance_scale'], 'seconds': frozen_generation['seconds'], 'adapted': frozen_generation['model']['adapted']}})\n"
                "frozen_scores = score_generations(scorer, frozen_generation['images'], references=test_records)\n"
                "real_ceiling = real_photo_baseline(scorer, test_records)\n"
                "print({{'frozen_generations': {{k: frozen_scores[k] for k in ('clip_prompt_similarity', 'label_accuracy', 'reference_similarity')}}}})\n"
                "print({{'real_photo_ceiling': {{k: real_ceiling[k] for k in ('clip_prompt_similarity', 'label_accuracy', 'reference_similarity')}}}})\n"
                "for entry in frozen_scores['per_image'][::IMAGES_PER_PROMPT]:\n"
                "    print({{'prompt': entry['prompt'][:42], 'clip': entry['clip_prompt_similarity'], 'nearest': entry['nearest_prompt'][13:40], 'correct': entry['correct'], 'reference_similarity': entry['reference_similarity']}})\n"
                "print({{'grid': str(grid([g['image'] for g in frozen_generation['images']], 'outputs/{stem}_frozen_grid.jpg'))}})"
            ),
        },
        {
            "md": (
                "## 7. Bounded LoRA fine-tuning\n\n"
                "`pipe.adapt` trains the 176 LoRA tensors (rank 8, 1,646,592 parameters — 0.13 % of the UNet) that "
                "`peft` attached to the attention projections of the UNet, and nothing else; the UNet base, the MoVQ "
                "and the prior are frozen. Each step takes one training photograph's latent (MoVQ-encoded once, seeded), "
                "draws a timestep uniformly from the 1,000-step schedule and a noise tensor (both seeded), adds the noise, "
                "and minimises the MSE between the predicted and the true noise; AdamW at a fixed learning rate, gradient-norm "
                "clipping at 1.0, float16 autocast with loss scaling on CUDA. Epoch 0 records the frozen model's validation loss, "
                "and the epoch with the lowest validation denoising loss is kept."
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
                "print({{'trainable_parameters': adapt_result['adapter']['n_trainable'], 'total_parameters': adapt_result['adapter']['n_total'], 'steps': adapt_result['steps'], 'best_epoch': adapt_result['best_epoch'], 'precision': adapt_result['adapter']['precision'], 'seconds': adapt_seconds, 'gpu_memory_gb': gpu_memory_gb()}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation: the paired comparison\n\n"
                "The test photographs were never used for training or epoch selection. The adapted model is scored exactly as "
                "the frozen model was in Section 6 — the same seed, so the same latents, noise and timesteps, and the same "
                "twelve prompt/seed pairs for generation — and the table puts the frozen, the adapted and the real-photo "
                "numbers side by side. The cell asserts only what the procedure guarantees — the kept epoch's validation loss is no "
                "higher than the frozen model's (epoch 0) and the re-scored validation loss matches the history."
            ),
            "code": (
                "adapted_val = pipe.evaluate(val_records, seed=EVAL_SEED)\n"
                "adapted_test = pipe.evaluate(test_records, seed=EVAL_SEED)\n"
                "adapted_generation = pipe.generate(generation_prompts, seed=1000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "adapted_scores = score_generations(scorer, adapted_generation['images'], references=test_records)\n"
                "comparison = {{\n"
                "    'denoising_mse_validation': {{'frozen': frozen_val['denoising_mse'], 'adapted': adapted_val['denoising_mse']}},\n"
                "    'denoising_mse_test': {{'frozen': frozen_test['denoising_mse'], 'adapted': adapted_test['denoising_mse']}},\n"
                "    'denoising_mse_test_by_timestep': {{t: {{'frozen': frozen_test['by_timestep'][t], 'adapted': adapted_test['by_timestep'][t]}} for t in adapted_test['by_timestep']}},\n"
                "    'clip_prompt_similarity': {{'frozen': frozen_scores['clip_prompt_similarity'], 'adapted': adapted_scores['clip_prompt_similarity'], 'real_photos': real_ceiling['clip_prompt_similarity']}},\n"
                "    'label_accuracy': {{'frozen': frozen_scores['label_accuracy'], 'adapted': adapted_scores['label_accuracy'], 'real_photos': real_ceiling['label_accuracy']}},\n"
                "    'reference_similarity': {{'frozen': frozen_scores['reference_similarity'], 'adapted': adapted_scores['reference_similarity'], 'real_photos': real_ceiling['reference_similarity']}},\n"
                "}}\n"
                "for name, row in comparison.items():\n"
                "    print({{name: row}})\n"
                "for before, after in zip(frozen_scores['per_image'][::IMAGES_PER_PROMPT], adapted_scores['per_image'][::IMAGES_PER_PROMPT]):\n"
                "    print({{'prompt': before['prompt'][:42], 'reference_similarity': {{'frozen': before['reference_similarity'], 'adapted': after['reference_similarity']}}, 'correct': {{'frozen': before['correct'], 'adapted': after['correct']}}}})\n"
                "print({{'grid': str(grid([g['image'] for g in adapted_generation['images']], 'outputs/{stem}_adapted_grid.jpg'))}})\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'components': {{'id': PRIOR_ID, 'revision': PRIOR_REVISION}},\n"
                "    'scorer': frozen_scores['scorer'],\n"
                "    'data_source': data_source,\n"
                "    'dataset': dataset_report,\n"
                "    'generation': {{'steps': STEPS, 'guidance_scale': GUIDANCE_SCALE, 'images_per_prompt': IMAGES_PER_PROMPT, 'seed': 1000}},\n"
                "    'frozen': {{'validation': frozen_val, 'test': frozen_test, 'generations': frozen_scores}},\n"
                "    'adapted': {{'validation': adapted_val, 'test': adapted_test, 'generations': adapted_scores}},\n"
                "    'real_photo_ceiling': real_ceiling,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
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
        },
        {
            "md": (
                "## 9. A new prompt, artifact export and fresh reload\n\n"
                "The adapted model renders `NEW_PROMPT` — a composition that appears in no training caption — at two seeds; "
                "the CLIP prompt similarity is printed as a sanity check, not an evaluation.\n\n"
                "`pipe.save_artifact` writes the 176 trained tensors (about 6.6 MB) as `adapter.safetensors` with a "
                "`manifest.json` recording the artifact format, the decoder's id and revision, the prior snapshot's id "
                "and revision, the LoRA configuration, the tensor names, the file size and SHA-256, the training configuration "
                "and the epoch history (OUT8). `KandinskyInpaintPipeline.from_artifact` re-verifies both snapshots, checks the "
                "manifest, the LoRA scope and the digest **before** deserialising, loads a fresh UNet with the adapter "
                "attached and overlays the tensors — a new object from files, not the in-memory model (VER2). The fresh "
                "pipeline adopts the prompt embeddings already encoded, and the cell asserts that it reproduces the same "
                "held-out denoising loss and the same image for the same prompt and seed (VER4)."
            ),
            "code": (
                "import platform\n"
                "import shutil\n\n"
                "new_generation = pipe.generate([NEW_PROMPT, NEW_PROMPT], seed=2000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "new_scores = score_generations(scorer, new_generation['images'])\n"
                "print({{'new_prompt': NEW_PROMPT, 'clip_prompt_similarity': new_scores['clip_prompt_similarity'], 'seconds': new_generation['seconds'], 'note': 'sanity check, not an evaluation'}})\n"
                "for i, entry in enumerate(new_generation['images']):\n"
                "    entry['image'].save(f'outputs/{stem}_new_prompt_{{i}}.png')\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': artifact_manifest['weights']['n_tensors'], 'bytes': artifact_manifest['weights']['bytes'], 'sha256': artifact_manifest['weights']['sha256'][:16] + '...'}})\n\n"
                "reloaded = KandinskyInpaintPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, prior_dir=PRIOR_WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded.import_prompt_cache(pipe.export_prompt_cache())\n"
                "reloaded_test = reloaded.evaluate(test_records, seed=EVAL_SEED)\n"
                "before = pipe.generate([prompts[0]], seed=3000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)['images'][0]['image']\n"
                "after = reloaded.generate([prompts[0]], seed=3000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)['images'][0]['image']\n"
                "parity = {{'denoising_mse_diff': round(abs(reloaded_test['denoising_mse'] - adapted_test['denoising_mse']), 8), 'mean_abs_pixel_diff': round(float(np.abs(np.asarray(before, dtype=np.float32) - np.asarray(after, dtype=np.float32)).mean()), 4)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['denoising_mse_diff'] < 1e-6 and parity['mean_abs_pixel_diff'] < 1.0\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'precision': str(pipe.dtype).replace('torch.', ''), 'source': pipe.source}},\n"
                "    'components': {{**evaluation_report['components'], 'license': PRIOR_LICENSE}},\n"
                "    'scorer': {{**evaluation_report['scorer'], 'license': SCORER_LICENSE}},\n"
                "    'provenance': {{\n"
                "        'snapshots': {{'decoder': len(MANIFEST['files']), 'prior': len(PRIOR_MANIFEST['files']), 'scorer': len(SCORER_MANIFEST['files'])}},\n"
                "        'safetensors_only': True,\n"
                "        'remote_code_executed': False,\n"
                "        'prior_released_before_training': released,\n"
                "        'data_base_url': CORPUS_BASE_URL,\n"
                "        'data_license': CORPUS_LICENSE,\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'diffusers': diffusers.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__}},\n"
                "    'data_source': data_source,\n"
                "    'comparison': comparison,\n"
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
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A LoRA of 1.6 million parameters trained for a few minutes on 36 photographs lowers the held-out denoising loss on "
        "twelve photographs the model never saw and moves its generations towards the held-out real photographs of the same "
        "species. That is the claim: the adaptation contract teaches the UNet a narrow visual domain from a handful of captioned "
        "images, the held-out objective is measured on identical inputs before and after, and the artifact is ~6.6 MB.\n\n"
        "The numbers are sample-sanity evidence. A denoising loss is the training objective, not a quality score; CLIP "
        "similarity and CLIP's nearest-caption vote are a frozen model's opinion, not a human judgement, and CLIP itself has "
        "biases about what a species name looks like; twelve images per model from one seeded run give no dispersion "
        "estimate; and nothing here measures aesthetics, diversity, artefacts or prompt fidelity beyond the six captions. "
        "Fine-tuning on a narrow domain can also erode the model elsewhere — the new prompt in Section 9 is a sanity check on "
        "one composition, not a test of generality.\n\n"
        "Three things to carry to real data. **Captions are the contract:** the adapter learns the association between the "
        "caption text and the images; a caption that does not describe its image, or one caption for very different images, "
        "teaches noise. **Hold out by caption, not by image:** the split keeps every caption's images across sets so the "
        "held-out loss measures generalisation within the domain; a caption with one image cannot be evaluated. **Licences "
        "travel with the outputs:** the Kandinsky weights are Apache-2.0, the training photographs here are CC0 — with your own "
        "data, the rights to the images and to what the adapter produces are yours to establish.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can stage and digest-verify three pinned safetensors snapshots, fetch and validate digest-pinned real "
        "photographs, encode prompts and release the prior, execute bounded LoRA fine-tuning, evaluate the frozen and the "
        "adapted model on identical held-out inputs with a real-photo ceiling, and emit the shown machine-readable artifacts — "
        "without the repository being reachable. It does **not** establish benchmark superiority, production fitness, or image "
        "quality beyond the checks shown.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/kandinsky-inpainting-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/kandinsky-inpainting-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights notes: https://github.com/kurtvalcorza/kandinsky-inpainting-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face decoder repository: https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint (revision `{MODEL_REVISION}`)\n"
        "- Hugging Face prior repository: https://huggingface.co/kandinsky-community/kandinsky-2-2-prior (revision 9fc51ad5732afc5d031724219d22e6c42179c5a8)\n"
        "- Shakhmatov, A., et al. (2023). Kandinsky 2.2: https://github.com/ai-forever/Kandinsky-2\n"
        "- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR: https://arxiv.org/abs/2106.09685\n"
        "- Cherti, M., et al. (2023). Reproducible scaling laws for contrastive language-image learning. CVPR (the LAION CLIP scorer): https://arxiv.org/abs/2212.07143\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)\n"
    ),
}
