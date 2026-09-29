---
license: apache-2.0
model_card_spec: "1.2"
pipeline_tag: image-to-image
task: "Text-Guided Image Inpainting (Kandinsky 2.2 inpainting UNet + MoVQ)"
base_model: kandinsky-community/kandinsky-2-2-decoder-inpaint
date_published: "2023-07-25"
date_published_source: "Hugging Face Hub commit `48ea15d787d96dd68682d436c23be99b34b18aca` (2023-07-25, 'Adding safetensors variant of this model (#3)') first published the two safetensors weight files this repository packages; their sizes and SHA-256 digests at that commit equal the committed manifest. The UNet weights were first uploaded in `.bin` form on 2023-07-05 (commit `88cbe94`). The pinned revision `db790ad5` is a later README-only edit (2023-10-09)."
---

# Kandinsky 2.2 Decoder Inpaint — Text-Guided Image Inpainting (Masked Photographs & Bounded LoRA Fine-Tuning)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-kandinsky--community%2Fkandinsky--2--2--decoder--inpaint-ffcc4d?style=flat)](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-ai--forever%2FKandinsky--2-181717?style=flat&logo=github&logoColor=white)](https://github.com/ai-forever/Kandinsky-2)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2310.03502-b31b1b.svg)](https://arxiv.org/abs/2310.03502)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint/blob/main/README.md)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are fetched unmodified under their upstream Apache-2.0 licence; the accompanying code and notebook are Apache-2.0. Nothing here is validated for production, and no benchmark result is claimed.

> [!IMPORTANT]
> **Three pinned snapshots make one inpainting system, the UNet reads a keep mask, and inpainting has no single ground truth.** The decoder checkpoint ships the 5.01 GB inpainting UNet and the 0.27 GB MoVQ; the prior repository provides the CLIP encoders and the prior; a CLIP ViT-B/32 is pinned for evaluation only. `src/kandinsky_inpainting_pipeline/pipeline.py` stages and digest-verifies all three before loading them.

---

## Interactive Colab Tutorials

This pipeline provides a self-contained Google Colab notebook. It carries the repository's code in its own cells and runs end to end without cloning the repository:

- **End-to-End Inpainting Fine-Tuning Notebook (guided)**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/kandinsky-inpainting-pipeline/blob/main/tutorials/kandinsky_inpainting_colab.ipynb) [`kandinsky_inpainting_colab.ipynb`](https://github.com/kurtvalcorza/kandinsky-inpainting-pipeline/blob/main/tutorials/kandinsky_inpainting_colab.ipynb)  
  *60 digest-pinned CC0 iNaturalist bird photographs with the deterministic centre mask and a seeded split; validation with refusal probes; caption encoding with the prior released before training; held-out denoising loss, kept-region preservation and CLIP prompt similarity for the frozen model; bounded LoRA fine-tuning; a paired comparison on identical inputs; a new caption inpainted; safetensors adapter export with verified reload parity; and a mask-size activity.*

---

#### Description

`kandinsky-community/kandinsky-2-2-decoder-inpaint` at revision `db790ad5cbcabed886f069ef2710774657621702` is the inpainting decoder of Kandinsky 2.2, a two-stage latent diffusion model from ai-forever (Razzhigaev et al., 2023). The separately pinned prior `kandinsky-community/kandinsky-2-2-prior` first maps a caption to a CLIP image embedding. The decoder's `UNet2DConditionModel` (1,253,074,568 parameters) then removes noise from a 64 × 64 × 4 latent over a fixed number of scheduler steps, conditioned on that embedding.

The inpainting UNet reads nine input channels. They are the four noisy latent channels, the four latent channels of the input photograph with the repaint region set to zero, and one keep-mask channel that is 1 where the photograph is kept. A MoVQ autoencoder encodes the photograph into latents and decodes the final latent into a 512 × 512 RGB image. At inference the upstream `diffusers` pipeline resets the kept region's latents to the input photograph's latents, noised to the current step, after every step, and restores them exactly after the last step. The repainted region is therefore generated, and the kept region is carried through the MoVQ.

The upstream weights do not change at inference time. Adaptation in this repository is gradient training of a rank-8 LoRA adapter (`peft`) on the UNet attention projections `to_q`, `to_k`, `to_v` and `to_out.0`. The adapter has 176 tensors and 1,646,592 trainable parameters, kept in float32. The prior, the MoVQ and the base UNet weights stay frozen.

This repository adds the `KandinskyInpaintPipeline` class in `src/kandinsky_inpainting_pipeline/pipeline.py`, the masked-photograph dataset contract in `samples.py`, and the preservation and CLIP scoring in `metrics.py`. When a reader runs this code, it does the following:

- verifies three Hub snapshots against committed manifests before any model library is imported;
- validates records and attaches each record's mask, or the deterministic centre mask when a record has none;
- encodes captions through the prior, seeding each run from the caption's SHA-256, then releases the prior;
- inpaints seeded images and reports kept-region PSNR and SSIM for each;
- evaluates held-out denoising MSE with the same UNet input layout that `diffusers` uses at inference;
- fine-tunes the LoRA adapter with validation-loss epoch selection;
- writes a safetensors adapter artifact whose digest is checked before it is loaded again.

#### Intended Use and Limitations

The pipeline is a teaching and research reference for adapting a latent diffusion inpainting model to a small set of captioned photographs.

###### Primary Intended Uses

The pipeline exposes two tasks:

- **Text-guided inpainting.** The input is a list of records, each an RGB photograph, a greyscale mask of the same size (255 = repaint, 0 = keep) and a caption of 1..1,000 characters. A seed, a step count (default `20`) and a guidance scale (default `4.0`) complete the request. The output is one 512 × 512 RGB image per record, with its `unmasked_psnr_db` and `unmasked_ssim`.
- **Adaptation to captioned, masked photographs.** The input is a list of `{id, image, caption}` records with optional `mask_image` (4..2,000 records). The output is a LoRA adapter trained on the noise-prediction objective, written as `adapter.safetensors` with a `manifest.json`.

The application domains envisioned during development are these:

- teaching how masked diffusion editing works and how parameter-efficient fine-tuning changes it;
- research on adapting an inpainting model to a narrow photographic subject, such as a set of species photographs, and on measuring what the adaptation changes;
- producing illustrative edited images for such research, labelled as synthetic.

The design role is a reference implementation that a reader's own editing or training code can embed or copy. It is not a production image-editing service.

###### Primary Intended Users

The intended users are machine-learning engineers, researchers and students who work with diffusion models and parameter-efficient fine-tuning. The envisioned settings are research, teaching and self-hosted experimentation on a GPU the user controls.

The pipeline assumes the following user competencies:

- how diffusion sampling, classifier-free guidance and masks affect an edited image;
- that a lower denoising loss does not imply better-looking edits;
- that kept-region PSNR and CLIP similarity are automated proxies and not a human judgement;
- how to check the licence and consent status of the photographs they edit or train on.

The pipeline enforces only structural limits on records, masks and captions. It cannot judge whether a photograph, a mask or a caption is appropriate. It is therefore not robust to careless or adversarial inputs.

###### Out-of-scope use cases

1. **Capability boundary:** the pipeline repaints a masked region of an existing photograph at 512 × 512 and fine-tunes a LoRA on the inpainting UNet. It does not generate images from text alone. That capability is covered by the public [`kandinsky-generation-pipeline`](https://github.com/kurtvalcorza/kandinsky-generation-pipeline) repository. It does not perform outpainting, depth-conditioned generation or editing at other resolutions.
2. **Input boundary:** photographs must have sides within 256..4,096 px, captions 1..1,000 characters, and masks the same width and height as their photograph. A dataset must have 4..2,000 records with unique ids of at most 64 characters. Each photograph and mask is resized so the shorter side is 512 px and centre-cropped to 512 × 512, so content outside the central square is neither edited nor learned.
3. **Mask boundary:** masks should be binary. `diffusers` treats a mask value of 128 or more as repaint at inference, while `evaluate` and `adapt` pass grey values through as fractional keep-mask values, and preservation is scored only where the mask is exactly 0.
4. **Data-size boundary:** the adaptation is designed for tens to hundreds of photographs and at most 50 epochs. It is not a full fine-tuning recipe, and it is not tested on thousands of photographs.
5. **Decision boundary:** not for producing edited images presented as unedited photographs, or for any decision that treats an edited image as factual evidence, such as news, legal, insurance, medical or scientific evidence.
6. **Provenance boundary:** not for use with weights that do not match the committed manifests. The loader refuses a mismatch rather than loading it.

---

#### Factors

The pipeline's behaviour varies with the subject named in the caption, the photographs and masks used for adaptation, and the hardware it runs on.

###### Groups

The pipeline is not human-centric by design. The bundled sample contains photographs of six North American bird species, and the evaluation measures no demographic attribute. The model can still repaint people into a photograph, or edit people who appear in one, when the caption or the input asks for it.

The upstream training corpus is web image–text data that the upstream authors did not audit or document by demographic group. How people of different ages, genders, skin tones or cultures are represented in it is unknown. This repository therefore makes no group-level fairness claim.

An operator who edits or adapts photographs of people takes on that audit. Before relying on the output, the operator should compare edits across the groups relevant to their use. One method is fixed masks and captions that vary only the described group, reviewed by people from those groups.

###### Instrumentation

The upstream model was trained on image–text pairs collected from the web, captured by unknown cameras and captioned by their publishers or by automated tools. The upstream authors do not document the capture instruments.

The tutorial sample consists of 60 research-grade iNaturalist photographs taken by volunteer observers, typically with consumer cameras and phones. They vary in resolution, focus and lighting. The iNaturalist bucket serves them at about 500 px on the longer side.

The pipeline sees only decoded RGB pixels after a resize and centre crop. It cannot detect instrument defects such as blur, compression artefacts, watermarks or colour casts. An adapter trained on photographs with such defects learns to reproduce them inside the repaint region.

Masks and captions are instruments too. The sample has no hand-drawn masks, so every sample record uses the centre mask, which covers the middle half of each side. The sample captions come from one fixed template per species and describe neither pose, background nor lighting.

###### Environment

**Operating environment.** Python 3.12 with the pinned packages `torch==2.14.0`, `diffusers==0.40.0`, `transformers==5.17.0`, `peft==0.21.0`, `accelerate==1.15.0`, `safetensors==0.8.0`, `huggingface-hub==1.32.0`, `numpy==2.5.3` and `pillow==11.3.0`. On CUDA the UNet, MoVQ and prior run in float16. The LoRA tensors stay in float32, and training uses float16 autocast with `torch.amp.GradScaler`.

The tutorial targets a CUDA GPU with at least 15 GB of memory, such as a 16 GB T4, and about 20 GB of free disk for the 16.47 GB of pinned weights. No run on that hardware has been recorded yet. On CPU the code runs in float32 but is impractically slow for the tutorial.

**Data environment.** Adaptation assumes that the photographs the adapter is trained on resemble the photographs the user later wants to edit. That means the same kind of subject, framing and photographic style, with masks of a similar shape and size. Captions at inference time should follow the pattern of the training captions. Masks or subjects far from the adaptation data move the output back towards the base model's behaviour. Held-out denoising loss is meaningful only when the validation photographs come from the same distribution as the training photographs.

---

#### Metrics

The metrics are chosen because a repainted region has no single correct answer. Each metric measures one property of the model, and none of them measures edit quality as a person would judge it.

###### Performance Measures

The code reports these measures. Names are given as the code reports them.

1. **`denoising_mse`** from `evaluate`, with the per-timestep breakdown `by_timestep`. It is the mean squared error between the UNet's noise prediction and the true Gaussian noise added to MoVQ latents of held-out photographs, at timesteps `100`, `300`, `500`, `700` and `900`. The UNet receives the masked latents and the keep-mask channel in the layout of `diffusers` v0.40.0 `KandinskyV22InpaintPipeline`. It is the training objective measured on photographs the adapter never trained on, and the one measure that compares the frozen and the adapted model on identical inputs.
2. **`unmasked_psnr_db`** and **`unmasked_ssim`** from `generate`, averaged as `mean_unmasked_psnr_db` and `mean_unmasked_ssim` by `score_inpainting_preservation`. PSNR is in decibels, capped at `100.0` for an exact match. SSIM here is one global statistic over the kept pixels, not the windowed SSIM of Wang et al. (2004). Both measure how faithfully the kept region survives, which in this system is the MoVQ encode-decode round trip.
3. **`mean_prompt_similarity`** and **`prompt_similarities`** from `score_generations`. Each value is the pinned CLIP ViT-B/32 model's `logits_per_image` score between a whole output image and its caption: a cosine similarity multiplied by CLIP's learned temperature. It captures agreement between the image and the caption.

The tutorial frames the CLIP measure with two references computed the same way. The floor is a mean-colour fill of the repaint region, and the ceiling is the original, unmasked photograph.

The measures are complementary. The denoising loss is sensitive to the adaptation but says nothing about how an edit looks. Preservation shows whether the kept region is disturbed but ignores the repainted region. CLIP describes the whole image but depends on one automated scorer and on the caption wording. Reading only the loss would miss a model that fits noise better but repaints worse. Reading only CLIP would reward an image that matches the caption while damaging the kept region.

One difference from inference remains in `denoising_mse` and training. At inference `diffusers` also shrinks the kept region by up to one latent pixel (8 px) around the repaint region; `evaluate` and `adapt` do not. No FID, LPIPS or human preference score is computed. The 12 held-out photographs are far too few for FID.

###### Decision thresholds

The pipeline applies three implicit decision rules:

- At inference `diffusers` binarises the mask: a value of 128 or more is repainted, and anything lower is kept.
- Preservation scores count a pixel as kept only where the mask is exactly 0.
- Adaptation keeps the epoch with the lowest validation `denoising_mse`, including epoch 0, the frozen model. The kept adapter therefore never has a higher validation loss than the frozen model.

No acceptance threshold on any measure was set during development, and no quality threshold is shipped. A PSNR, a similarity score or a loss value from this pipeline cannot, on its own, decide whether an edited image is fit for use. The operator who publishes edited images owns any acceptance rule.

A false accept, where a misleading or damaged edit is published, usually costs more than a false reject, where a usable edit is discarded. The operator should therefore combine automated scores with human review, and set any threshold on their own validation images.

###### Approaches to uncertainty and variability

Every number the tutorial reports comes from a single run on one seeded split of the tutorial sample. The split is 36 training, 12 validation and 12 test photographs, 6, 2 and 2 per species with seed `42`. No repeated runs, cross-validation or bootstrap are performed, and no standard deviation or confidence interval is reported. A difference between the frozen and the adapted model is one observation on 12 photographs, not an estimate of a population effect.

Seeds control most of the randomness:

- the data split (`SAMPLE_SEED = 42`, and `seed` in `split_dataset` for user data);
- the evaluation noise, derived from the evaluation `seed`, the timestep and the batch position;
- the training noise, timesteps and ordering (`seed`);
- each inpainted image (`seed + index`);
- each caption's prior run, seeded with `prompt_seed(caption)`, the first four bytes of the caption's SHA-256 masked to 31 bits.

A caption's embedding therefore does not depend on the other captions encoded with it or on Python's per-process hash salt. Non-deterministic GPU kernels and float16 autocast remain as sources of variability. They can change low-order digits between runs on the same hardware, and more between different GPUs.

CLIP scores are scaled similarities, not probabilities, and nothing in the pipeline is calibrated. A caller who needs a calibrated measure of edit quality must collect human ratings of their own edits and calibrate against them.

---

#### Ethical considerations and biases

No external ethics board or group review has assessed this pipeline. The considerations below are the developers' own.

###### Data

The upstream Kandinsky 2.2 model was trained on large web-scale image–text datasets. The upstream authors describe these only at a high level and do not list their sources or filtering in the model repository. It is therefore not ruled out that the training data includes personal images, faces, copyrighted works or other sensitive material. Whether it does is unknown.

This repository distributes code, snapshot manifests, configuration files and documentation. It does not distribute model weights in Git; they are fetched from the Hugging Face Hub at the pinned revisions. It does not distribute photographs either. The tutorial downloads 60 CC0 1.0 photographs from the iNaturalist open-data bucket at run time, and the adapter it exports is trained only on those.

The operator is responsible for the photographs, masks and captions they supply. The pipeline does not check them for faces, personal data, copyrighted content or confidential material. An adapter can memorise and reproduce its training photographs, so an adapter trained on restricted photographs must be treated as restricted too.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life. It produces edited, partly synthetic images. No edited image should be treated as a record of a real person, place or event.

Nobody has validated the pipeline for any such domain. Its only checks are offline unit tests and the tutorial on bird photographs, and the tutorial has no recorded clean run yet.

Foreseeable misuse in a sensitive domain includes altering medical, forensic or news photographs. Such use would require, at minimum, human review of every image, disclosure that the image was edited, and validation by the responsible domain authority. This repository provides none of these.

###### Mitigations

1. **Supply-chain integrity:** `MODEL_REVISION`, `PRIOR_REVISION` and `SCORER_REVISION` are immutable 40-character commit hashes. Each snapshot is checked against its committed `dimer-base-manifest.json`, and every file's byte count and SHA-256 must match before loading. Staging refuses a manifest that names a different model or revision, and downloads only at the pinned revision.
2. **No executable serialization:** every weight file is safetensors, and the snapshot check rejects file types outside `.safetensors`, `.json`, `.model`, `.txt` and `.md`. No pickle is deserialized, and no Hub-hosted code runs: the model classes come from `diffusers`, `transformers` and `peft`.
3. **Adapter integrity:** `load_adapter` refuses an artifact whose `format` is not `org.valcorza.kandinsky-inpainting.adapter.v1`, whose recorded base model is not the pinned decoder revision, or whose `adapter.safetensors` SHA-256 differs from its manifest.
4. **Input integrity:** `validate_dataset` rejects datasets outside 4..2,000 records, duplicate ids, missing fields, photograph sides outside 256..4,096 px, captions outside 1..1,000 characters and masks whose size differs from their photograph, before any model runs. `load_byod_dataset` refuses a missing, unreadable or mis-sized file and names the table row, the id and the file.
5. **Consistent conditioning:** `unet_keep_mask` converts the repository's repaint mask into the UNet's keep-mask channel, so `evaluate` and `adapt` present the UNet with the same channel convention as `generate`. An offline test asserts this layout.
6. **Bounded, numerically stable adaptation:** `adapt` refuses more than 50 epochs, a learning rate above `1e-2` or a batch size above 8. It keeps the epoch with the lowest validation loss, so an adaptation that makes the model worse on held-out data is not exported. The LoRA tensors stay in float32 under float16 autocast with a gradient scaler, and gradients are clipped at norm 1.0.
7. **Reproducibility:** the split, the training and evaluation noise, each inpainted image and each caption's prior run are seeded. Runtime packages are pinned exactly in `pyproject.toml` and in the notebook. The exported manifest records the base model identity and the adapter configuration.

The pipeline has no content filter or safety checker on captions or edited images, and it adds no watermark or provenance metadata to edited images.

###### Risks and harms

1. **Misleading edited photographs.** Inpainting alters a real photograph while keeping most of it intact, which makes the edit more believable than a fully generated image. Third parties who see the image without disclosure bear the harm, and the operator bears the reputational harm. The risk is likely under normal use because this pipeline adds no watermark. The magnitude ranges from minor confusion to serious harm when the image is used as evidence.
2. **Harmful or non-consensual edits.** No content filter runs, so a caption and mask can insert or remove objects, clothing or people in a photograph of a real person. The people depicted bear the harm. Its likelihood depends on who can submit inputs, and its magnitude can be severe.
3. **Bias amplification.** Web-trained generators reproduce stereotypes in how they depict people, occupations and cultures, and an adapter trained on a skewed set of photographs narrows the output further. The groups depicted and the viewers bear the harm. It is likely whenever people are repainted without the audit described under *Groups*.
4. **Training-data leakage.** A LoRA trained on a few photographs can reproduce their content inside the repaint region. If those photographs are private or copyrighted, the adapter and its outputs can leak them. The data subjects and rights holders bear the harm, and it is likely with small datasets and many epochs.
5. **Automation bias.** Users may read a high kept-region PSNR, a rising CLIP score or a falling denoising loss as proof of a good edit. The operator then accepts worse edits, and the downstream audience bears the cost. It is likely when scores are reported without human review.
6. **Out-of-distribution degradation.** Masks, subjects or photographs far from the adaptation data produce edits of unknown quality with no warning. The operator bears the harm, which is usually a wasted or misleading edit.

###### Use cases

The following uses are unacceptable even where the pipeline would work:

1. editing photographs of real people to create sexual or intimate imagery, or any sexual imagery of minors;
2. altering photographs of real people or events and presenting them as authentic, including disinformation, fabricated evidence and impersonation;
3. removing or altering identifying marks, watermarks, signatures or timestamps to misrepresent a photograph's origin or ownership;
4. generating harassment, hate imagery or material intended to intimidate or demean a person or group;
5. surveillance, biometric identification or demographic profiling, and training adapters on photographs of people collected without their consent;
6. producing images used to discriminate in employment, housing, credit, insurance, education or healthcare access;
7. deceptive, manipulative or fraudulent applications, such as altered product photographs, insurance-claim photographs or identity documents;
8. any use that violates the upstream Apache-2.0 licence, the rights attached to the input photographs, or applicable law.

## Immutable provenance

- Model: `kandinsky-community/kandinsky-2-2-decoder-inpaint`
- Revision: `db790ad5cbcabed886f069ef2710774657621702`
- Manifest: `weights/kandinsky-2-2-decoder-inpaint/dimer-base-manifest.json`, format `dimer_hf_snapshot` v1, 7 files, `totalBytes` 5283767688
- UNet `unet/diffusion_pytorch_model.safetensors` (5,012,378,704 bytes) SHA-256: `098b846d2378b4b44a33e0bd47f89f2886b72a10200cbf9e96d5fae3c471543f`; 1,253,074,568 parameters, 9 input and 8 output channels
- MoVQ `movq/diffusion_pytorch_model.safetensors` (271,380,364 bytes) SHA-256: `43a5860fea195a7116f2471396c5cc9535fade9b63c4857d8a192ffd924b7002`
- Prior: `kandinsky-community/kandinsky-2-2-prior` at `9fc51ad5732afc5d031724219d22e6c42179c5a8`; manifest `weights/kandinsky-2-2-prior/dimer-base-manifest.json`, 14 files, `totalBytes` 10574964619
- Scorer (evaluation only): `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` at `1a25a446712ba5ee05982a381eed697ef9b435cf`; manifest `weights/clip-vit-b-32-laion2b/dimer-base-manifest.json`, 9 files, `totalBytes` 608782299
- Publication date: the two safetensors weight files were first published at Hub commit `48ea15d787d96dd68682d436c23be99b34b18aca` on 2023-07-25, with the same byte sizes and SHA-256 digests as the manifest.
- Upstream references: https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint · https://huggingface.co/kandinsky-community/kandinsky-2-2-prior · https://github.com/ai-forever/Kandinsky-2

## Input/output contract

- `KandinskyInpaintPipeline.from_pretrained(weights_dir, prior_dir, *, device=None, use_lora=False, allow_download=False)`: verifies the decoder and prior snapshots and loads the UNet (with an untrained LoRA when `use_lora=True`), the MoVQ and the scheduler configuration.
- `encode_prompts(prompts) -> dict`, `release_prior() -> bool`, `export_prompt_cache() -> dict`, `import_prompt_cache(cache) -> int`: caption encoding with the prior, seeded by `prompt_seed`.
- `generate(records, *, seed=0, steps=20, guidance_scale=4.0) -> dict`: records carry `image`, `mask_image` and `caption`; each result carries `id`, `prompt`, `image`, `unmasked_psnr_db` and `unmasked_ssim`.
- `evaluate(records, *, seed=0, batch_size=4) -> dict`: held-out `denoising_mse` with `by_timestep` and `per_record`.
- `adapt(train, val=None, *, epochs=4, lr=1e-4, batch_size=1, seed=0, progress=None) -> dict`: bounded LoRA fine-tuning with a `history` and the kept `best_epoch`.
- `save_artifact(output_dir, *, metadata=None) -> dict`, `load_adapter(artifact_dir) -> dict`, `from_artifact(artifact_dir, ...)`: `adapter.safetensors` (176 LoRA tensors) plus a digest-bound `manifest.json`.
- `load_byod_dataset(path)`: a directory or zip with `captions.csv` (`id`, `file`, `caption`, optional `mask`).

## Deployment notes

| Field | Status |
|---|---|
| Licence | Apache-2.0 for decoder and prior; MIT for the scorer; repository code Apache-2.0 |
| Weights | About 15.86 GB for inpainting (5.28 GB decoder and 10.57 GB prior), plus the 0.61 GB evaluation scorer |
| Remote code | Not required: standard `diffusers` and `transformers` classes |
| Executable serialization | None: safetensors only |
| Runtime | PyTorch 2.14, `diffusers`, `transformers`, `peft`; float16 on CUDA with float32 LoRA tensors |

## Verification records

`docs/release-verification.md` holds the procedure and every record, including a CPU pre-flight against stub models. The clean-runtime run of the tutorial notebook:

- **Date:** 2026-09-29
- **Subject:** `tutorials/kandinsky_inpainting_colab.ipynb` at commit `6fd3ab4`, blob `c8930286cd79` (full identifiers in `docs/release-verification.md`)
- **Runtime:** Kaggle batch kernel on a Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`
- **Procedure:** the notebook was fetched at that commit and run with `Run all` in a fresh interpreter, with an empty Hugging Face cache and no repository checkout. Form fields were at their defaults (`USE_BYOD = False`). The install cell's restart guard fired once because the kernel had preloaded older `numpy` and `protobuf`, and the kernel was restarted and run again from the top.
- **Observed result:** 12 of 12 code cells ran without error in 906.0 s. Held-out test `denoising_mse` was 0.028245 for the frozen model and 0.027997 after adaptation. Kept-region PSNR rose from 25.30 to 25.59 dB and SSIM from 0.9379 to 0.9420. The reloaded adapter gave the same denoising loss (`denoising_mse_diff` 0.0) and a `mean_abs_pixel_diff` of 0.068 on the 0–255 scale, inside the asserted tolerance of 1.0.
- **Caveats:** one run on one seeded split. This is sample-sanity evidence, not a benchmark. The BYOD branch was not exercised, so the status remains `Candidate`.

## References

- Razzhigaev, A., et al. (2023). Kandinsky: An improved text-to-image synthesis with image prior and latent diffusion. arXiv:2310.03502. https://arxiv.org/abs/2310.03502
- Rombach, R., et al. (2022). High-resolution image synthesis with latent diffusion models. CVPR.
- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR. https://arxiv.org/abs/2106.09685
- Cherti, M., et al. (2023). Reproducible scaling laws for contrastive language-image learning. CVPR. https://arxiv.org/abs/2212.07143
- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4), 600–612. https://doi.org/10.1109/TIP.2003.819861
- Pinned repositories: https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint · https://huggingface.co/kandinsky-community/kandinsky-2-2-prior · https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
