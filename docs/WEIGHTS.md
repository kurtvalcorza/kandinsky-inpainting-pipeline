# Weight provenance and the three pinned snapshots

This repository pins **three** Hugging Face snapshots, each with its own `dimer-base-manifest.json` (format `dimer_hf_snapshot` v1: byte size and SHA-256 per file, the immutable revision) and each staged and verified separately by `src/kandinsky_inpainting_pipeline/pipeline.py`. Every weight file is safetensors; nothing is unpickled, and the model classes come from `diffusers`, `transformers` and `peft` on PyPI — no Hub-hosted code is executed. Large tensors are not committed to Git: `weights/` holds the manifests and the small upstream configuration, tokenizer and model-card files, and `.gitattributes` keeps `weights/** -text` so their bytes match the recorded digests on every platform.

## 1. The decoder — Kandinsky 2.2 Decoder Inpaint

- Upstream: `kandinsky-community/kandinsky-2-2-decoder-inpaint`
- Immutable revision: `db790ad5cbcabed886f069ef2710774657621702` (a README-only commit on 2023-10-09; the two safetensors weight files were first published at commit `48ea15d` on 2023-07-25 with the same sizes and digests, and the UNet was first uploaded as `.bin` on 2023-07-05 at commit `88cbe94`).
- Upstream weight licence: **Apache-2.0** (`license: apache-2.0` in the pinned README front matter and in the manifest).
- Local layout: `weights/kandinsky-2-2-decoder-inpaint/` holds the 7 manifest entries — upstream `README.md`, `model_index.json`, `scheduler/scheduler_config.json`, `movq/config.json`, `movq/diffusion_pytorch_model.safetensors` (271,380,364 bytes, SHA-256 `43a5860f…`), `unet/config.json`, and `unet/diffusion_pytorch_model.safetensors` (5,012,378,704 bytes, SHA-256 `098b846d…`); 5,283,767,688 bytes in total. `verify_snapshot()` checks every entry by byte size and SHA-256 and refuses on the first mismatch; `stage_missing_files(allow_download=True)` fetches only absent entries, only at the pinned revision.
- Architecture: `UNet2DConditionModel` with 1,253,074,568 parameters in 724 tensors (asserted by `build_unet`), `in_channels` 9 (4 noisy latent + 4 masked latent + 1 keep mask) and `out_channels` 8 (the first 4 are the noise prediction), conditioned on CLIP image embeddings; MoVQ `VQModel` encoder/decoder. Loaded in float16 on CUDA, float32 on CPU. The upstream repository also ships `.bin` duplicates; only the safetensors files are in the manifest and staged.

## 2. The prior — Kandinsky 2.2 Prior (shared)

- Upstream: `kandinsky-community/kandinsky-2-2-prior`
- Immutable revision: `9fc51ad5732afc5d031724219d22e6c42179c5a8`
- Upstream licence: **Apache-2.0** in the pinned README front matter.
- Local layout: `weights/kandinsky-2-2-prior/` holds 14 manifest entries (10,574,964,619 bytes), including `prior/diffusion_pytorch_model.safetensors` (4,104,940,968 bytes), `image_encoder/model.safetensors` (3,689,912,664 bytes), `text_encoder/model.safetensors` (2,778,702,976 bytes), the tokenizer, image-processor and scheduler configuration.
- The prior is loaded by `encode_prompts` (each caption's sampler seeded by `prompt_seed`, the first four bytes of the caption's SHA-256) and released by `release_prior` before training.

## 3. The scorer — CLIP ViT-B/32 (evaluation only)

- Upstream: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, revision `1a25a446712ba5ee05982a381eed697ef9b435cf`, MIT licence.
- Local layout: `weights/clip-vit-b-32-laion2b/`, 9 manifest entries (608,782,299 bytes), including `model.safetensors` (605,157,884 bytes). Loaded as `transformers.CLIPModel` in float32 by `metrics.ClipScorer` only — never part of inpainting, never trained; its scores are not a human judgement.

## Fidelity

No upstream regression fixture is published for the inpainting UNet. The evidence available in this repository is the digest verification of every file, the parameter-count and tensor-count assertion in `build_unet`, and the unit tests on stub models. No GPU run of the pinned weights has been recorded yet (see `docs/release-verification.md`).

## Runtime facts

- Precision: UNet, MoVQ and prior float16 on CUDA (float32 on CPU, unsupported for the tutorial); LoRA tensors float32; training under float16 autocast with `torch.amp.GradScaler`, loss in float32.
- Masks: the repository convention is 255 / 1 = repaint. `unet_keep_mask` converts it to the UNet's keep-mask channel (1 = keep), the convention of `diffusers` v0.40.0 `pipeline_kandinsky2_2_inpainting.py` (L238 `mask = 1 - mask`, L452, L479), so `evaluate`, `adapt` and `generate` condition the UNet the same way.
- Inpainting: 20 steps at guidance 4.0 by default through `diffusers.KandinskyV22InpaintPipeline` with a DDPM scheduler built from the pinned configuration; per-record seeded generator (`seed + index`).
- Evaluation: held-out denoising MSE on MoVQ-encoded latents at timesteps 100, 300, 500, 700, 900 with seeded noise; kept-region PSNR and SSIM on every inpainted image.
- Adaptation: LoRA rank 8 / alpha 8 on `to_q`, `to_k`, `to_v`, `to_out.0` — 176 tensors, 1,646,592 parameters — injected via `peft`; AdamW, gradient clipping at norm 1.0, uniform timesteps, validation-loss epoch selection.
- `diffusers`, `transformers`, `peft` and `accelerate` are imported lazily so manifest verification and input validation run first.

## Licences

Apache-2.0 for the decoder and prior weights, MIT for the scorer, Apache-2.0 for the repository code. The tutorial photographs are CC0 1.0 and are fetched at run time, not redistributed.
