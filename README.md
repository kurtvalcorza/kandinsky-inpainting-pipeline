# Kandinsky 2.2 Inpainting Pipeline

A DIMER pipeline for two-stage image inpainting with
[`kandinsky-community/kandinsky-2-2-decoder-inpaint`](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint)
and the shared Kandinsky 2.2 prior. It provides immutable snapshot verification, prompt-embedding caching,
masked generation, preservation metrics, bounded LoRA adaptation, and portable safetensors artifacts.

## Pinned components

| Component | Revision |
| --- | --- |
| `kandinsky-community/kandinsky-2-2-decoder-inpaint` | `db790ad5cbcabed886f069ef2710774657621702` |
| `kandinsky-community/kandinsky-2-2-prior` | `9fc51ad5732afc5d031724219d22e6c42179c5a8` |
| `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | `1a25a446712ba5ee05982a381eed697ef9b435cf` |

Each snapshot is accepted only when every manifest entry matches its recorded byte count and SHA-256 digest.
Executable and pickle-based model files are rejected.

## Development setup

Python 3.12 is required.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
```

The package validates records before importing Torch, Diffusers, Transformers, or PEFT. An inpainting record
contains `id`, `image`, `caption`, and optionally `mask_image`; a deterministic center mask is used when the mask
is omitted. Images and masks are resized together and center-cropped to 512 x 512.

## Weights

Large tensors are intentionally excluded from Git. The repository commits only manifests and small upstream
configuration, tokenizer, and model-card files under `weights/`. Use the staging helpers with
`allow_download=True` to fetch missing files at the pinned revisions.

## Adaptation artifacts

LoRA adaptation targets only the UNet attention projections. `save_artifact()` writes `adapter.safetensors` and
a digest-bound `manifest.json`; `load_adapter()` rejects altered weights or a mismatched base-model identity.

## Status

**Initial development.** Offline validation, sample-corpus, import-boundary, adaptation, and artifact tests are
available. The standalone tutorial, model card, and clean-runtime accelerator evidence have not been published,
so the release-asset validator and notebook parity check are not CI gates yet.

## License

Repository code is licensed under Apache-2.0. Upstream model and sample-data licenses remain governed by their
respective sources and are recorded in the committed model cards and dataset metadata.

## AI assistance disclosure

Development used AI-assisted implementation and review. Tests, pinned identities, manifests, and source files
remain the authoritative evidence.
