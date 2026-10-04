# Release status

Current status: **Candidate** — the notebook was regenerated after the 2026-10-02 notebook review (findings KIP-m1..m4), so it needs a hosted `Run all` of its current revision before it can be promoted again. Its previous revision was release-grade: at `507e06d` the `E2E` / `GUIDED` tutorial notebook `tutorials/kandinsky_inpainting_colab.ipynb` runs its stages in an isolated hash-locked environment and installs nothing into the kernel. It passed `Run all` in one pass on Google Colab (T4, 2026-09-30) and on a clean Kaggle T4 in strict single-pass mode, and the REL12 BYOD journey on Kaggle (2026-09-29); all recorded in `docs/release-verification.md`.

Size note: the served weights for inpainting are about 15.86 GB (decoder 5.28 GB + shared prior 10.57 GB), plus the 0.61 GB evaluation scorer. Provenance for all three snapshots is recorded in `docs/WEIGHTS.md` and `MODEL_CARD.md`.
