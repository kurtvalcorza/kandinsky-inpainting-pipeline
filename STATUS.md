# Release status

Current status: **Candidate** — a clean-runtime `Run all` of the `E2E` / `GUIDED` tutorial notebook `tutorials/kandinsky_inpainting_colab.ipynb` at `6fd3ab4` passed on a Kaggle T4 on 2026-09-29, with 12/12 code cells, fine-tuning, evaluation, export and a reload within tolerance (`docs/release-verification.md`). Promotion to **Release-grade** still needs the BYOD positive and negative checks (REL12), run through the `BYOD_PATH` field.

Size note: the served weights for inpainting are about 15.86 GB (decoder 5.28 GB + shared prior 10.57 GB), plus the 0.61 GB evaluation scorer. Provenance for all three snapshots is recorded in `docs/WEIGHTS.md` and `MODEL_CARD.md`.
