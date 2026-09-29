# Release status

Current status: **Release-grade** — at `6fd3ab4` the `E2E` / `GUIDED` tutorial notebook `tutorials/kandinsky_inpainting_colab.ipynb` passed a clean-runtime `Run all` of the default path on a Kaggle T4 and the REL12 BYOD journey (representative photographs and masks accepted through `BYOD_PATH` and carried through adaptation, evaluation, export and reload; three incompatible inputs refused), both on 2026-09-29 and recorded in `docs/release-verification.md`.

Size note: the served weights for inpainting are about 15.86 GB (decoder 5.28 GB + shared prior 10.57 GB), plus the 0.61 GB evaluation scorer. Provenance for all three snapshots is recorded in `docs/WEIGHTS.md` and `MODEL_CARD.md`.
