# Release status

Current status: **Candidate** — the `E2E` / `GUIDED` tutorial notebook `tutorials/kandinsky_inpainting_colab.ipynb` is generated from the package and passes the static validator (`tools/validate_release_assets.py`), the generator parity check (`tools/build_notebook.py --check`), `ruff` and the offline unit suite. Its stage cells have also run on CPU against stub models (a pre-flight that exercises the notebook's code paths, not the model; see `docs/release-verification.md`). No clean-runtime GPU execution of the notebook has been recorded. Promotion to **Release-grade** requires a top-to-bottom `Run all` in a fresh supported GPU runtime, recorded in `docs/release-verification.md`, including the BYOD positive and negative checks.

Size note: the served weights for inpainting are about 15.86 GB (decoder 5.28 GB + shared prior 10.57 GB), plus the 0.61 GB evaluation scorer. Provenance for all three snapshots is recorded in `docs/WEIGHTS.md` and `MODEL_CARD.md`.
