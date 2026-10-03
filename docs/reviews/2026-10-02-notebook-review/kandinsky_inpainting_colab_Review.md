# Kandinsky 2.2 inpainting guided notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/kandinsky-inpainting-pipeline`  
**Notebook:** `tutorials/kandinsky_inpainting_colab.ipynb`  
**Reviewed commit:** `2d19352a60309c2375a481e59e79e7c74baa2e4f` (`main`, merge of PR #2)  
**Notebook Git blob:** `447723996da8868d563af5edb9587238cfdcac10`  
**Finding prefix:** `KIP`  
**Framework:** Notebook Review Framework v1; requirements baseline DIMER Notebook Specification **2.2** (2026-09-26, `ml-worker` `origin/main` `b1cfe13`)

## Executive assessment

This is a well-built notebook. Every stage runs in an isolated hash-locked `uv` environment, so nothing is installed
into the kernel and no restart is needed. The code is carried byte for byte with hash checks, and three snapshots are
pinned by commit and digest. The notebook keeps its three kinds of number apart (held-out denoising loss, kept-region
preservation, CLIP prompt similarity) and calls none of them image quality. The comparison is paired on identical
latents, noise, masks and seeds. The epoch is chosen on validation, and the fresh-process reload has explicit parity.
The default `Run all` path and the REL12 BYOD journey both have hosted evidence **at this exact notebook blob**, and
both passed in one pass with no restart.

**There is no Blocker and no Major finding.** The two sibling defects were checked:

- **Loss trend (sibling KCD-M1):** this does not recur. Section 6 says noise is "harder to predict at low timesteps
  (little noise)", and the hosted run agrees: 0.1046 at t = 100 falling monotonically to 0.0002 at t = 900 (probe `P2`).
- **Real-photo ceiling (sibling KGN-M1):** the mechanism behind KGN-M1 is absent here. This repository has no
  `real_photo_baseline` and no reference mean. Its "original-photograph ceiling" is each unmasked test photograph's own
  CLIP score against its own caption, which is a legitimate per-image reference (probe `P1`). The **label** "ceiling" is
  still wrong, as it was in row 33 (KCD-m1). The frozen model scores above it on the default run (31.08 against 30.27),
  and on the BYOD run both models score far above it (33.96 and 33.66 against 26.76). Four learner-facing passages say
  the outputs sit "between" the floor and the ceiling. This is **KIP-m1**, rated **Minor** to match KCD-m1. The
  severity reasoning is under KIP-m1.

Four Minor findings in all. Two of them leave an applicable spec **MUST** unmet: SPL3 (KIP-m2, the split's independence
assumption is unstated) and DAT12 (KIP-m3, the BYOD minimum the notebook states is refused by the code). The sibling
DAT9 finding (KCD-m3 / KGN-m2) does not recur: *Interpretation and limits* states that pretraining overlap cannot be
ruled out.

None of these findings says the default run fails. The hosted evidence shows that it passes.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Revision | `2d19352` (merge of PR #2). The notebook blob equals the blob of the recorded hosted runs at `507e06d`; `507e06d..2d19352` changes only `MODEL_CARD.md`, `README.md`, `STATUS.md`, `docs/release-verification.md` and `tutorials/README.md` (probe `P0`). The carrier records generating revision `c29c902f018e` |
| Profile / mode | `E2E` / `GUIDED`, declared in the opening and `metadata.dimer` |
| Spec declared / applied | 2.2 / 2.2 |
| Audience | Can open a hosted notebook, run cells in order and read short Python; no prior diffusion-model experience assumed |
| Prerequisites stated | Linux x86_64 CUDA GPU ≥ 15 GB (T4), about 30 GB disk; CPU-only explicitly unsupported |
| Supported runtime | Google Colab T4 (primary), Kaggle T4, Linux Jupyter with CUDA |
| Promised outcomes | Three digest-verified snapshots; 60 CC0 bird photographs with the deterministic centre mask, validated and split 36 / 12 / 12 with four refusal probes; captions encoded and the prior released; frozen baseline (held-out denoising loss, kept-region PSNR/SSIM, CLIP prompt similarity between a mean-fill floor and an "original-photograph ceiling"); bounded LoRA (rank 8, 1,646,592 parameters); paired comparison from the exported adapter in a fresh process; fresh-process reload with parity and a new caption; an optional mask-size activity; BYOD through the same stages |
| Generator | `tools/build_notebook.py` (3.0) from `tools/notebook_template.py`; stage logic in `tools/tutorial_stages.py` (carried) |

### Evidence actually obtained

**Source inspection.** All 37 cells (11 code, 26 markdown), the carried stage runner, `samples.split_dataset`,
`metrics.py`, the template, `README.md`, `tutorials/README.md`, `docs/release-verification.md` and `STATUS.md`.

**Documented execution evidence (this exact blob).** `docs/release-verification.md` records three runs of blob
`447723996…`: Google Colab T4 (2026-09-30, 11/11 code cells, no restart), a strict single-pass Kaggle T4 run
(2026-09-29, 904.7 s) and the REL12 BYOD journey (positive zip plus three refusals, 702.9 s, strict single pass). The
Kaggle executor outputs are archived under `.agent/backups/kandinsky-kaggle-2026-09-29/out-strict/` (`LEDGER.md` row:
`507e06d`, blob `447723996…`, PASS, 11/11). This review read both runs' `kandinsky_inpainting_evaluation_report.json`
and the default run's `activity.json`, `reload.json` and `prepare.json` for the values used below (sha256 in
`source_manifest.json`). The executed Colab notebook was not re-read; its row in the release record does not list CLIP
values.

**Direct execution (CPU, Windows, the repository `.venv`: CPython 3.12.12, numpy 2.5.3, Pillow 12.3.0, torch 2.14.0+cpu;
no diffusers sampling, no model weights).** `run_probes.py` probes `P0`–`P6`: blob identity; the CLIP references from
the archived report; the per-timestep losses; the carried BYOD splitter on synthetic records (minimum size,
near-duplicates, split semantics); and text checks on the notebook, `tutorials/README.md` and the release record. The
static gates (`P7`) were also run: the repository's offline suite (47 passed, 0 skipped, `PYTHONPATH=src`),
`tools/validate_release_assets.py` (PASS) and `tools/build_notebook.py --check` (up to date). These are **not**
execution evidence under REL8. Probe wall-clock time was about one minute, well under the 20-minute cap.

**Not verified.** Any GPU stage in this review; the `small` activity mask on real weights; learner understanding.

### Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Well scaffolded: audience, how-to-use, Where the code runs, I/O contract, roadmap, glossary, five predictions, What to notice, Check your reasoning, troubleshooting and a conclusion template. The loss-by-timestep explanation is correct. The CLIP "ceiling" framing is contradicted by the learner's own default output (KIP-m1) |
| Clean default | Documented execution evidence at this blob | Passed on Colab T4 and strict Kaggle T4 in one pass with no restart. Reload parity: `denoising_mse_diff` 0.0, `mean_abs_pixel_diff` 0.0766 / 0.0723 (tolerance 1.0). Not re-run here |
| Active learning | Documented execution evidence for the default `large` mask at this blob; source inspection | `large` mask recorded: PSNR 23.92 → 17.76 and 25.60 → 23.13 dB, CLIP 32.75 → 34.59 and 35.86 → 34.66. These are consistent with the hedged sample answer. The activity writes only `outputs/activity.json` and its own grid, and nothing reads them back; `result.json` is written before it (probe `P9`). Switching the mask and rerunning therefore cannot leave a stale block in another export, so row 33's KCD-m5 does not recur. `small` on real weights is not verified |
| Reuse and recovery | Documented REL12 evidence at this blob; direct CPU execution of the carried splitter | Positive BYOD (12 records, 6 own masks) reached export and reload, and three refusals named the row, id and file. The stated minimum dataset size is refused by the code (KIP-m3). A one-pixel near-duplicate is kept and can land in test while its original is in train (KIP-m2). On BYOD the "ceiling" is 7 points below both models (KIP-m1) |

## 2. Separate judgments

| Dimension | Judgment |
|---|---|
| Technical correctness | Sound. The environment is isolated, the carrier matches the repository, snapshots are digest-pinned, only safetensors are loaded, and each stage is its own process. Evaluation is paired and seeded, conditioning matches inference (nine channels, keep-mask convention), the epoch is chosen on validation, and the fresh-process reload has explicit parity. No computational defect found |
| Promise fulfilment | Every promised stage runs and is evidenced at this blob, including BYOD and the mask activity. The one promise the evidence contradicts is that CLIP places the outputs "between" the floor and the ceiling (KIP-m1) |
| Scientific validity | Comparisons are paired and labelled sample-sanity; guaranteed results are kept apart from observations; score semantics are stated. Gaps: the split's independence assumption and near-duplicate handling (KIP-m2) |
| Learner experience | Above fleet average. The timestep explanation is correct, and the activity's answer is honestly hedged. The ceiling framing and the Section 8 hint that the ceiling caps CLIP teach a wrong reading of the reference (KIP-m1) |
| Spec conformance | RUN, ST, ENV, MOD, VER, REL, GDL and DAT9 requirements are met on the evidence above. **Unmet MUSTs:** SPL3 (KIP-m2) and DAT12 (KIP-m3, the stated limit is inaccurate) |

## 3. Findings

No Blocker or Major findings.

### KIP-m1 — Minor (EVAL10): the "original-photograph ceiling" is exceeded, but four passages place the outputs between floor and ceiling

**Location:** opening (cell 0, "CLIP prompt similarity between a mean-colour-fill floor and the original-photograph
ceiling", twice); Section 6 bullet and prediction (cell 21); Section 6 answer (cell 23, "a working inpainting model lands
between the two, often close to the ceiling"); Section 8 prediction (cell 27, "Will CLIP prompt similarity move towards
the ceiling?"); Section 8 What to notice (cell 29, "the CLIP row places both models between the floor and the ceiling")
and its answer ("If the frozen model already sits near the ceiling, there is little room for CLIP to rise at all");
*Interpretation and limits* (cell 36, "CLIP places the outputs between a mean-fill floor and the original photographs.
That is the claim"); conclusion template (cell 36, "between a mean-fill floor of [floor] and an original-photograph
ceiling of [ceiling]"); closing statement. Generator: `tools/notebook_template.py` lines 124, 187, 383–386, 404, 468,
480, 490, 573, 590, 662. Code: `tools/tutorial_stages.py` `stage_frozen` (`original_photo_ceiling`, line 414).

**Observed issue:** the reference is computed correctly, but it is not a ceiling.

| Run (blob `447723996…`) | Mean-fill floor | Frozen | Adapted | "Ceiling" (unmasked photographs) |
|---|---|---|---|---|
| Strict Kaggle T4, default | 24.026 | **31.078** | **30.390** | 30.274 |
| Strict Kaggle T4, REL12 BYOD | 22.727 | **33.959** | **33.658** | 26.759 |
| Kaggle T4 at `6fd3ab4` (earlier blob, release record) | 24.026 | **31.078** | **30.298** | 30.274 |

Per image on the default run, the frozen output beats its own original photograph on 7 of 12 test photographs and the
adapted output on 5 of 12. The floor never beats the original (0 of 12) (probe `P1`). A generator repaints the masked
region to match the caption literally, while a field photograph only has to contain the bird, so the unmasked photograph
is a **reference**, not an upper bound. The Section 8 answer's "little room for CLIP to rise" treats it as a cap.

**Consequence:** on the default run the learner's own output contradicts the What to notice note. The frozen model sits
above the "ceiling", not between, and the BYOD user sees both models about 7 points above it. A learner who believes the
answer box learns that an original photograph bounds CLIP similarity, and fills the conclusion template with a value
outside the stated interval. The paired-comparison conclusion itself is not affected: the notebook already says a CLIP
change of a few tenths is within seed noise.

**Severity and relation to the siblings.** KGN-M1 (row 34) was Major because its reference was **computationally
inflated**: `real_photo_baseline` included each photograph in its own reference mean (88.66 against a leave-one-out
estimate of about 57.7), contrary to its docstring, and that reference carried objective 3. The learner was misled about
what the numbers *are*. Here the reference is a correct per-image measurement (`real_clip = score_generations(scorer,
test_images, test_prompts)`, with no self-match), it is not a learning objective, and the numbers are printed side by
side. Only the word "ceiling" and the "between" prose are wrong. That matches KCD-m1 (row 33: generated 32.305 against a
30.25 "ceiling", with a contradicting sample answer), which was rated Minor. Under framework §6 this is a misleading
simplification that causes localized confusion. It does not leave a core promise or objective undelivered, so it is
Minor.

**Evidence:** documented execution evidence (default and BYOD `evaluation_report.json`, blob `447723996…`; the release
record's `6fd3ab4` row); source inspection of `tutorial_stages.py`; probe `P1`.

**Recommended correction:** rename the reference to **original-photograph reference** throughout the template and the
stage-runner keys (`original_photo_reference`). Say once, in Section 6, that a repainted photograph can score above its
original, because the generator renders the caption more literally than a field photograph does. Rewrite the Section 6
answer, the Section 8 prediction, What to notice and answer, the *Interpretation* claim and the conclusion template so
that none of them says the outputs lie between the two references or that the reference limits how far CLIP can rise.
Optionally print the per-image count of outputs above their original.

**Acceptance check:** no learner-facing text, and no `tutorials/README.md` sentence, calls the original-photograph value
a ceiling or says the outputs lie "between" the floor and it; the Section 8 answer no longer implies a cap; the
conclusion template asks for the floor and reference values without an interval; `tools/build_notebook.py --check` and
`tools/validate_release_assets.py` pass (the validator's required phrase "original-photograph ceiling",
`validate_release_assets.py` line 176, is updated with the template).

### KIP-m2 — Minor (spec MUST SPL3; SPL10): the split's independence assumption is unstated, and near-duplicates can cross splits

**Location:** Section 4 (cell 16, "a seeded stratified split — 6 / 2 / 2 per species"); opening BYOD paragraph (cell 0,
"Your records are split by caption into training, validation and test sets"). Generator: `tools/notebook_template.py`
lines 141 and 301; `samples.split_dataset` docstring ("grouped by caption", line 796); `tutorials/README.md` *Split
integrity* ("In BYOD mode the split is by caption").

**Observed issue:** both splits are seeded shuffles **stratified within caption**, so every caption appears in train,
validation and test (probe `P4`: 2 captions, both in every split). "Split by caption" and "grouped by caption" read as a
group split that holds whole captions out, which is the opposite mechanism. The word "independent" appears nowhere in
the notebook (probe `P6`), so SPL3 is unmet. The sample corpus does mitigate the risk by design (one photograph per
observer per species), but the notebook never connects that design to the independence assumption. De-duplication is by
decoded-pixel digest only. A one-pixel near-duplicate was kept, and with seed 0 it landed in **test** while its original
was in **train** (probe `P4`).

**Consequence:** a BYOD user with burst shots or several crops of one scene gets near-copies across train and test, and
the held-out loss then overstates transfer. The notebook gives no warning.

**Evidence:** source inspection; direct CPU execution of the carried `split_dataset` on synthetic images (probe `P4`).

**Recommended correction:** in Section 4, add one sentence saying that the random split assumes the photographs are
independent, that the sample enforces this with one photograph per observer per species, and that burst shots or
near-duplicates should be removed or kept within one split. Replace "split by caption" / "grouped by caption" with
"split within each caption (every caption appears in every split)". Optionally add a perceptual-hash near-duplicate
check to `split_dataset`.

**Acceptance check:** the notebook text names the independence assumption and the near-duplicate risk; no text or
docstring describes the BYOD split as grouped or held out by caption; the build check passes.

### KIP-m3 — Minor (spec MUST DAT12): the stated BYOD minimum is refused by the code, and two rules disagree

**Location:** opening BYOD paragraph (cell 0, "you need at least four photographs, and at least one caption with three or
more photographs"); Troubleshooting (cell 36, "give at least one caption three or more photographs and provide at least
four training photographs in total"). Generator: `tools/notebook_template.py` lines 139 and 657. Code:
`samples.split_dataset` (20 % / 20 % per caption) and `pipeline.MIN_TRAIN_RECORDS = 4`.

**Observed issue:** the opening's minimum (four photographs, one caption with three) is refused with `split leaves 2
training records; at least 4 are required`, and so are four photographs of one caption (probe `P3`). The smallest
accepted inputs are 6 photographs of one caption, or 4 + 4 across two captions. Troubleshooting states a different rule
again ("four **training** photographs"), which a learner cannot check before the split runs.

**Consequence:** a learner who prepares exactly the stated minimum is refused. The refusal message is actionable, so
this is friction, not a dead end, but DAT12 requires the stated limits to be accurate before upload.

**Evidence:** direct CPU execution of the carried `split_dataset` on synthetic records (probe `P3`); source inspection.

**Recommended correction:** state the rule the code enforces, once, in the opening and in Troubleshooting. For example:
"at least six photographs of one caption, or four of each of two captions; each caption with three or more photographs
gives one test and one validation photograph, and at least four must remain for training." Alternatively, change the
code to match the text, which is Kurt's call.

**Acceptance check:** a zip built to the stated minimum passes `split_dataset`, and one photograph fewer is refused; the
opening and Troubleshooting state the same rule.

### KIP-m4 — Minor (UX12): "no timing recorded" statements contradict the release record

**Location:** opening (cell 0, "No clean-runtime timing of this version has been recorded yet"); `tutorials/README.md`
table, Run-all column ("not yet recorded"). Generator: `tools/notebook_template.py` line 130.

**Observed issue:** `docs/release-verification.md` records 904.7 s for a strict single-pass Kaggle T4 run of this exact
blob, 702.9 s for the BYOD journey, and 79 s of environment setup on Colab (probe `P5`). The same `tutorials/README.md`
row cites these runs in its Release-status column.

**Consequence:** the learner has no idea how long `Run all` takes, although the downloads alone are about 16.5 GB, and
the README contradicts itself.

**Evidence:** source inspection; probe `P5`.

**Recommended correction:** state the measured duration with its environment, e.g. "about 15 minutes on a Kaggle T4
with an empty cache (904.7 s, recorded in `docs/release-verification.md`)", and fill the README's Run-all column.

**Acceptance check:** neither the notebook nor `tutorials/README.md` says the timing is unrecorded while the release
record holds a timed run of the same blob; any duration stated names its runtime.

### KIP-S1 — Suggestion: point Section 8 at the per-timestep paired change

Section 6 tells the learner to "compare timestep by timestep later", but the Section 8 What to notice reads only the
rows as a whole. The test mean (0.028245) is about 74 % the t = 100 term (probe `P2`). In the recorded run the adapted
loss is lower at **every** timestep (t = 100: 0.10464 → 0.10398; t = 900: 0.000216 → 0.000210). One sentence pointing at
`denoising_mse_test_by_timestep` would deliver the comparison Section 6 promises.

### KIP-S2 — Suggestion: give the new-caption CLIP values a reference

Section 9 prints CLIP 27.87 and 17.86 for the two new-caption inpaintings (recorded run). The second is below the
default run's mean-fill floor, but the learner has nothing to read it against, since the floor and the original are
scored only with the training captions. Scoring the same two photographs' mean-fill and original against `NEW_PROMPT`
would make the "sanity check" readable.

## 4. Readiness

**Needs revision.** There is no Blocker and no Major. The default and BYOD journeys have hosted evidence at this exact
blob with no restart. Two applicable MUSTs are unmet: SPL3 (KIP-m2) and DAT12 (KIP-m3). Both are fixed by prose or
docstring changes. KIP-m1 and KIP-m4 are template prose, plus one validator phrase. Because the notebook bytes would
change, a regenerated notebook needs a new hosted `Run all` record before it returns to Release-grade.

## 5. Verified versus inferred

- **Verified:** the notebook blob equals the hosted-evidence blob; the static gates pass (47 tests, validator,
  build check); the CLIP references, per-image scores, per-timestep losses and activity values come from the executor's
  own reports for this blob; the ceiling is computed per image with no self-match (code read); the BYOD minimum-size
  boundary and the near-duplicate behaviour were directly executed on CPU.
- **Inferred:** that the original photograph is exceeded because the generator renders the caption more literally (a
  plausible mechanism, not isolated); the effect of a one-pixel near-duplicate on the held-out loss.
- **Only Kurt can confirm:** whether the intended BYOD contract is the code's 20 % / 20 % per-caption split (fix the
  text) or the text (fix the code).
- **Most likely to be wrong:** KIP-m1's severity. On BYOD the "ceiling" sits 7 points below both models, and the
  Section 8 answer teaches that it caps CLIP. A reader could call that materially misleading (Major). It is rated Minor
  for consistency with KCD-m1, because unlike KGN-M1 the reference is computed correctly.
