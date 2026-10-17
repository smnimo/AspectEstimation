# TODO for matlab-ver

Open issues only (checked 2026-10-18). Background, pipeline and the changes from the 2019 scripts used for the paper are in
[README.md](README.md); the 2019 scripts and outputs are kept in `python-ver/refs/` (git-ignored).

## T1. One run in MATLAB
The corrected scripts were run with GNU Octave 10.3 (+ statistics package) on the paper's `gazedata.mat`, but never in MATLAB.
- Run `main_mask.m` with the full `train_num = 100` (Octave was run with 5 restarts in a local copy, ~9 min). Expected: step 1
  identical to the paper's outputs, 4 sessions without AOF, Table 2 within about 2 sessions (restarts are now seeded, so the
  numbers differ slightly from the paper), cosine about 0.82 / 0.86 / 0.86 with AOF and 0.61 / 0.69 / 0.58 without.
- Check the figure scripts in MATLAB (Octave cannot reproduce some plotting functions, e.g. `histogram` in
  `plot_duration_segmentation_trialID.m`).
- `ReadwithTimestamp.m` / `Set_selected.m` need the raw data of all 111 sessions (only one session is available; with it, the
  script stops in `Set_selected.m`, which has a hard-coded table for 37 participants).

## T2. How to report the correction of paper Fig. 8
The paper's Fig. 8 (`[37 20 21 31 28 30 40 36 36 30]`) was computed with a linear index into the AOF masks (`for_rev2.m`).
The corrected count per temporal segment is `[579 779 818 878 802 962 977 919 932 791]` (time points with AOF:
`[319 360 374 380 354 390 396 377 369 325]`); the qualitative statement of the paper still holds. Decide with the co-author
whether to publish an erratum; README lists the values as an erratum candidate.

## T3. Supplementary and unused scripts (keep or remove)
- `for_rev2.m` (Fig. 8 and reviewer plots; hard-codes 37 participants / 111 sessions), `test_cor_p.m`,
  `learn_aspects_from_part.m` (pLSA on 10 temporal parts of the sessions without AOF), `confirm_bias.m` (p-values of the window
  counts, used to choose thr): keep as documented supplementary scripts or remove.
- `LoadQuestionnair.m`: questionnaire data unrelated to the paper; labels in Shift-JIS (garbled in UTF-8).
- `multi_choose_prob.m`, `multinomial.m`: not called by any script; `HomogenousProduct.m` is only used by `confirm_bias.m`.
