# TODO for matlab-ver (preparation for the public release)

Reference: Shimonishi & Kawashima (2020), "A Two-step Approach for Interest Estimation from Gaze Behavior in Digital Catalog
Browsing", J. Eye Movement Research 13(1):4 ("the paper"). The scripts and outputs used for the paper are in `python-ver/refs/`
(git-ignored; "refs"). Items marked [py] are handled in `python-ver` as well (see `python-ver/TODO.md`, T9).

**Status (2026-10-17):** A1, B1-B7, C1-C14 and P2 are fixed in the `.m` / `.R` scripts; D (clean-up) is open.
MATLAB is not available in the development environment, so the fixed scripts have NOT been run. They were checked by
reading and by re-implementing the changed computations in Python against `refs` (`python-ver/tests`): with the fixes,
`calc_lkh.m` / `calc_segments.m` (incl. P2, C3, C4) still give the paper-time `lkh_win_5.mat` / `segsessions.mat` exactly;
the fixed `plot_conf_matrix.m` rule still gives Table 2; `for_rev2.m` now gives the corrected Fig. 8 values below.
**Run `main_mask.m` (and the figure scripts) once in MATLAB before publishing.** Behavior changes w.r.t. the paper-time
scripts: Fig. 8 (A1), `cortop` excludes sessions without AOF (C10; mean cortop spe K=3: 0.754 -> 0.761 with the paper's p),
error bars of `plot_sim.m` are std (C13), the aspect order of `calc_matching.m` may differ in ties among unmatched aspects
for K > 3 (C8), gaze samples with only the left eye are kept (C1), restarts are seeded (C7, different random results).

## 1. refs (paper-time implementation) vs matlab-ver

- Scripts: every `.m` file in refs is byte-identical to `matlab-ver` (including line endings). refs lacks `exact_prob.m`,
  `HomogenousProduct.m` (they were in `../functions`, added by `pathfunc.m` / `addpath('../functions')`) and keeps
  `plot_org_gaze.m` in `for_plot_org_data/script/` (identical).
- Data / outputs that the shipped scripts do not reproduce:
  - `gazedata.mat` sessions have the fields `session_id` and `selected`; the shipped `ReadwithTimestamp.m` does not set
    `session_id` (needed by `plot_dur_usr.m`, `plot_duration_segmentation_trialID.m`). `selected` comes from `Set_selected.m`.
  - `task_att.mat` (`giventask`, `task_att`) is not created by any script (it equals paper Table 1).
  - `results/Akq_*.csv` were edited by hand for `plot_aspect.R`: a header row, a row-index column and a second copy of `p.vz`
    multiplied by 2 (`csvwrite` in `learn_aspects_from_mask.m` writes only the 15 x K values).
  - `results/cor.mat` (`CalcSim.m`) was computed before `p_spe_topic5` / `p_all_topic4` were reordered by `calc_matching.m`
    (same values, permuted columns), i.e. the scripts were run in a different order than `main_mask.m`.
  - `distances.mat` (`plot_eval_AoF.m`) and `p_part*`, `cor-*_topic3.mat` (`learn_aspects_from_part.m`, `calc_matching.m`,
    `test_cor_p.m`) are reviewer-response material, not paper figures.

## 2. Paper vs code

Consistent (checked with refs and python-ver): Eq. 1 (`calc_lkh.m`, `exact_prob.m`), Eq. 2/3/8 (`plsa.m`, `estep.m`, `mstep.m`),
l = 5 and thr = 0.03, Table 2 (from `p_spe_topic3.mat`, accuracy 93/111 = 83.8 %), Fig. 10 (`plot_sim_task_asp.m`),
Fig. 9 (`plot_aspect.R` on unscaled `p.vz`: the color bars 0..0.2 / 0..0.33 equal max(vz); `refs/results/spe3.pdf`),
Fig. 7 (= session 17 of `disp_mask_for_paper.m`), "4 sessions without AOF, all shorter than l" (their lengths are 3, 4, 3, 1).
There is no evidence that the AOF masks were changed for the paper figures: `segsessions.mat` equals `calc_segments.m` output.

Differences:

| # | Paper | Code | Effect |
|---|---|---|---|
| P1 | Fig. 8: "total number of detected AOF in each temporal position" | `for_rev2.m` sums `seqmask{s}(l)`, a linear index into the 15 x T mask (elements of the first columns), not the AOF at time l | **Fig. 8 shows the buggy values** `[37 20 21 31 28 30 40 36 36 30]`. Correct: number of AOF `[579 779 818 878 802 962 977 919 932 791]`, time points with AOF `[319 360 374 380 354 390 396 377 369 325]` (refs/segsessions.mat). The qualitative statement (more comparison in the latter part, decreasing at the end) still holds |
| P2 | AOF if x/l **>** phi | `calc_segments.m`: `>=` (fixed: `>`) | none for this data (phi = 1/3 and x/5 never equals 1/3) |
| P3 | time = transitions between browsed items | `LoadGaze.m` also drops gaze runs shorter than 150 ms and merges repeated targets | not described in the paper; document |
| P4 | 60 Hz samples, both eyes | `LoadGaze.m` never fills a missing left eye (H1, see C1) | samples with only the left eye are dropped |

## 3. Fixes

### A. Results / figures of the paper
- [done] A1 (P1). `for_rev2.m`: now sums `sum(seqmask{s}(:, l))`, the number of AOF at time l, as the paper text says
  ("total number of detected AOF"); the absolute path `/Users/simonisi/.../cor.mat` (unused) was replaced by `load gazedata`
  (`session` was used without being loaded). Open: regenerate Fig. 8 and decide on an erratum note (with the co-author).

### B. The shipped pipeline does not run as is
- [done] B1. `main_mask.m` -> `calc_matching.m`: the `spe` / `all` matching blocks are commented out (`%{ ... %}`) and the active block
  loads `p_part*-10_topic3.mat`, which only `learn_aspects_from_part.m` creates -> error. `CalcSim.m` / `plot_conf_matrix.m`
  need matched `p_spe_topicK` / `p_all_topicK`. Activate the two blocks, move the part block to `learn_aspects_from_part.m`. [py]
- [done] B2. `task_att.mat` is not generated: add a script (giventask from paper Table 1: task 1 {CPU 3, memory 3}, task 2 {screen 1,
  weight 1}, task 3 {price 1, memory 3}). [py: `evaluation.GIVENTASK`]
- [done] B3. `ReadwithTimestamp.m`: set `session{usr, task}.session_id = session_id` (as in refs `gazedata.mat`); input paths
  `../knowledge/knowledge.csv`, `../experiment/` are outside the repository (make them parameters / document). [py]
- [done] B4. `plot_dur_usr.m` uses `S` without defining it (`S = numel(session)`), and `session_id` (B3).
- [done] B5. `plot_aspect.R` reads hand-edited CSVs (header, index column, x 2 copy; `Cs <- C/2+1.5` selects the copy). Read the
  `csvwrite` output directly and plot `p.vz` without the factor 2 (the paper used unscaled values). [py]
- [done] B6. `learn_aspects_from_mask.m` loads `segsessions.mat` if it exists (`try load('segsessions')`): a stale cache silently
  replaces the masks when `thr` / `win_size` / data change. Recompute, or check the saved `thr` / `win_size`.
- [done] B7. `pathfunc.m` / `addpath('../functions')`: functions outside the repository; `exact_prob.m`, `normalize.m`,
  `HomogenousProduct.m` are now in `matlab-ver`, so remove the dependency.

### C. Latent bugs (no effect on the paper data)
- [done] C1 (H1). `LoadGaze.m` / `plot_org_gaze.m`: the second line of the missing-eye handling,
  `x(x(:, 1)==-1, 2) = x(x(:, 1)==-1, 1)`, runs after column 1 was filled, so a missing left eye is never filled and the
  sample becomes (x_R - 1)/2 < 0 (dropped). 24 samples in `sample_data`. Fix with the mask of column 2. [py]
- [done] C2. `LoadGaze.m`: x == 1 or y == 1 is accepted (`x > 1` is the rejection test) but maps to a non-existent cell
  (next row, or target > 12 -> index error in `list(f)`). Clip to the last column / row. Not present in the data. [py]
- [done] C3. `calc_segments.m:8`: `N_value` is taken from the browsed values of session 1 only (fewer mask rows if session 1 never
  shows a value); use all sessions / `semantic_attributes` as `calc_lkh.m`. Harmless on the paper data. [py]
- [done] C4. `calc_lkh.m`, `calc_segments.m`: `hist(..., 1:size(prob_att{1}, 2))` uses the value range of attribute 1 for all
  attributes, and `allpair3` assumes 3 values ("N_value(1) ?????" comment); `learn_aspects_from_mask.m` /
  `disp_mask*.m` hard-code the offsets +3, +6, +9, +12 (with wrong comments, e.g. "[5 -- 8]"). Fine for 5 x 3 values only.
- [done] C5. `exact_prob.m` / `multi_choose_prob.m`: `prob_vec = normalize(prob)` is overwritten on the next line (dead code), and
  `return` on invalid input leaves `out` undefined (error message in the caller instead of a clear error). Use `error(...)`.
- [done] C6. `plsa.m`: stops when the relative increase is <= `lim` = 1e-4; the saved `p` is not converged (continuing EM changes
  the spe accuracy from 93 to 94 sessions) and `L(end)` belongs to the parameters before the last M-step. Document, or use a
  stricter `lim`. [py: same behavior, documented]
- [done] C7. `train.m`: no random seed (`rng`) -> results cannot be reproduced; `parfor` streams differ between runs.
- [done] C8. `calc_matching.m`: ties among the unmatched aspects (K > 3) are resolved in `perms` order, so running it twice
  reverses them again (not idempotent); it overwrites the training results (`p_*_topicK.mat`, `Akq_*.csv`).
- [done] C9. `plot_conf_matrix.m`: "no comparison" is detected by `val == 1/n_task` (exact float comparison, works only because
  `mstep.m` sets exactly `normalize(ones(1, K))`); it also assumes topic == n_task. Better: sessions with no AOF column. [py: same rule]
- [done] C10. `CalcEval.m`: sessions without AOF (uniform zs) contribute `cor(1)` (aspect 1) to `cortop`; exclude them.
- [done] C11. `CalcSim.m`: `save(sprintf('./%s/cor', loaddir))` saves the whole workspace (KB, lkh_ses, session, ...);
  `cell(4, 1)` for 5 topics.
- [done] C12. `learn_aspects_from_mask.m`: in the `seglabel` clean-up, `n = n - 1` inside `for n = ...` has no effect (MATLAB for
  loops ignore it). `segsession` / `seglabel` are saved but not used for training.
- [done] C13. `plot_sim.m`: the error bars are the variance (`var`), not the standard deviation; the first figure is commented out.
- [done] C14. `plot_eval_AoF.m`: `cos_dist_mset` is NaN (0/0) for sessions without AOF and they are removed implicitly by
  `cos_dist_mset >= 0`; make it explicit.

### D. Paper-figure / exploratory scripts to clean up for publication
- D1. `disp_mask_for_paper.m`, `plot_mask_for_paper.m`, `disp_mask.m`: three copies of the `session_all` / `calc_segments`
  code with hard-coded sessions (17 = Fig. 7; 34, 46) and `N_value = [3, 3, 3, 3, 3]`; merge into one function
  (session index as argument) that uses the saved masks.
- D2. `for_rev2.m` (Fig. 8 and reviewer plots), `test_cor_p.m`, `learn_aspects_from_part.m` (pLSA on 10 temporal parts of
  `session_all`, i.e. without AOF), `confirm_bias.m` (p-values of the multinomial windows, used to choose thr): keep as
  documented supplementary scripts or remove.
- D3. `LoadQuestionnair.m`: questionnaire data unrelated to the paper, labels in Shift-JIS (garbled in UTF-8), CRLF.
- D4. `multi_choose_prob.m`, `multinomial.m`: not used by the pipeline (`multi_choose_prob` is NOT `exact_prob`).
- D5. `plot_duration_segmentation_trialID.m`: uses `session_id` (B3) and `pathfunc` (B7).
- D6. Mixed line endings (several scripts use CRLF, others LF); README (done: usage and data layout).

## 4. Next steps
1. Run `main_mask.m`, `for_rev2.m`, `plot_eval_AoF.m`, `plot_dur_usr.m` and `plot_aspect.R` in MATLAB / R and fix any error
   (none of the fixed scripts could be executed here).
2. Decide how to treat Fig. 8 (A1) in the README / an erratum.
3. D1-D6 (clean-up of the paper-figure and exploratory scripts).
