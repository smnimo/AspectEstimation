# TODO for matlab-ver (preparation for the public release)

Reference: Shimonishi & Kawashima (2020), "A Two-step Approach for Interest Estimation from Gaze Behavior in Digital Catalog
Browsing", J. Eye Movement Research 13(1):4 ("the paper"). The 2019 scripts and outputs used for the paper are in
`python-ver/refs/` (git-ignored; "refs"). The Python port follows these scripts (`python-ver/TODO.md`).

Status (checked 2026-10-18): the bugs found in the 2019 scripts are fixed (section 3). MATLAB is not available in the
development environment; the fixed scripts were run with **GNU Octave 10.3** (+ statistics package) on refs' `gazedata.mat`
(section 3, 2026-10-18) and checked against refs and python-ver. Open items are in section 1.

## 1. Open tasks (verified against the code on 2026-10-17)

- **O1. One run in MATLAB** (all scripts already run in Octave, see section 3): `main_mask.m` with the full 100 restarts
  (Octave was run with 5), and the figures (Octave cannot show the MATLAB-only plotting details, e.g. `histogram`).
  `ReadwithTimestamp.m` / `Set_selected.m` need the 111 raw sessions (only one is available).
- **O2. Paper Fig. 8**: regenerate it with the corrected `for_rev2.m` and decide with the co-author how to report the
  correction (README already lists the values, see section 2 / P1).
- **O4. Clean-up of the paper-figure / exploratory scripts** (not bugs; the scripts are unchanged since 2019):
  - D2. `for_rev2.m` (Fig. 8 and reviewer plots; hard-coded 37 users / 111 sessions), `test_cor_p.m`, `learn_aspects_from_part.m`
    (pLSA on 10 temporal parts of `session_all`, i.e. without AOF), `confirm_bias.m` (p-values of the window counts, used to
    choose thr): keep as documented supplementary scripts or remove.
  - D3. `LoadQuestionnair.m`: questionnaire data unrelated to the paper, labels in Shift-JIS (garbled in UTF-8).
  - D4. `multi_choose_prob.m`, `multinomial.m`: not called by any script (`multi_choose_prob` is NOT `exact_prob`);
    `HomogenousProduct.m` is only used by `confirm_bias.m`.

## 2. Paper vs code

Consistent (checked with refs and python-ver): Eq. 1 (`calc_lkh.m`, `exact_prob.m`), Eq. 2/3/8 (`plsa.m`, `estep.m`, `mstep.m`),
l = 5, thr = 0.03, Table 2 (from `p_spe_topic3.mat`, 93/111 = 83.8 %), Fig. 10 (`plot_sim_task_asp.m`; printed as a second
"Figure 8" in the paper), Fig. 9 (`plot_aspect.R` on unscaled `p.vz`: the color bars 0..0.2 / 0..0.33 equal max(vz);
`refs/results/spe3.pdf`), Fig. 7 (session 17), "4 sessions without AOF, all shorter than l" (lengths 3, 4, 3, 1).
The AOF masks were not changed for the paper figures: refs' `segsessions.mat` equals the `calc_segments.m` output.

| # | Paper | 2019 code | Now |
|---|---|---|---|
| P1 | Fig. 8: "total number of detected AOF in each temporal position" | `for_rev2.m` summed `seqmask{s}(l)`, a linear index into the 15 x T mask; Fig. 8 shows these values `[37 20 21 31 28 30 40 36 36 30]` | number of AOF per time point: `[579 779 818 878 802 962 977 919 932 791]` (time points with AOF `[319 360 374 380 354 390 396 377 369 325]`); the qualitative statement of the paper still holds (O2) |
| P2 | AOF if x/l **>** phi | `>=` | `>` (no effect on this data: phi = 1/3, x/5 never equals 1/3) |
| P3 | time = transitions between browsed items | `LoadGaze.m` also drops gaze runs < 150 ms and merges repeated targets | unchanged; documented in README |
| P4 | gaze of both eyes | a missing left eye was never filled (sample dropped) | filled from the right eye (C1) |

## 3. Done (2026-10-17; not executed, see O1)

refs vs matlab-ver before the fixes: all `.m` files byte-identical; refs' `exact_prob.m` / `HomogenousProduct.m` lived in
`../functions`; refs' `gazedata.mat` has `session_id` / `selected` (not set by the 2019 `ReadwithTimestamp.m`); `task_att.mat` was
created by no script; `Akq_*.csv` were edited by hand (header, index column, copy x 2); `cor.mat` was computed before
`p_spe_topic5` / `p_all_topic4` were reordered.

- A1 `for_rev2.m`: Fig. 8 counts `sum(seqmask{s}(:, l))`; loads `gazedata` (was used unloaded), no absolute path.
- B1 `calc_matching.m`: `spe` / `all` matching active (was commented out; the active part block needed files of
  `learn_aspects_from_part.m`, now in that script). B2 `make_task_att.m` creates `task_att.mat`. B3 `ReadwithTimestamp.m` sets
  `session_id`, input paths `knowledge_csv` / `expdir` can be set, `fclose`. B4 `plot_dur_usr.m` defines `S`. B5 `plot_aspect.R`
  reads the `csvwrite` output and plots unscaled values. B6 `learn_aspects_from_mask.m` always recomputes the masks (no stale
  `segsessions.mat`), creates `results/`. B7 no `../functions` (`pathfunc.m` adds `matlab-ver`).
- C1 `LoadGaze.m` / `plot_org_gaze.m` fill a missing left eye. C2 x == 1 / y == 1 -> last column / row. C3 `calc_segments.m`:
  `N_value` from the catalogs of all sessions. C4 per-attribute value ranges (`calc_lkh.m` calls `exact_prob` directly instead
  of the 3-value `allpair3` table; `learn_aspects_from_mask.m` offsets from `cumsum(N_value)`). C5 `exact_prob.m` /
  `multi_choose_prob.m`: `error` on invalid input, dead `normalize` removed. C6 `plsa.m`: stopping rule documented.
  C7 `train.m`: optional seed (`rng(seed + i)`, `seed = 0` in `learn_aspects_from_mask.m`). C8 `calc_matching.m`:
  lexicographic order (idempotent), saves `perm`. C9 `plot_conf_matrix.m`: "no comparison" = sessions without AOF (masks).
  C10 `CalcEval.m`: `cortop` NaN for sessions without AOF. C11 `CalcSim.m` saves only the results. C12 `seglabel` renumbering.
  C13 `plot_sim.m` error bars = std. C14 `plot_eval_AoF.m` filters NaN explicitly.
- Effects w.r.t. the 2019 scripts: Fig. 8 (P1); mean cortop spe K=3 with the paper's p 0.754 -> 0.761 (C10); std error bars
  (C13); ties among unmatched aspects for K > 3 (C8); left-eye-only samples kept (C1); seeded restarts (C7). Table 2 and
  Fig. 10 are unchanged on refs. `plot_duration_segmentation_trialID.m` (former D5) is no longer blocked (B3 / B7; not run, O1).
- README rewritten (requirements, data layout, pipeline, scripts, paper results, Fig. 8 erratum candidate).

2026-10-18:
- O3 `test_cor_p.m` loads `cor-<N_part>_topic3.mat` with `N_part = 10` as `learn_aspects_from_part.m` (it loaded `cor-5`,
  a file of an earlier run with 5 parts).
- D1 `disp_mask.m`, `disp_mask_for_paper.m`, `plot_mask_for_paper.m` (three copies of the mask computation with hard-coded
  offsets and sessions) replaced by the function `plot_mask(sessions)`, which plots the saved `segsessions.mat`
  (`plot_mask(17)` = Fig. 7; the color range is fixed to [0, 1]).
- D6 all `.m` / `.R` files use LF; `.gitattributes` (repository root) keeps LF for code / documents and leaves the raw data
  (`*.csv`, `*.list`, CRLF) untouched.
- Octave run (O1, 2026-10-18; Octave 10.3 + statistics 1.8.2, R 4 + corrplot 0.95 from conda-forge; refs' `gazedata.mat`):
  - `main_mask.m` runs end to end (with `train_num = 5` in a local copy; 9 min). `calc_lkh.m`: `lkh_win_5.mat` identical to refs
    (max diff 0); `calc_segments.m`: masks identical; `plot_conf_matrix.m` (spe): `[[32 0 3 2] [0 29 8 0] [4 0 31 2]]` (one
    session from Table 2, within the python-ver criterion), cosine 0.824 / 0.856 / 0.859; with 5 restarts the "all" model ends in
    another local optimum (cosine 0.68 / 0.63 / 0.72; the paper used 100 restarts). The `.mat` outputs are read by python-ver.
  - `calc_matching.m` run twice keeps the order (C8); `CalcSim.m`: 4 NaN in `cortop` (C10); `plot_eval_AoF.m`: `distances.mat`
    equal to refs (max diff 2e-16); `for_rev2.m`: Fig. 8 `[579 779 818 878 802 962 977 919 932 791]`; `ReadwithTimestamp.m` on the
    raw session: `f` equal to refs (usr 2 / task 1), `session_id` set, `dur` equal to python-ver's fixed reader (5e-15), then
    stops in `Set_selected.m` (needs 37 users); `plot_org_gaze.m`, `plot_L_topic.m`, `plot_sim_task_asp.m`, `plot_mask.m`,
    `confirm_bias.m`, `learn_aspects_from_part.m`, `test_cor_p.m`, `plot_duration_segmentation_trialID.m` (with a `histogram`
    shim) run.
  - Fixed while running: `plot_sim.m` used `topic`, which was only set inside the commented-out block (error in MATLAB too);
    `save` calls now give the `.mat` extension explicitly (Octave does not add it); `plot_aspect.R` uses `col.lim` for
    corrplot >= 0.90 (`cl.lim` was ignored: color bar -1..1 instead of 0..max), output matches paper Fig. 9;
    `plot_dur_usr.m` / `test_cor_p.m` use calls that work in MATLAB and Octave (`set(gca, 'XTickLabel', ...)` instead of
    boxplot group labels, `patch(..., 'FaceAlpha', 0.3)` instead of `alpha`, `set(er, 'LineStyle', ...)`).
