# AspectEstimation - MATLAB Version

MATLAB code of

> Shimonishi & Kawashima (2020), "A Two-step Approach for Interest Estimation from Gaze Behavior in Digital Catalog
> Browsing", *Journal of Eye Movement Research* 13(1):4. doi: 10.16910/jemr.13.1.4

Step 1 ("MSET" in the code, AOF detection in the paper) detects attributes-of-focus by a likelihood-based short-term analysis;
step 2 learns aspects and per-session interests with pLSA (the pIAF model). A Python port is in `../python-ver`.

**Status:** the scripts used for the paper (kept in `../python-ver/refs`, not in git) were corrected for this release:
the pipeline runs end to end, latent bugs were fixed, and paper Fig. 8, which was computed with an indexing bug, is now computed
as described in the paper. The corrected scripts were checked against the paper-time outputs (through the Python port and by
running them in GNU Octave); they have not been run in MATLAB. Changes are summarized in
[Changes from the 2019 scripts](#changes-from-the-2019-scripts); open issues are in [TODO.md](TODO.md).

## Requirements

- MATLAB with the Statistics and Machine Learning Toolbox (`pdist`, `squareform`, `ttest2`, `combnk`, `boxplot`).
  `train.m` uses `parfor` (Parallel Computing Toolbox; runs serially without it).
- R with `corrplot` for `plot_aspect.R` (optional).
- GNU Octave (>= 10, with the `statistics` package) also runs the scripts: start with `pkg load statistics` and
  `save_default_options('-mat7-binary')` (MATLAB-compatible `.mat` files); `parfor` runs serially and `histogram`
  (`plot_duration_segmentation_trialID.m`) is not available.

## Data layout

The scripts are run from `matlab-ver/` and expect (paths are hard-coded):

```
../knowledge/knowledge.csv     KB: 12 items x 5 attributes (price, screen size, CPU score, memory capacity, weight), values 1..3
../experiment/timestamp.list   one session name per line, 4 lines per participant (the 4th session is skipped)
../experiment/<name>.csv       gaze: timestamp (us), right_x, right_y, left_x, left_y (normalized 0..1, -1 = missing)
../experiment/<name>.list      the 12 item ids in grid order (3 x 4, row-major); hundreds digit = task (e.g. 101 = task 1, item 1)
../experiment/<name>.jpg       screenshot (plot_org_gaze.m only)
```

The input paths of `ReadwithTimestamp.m` can be changed by setting `knowledge_csv` and `expdir` before running it.
`results/` is created by `learn_aspects_from_mask.m`; `task_att.mat` (paper Table 1) by `make_task_att.m`.

`../sample_data/` contains one session (`170116024150092.*`) and `knowledge.csv`.
Feature index of attribute `a` (1..5) and value `v` (1..3): `(a - 1) * 3 + v` (15 features).
Sessions are stored as `{usr, task}` cells (37 x 3) and traversed with `(:)`, i.e. by task, then by user.

## Pipeline (`main_mask.m`)

| Order | Script | Input -> output |
|---|---|---|
| 1 | `ReadwithTimestamp.m` (+ `LoadGaze.m`, `Set_selected.m`) | raw gaze -> `gazedata.mat` (`session` with `f`, `dur`, `list`, `task`, `usr`, `session_id`, `totalduration`, ...; `semantic_attribute_seq{usr,task}` 5 x T, `semantic_attributes{usr,task}` 12 x 5, `KB`) |
| 2 | `calc_lkh.m` (+ `exact_prob.m`) | likelihood of each window (l = 5) per attribute -> `lkh_win_5.mat` (`lkh_ses`) |
| 3 | `learn_aspects_from_mask.m` (+ `calc_segments.m`, `train.m`, `plsa.m`, `estep.m`, `mstep.m`, `normalize.m`) | AOF masks (thr = 0.03) -> `segsessions.mat`; pLSA with 1..5 aspects, 100 restarts (seeded), on the masks ("spe") and on all values ("all") -> `results/p_{spe,all}_topicK.mat` (`p.vz` aspects, `p.zs` interests, `L` log-likelihood), `results/Akq_*.csv` |
| 4 | `make_task_att.m` | `task_att.mat` (`giventask`, 3 x 15) |
| 5 | `calc_matching.m` | reorder the aspects so that aspect k matches task k (K = 2..5; idempotent; saves `perm`) |
| 6 | `CalcSim.m` (+ `CalcEval.m`) | cosine similarity between tasks and aspects -> `results/cor.mat` (`cortop` is NaN for sessions without AOF) |
| 7 | `plot_conf_matrix.m` | task estimation (3 aspects): confusion matrix (paper Table 2); sessions without AOF = "no comparison" |

`main_mask.m` runs steps 2-7 (step 1, which needs the raw data, is commented out) and starts from an existing `gazedata.mat`.

Processing details:
- `LoadGaze.m`: fills a missing eye from the other one, averages both eyes, maps the gaze to the 3 x 4 grid, drops runs shorter
  than 150 ms and merges consecutive identical targets; time = transitions between browsed items.
- `calc_lkh.m`: multinomial likelihood of the value counts in each window under phi_k = N_k / N (paper Eq. 1); the first
  `l - 1` time points keep likelihood 1.
- `calc_segments.m`: windows with likelihood < thr are distinctive; values with count / l > phi_k become AOF for the whole
  window (paper; the 2019 scripts used `>=`, identical for this data).
- `plsa.m`: EM until the relative increase of the log-likelihood is <= 1e-4 (at most 999 iterations; the result is not
  necessarily converged); `train.m` keeps the best of the random restarts (`rng(seed + i)`, `seed = 0` in
  `learn_aspects_from_mask.m`; the paper's results were not seeded).

## Figures and supplementary scripts

| Script | Content |
|---|---|
| `plot_L_topic.m` | final log-likelihood vs number of aspects; aspects as images |
| `plot_aspect.R` | aspects as a circle matrix (paper Fig. 9) from `results/Akq_*.csv`, values not scaled |
| `plot_sim_task_asp.m`, `plot_sim.m` | cosine similarity between tasks and matched aspects (paper Fig. 10); `plot_sim.m` error bars = std |
| `plot_mask.m` | function: all values and AOF of given sessions from `segsessions.mat` (`plot_mask(17)` = paper Fig. 7) |
| `plot_org_gaze.m` | gaze trajectory over the screenshot (paper Fig. 6) |
| `plot_eval_AoF.m` | similarity of AOF detected by All / Top / Comparison / LD to the task, t-tests -> `distances.mat` |
| `plot_dur_usr.m`, `plot_duration_segmentation_trialID.m` | session durations, number of browsed items, AOF statistics (need `session_id`) |
| `pathfunc.m` | adds `matlab-ver` to the path (used by `plot_duration_segmentation_trialID.m`) |

## Results of the paper

With the outputs used for the paper: 111 sessions (37 participants x 3 tasks), 4 sessions without AOF (shorter than the window),
task estimation accuracy 83.8 % (Table 2), cosine similarity between tasks and matched aspects 0.82 / 0.86 / 0.86 with AOF detection
and 0.61 / 0.69 / 0.58 without (Fig. 10).

Erratum candidate: paper Fig. 8 ("total number of detected AOF in each temporal position", values 37, 20, 21, 31, 28, 30, 40, 36,
36, 30) was computed with a linear index into the AOF masks. The number of AOF per segment of the 10 temporal segments is
579, 779, 818, 878, 802, 962, 977, 919, 932, 791 (time points with AOF: 319, 360, 374, 380, 354, 390, 396, 377, 369, 325); the
statement that comparison happens mainly in the latter part and decreases at the end still holds.

## Changes from the 2019 scripts

The scripts used for the paper (2019) were corrected for this release (2026-10). With the paper's `gazedata.mat`, step 1
(`lkh_win_5.mat`, AOF masks) is unchanged, and Table 2 / Fig. 10 do not change when evaluated on the paper's learned models.

- Results that change: paper Fig. 8 (indexing bug, see above; its script was removed, the corrected count is computed by
  `python-ver`, `main.py --plot`); `cortop` of `CalcEval.m` excludes sessions without
  AOF (they used aspect 1); `plot_sim.m` error bars show the standard deviation (was the variance); the random restarts are
  seeded (`train.m`, `seed = 0`), so learned models differ from the paper's by random-restart variation.
- Gaze reading (`LoadGaze.m`, `plot_org_gaze.m`): a missing left eye is filled from the right eye (such samples were dropped);
  gaze exactly at x = 1 or y = 1 belongs to the last column / row (it pointed to a non-existent cell).
- AOF rule `count / l > phi_k` as in the paper (was `>=`; identical on this data); value ranges are taken from the catalogs of
  all sessions and handled per attribute (no hard-coded 3 values / offsets).
- Pipeline: `main_mask.m` runs end to end (`calc_matching.m` matching blocks re-enabled, `make_task_att.m` added); matching is
  idempotent (lexicographic order) and saves the permutation; "no comparison" in `plot_conf_matrix.m` uses the AOF masks;
  `ReadwithTimestamp.m` stores `session_id`; the masks are always recomputed (no stale `segsessions.mat`); no dependency on
  `../functions`; `save` always writes `.mat` files; `exact_prob.m` raises errors on invalid input.
- Figures: `plot_aspect.R` reads the `csvwrite` output and plots unscaled values (the 2019 CSVs held a copy x 2 for
  visibility), with `col.lim` for corrplot >= 0.90; `plot_sim.m` defines `topic` for its active figure (was an error);
  `plot_mask.m` replaces three mask scripts; `plot_dur_usr.m` also runs in Octave; all files use LF.
- Removed: supplementary and unused scripts that are not needed for the paper's results: `for_rev2.m` (Fig. 8 and reviewer
  plots), `learn_aspects_from_part.m` / `test_cor_p.m` (pLSA on temporal parts, reviewer response), `confirm_bias.m` (choice of
  thr), `LoadQuestionnair.m` (unrelated questionnaire), `multi_choose_prob.m`, `multinomial.m`, `HomogenousProduct.m` (unused).

Details of every change are in the commit history and pull requests #2 and #3.
