# TODO / handover notes for python-ver

Goal: `python-ver` reproduces the computations of `matlab-ver`, the code of *Shimonishi & Kawashima (2020), "A Two-step
Approach for Interest Estimation from Gaze Behavior in Digital Catalog Browsing", J. Eye Movement Research 13(1):4*.
The corrected `matlab-ver` is the reference (its open items: `../matlab-ver/TODO.md`); the paper is the specification
where MATLAB is ambiguous; `refs/` holds the 2019 scripts and outputs used for the paper.

Status (checked 2026-10-17): steps 1 and 2 and the evaluation equal the MATLAB outputs of the paper (section 5);
open items are in section 1.

## 1. Open tasks (verified against the code on 2026-10-17)

- **O1. Paper Fig. 8 (`for_rev2.m`) is not ported.** Port the corrected count (number of AOF per time point, summed over
  10 temporal segments of each session) to `evaluation` / `plotter` / `main.py --plot` and test it on `refs/segsessions.mat`:
  expected `[579 779 818 878 802 962 977 919 932 791]` (the paper's `[37 20 21 31 28 30 40 36 36 30]` comes from a linear-index bug).
- **O2. `SessionProcessor.create_session_all` hard-codes the offsets +3, +6, +9, +12** (and `main.learn_aspects` raises unless
  every attribute has 3 values). The fixed `learn_aspects_from_mask.m` uses `offset = [0; cumsum(N_value(1:end-1))]`;
  generalize the same way and drop the check. No effect on the paper data (5 x 3 values).
- **O3 (D3). `.gitattributes`** (`python-ver/** text eol=lf`) to keep LF on Windows: not added yet (all files are LF now).
- **O4. Re-check after the first MATLAB run of the fixed matlab-ver.** The `.m` fixes were never executed; if running them
  reveals differences (e.g. `rng(seed + i)` streams, error fixes), update python-ver and section 4.
- Optional: `plot_duration_segmentation_trialID.m`, `disp_mask.m`, `learn_aspects_from_part.m` / `test_cor_p.m`,
  `confirm_bias.m` are not ported (exploratory / reviewer-response scripts, not needed for the main flow).

## 2. Background (what is computed)

A participant looks at 12 items (laptops, 3x4 grid) to fulfil one of 3 tasks. Each item has 5 attributes (price, screen size,
CPU score, memory capacity, weight), each with 3 values (1 = low/small/light ... 3 = high/large/heavy); every value is shared by
4 of the 12 items. A *session* = one participant x one task (37 x 3 = 111).

- **Step 1 - attributes-of-focus (AOF)** (`calc_lkh.m`, `calc_segments.m`; paper Eq. 1): gaze -> browsed items (time = item
  transitions) -> attribute values `v_t` (5 x T). A window of length l = 5 gives a count vector per attribute; its likelihood
  under a multinomial with phi_k = N_k / N is computed. If it is < thr (0.03), every value with `count/l > phi_k` becomes an AOF
  for the whole window. Result: binary mask `seqmask` (15 x T, rows attribute-major: price v1..v3, screen, CPU, memory, weight).
- **Step 2 - pIAF / pLSA** (`plsa.m`, `estep.m`, `mstep.m`; Eq. 2, 3, 8): each time point is a multi-hot f_t;
  P(f_t | r) = n_t! prod_k p_{r,k}^{f_t,k}; EM learns `p.vz` (aspects) and `p.zs` (interests theta(s)), on `seqmask` ("spe")
  and on all values `session_all` ("all").
- **Evaluation** (`calc_matching.m`, `CalcEval.m`, `CalcSim.m`, `plot_conf_matrix.m`, `plot_eval_AoF.m`; Eq. 4, Table 2,
  Fig. 10 (printed as a second "Figure 8" in the paper)): aspect-task matching, cosine similarity, task estimation = argmax theta(s).
  Paper: cosine 0.82 / 0.86 / 0.86 with AOF vs 0.61 / 0.69 / 0.58 without; Table 2 = `[[32,0,3,2],[0,30,7,0],[4,0,31,2]]`, 83.8 %.

## 3. MATLAB -> Python map

| MATLAB (`matlab-ver/`) | Python (`python-ver/`) | State |
|---|---|---|
| `ReadwithTimestamp.m`, `LoadGaze.m` | `src/data/gaze_reader.py` (`load_all_sessions`, `process_session`, `save_gazedata`, `gaze_xy`; CLI) | OK |
| `Set_selected.m` | - | not ported (`selected` is not used elsewhere) |
| `calc_lkh.m`, `exact_prob.m` | `main.calc_lkh`, `src/data/data_loader.py` `LikelihoodCalculator`, `src/utils/math_utils.py` | OK |
| `calc_segments.m` | `src/models/segments.py` `SegmentCalculator` | OK |
| `seqmask(:, sum==0) = []`, `session_all` | `SessionProcessor.remove_empty_columns`, `create_session_all` | OK (O2: 3 values per attribute only) |
| `plsa.m`, `estep.m`, `mstep.m`, `normalize.m`, `train.m` | `src/models/plsa.py` `FourLayerPLSA`, `main.train` | OK |
| `main_mask.m`, `learn_aspects_from_mask.m`, `make_task_att.m` | `main.py` (`learn_aspects`), `evaluation.GIVENTASK` | OK |
| `calc_matching.m`, `CalcEval.m`, `CalcSim.m`, `plot_conf_matrix.m`, `plot_eval_AoF.m` | `src/models/evaluation.py`, `main.py --eval` / `--eval-matlab` | OK |
| `plot_L_topic.m`, `plot_conf_matrix.m`, `plot_sim.m`, `plot_sim_task_asp.m`, `plot_eval_AoF.m`, `plot_dur_usr.m`, `plot_org_gaze.m`, `disp_mask_for_paper.m`, `plot_aspect.R` | `src/visualization/plotter.py`, `main.py --plot` | OK (checked visually, smoke tests) |
| `for_rev2.m` (Fig. 8) | - | O1 |

## 4. Conventions and verification

- `gazedata.mat`: `semantic_attributes{usr,task}` = 12x5 `KB(list,:)`; `semantic_attribute_seq{usr,task}` = 5xT `KB(list(f),:)'`;
  `session{usr,task}` with `f dur list task usr session_id totalduration candidate selected`. Cells are traversed column-major
  (by task, then user): rows `(task-1)*n_usr+1 : task*n_usr` belong to `task` (`main.load_gazedata_mat`, `from_reader_sessions`,
  `load_gazedata_sessions`, `evaluation.session_tasks`).
- `knowledge.csv` = KB (12 x 5, values 1..3; columns price, screen, CPU, memory, weight). `.list` = 12 item ids in grid order,
  task in the hundreds digit. Gaze CSV = `timestamp(us), right_x, right_y, left_x, left_y`, 0..1, `-1` = missing
  (the right/left order is taken from the docstrings; MATLAB does not name the columns).
- Task-related values as 15-dim indices (attribute*3 + value-1): task 1 {8, 11}, task 2 {3, 12}, task 3 {0, 11} (= `refs/task_att.mat`).
- Python follows the corrected matlab-ver; compared with the 2019 scripts of the paper (none changes Table 2 / Fig. 10 on refs):
  a missing left eye is filled (2019: dropped), x == 1 / y == 1 map to the last column / row, AOF rule `>` (2019 `>=`), `N_value`
  from the catalogs of all sessions, matching in lexicographic order (idempotent), "no comparison" and `cortop` from the AOF masks
  (`no_aof`; mean cortop spe K=3 with the paper's p 0.754 -> 0.761), `plot_sim` error bars = std, invalid inputs raise errors.
- Random restarts are seeded (`--seed`, restart i uses seed + i); numpy and MATLAB streams differ, so learned aspects are compared
  statistically: accepted when <= 2 sessions move vs Table 2 and cosine is within 0.01 (decision D6).

```bash
cd python-ver
uv venv && uv pip install -r requirements.txt          # or python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q                    # 31 tests, ~8 s (test_refs.py is skipped without refs/)
.venv/bin/python main.py --mat refs/gazedata.mat --plot --savedir /tmp/res   # real data, all figures, ~15 s
.venv/bin/python main.py --eval-matlab refs/results --plot                   # evaluate the MATLAB results of the paper
.venv/bin/python src/data/gaze_reader.py --timestamp-list ../experiment/timestamp.list --knowledge ../knowledge/knowledge.csv --out gazedata.mat
```

- Tests: `test_matlab_equivalence.py` / `test_evaluation.py` compare with line-by-line re-implementations of the MATLAB code;
  `test_refs.py` with the outputs of the paper in `refs/` (git-ignored, never commit it; only one raw gaze session is included);
  `test_plotter.py` are smoke tests. New ported code should get the same kind of test (reference = MATLAB semantics).
- `results/`, `.venv`, `refs/` are git-ignored; all python-ver files are LF.

## 5. History (done, 2026-10-17)

- Copilot's fixes verified: first window of `calc_lkh` (`range(win_size - 1, T)`), `n!` of `exact_prob`, removal of all-zero AOF
  columns, vectorized E/M-steps, equivalence tests.
- Real-data verification with `refs/`: `lkh_ses` (max diff 1e-17), `seqmask`, `session_all` identical; 4 sessions without AOF;
  MATLAB `p_spe_topic3` gives Table 2 / 83.8 %; `cor.mat` and `distances.mat` reproduced; the raw session usr 2 / task 1 equals
  `gazedata.mat` with the 2019 gaze handling (`ml_load_gaze(matlab_bug=True)`).
- T1 evaluation ported (`evaluation.py`, `--eval`, `--eval-matlab`); `main.learn_aspects` crash when saving `segsessions.npz` fixed.
- T7 training: E/M-steps from MATLAB's p equal `plsa.m` on real data (rel 1e-12); the saved MATLAB p is not converged
  (`lim = 1e-4`; continuing EM changes spe accuracy 93 -> 94); 20 seeds: spe 92-95 correct, <= 2 sessions moved; criterion D6.
- H1 / T2 gaze reading: missing eye filled both ways, edge coordinates clipped, invalid ids raise; `load_all_sessions` follows
  `ReadwithTimestamp.m` (with `session_id`), fake-data fallbacks removed, `save_gazedata` writes a gazedata.mat.
- T3-T5 clean-up: single training API (`main.train`, `FourLayerPLSA.fit` with MATLAB defaults; unused P(o|v) layer removed),
  unused / diverging code and silent fallbacks removed, stale mock-data scripts deleted, README rewritten, unused dependencies removed.
- T8 figures: `plotter.py` rewritten (the old functions were mostly not ports), `main.py --plot`; `plot_aspect_circles` plots
  unscaled vz like paper Fig. 9 (the x 2 of the hand-edited 2019 CSVs is not used).
- T9 sync with the corrected matlab-ver (P2, B3, B5, B6, C1-C3, C6-C10, C13, C14; see `../matlab-ver/TODO.md`), except O1 / O2.

## 6. Decisions

- Open: D3 (`.gitattributes`, O3).
- Resolved: D1 AOF rule `>` (paper; no effect on this data), D2 `task_att.mat` = `evaluation.GIVENTASK`, D4 stale scripts deleted,
  D5 `refs/` is git-ignored, D6 training acceptance criterion (section 4), H1 fixed in both versions, aspect plots without scaling.
