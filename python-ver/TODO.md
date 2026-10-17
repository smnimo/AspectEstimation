# TODO / handover notes for python-ver

Goal: `python-ver` must reproduce the computations of `matlab-ver` (the code behind the paper
*Shimonishi & Kawashima (2020), "A Two-step Approach for Interest Estimation from Gaze Behavior in Digital
Catalog Browsing", J. Eye Movement Research 13(1):4*). The MATLAB code is the reference; the paper is the
specification where MATLAB is ambiguous. Read sections 1-4 first, then pick a task from section 6.

## 1. Background (what is computed)

A participant looks at 12 items (laptops, 3x4 grid) to fulfil one of 3 tasks. Each item has 5 attributes
(price, screen size, CPU score, memory capacity, weight), each with 3 values (1 = low/small/light ... 3 = high/large/heavy);
every value is shared by 4 of the 12 items. A *session* = one participant x one task (paper: 37 x 3 = 111).

- **Step 1 - attributes-of-focus (AOF)** (`calc_lkh.m`, `calc_segments.m`; paper Eq. 1):
  gaze -> sequence of browsed items (time = item transitions) -> attribute-value sequence `v_t` (5 x T).
  A window of length l = 5 (shift 1) gives a count vector per attribute; its likelihood under a multinomial with
  phi_k = N_k / N (neutral browsing) is computed. If likelihood < thr (0.03), the window is "distinctive" and every
  value with `count/l >= phi_k` becomes an AOF for all time points of that window. Result: binary mask `seqmask`
  (15 x T, rows = attribute-major: price v1..v3, screen v1..v3, CPU, memory, weight).
- **Step 2 - pIAF / pLSA** (`plsa.m`, `estep.m`, `mstep.m`; paper Eq. 2, 3, 8):
  each time point is a multi-hot vector f_t (the AOF); P(f_t | aspect r) = n_t! prod_k p_{r,k}^{f_t,k} (`exact_prob.m`);
  EM learns `p.vz` (aspect x 15, the aspects) and `p.zs` (session x aspect, the interest theta(s)).
  Trained on `seqmask` ("spe", with the AOF step) and on all attribute values `session_all` ("all", without the step).
- **Evaluation** (`calc_matching.m`, `CalcEval.m`, `CalcSim.m`, `plot_conf_matrix.m`; paper Eq. 4, Fig. 8, Table 2):
  match learned aspects to the 3 tasks, cosine similarity to task-related attribute values, task estimation = argmax theta(s).
  Paper results (3 aspects): cosine ~0.82-0.86 with AOF vs ~0.58-0.69 without (read from Fig. 8); accuracy 83.8 % with
  Table 2 = `[[32,0,3,2],[0,30,7,0],[4,0,31,2]]` (last column = "no comparison").

## 2. MATLAB -> Python map

| MATLAB (`matlab-ver/`) | Python (`python-ver/`) | State |
|---|---|---|
| `ReadwithTimestamp.m`, `LoadGaze.m` | `src/data/gaze_reader.py` `GazeDataReader.load_session_from_files` / `_process_session_data` | OK (verified on `sample_data`); `load_all_sessions` wrong, see T2 |
| `Set_selected.m` | - | not ported (T2) |
| `calc_lkh.m`, `exact_prob.m`, `HomogenousProduct.m` | `main.calc_lkh`, `src/data/data_loader.py` `LikelihoodCalculator`, `src/utils/math_utils.py` | OK |
| `calc_segments.m` | `src/models/segments.py` `SegmentCalculator.calc_segments` | OK |
| `seqmask(:, sum==0) = []` | `SessionProcessor.remove_empty_columns` | OK |
| `session_all` | `SessionProcessor.create_session_all` | OK (3 values per attribute assumed, as in MATLAB) |
| `plsa.m`, `estep.m`, `mstep.m`, `normalize.m` | `src/models/plsa.py` `FourLayerPLSA`, `multinomial_probs` | OK (vectorized) |
| `train.m` (best of 100 restarts, parfor) | `main.train` (process pool) | OK; old `plsa.train_multiple_runs` differs, see T3 |
| `main_mask.m`, `learn_aspects_from_mask.m` | `main.py` (`learn_aspects`, CLI) | OK |
| `calc_matching.m`, `CalcEval.m`, `CalcSim.m`, `plot_conf_matrix.m` | `src/models/evaluation.py`, `main.py --eval` / `--eval-matlab` | OK (verified on `refs/`) |
| `plot_eval_AoF.m`, `plot_sim.m` | - | not ported (plots only) |
| `plot_*.m` | `src/visualization/plotter.py` | exists, not re-verified against MATLAB |
| `confirm_bias.m`, `learn_aspects_from_part.m`, `for_rev2.m`, `test_cor_p.m`, `multi_choose_prob.m` | - | not ported / not needed for the main flow (`multi_choose_prob.m` is NOT `exact_prob`) |

## 3. Data conventions (easy to get wrong)

- `gazedata.mat` (made by `ReadwithTimestamp.m`): `semantic_attributes{usr,task}` = 12x5 `KB(list,:)`;
  `semantic_attribute_seq{usr,task}` = 5xT `KB(list(f),:)'`; `session{usr,task}` with `.f .dur .list .task .usr .candidate`.
  MATLAB traverses cells with `(:)` = **column-major**, i.e. ordered by task, then user (`calc_lkh.m`, `plot_conf_matrix.m`:
  rows `(task-1)*n_usr+1 : task*n_usr` belong to `task`). `main.load_gazedata_mat` and `main.from_reader_sessions` follow this.
- `knowledge.csv` = KB, 12 items x 5 attributes, values 1..3. Column order (verified with the `candidate` items of
  `ReadwithTimestamp.m`): price, screen size, CPU score, memory capacity, weight.
- `.list` file = the 12 item ids of the session, task in the hundreds digit (e.g. `101 102 103 109 ...` = task 1);
  `item - task*100` indexes KB; position in the list = grid position (1..12, row-major 3x4) used by `f`.
- Gaze CSV = `[timestamp(us), right_x, right_y, left_x, left_y]`, normalized 0..1, `-1` = missing.
- Tasks (paper Table 1) and task-related values (verified: the `candidate` items of `ReadwithTimestamp.m` satisfy them):
  task 1 (home, movies/games): CPU = 3, memory = 3; task 2 (carry, notes): screen = 1, weight = 1; task 3 (low cost, web): price = 1, memory = 3.
  As 15-dim vectors (index = attribute*3 + value-1): task 1 -> {8, 11}, task 2 -> {3, 12}, task 3 -> {0, 11}.
  `task_att.mat` (`giventask`, 3x15) in `refs/` equals these vectors (`evaluation.GIVENTASK`).
- Time index: MATLAB is 1-based, windows are `t = win_size:T` -> 0-based `win_size-1 .. T-1` (this was a bug, fixed).

## 4. Environment and verification

```bash
cd python-ver
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # numpy is not installed system-wide
.venv/bin/python -m pytest tests -q          # MATLAB-equivalence tests (25 tests, ~5 s; test_refs.py is skipped without refs/)
.venv/bin/python main.py --demo --train-num 5 --max-topics 3 --savedir /tmp/res   # smoke test, synthetic data
.venv/bin/python main.py --mat refs/gazedata.mat --eval                           # real data (MATLAB defaults: 100 restarts, 5 topics), ~15 s
.venv/bin/python main.py --eval-matlab refs/results                               # evaluate the MATLAB results of the paper
```

- `tests/test_matlab_equivalence.py` contains line-by-line numpy re-implementations of `calc_lkh`, `calc_segments`, `plsa/estep/mstep`,
  `exact_prob` and `LoadGaze`; new ported code should get the same kind of test (reference = MATLAB semantics, not the Python code).
- MATLAB is not available in the dev environment. `refs/` (git-ignored, provided by the user) holds the MATLAB scripts and outputs
  used for the paper (`gazedata.mat`, `lkh_win_5.mat`, `segsessions.mat`, `task_att.mat`, `results/p_*_topicK.mat`, `results/cor.mat`, ...);
  `tests/test_refs.py` compares against them. Only one raw gaze session is in `refs/` (`for_plot_org_data/experiment`, = `sample_data`).
- Runtime: 111 sessions, 100 restarts, K = 1..5, spe + all: ~13 s in total on this machine.
- `python-ver/results/`, `aspectmodel/`, `.venv` are git-ignored; do not commit outputs.
- Line endings: all python-ver files are LF.

## 5. Done

- `calculate_session_likelihood` skipped the first window (`range(win_size, T)`); now `range(win_size - 1, T)`.
- `exact_prob` for 0/1 vectors lacked `n!`; now equal to `exact_prob.m` (`plsa.py` uses `multinomial_probs`, the vectorized version).
- All-zero AOF columns are removed before training (`remove_empty_columns`).
- `main.py` is the `main_mask.m` pipeline; old broken `main.py` (random sessions, wrong call) removed.
- E-step/M-step vectorized (identical iterations / log-likelihood / parameters to the loop version, ~50x faster).
- `tests/test_matlab_equivalence.py` added.
- T1: evaluation ported (`src/models/evaluation.py`; matching visits `perms(1:K)` order so ties are broken as in MATLAB;
  "no comparison" uses the exact `max == 1/n_task` test as MATLAB). Tests: `tests/test_evaluation.py` (synthetic, MATLAB references) and `tests/test_refs.py`.
- `main.learn_aspects` crashed on real data when saving `segsessions.npz` (`np.array(list, dtype=object)` with equal first dimensions); fixed with `_object_array`.
- Verified on `refs/` (real data, 111 sessions): `lkh_ses` (max diff 1e-17), `seqmask` and `session_all` (exact), 4 sessions without AOF;
  the MATLAB `p_spe_topic3` gives paper Table 2 and 93/111 = 83.8 %; `cor.mat` (CalcSim) is reproduced.
  Python training (seed 0): spe K=3 log-likelihood -13249.52 (MATLAB -13248.47), Table 2 except one task-2 session (accuracy 82.9 %).
- `GazeDataReader.load_session_from_files` on the raw session of `refs/` equals `gazedata.mat` usr 2 / task 1 (`f`, `dur`).

## 6. Open tasks (suggested order: T2 (+ H1), then T3-T5)

### T7. (DONE) Python vs MATLAB training - investigated, no bug found; acceptance criterion D6
- Deterministic (`test_refs.py::test_em_from_matlab_p`): from MATLAB's `p_{spe,all}_topic3`, three E/M-steps of `FourLayerPLSA` equal a loop
  re-implementation of `plsa.m` on the real data (rel 1e-12); the first log-likelihood is >= MATLAB `L(end)` (`L(end)` belongs to the parameters
  before the last M-step); MATLAB's `L` satisfies the stopping rule `|inc / L_new| <= 1e-4` for all saved models.
- The saved MATLAB `p` is NOT converged: continuing EM from it (2000 iterations) raises L by ~1.7 (spe K=3) and changes the result
  (spe accuracy 93 -> 94 sessions, all 84 -> 83). Differences of 1-2 sessions are therefore within the effect of `lim = 1e-4` alone.
- Spread over 20 seeds (100 restarts each, K=3): spe 92-95 correct (paper 93), sessions moved vs Table 2: 0 (2 seeds), 1 (13), 2 (5);
  all 82-84 (MATLAB 84); cosine within +-0.003 of MATLAB.
- `test_refs.py::test_python_training_close_to_paper` uses the agreed criterion (D6) "<= 2 sessions moved vs Table 2, cosine within 0.01" (seed 0).

### H1. `LoadGaze.m` drops samples with only the left eye missing (decision: FIX in Python)
- `x(x(:,1)==-1, 2) = x(x(:,1)==-1, 1)` runs after column 1 was filled, so it only touches rows where both eyes are -1; a missing *left* eye
  (column 2) is averaged as (x_R - 1)/2 < 0 and the sample becomes invalid. `gaze_reader.gaze_to_items` reproduces this (24 samples in `sample_data`).
- Fix both directions; update `ml_load_gaze` / `test_gaze_reader_matches_loadgaze` to the fixed specification. The MATLAB-compatible result
  equals `gazedata.mat` (usr 2 / task 1); after the fix the reader no longer matches `gazedata.mat` (the `--mat` pipeline is unaffected).
  Its effect on the paper results cannot be measured (only one raw session). Column order right/left is from the docstrings only.
- Also: targets outside 1..12 (x or y exactly 1.0) are silently mapped to KB value 0 (MATLAB errors); mark them invalid instead. Not present in `refs/`.

### Notes on `refs/`
- `cor.mat` was computed before `p_spe_topic5` / `p_all_topic4` were reordered by `calc_matching.m` (same values, permuted columns).
- `calc_matching.m` is not idempotent for K >= 5 (ties among the unmatched aspects are resolved in `perms` order).
- `segsessions.mat` has thr = 0.03, win = 5; session 1 contains every value, so the `N_value` of `calc_segments.m:8` (session 1 only) is harmless here.

### T1. (DONE, see section 5) Port the evaluation - original notes kept for reference
Where: new module (e.g. `src/models/evaluation.py`) + an option/script that runs it on `main.learn_aspects` results.
Reference: `calc_matching.m`, `CalcEval.m`, `CalcSim.m`, `plot_conf_matrix.m`, `plot_sim.m`, `plot_eval_AoF.m`.
- `giventask`: 3x15 binary matrix from section 3 (or load `task_att.mat` if the user provides it).
- Matching: for topic K = 2..5 (and `spe`/`all`) choose the permutation of aspects maximizing `sum(diag(cosine(giventask_normalized, vz')))` over the first
  min(3, K) aspects (`perms` in MATLAB; `scipy.optimize.linear_sum_assignment` gives the same optimum) and permute both `vz` and `zs`.
  NOTE: in the shipped `calc_matching.m` these two blocks are inside `%{ ... %}` and only the `p_part*` block (`learn_aspects_from_part.m`, 10 parts) is active;
  `CalcEval.m` expects matched `p_spe_topicK` / `p_all_topicK`, so implement the (commented) `spe`/`all` blocks.
- Similarity (`CalcEval`): `cor_task = cosine(normalize(giventask,2) rows, vz columns)` (3 x K); `cortop(s) = cor_task(task(s), argmax_r zs(s,r))`.
- Confusion matrix (`plot_conf_matrix`, topic = 3): sessions of `task` are rows `(task-1)*n_usr+1 : task*n_usr` of `zs`;
  `est = argmax zs`; if `max zs == 1/n_task` (uniform -> session without AOF) the estimate is "no comparison" (use `np.isclose`).
  Result: `n_task x (n_task+1)` counts, accuracy = trace / total (paper: 93/111 = 83.8 %).
- Acceptance: unit tests with synthetic data where the true permutation is known; on real data the accuracy and cosines should be in the range of the paper
  (random restarts, so not exact). Needs the 111-session `gazedata.mat` (T6) or a user-provided result.

### T2. `GazeDataReader.load_all_sessions` does not follow `ReadwithTimestamp.m` / `LoadGaze.m`
Where: `src/data/gaze_reader.py`.
- `load_all_sessions` -> `process_session` uses the raw ~60 Hz grid samples as `f` (no 150 ms filter, no merge of identical consecutive targets),
  takes the task from `task_id` instead of `floor(cont(1)/100)`, and `dur` is not the filtered duration. Should call `_process_session_data` (correct).
- `_load_available_data_files` processes only the first csv/list pair it finds (they are not even paired by name) and silently falls back to
  fake sessions (`_create_sample_sessions`); replace by an explicit directory + `timestamp.list` loop:
  `session_id = mod(count,4)+1`, new user when `session_id == 1`, skip `session_id == 4`, `{usr, task}` keys.
- `save_gazedata` stores sessions in dict order; MATLAB order is column-major (`main.from_reader_sessions`).
- Not ported: `Set_selected.m` (hard-coded `session{usr,task}.selected = list(idx)` table for 37 users x 3 tasks).
- Acceptance: for `sample_data/170116024150092.*` the result equals `load_session_from_files` (test exists for that path); add a multi-session test with a tiny fake `timestamp.list`.

### T3. Old training API differs from MATLAB
Where: `src/models/plsa.py` `train_multiple_runs`, `train_standard_plsa`; `FourLayerPLSA.fit`.
- Defaults `train_lim=1e-6`, `train_num=10`, `max_itr=100` (MATLAB `1e-4`, `100`, `1000`; `main.py` already uses MATLAB values).
- `fit` prints every run (and a warning when no observation model is set); `train_multiple_runs` silently skips runs with empty history (MATLAB errors).
- `train_standard_plsa` passes `segments=None` positionally and is unused. Consider making `main.train` the single implementation.

### T4. Small differences from MATLAB
- `data_loader.GazeDataLoader.calculate_attribute_probabilities` uses the value range present in the session (`min..max`); MATLAB uses `1..N_value` over all sessions
  (`main.calc_lkh` already does the latter). Align or remove.
- `SegmentCalculator.calc_segments`: `N_value` from all sessions (MATLAB: session 1) and fallback `[3,4,3,5,4]` / 5 attributes when empty.
- `SessionProcessor.create_segsessions` / `_clean_segment_labels` are not used (MATLAB saves `segsession` but trains on `seqmask`); `create_segsessions` has a
  fallback vocabulary of 57 and masks only the first `min_rows` rows.
- AOF rule: code/MATLAB use `count/l >= phi_k`, the paper writes `>` (see decision D1).
- `exact_prob` renormalizes `prob` if it does not sum to 1; `exact_prob.m` computes `prob_vec = normalize(prob)` and then overwrites it (no normalization). Irrelevant for valid input.
- `plsa.py` still imports `exact_prob` (unused).

### T5. Stale scripts and README
- Confirmed README errors (Quick Start): `load_session_from_files(gaze_csv_path=..., list_file_path=...)` (actual: `csv_file`, `list_file`);
  `reader.knowledge_data.values` does not exist (`reader.knowledge_base`, a 12x5 int array = KB, not a P(o|v) matrix);
  the example trains on `semantic_attribute_seq_onehot` (that is the "all" setting, not AOF masks); "Data Format" describes `knowledge.csv` as a
  "Feature->Observation mapping" and a `.list` as `[101, 102, 103, 201, ...]` (all ids of a session share the task digit).
  "Numerical results match MATLAB" is only true for the parts in the map above. The "4-layer" wording: `P(o|v)` is an identity mapping and does not change any computation.
- `main_four_layer.py`, `test_four_layer.py`, `test_comprehensive_visualization.py` use random/mock data; `test_with_arguments.py` feeds the raw 5xT attribute
  sequence (not an AOF mask) to `fit`; `test_actual_data.py` / `test_likelihood_convergence.py` depend on `load_all_sessions` (T2). Fix or delete; they do not test MATLAB equivalence.
- `gaze_reader.main_read_gaze_data` uses `GazeDataReader("../../..")` as base path.

### T6. (mostly DONE with `refs/`, see section 5 and T7) End-to-end verification against real MATLAB output
- Only synthetic data (multi-session) and one real session (`sample_data/`, gaze -> AOF mask looks right: CPU-high and memory-large rows lit for task 1) were checked.
- Needs from the user: `gazedata.mat` (111 sessions) and, ideally, MATLAB's `lkh_win_5.mat`, `segsessions.mat`, `results/p_*_topicK.mat`.
  Compare `lkh_ses`, `seqmask` exactly; compare learned aspects statistically (random init) via T1 metrics.
- `main.load_gazedata_mat` has never been run on a real `.mat`: it assumes `scipy.io.loadmat` returns object arrays of shape (37, 3) with 2-D numeric cells.
- Other MATLAB scripts not ported: `confirm_bias.m`, `learn_aspects_from_part.m`, `for_rev2.m`, `test_cor_p.m`.

## 7. Decisions needed from the user

- D1. AOF threshold `>=` vs `>`: irrelevant for this data (phi_k = 1/3 for every value, count/5 never equals 1/3); kept `>=`.
- D2. Resolved: `refs/task_att.mat`.
- D3. Add a `.gitattributes` (`python-ver/** text eol=lf`) to keep LF on Windows?
- D4. Stale scripts in T5: fix or delete?
- D5. Resolved: `refs/` (git-ignored, not to be committed).
- D6. Resolved: Python training is accepted when <= 2 sessions move vs Table 2 and cosine is within 0.01 (seed 0; `test_python_training_close_to_paper`).
