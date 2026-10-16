# TODO (python-ver vs matlab-ver)

Status of the comparison with `matlab-ver` / the paper (Shimonishi & Kawashima, JEMR 2020).

## Done

- Window loop of `LikelihoodCalculator.calculate_session_likelihood` started one step late (first window skipped); now `range(win_size - 1, T)` like `t = win_size:T`.
- `exact_prob` for 0/1 vectors lacked `n!`; now identical to `exact_prob.m`.
- Time points without AOF are removed before training (`SessionProcessor.remove_empty_columns`).
- `main.py` rewritten as the pipeline of `main_mask.m` (likelihood -> AOF mask -> pLSA `spe`/`all`, MATLAB default parameters, parallel restarts).
- E-step / M-step of `FourLayerPLSA` vectorized (same results as the loop version, ~50x faster).
- Python files in `python-ver/` converted to LF.

## Open

### 1. Matching aspects with tasks / evaluation (not ported)
- `calc_matching.m`: permutation of learned aspects that maximizes the cosine similarity to `giventask` (`task_att`).
- `CalcEval.m` / `CalcSim.m`: cosine similarity between task-related attribute values and aspects (paper Eq. 4, Fig. 8), task estimation by `argmax_r theta(s)` (paper Table 2, 83.8 %), for `spe` and `all`, topics 1..5.
- `plot_conf_matrix.m` input (confusion matrix) depends on the above.
- `task_att.mat` / `giventask` has to be provided or re-created from paper Table 1.

### 2. `GazeDataReader.load_all_sessions` does not follow `LoadGaze.m`
- `load_all_sessions` -> `_load_available_data_files` / `process_session` uses raw ~60 Hz grid samples as `f`: no 150 ms duration filter, no merging of consecutive identical targets. The window length 5 then means ~83 ms instead of 5 item transitions.
- `process_session` uses `task_id` for `list - task*100` instead of `floor(cont(1)/100)`, and `dur` is not the filtered duration.
- `load_session_from_files` (-> `_process_session_data`) is correct (checked on `sample_data`: identical to `LoadGaze.m`); `load_all_sessions` should call it.
- Not ported: `Set_selected.m` (`session{usr,task}.selected`), `timestamp.list` handling is only partially checked.
- `GazeDataReader.save_gazedata` stores sessions in dict order, not in MATLAB's column-major `cell(:)` order; `main.from_reader_sessions` sorts by (task, usr).

### 3. Training parameters of the old API differ from MATLAB
- `train_multiple_runs` defaults: `train_lim=1e-6`, `train_num=10`, `max_itr=100` (MATLAB: `1e-4`, `100`, `1000`). `main.py` already uses the MATLAB values.
- `FourLayerPLSA.fit` / `train_multiple_runs` print every run; `train_multiple_runs` silently skips runs whose history is empty (MATLAB would error).
- `train_standard_plsa` passes `segments=None` positionally and is unused.

### 4. Small differences from MATLAB
- `LikelihoodCalculator` / `GazeDataLoader.calculate_attribute_probabilities` use the value range present in the session (`min..max`); MATLAB uses `1..N_value` over all sessions (`main.calc_lkh` already follows MATLAB). The loader method should be aligned.
- `SegmentCalculator.calc_segments` takes `N_value` from the sequences of all sessions (MATLAB: session 1 only) and has hard-coded fallbacks (`[3,4,3,5,4]`).
- `SessionProcessor.create_session_all` offsets (+3, +6, ...) assume 3 values per attribute (same as MATLAB); `main.py` raises otherwise.
- `SessionProcessor.create_segsessions` / `_clean_segment_labels` are not used by the pipeline (MATLAB saves `segsession` but trains on `seqmask`); `create_segsessions` has a fallback vocabulary size of 57 and masks only the first `min_rows` rows.
- AOF rule is `x/l >= phi` in the code (MATLAB), `>` in the paper (Sec. "AOF detection").
- `exact_prob` normalizes `prob` when it does not sum to 1; `exact_prob.m` discards the normalization. Irrelevant for valid input.
- `plsa.py` still imports `exact_prob` (unused after vectorization).

### 5. Stale scripts / documentation
- `main_four_layer.py`, `test_four_layer.py`, `test_comprehensive_visualization.py` use random/mock data, identity observation model, and (in `test_with_arguments.py`) feed the raw 5xT attribute sequence to `fit` instead of AOF masks. They do not exercise the MATLAB pipeline.
- `test_actual_data.py` / `test_likelihood_convergence.py` rely on `load_all_sessions` (see 2).
- `README.md`: "Numerical results match MATLAB" is only true for the parts listed under Done; Quick Start uses argument names (`gaze_csv_path`, `list_file_path`) and attributes (`reader.knowledge_data`) that need to be checked against the code; the "4-layer" description is an identity mapping `P(o|v)` and does not change the computation.
- `gaze_reader.main_read_gaze_data` uses `GazeDataReader("../../..")` as base path.

### 6. Verification still missing
- No end-to-end comparison with MATLAB output: run `main_mask.m` and `main.py --mat gazedata.mat` on the same `gazedata.mat` (all 111 sessions) and compare `lkh_ses`, `seqmask`, and the learned `p.vz` (same initialization needed, or compare log-likelihood / aspect similarity).
- `sample_data` contains one session only; the multi-session training has only been tested on synthetic data.
- Other MATLAB scripts not ported: `confirm_bias.m`, `learn_aspects_from_part.m`, `for_rev2.m`, `test_cor_p.m`, `plot_*` variants for the paper figures.
