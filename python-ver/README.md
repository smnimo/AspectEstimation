# AspectEstimation - Python Version

Python port of `matlab-ver`, the code of

> Shimonishi & Kawashima (2020), "A Two-step Approach for Interest Estimation from Gaze Behavior in Digital
> Catalog Browsing", *Journal of Eye Movement Research* 13(1):4.

The MATLAB code is the reference. Step 1, step 2 and the evaluation reproduce the MATLAB outputs used for the paper
(see [Verification](#verification)). Status, known differences and open tasks are in [TODO.md](TODO.md).

## Pipeline

| Step | MATLAB | Python |
|---|---|---|
| Raw gaze -> browsed items -> attribute sequences (`gazedata.mat`) | `ReadwithTimestamp.m`, `LoadGaze.m` | `src/data/gaze_reader.py` (`GazeDataReader`) |
| Step 1: window likelihoods | `calc_lkh.m` | `main.calc_lkh`, `src/data/data_loader.py` (`LikelihoodCalculator`) |
| Step 1: attributes-of-focus (AOF) masks | `calc_segments.m` | `src/models/segments.py` (`SegmentCalculator`) |
| Step 2: pLSA on AOF masks ("spe") and on all values ("all") | `learn_aspects_from_mask.m`, `train.m`, `plsa.m` | `main.learn_aspects`, `main.train`, `src/models/plsa.py` |
| Evaluation: aspect-task matching, cosine similarity, confusion matrix | `calc_matching.m`, `CalcEval.m`, `plot_conf_matrix.m` | `src/models/evaluation.py`, `main.py --eval` |
| Evaluation of the detected AOF (All / Top / Comparison / LD) | `plot_eval_AoF.m` | `evaluation.aof_similarities`, `aof_ttests` |
| Figures | `plot_*.m`, `disp_mask_for_paper.m`, `plot_aspect.R` | `src/visualization/plotter.py`, `main.py --plot` |

Setting: 12 items (laptops, 3 x 4 grid) with 5 attributes (price, screen size, CPU score, memory capacity, weight) of 3 values each,
3 tasks, 37 participants (111 sessions). Features are 15-dimensional (index = attribute * 3 + value - 1).

## Installation

```bash
cd python-ver
uv venv && uv pip install -r requirements.txt      # or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# raw gaze data -> gazedata.mat (ReadwithTimestamp.m)
.venv/bin/python src/data/gaze_reader.py --timestamp-list ../experiment/timestamp.list \
    --knowledge ../knowledge/knowledge.csv --out gazedata.mat

# full pipeline (main_mask.m): step 1, step 2 (100 restarts, 1..5 aspects) and evaluation; ~15 s
.venv/bin/python main.py --mat gazedata.mat --eval --savedir results

# + figures (results/figures/*.png)
.venv/bin/python main.py --mat gazedata.mat --plot --savedir results

# evaluate MATLAB results (p_{spe,all}_topicK.mat), optionally with figures
.venv/bin/python main.py --eval-matlab refs/results --plot

# synthetic data (smoke test)
.venv/bin/python main.py --demo --train-num 5 --max-topics 3 --eval
```

`main.py --help` lists the parameters (window size 5, threshold 0.03, `lim` 1e-4, `--task-att task_att.mat`, ...; defaults are the MATLAB values).
Outputs in `--savedir`: `segsessions.npz`, `p_{spe,all}_topicK.npz` (`zs`, `vz`, `L`), `Akq_{spe,all}_topicK.csv` and, with `--eval`,
`evaluation.npz` (matched aspects, cosine similarities, confusion matrix and, with `--mat`, the AOF detection similarities
of `plot_eval_AoF.m`); with `--plot`, `figures/*.png`. Training outputs are not overwritten by the matching
(unlike `calc_matching.m`).

From Python:

```python
import main
attrs, seqs = main.load_gazedata_mat('gazedata.mat')      # MATLAB cell(:) order: by task, then by user
results = main.learn_aspects(attrs, seqs, savedir='results')
```

## Data format

- `knowledge.csv`: KB, 12 items x 5 attributes, values 1..3 (columns: price, screen size, CPU score, memory capacity, weight).
- `<session>.list`: the 12 item ids of a session in grid order (row-major 3 x 4); the hundreds digit is the task
  (e.g. `101 102 103 109 ...` = task 1), `id - task * 100` indexes KB.
- `<session>.csv`: `timestamp (us), right_x, right_y, left_x, left_y`, normalized to 0..1, `-1` = missing.
- `timestamp.list`: one session name per line, 4 lines per participant (the 4th is skipped, as in `ReadwithTimestamp.m`).
- `gazedata.mat`: `semantic_attributes{usr, task}` (12 x 5), `semantic_attribute_seq{usr, task}` (5 x T), `KB`.

## Verification

```bash
.venv/bin/python -m pytest tests -q
```

- `tests/test_matlab_equivalence.py`, `tests/test_evaluation.py`: comparisons with line-by-line re-implementations of the MATLAB code.
- `tests/test_refs.py`: comparisons with the MATLAB outputs used for the paper in `refs/` (git-ignored; skipped when absent):
  window likelihoods, AOF masks and `session_all` are identical; the MATLAB results give the paper's Table 2 (accuracy 83.8 %);
  Python training is accepted when at most 2 sessions differ from Table 2.

Intentional differences from MATLAB (details in TODO.md): a missing left eye is filled from the right eye (`LoadGaze.m` drops such samples),
gaze coordinates equal to 1 belong to the last grid column / row, invalid inputs raise errors instead of producing wrong values.

## Figures

`src/visualization/plotter.py` has one function per MATLAB figure; `main.py --plot` saves them to `<savedir>/figures`:

| File | MATLAB | Content |
|---|---|---|
| `L_topic.png`, `aspects_{all,spe}.png` | `plot_L_topic.m` | final log-likelihood vs number of aspects; learned aspects and the given tasks |
| `conf_{spe,all}.png` | `plot_conf_matrix.m` | task estimation (3 aspects), last column = no comparison |
| `sim.png`, `sim_task_asp.png` | `plot_sim.m`, `plot_sim_task_asp.m` | cosine similarity between tasks and aspects, with / without MET |
| `aspect_circles_{tag}_topicK[_x2].png` | `plot_aspect.R` | aspects as circles (`_x2`: values x 2, as `refs/results/*_2.pdf`) |
| `eval_aof.png` (`--mat`) | `plot_eval_AoF.m` | similarity of the detected AOF (All / Top / Comparison / LD) to the task |
| `dur_usr.png` (`--mat`) | `plot_dur_usr.m` | session durations per participant |
| `mask_session{17,34,46}.png` | `plot_mask_for_paper.m`, `disp_mask_for_paper.m` | all values and AOF mask of one session |
| `likelihood_convergence_{all,spe}.png` | - | log-likelihood over EM iterations |

`plotter.plot_org_gaze` (`plot_org_gaze.m`) draws a raw gaze trajectory over the catalog screenshot.
