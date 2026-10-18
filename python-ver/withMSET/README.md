# AspectEstimation - Python Version

The two-step approach of

> Shimonishi & Kawashima (2020), "A Two-step Approach for Interest Estimation from Gaze Behavior in Digital
> Catalog Browsing", *Journal of Eye Movement Research* 13(1):4.

using MSET for the detection of attributes-of-focus (step 1):

> Shimonishi & Kawashima (2020), "MSET: Multi-Scale Exact Test for Detecting Distinctive Periods in Choice Behavior",
> *AAAI 2020*.

It is a Python port of `matlab-ver/withLD` (the code of the JEMR paper) with step 1 replaced.

The MATLAB code is the reference for step 2 (pLSA), the evaluation and the reading of the gaze data, which reproduce
the MATLAB outputs used for the paper (see [Verification](#verification)). Two things differ from the MATLAB code:

- **Step 1 uses MSET** instead of the likelihood-based short-term analysis of the JEMR paper and `matlab-ver/withLD`
  (likelihood of windows of 5 items < 0.03, `calc_lkh.m` + `calc_segments.m`), which is not ported.
  See [Step 1: MSET](#step-1-mset).
- **Other catalogs**: the number of items, the attributes and their kinds (categorical, possibly multi-valued, or
  ordinal) are read from `knowledge.csv` (see [Other catalogs and attributes](#other-catalogs-and-attributes)).

## Pipeline

| Step | MATLAB | Python |
|---|---|---|
| Item attributes (`knowledge.csv`) | `csvread` | `src/data/knowledge.py` (`load_knowledge`, `Knowledge`) |
| Raw gaze -> browsed items -> attribute sequences (`gazedata.mat`) | `ReadwithTimestamp.m`, `LoadGaze.m` | `src/data/gaze_reader.py` (`GazeDataReader`) |
| Step 1: attributes-of-focus (AOF) masks | `calc_lkh.m`, `calc_segments.m` (likelihood-based; replaced) | `src/models/aof.py` (MSET), `main.detect_aof` |
| Step 2: pLSA on AOF masks ("spe") and on all values ("all") | `learn_aspects_from_mask.m`, `train.m`, `plsa.m` | `main.learn_aspects`, `main.train`, `src/models/plsa.py` |
| Evaluation: aspect-task matching, cosine similarity, confusion matrix | `calc_matching.m`, `CalcEval.m`, `plot_conf_matrix.m` | `src/models/evaluation.py`, `main.py --eval` |
| Evaluation of the detected AOF (All / Top / Comparison / LD) | `plot_eval_AoF.m` | `evaluation.aof_similarities`, `aof_ttests` |
| Figures | `plot_*.m`, `plot_mask.m`, `plot_aspect.R` | `src/visualization/plotter.py`, `main.py --plot` |

Setting of the paper: 12 items (laptops, 3 x 4 grid) with 5 attributes (price, screen size, CPU score, memory capacity, weight)
of 3 values each, 3 tasks, 37 participants (111 sessions). Features are 15-dimensional (index = attribute * 3 + value - 1).

## Installation

```bash
cd python-ver
uv venv && uv pip install -r requirements.txt      # or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd withMSET
```

The commands below are run from `python-ver/withMSET`. Code shared with `withLayout` (normalization, random restarts,
circle plots, grouped bar charts, log-likelihood plots) is in `python-ver/common`.

## Usage

```bash
# raw gaze data -> gazedata.mat (ReadwithTimestamp.m)
../.venv/bin/python src/data/gaze_reader.py --timestamp-list ../experiment/timestamp.list \
    --knowledge ../knowledge/knowledge.csv --out gazedata.mat

# full pipeline (main_mask.m): step 1, step 2 (100 restarts, 1..5 aspects) and evaluation; ~15 s
../.venv/bin/python main.py --mat gazedata.mat --eval --savedir results

# + figures (results/figures/*.png)
../.venv/bin/python main.py --mat gazedata.mat --plot --savedir results

# evaluate MATLAB results (p_{spe,all}_topicK.mat), optionally with figures
../.venv/bin/python main.py --eval-matlab ../../matlab-ver/withLD/refs/results --plot

# synthetic data (smoke test)
../.venv/bin/python main.py --demo --train-num 5 --max-topics 3 --eval

# another catalog: raw gaze data, typed knowledge.csv, grid and task-related values (see below)
../.venv/bin/python main.py --raw ../experiment/timestamp.list --knowledge examples/knowledge_typed.csv \
    --grid 3 4 --task-att examples/task_typed.csv --plot --savedir results_typed
```

`main.py --help` lists the parameters (`--alpha 0.05`, `--scales 1 10`, `lim` 1e-4, `--task-att task_att.mat`, ...; the step 2 and
evaluation defaults are the MATLAB values).
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

- `knowledge.csv`: item attributes. The paper's file has no header: KB, 12 items x 5 attributes, values 1..3
  (columns: price, screen size, CPU score, memory capacity, weight). For the typed format, see below.
- `<session>.list`: the item ids of a session in grid order (row-major, rows x columns of `--grid`, 3 x 4 in the paper);
  the hundreds digit is the task (e.g. `101 102 103 109 ...` = task 1), `id - task * 100` is the row of `knowledge.csv`.
- `<session>.csv`: `timestamp (us), right_x, right_y, left_x, left_y`, normalized to 0..1, `-1` = missing.
- `timestamp.list`: one session name per line, 4 lines per participant (the 4th is skipped, as in `ReadwithTimestamp.m`).
- `gazedata.mat`: `session{usr, task}` (`f`: browsed grid positions, `dur`, `list`, `task`, `usr`, `session_id`,
  `totalduration`) and, when every item has one value per attribute, `semantic_attributes{usr, task}` (items x attributes),
  `semantic_attribute_seq{usr, task}` (attributes x T) and `KB`.

## Step 1: MSET

Every window of l consecutive browsed items, for l = 1..10 (`--scales MIN MAX`), is tested against neutral browsing
(items drawn uniformly from the displayed catalog) by an exact test (AAAI 2020 Eq. 2): the p-value is the probability
under neutral browsing of the outcomes as likely as the observed one or less likely. A window with p-value < 0.05
(`--alpha`) is distinctive; the AOF of all distinctive windows of all scales are united at each time point.

| Kind | Distribution under neutral browsing | AOF of a distinctive window |
|---|---|---|
| categorical | multinomial over the values of the displayed items (with several values per item: over their combinations) | values whose share in the window exceeds their share in the catalog |
| ordinal | for each cut between levels, binomial for the number of items above the cut (exact binomial test per cut) | levels on the side of the cut the window is biased to; united over the distinctive cuts |

A headerless `knowledge.csv` and `--mat` without `--knowledge` treat every attribute as categorical. The p-values agree
with `MSET/scripts/mset.py` of the authors (`MSET_cat`) to 1e-15. A p-value equal to alpha (up to rounding) is not
significant.

On the paper's data (`refs/gazedata.mat`, 100 restarts, seed 0), MSET with the defaults finds no AOF in 6 sessions
(likelihood-based: 4) and gives a task estimation accuracy of 77.5 % with 3 aspects (likelihood-based, paper Table 2: 83.8 %).

## Other catalogs and attributes

The paper's setting is fixed in the MATLAB code (12 items, 5 attributes x 3 values). The Python version takes it from the inputs:

- **Catalog and attributes**: `knowledge.csv` with two header rows (attribute names, then `ordinal` or `categorical`),
  then one row per item. An ordinal cell is one level 1..N; a categorical cell holds one or more values separated by `;`
  (integers 1..N or labels). Example ([examples/knowledge_typed.csv](examples/knowledge_typed.csv)):

  ```
  price,screen,CPU,memory,weight,color,use
  ordinal,ordinal,ordinal,ordinal,ordinal,categorical,categorical
  2,3,3,3,2,silver,business
  2,2,3,3,3,black,gaming;business
  ...
  ```

  A file without header rows is read as in the paper (every attribute single-valued categorical).
- **Screen grid**: `--grid ROWS COLS` (default 3 4); the `.list` files must have ROWS x COLS items.
- **Task-related values**: `--task-att` with a CSV of one row per task and one 0/1 column per feature, in the order
  attribute by attribute, value by value (an optional first row with labels such as `price 1` is skipped;
  [examples/task_typed.csv](examples/task_typed.csv)). The built-in values are used only for the paper's features.
  The evaluation expects every participant to have one session of each task.
- **Input**: `--raw timestamp.list --knowledge knowledge.csv` reads the raw gaze data; `--mat gazedata.mat --knowledge
  knowledge.csv` uses the session structs of a `gazedata.mat` (made by `ReadwithTimestamp.m` or `gaze_reader.py`).

Step 1 treats each attribute according to its kind ([Step 1: MSET](#step-1-mset)). The figures label the features with the attribute names and values of `knowledge.csv`.

## Verification

```bash
../.venv/bin/python -m pytest tests -q
```

- `tests/test_matlab_equivalence.py`, `tests/test_evaluation.py`: comparisons with line-by-line re-implementations of the
  MATLAB code (pLSA, `exact_prob.m`, `LoadGaze.m`, evaluation).
- `tests/test_refs.py`: comparisons with the MATLAB outputs used for the paper in `matlab-ver/withLD/refs/` (git-ignored; skipped
  when absent). On the MATLAB AOF masks (`segsessions.mat`): the MATLAB results give the paper's Table 2 (accuracy 83.8 %),
  the evaluation and pLSA reproduce the MATLAB values, and Python training is accepted when at most 2 sessions differ from
  Table 2. `session_all` is identical; MSET on the paper's data is checked against regression values.
- `tests/test_knowledge.py`: step 1 (MSET) for categorical (multi-valued) and ordinal attributes against references
  written with `scipy.stats`, multi-scale union, false-positive rate under neutral browsing; `knowledge.csv` parsing;
  reader, `gazedata.mat` and the whole pipeline with the example files.

Python follows the corrected matlab-ver (`../../matlab-ver/withLD/README.md`, "Changes from the 2019 scripts"); compared with the 2019 scripts used for the paper: a missing left eye
is filled from the right eye, gaze coordinates equal to 1 belong to the last grid column / row, the AOF rule is `>` (paper),
sessions without AOF are excluded from `cortop`, aspect matching keeps an already matched order, invalid inputs raise errors.
None of these change the paper's numbers (Table 2, Fig. 10) on the refs data (with the MATLAB AOF masks).

## Figures

`src/visualization/plotter.py` has one function per MATLAB figure; `main.py --plot` saves them to `<savedir>/figures`:

| File | MATLAB | Content |
|---|---|---|
| `L_topic.png`, `aspects_{all,spe}.png` | `plot_L_topic.m` | final log-likelihood vs number of aspects; learned aspects and the given tasks |
| `conf_{spe,all}.png` | `plot_conf_matrix.m` | task estimation (3 aspects), last column = no comparison |
| `sim.png`, `sim_task_asp.png` | `plot_sim.m`, `plot_sim_task_asp.m` | cosine similarity between tasks and aspects, with / without MET |
| `aspect_circles_{tag}_topicK.png` | `plot_aspect.R` | aspects as circles (paper Fig. 9; values not scaled) |
| `eval_aof.png` (`--mat`) | `plot_eval_AoF.m` | similarity of the detected AOF (All / Top / Comparison / LD) to the task |
| `dur_usr.png` (`--mat`) | `plot_dur_usr.m` | session durations per participant |
| `aof_temporal.png` | (2019 `for_rev2.m`, removed) | total number of AOF in each temporal position (paper Fig. 8, corrected; the paper's values came from an indexing bug) |
| `mask_session{17,34,46}.png` | `plot_mask.m` | all values and AOF mask of one session (session 17 = paper Fig. 7) |
| `likelihood_convergence_{all,spe}.png` | - | log-likelihood over EM iterations |

`plotter.plot_org_gaze` (`plot_org_gaze.m`) draws a raw gaze trajectory over the catalog screenshot.
