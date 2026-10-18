# AspectEstimation with Layout Effect - Python Version

Python port of `matlab-ver/withLayout`, the code of

> Shimonishi, Ishikawa, Yonetani, Kawashima & Matsuyama (2014), "Learning Aspects of Interest from Gaze"
> (視線運動解析による興味アスペクトの推定), *Transactions of the Human Interface Society* 16(2), 103-114.

The MATLAB code is the reference; the deterministic parts reproduce it exactly (see [Verification](#verification)).

## Model and code

| Part | Paper | MATLAB | Python |
|---|---|---|---|
| P(I\|V), P(R\|V) = (1 - alpha) P(I\|V) + alpha P(R; phi) | Eq. 4, 10 | `item_attr.m`, `reg_attr.m` | `src/layout_data.py` (`item_given_value`, `region_given_value`) |
| layout effect P(R; phi): relative gaze frequency | Eq. 11 | `make_area.m` | `layout_data.region_frequency` |
| EM learning of P(V\|Z), P(Z\|s), best of random restarts | Eq. 5-7, 3.3.1 | `train.m`, `plsa.m`, `estep.m`, `mstep.m` | `src/layout_model.py` (`train`, `fit`, `em_step`) |
| interests of a new session | 3.3.2 | `est.m` | `layout_model.estimate` |
| prediction of gazed regions / items of interest, P@k | Eq. 8, 9 | `eval_precision.m` | `src/layout_eval.py` (`eval_precision`) |
| leave-one-out evaluation | 4.2 | `main.m` | `layout_eval.leave_one_out` |
| comparison method (nearest-neighbor recommendation) | 4.3 | `main_comparison.m` | `layout_eval.leave_one_out_comparison` |
| figures | Fig. 5-9 | - | `src/layout_plot.py` (+ log-likelihood over EM iterations) |

The EM step does not form the K x Q x T posterior array of `estep.m` (only its sums are needed), which makes the
learning about 3 times faster with identical results.

## Usage

```bash
cd python-ver/withLayout
../.venv/bin/python main.py learn --attr 1 --alpha 0.5 --aspects 3 12            # aspects from all sessions (Fig. 8, 9)
../.venv/bin/python main.py evaluate --attr 1 2 3 --alpha 0 0.5 --aspects 6 12   # leave-one-out prediction (Fig. 5, 7)
../.venv/bin/python main.py evaluate --attr 1 --aspects 9 --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1   # Fig. 6
../.venv/bin/python main.py compare --attr 1 2 3 --method paper matlab           # comparison method
../.venv/bin/python main.py plot                                                 # figures -> results/figures
../.venv/bin/python main.py plot --matlab-results ../../matlab-ver/withLayout/result --savedir results/matlab
```

Options: `--data` (default `../../matlab-ver/withLayout/data`), `--savedir` (`results`), `--train-num` (100),
`--train-lim` (1e-2), `--est-lim` (1e-3), `--jobs` (number of CPUs), `--seed`, `--disjoint` (see below).
Attribute types `--attr`: 1 = semantic attributes (SA: category, calorie, steps), 2 = appearance attributes (AA: color
class), 3 = both (SA+AA). Outputs: `learn_attr<a>_alpha<100 alpha>.npz` (`vz_K<k>`, `zs_K<k>`, `L_K<k>`),
`loo_attr<a>_alpha<100 alpha>.npz` (`mpatk_{region,item}_K<k>`: 9 ratios x P@1..3, and per session `patk_...`),
`comparison_<method>_attr<a>.npz`.

Data (not in git): `data/experiment/timestamp.list`, `data/mat/<name>.mat`, `data/content.mat`, see
`matlab-ver/withLayout/README.md`.

## Differences from the paper and options

- The figures of the paper show the precision of the first prediction (P@1); all P@1..3 are saved.
- `content.mat` does not name its columns; the category labels of Fig. 8 (Rice, Meat, Fish, Vegetable, Pasta) follow
  the order in Fig. 8 of the paper.
- Comparison method: `--method matlab` is the MATLAB code (3 nearest training sessions in Bhattacharyya distance);
  `--method paper` follows the paper's description (single nearest session in cosine distance, Eq. 12).
- As `est.m`, the aspects are smoothed (+ 0.01) before estimating the interests.
- As `main.m`, the estimation part `f[:floor(x)]` and the evaluation part `f[ceil(x) - 1:]` share one time point when
  x = ratio * T / 10 is an integer; `--disjoint` uses disjoint parts.
- As `plsa.m`, the EM loop continues while the log-likelihood decreases; `max_itr` (10000) only prevents an endless loop.
- The restarts are seeded (`--seed`, restart i uses seed + i); MATLAB's `parfor` restarts are not seeded, so learned
  aspects agree statistically, not bit for bit.

## Verification

```bash
../.venv/bin/python -m pytest tests -q
```

- `tests/test_layout_equivalence.py`: comparisons with line-by-line re-implementations of the MATLAB code on synthetic data.
- `tests/test_data.py` (needs the data; skipped without it): the layout effect equals `area_s.mat`, and the comparison
  method reproduces the MATLAB results `result/{region,item}/MP_comp_type<a>.mat` exactly.
- The MATLAB functions were also run in GNU Octave on the real data with the same initial values: P(I\|V), P(R\|V),
  three EM iterations, the estimated interests and the precision agree with the Python port (differences <= 1e-13).
- Leave-one-out evaluation (100 restarts; SA / AA / SA+AA, alpha 0 / 0.5, 6 / 12 aspects; about 50 min on 16 cores):
  the P@1 values differ from the MATLAB results in `result/` by 0.036 on average (max 0.12; one session = 0.012), as
  much as two Python runs with different seeds differ from each other (0.030, max 0.10), i.e. by random-restart variation.
  As in the paper, the model predicts better than the comparison method in every setting.
