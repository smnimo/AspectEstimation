# AspectEstimation with Layout Effect - MATLAB Version

MATLAB code of

> Shimonishi, Ishikawa, Yonetani, Kawashima & Matsuyama (2014), "Learning Aspects of Interest from Gaze"
> (視線運動解析による興味アスペクトの推定), *Transactions of the Human Interface Society* 16(2), 103-114.

A probabilistic generative model of gaze targets: in each session the user has interests theta(s) over aspects; at each
time an aspect, then an attribute value, then a gazed region is drawn. The region probability mixes the items having the
attribute value with the layout effect of the content (paper Eq. 4):
P(R_n | V_q) = (1 - alpha) P(I_n | V_q) + alpha P(R_n; phi).

## Requirements

MATLAB with the Statistics and Machine Learning Toolbox; `train.m` uses `parfor` (Parallel Computing Toolbox; runs
serially without it).

## Data layout

The scripts are run from `matlab-ver/withLayout/` and read (the data are not included in the repository):

```
data/experiment/timestamp.list   one session name per line (81 sessions: 9 participants x 3 tasks x 3 contents)
data/mat/<name>.mat              f: gazed regions (1..12, 3 x 4 grid, row-major), one per gaze-target transition;
                                 list.index / list.num: content group and item number of the 12 displayed items;
                                 answer: the 3 selected items; content_id, task_id
data/content.mat                 content.category (5), content.calorie (4 levels), content.step (5), content.colhist
                                 (16 color classes) of the 4 x 15 items (made by data/makecontent.m)
```

Attribute types (`attr`): 1 = semantic attributes (SA: category, calorie, steps; 14 values), 2 = appearance attributes
(AA: color class; 16 values), 3 = both (SA+AA).

## Scripts

| Script | Content |
|---|---|
| `make_area.m` | layout effect P(R; phi) for the leave-one-out evaluation: gaze frequency of all sessions but one (paper Eq. 11) -> `area_s.mat` |
| `main.m` | leave-one-out evaluation (paper 4.2-4.4, Fig. 5-7): learn the aspects from 80 sessions, estimate the interests from the first 10 %, ..., 90 % of the left-out session, predict its gazed regions (Eq. 8) and selected items (Eq. 9) -> `result/{region,item}/MP_attr<attr>_alpha<100 alpha>.mat` (`mpatk{aspects, ratio}`: precision P@1..3) |
| `main_comparison.m` | comparison method (nearest-neighbor recommendation, paper 4.3) -> `result/{region,item}/MP_comp_type<attr>.mat` |
| `main_learning_aspects.m` | aspects and interests from all sessions (paper 4.4.3, Fig. 8, 9) -> `result/P_attr<attr>_alpha<100 alpha>.mat` (`vz{aspects}` = P(V\|Z), `zs{aspects}` = P(Z\|s)) |
| `train.m`, `plsa.m`, `estep.m`, `mstep.m`, `@prob` | EM learning (best of 100 random restarts; paper 3.3.1) |
| `est.m` | interests of a new session for learned aspects (paper 3.3.2) |
| `eval_precision.m` | precision of the predicted regions / items |
| `item_attr.m`, `reg_attr.m` | P(I\|V) (paper Eq. 10) and P(R\|V) (Eq. 4) |
| `colseq.m`, `@itemlist`, `normalize.m` | session loading, item list, normalization |
| `makeattr.m`, `est_cosine.m`, `bhattacharyya.m`, `eval_precision_comparison.m` | comparison method |

Run `make_area` first (`main.m` reads `area_s.mat`), then `main`, `main_comparison` and `main_learning_aspects`.
Parameters: 100 random restarts, EM thresholds 1e-2 (learning) and 1e-3 (estimation), 3, 6, 9, 12 aspects,
alpha 0 and 0.5, P@k with k = 1..3.

## Notes on the implementation

- The figures of the paper show the precision of the first prediction (P@1, `mpatk{aspects, ratio}(1)`).
- The comparison method of the code is not the one described in the paper: the code averages the 3 training sessions
  nearest in Bhattacharyya distance (weights 1 / distance) and ranks the items by Bhattacharyya distance, while the
  paper describes the single nearest session in cosine distance and items ranked by cosine distance.
- `est.m` smooths the aspects (`normalize(vz + 0.01, 2)`) before estimating the interests; this is not described in
  the paper.
- `main.m` estimates the interests from `f(1 : floor(x))` and evaluates on `f(ceil(x) : T)` (x = ratio * T / 10), so one
  time point is in both parts when x is an integer.
- The EM loops of `plsa.m` / `est.m` stop when the increase of the log-likelihood is between 0 and the threshold; they
  continue while the log-likelihood decreases.
