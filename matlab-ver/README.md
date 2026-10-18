# AspectEstimation - MATLAB Version

MATLAB code of two papers on estimating aspects of interest from gaze behavior in catalog browsing:

| Directory | Paper | Model |
|---|---|---|
| [withMSET](withMSET/README.md) | Shimonishi & Kawashima (2020), "A Two-step Approach for Interest Estimation from Gaze Behavior in Digital Catalog Browsing", *Journal of Eye Movement Research* 13(1):4 | attributes-of-focus detected by a likelihood-based short-term analysis ("MSET"), then pLSA on them |
| [withLayout](withLayout/README.md) | Shimonishi, Ishikawa, Yonetani, Kawashima & Matsuyama (2014), "Learning Aspects of Interest from Gaze", *Transactions of the Human Interface Society* 16(2) | pLSA-based generative model of gazed regions with the layout effect of the content |

## Requirements

MATLAB with the Statistics and Machine Learning Toolbox; `train.m` uses `parfor` (Parallel Computing Toolbox; runs
serially without it). `withMSET/plot_aspect.R` needs R with `corrplot` (optional).

## Usage

Each version is run from its own directory; the experiment data are not included in the repository (see the README files
for the expected data layout).

- `withMSET`: `main_mask.m` (likelihoods -> attributes-of-focus -> aspects -> matching -> similarity -> confusion matrix),
  starting from `gazedata.mat` made by `ReadwithTimestamp.m`.
- `withLayout`: `make_area.m`, then `main.m` (leave-one-out evaluation), `main_comparison.m` (comparison method) and
  `main_learning_aspects.m` (aspects from all sessions).
