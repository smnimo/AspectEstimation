# AspectEstimation - Python Version

Python ports of the MATLAB code in `../matlab-ver`:

| Directory | Paper | MATLAB |
|---|---|---|
| [withMSET](withMSET/README.md) | Shimonishi & Kawashima (2020), "A Two-step Approach for Interest Estimation from Gaze Behavior in Digital Catalog Browsing", *Journal of Eye Movement Research* 13(1):4 | `matlab-ver/withLD`, with the detection step (likelihood-based short-term analysis) replaced by MSET (Shimonishi & Kawashima, AAAI 2020); extended to other catalogs (items, categorical / ordinal attributes from `knowledge.csv`) |
| [withLayout](withLayout/README.md) | Shimonishi, Ishikawa, Yonetani, Kawashima & Matsuyama (2014), "Learning Aspects of Interest from Gaze", *Transactions of the Human Interface Society* 16(2) | `matlab-ver/withLayout` |
| `common` | code shared by both: `normalize` (`math_utils.py`), random restarts of the EM algorithms (`restarts.py`), figures (`plotting.py`: circle plots of aspects, grouped bar charts, log-likelihood over EM iterations) | `normalize.m`, `train.m` |

## Installation

```bash
cd python-ver
uv venv && uv pip install -r requirements.txt      # or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Each port is run from its own directory (`cd withMSET` or `cd withLayout`, then `../.venv/bin/python main.py ...`);
see their README files.
