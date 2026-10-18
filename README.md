# AspectEstimation

Code for estimating aspects of interest (combinations of item attributes that reflect a user's interest) from gaze
behavior during catalog browsing:

- Shimonishi & Kawashima (2020), "A Two-step Approach for Interest Estimation from Gaze Behavior in Digital Catalog
  Browsing", *Journal of Eye Movement Research* 13(1):4. doi: 10.16910/jemr.13.1.4 (`withMSET`)
- Shimonishi, Ishikawa, Yonetani, Kawashima & Matsuyama (2014), "Learning Aspects of Interest from Gaze",
  *Transactions of the Human Interface Society* 16(2), 103-114 (`withLayout`)

## Repository structure

| Directory | Content |
|---|---|
| [matlab-ver](matlab-ver/README.md) | **Reference implementation.** The MATLAB code used for the papers, cleaned up and corrected for this release (`withMSET`, `withLayout`). |
| [python-ver](python-ver/README.md) | Python port of `matlab-ver`, written with AI assistance (Claude Code) using the MATLAB code as the reference (`withMSET`, `withLayout`, and `common` for shared code). |
| `sample_data` | One gaze-recording session and the item attributes (`knowledge.csv`) of the 2020 experiment. |

`matlab-ver` is the reference: where the two versions differ, the MATLAB code defines the intended behavior.
`python-ver` was checked against it with tests that compare the Python code with line-by-line re-implementations of the
MATLAB code, with the MATLAB outputs of the papers and with the MATLAB code run in GNU Octave (see the README files of
`python-ver`). The deterministic computations agree with the MATLAB code; results that depend on random restarts of the
EM algorithms agree statistically.

The experiment data (except `sample_data`) and the MATLAB outputs are not included in the repository.
