# TODO for matlab-ver

Open issues only (checked 2026-10-18). Background, pipeline and the changes from the 2019 scripts used for the paper are in
[README.md](README.md). The corrected scripts were run in GNU Octave, not in MATLAB (accepted).

## T1. How to report the correction of paper Fig. 8
The paper's Fig. 8 ("total number of detected AOF in each temporal position", `[37 20 21 31 28 30 40 36 36 30]`) was computed
by the 2019 `for_rev2.m` with a linear index into the AOF masks (`seqmask{s}(l)` instead of `seqmask{s}(:, l)`). The corrected
count per temporal segment is `[579 779 818 878 802 962 977 919 932 791]` (time points with AOF:
`[319 360 374 380 354 390 396 377 369 325]`; `python-ver`, `main.py --plot`, `aof_temporal.png`). The statement of the paper
(comparison mainly in the latter part, decreasing at the end) still holds, and no other result depends on Fig. 8.
Decide with the co-author whether to publish an erratum in JEMR or only explain it in the README (currently listed there as an
erratum candidate).
