"""
Attributes-of-focus (AOF) detection by the multi-scale exact test (MSET; Shimonishi & Kawashima, AAAI 2020)

Every window of l consecutive browsed items, for every scale l in `scales` (default 1..10), is tested against neutral
browsing, i.e. items drawn uniformly from the displayed catalog. The p-value of a window is the probability under
neutral browsing of the outcomes as likely as the observed one or less likely (exact test, AAAI 2020 Eq. 2); a window
with p-value < alpha (default 0.05) is distinctive. The AOF of all distinctive windows of all scales that contain a
time point are united. The test depends on the kind of the attribute (data.knowledge):

- categorical: the combinations of values (patterns) of the browsed items follow a multinomial distribution with the
  shares of the patterns in the catalog (with one value per item, patterns are values: AAAI 2020 Eq. 1).
  AOF: the values whose share in the window exceeds their share in the catalog (x_n / l > theta_n).
- ordinal: one level per item. For each cut k between levels, the number of browsed items above k follows a binomial
  distribution (exact binomial test). The AOF of a distinctive cut are the levels on the side the window is biased to
  (levels > k or <= k); the AOF of all distinctive cuts are united.

Reference implementation of the test: MSET/scripts/mset.py of the authors (calc_pvalue_exact, find_dperiod_value).
"""

import itertools
import os
import sys
from functools import lru_cache
from math import lgamma
from typing import Dict, List, Sequence, Tuple

import numpy as np

from data.knowledge import CATEGORICAL, ORDINAL

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from common.math_utils import normalize  # noqa: E402

ValueSet = Tuple[int, ...]

DEFAULT_ALPHA = 0.05
DEFAULT_SCALES = (1, 10)
_TIES = 1 + 1e-9           # probabilities that differ only by rounding count as equal (ties, p-value == alpha)
_MAX_OUTCOMES = 5_000_000  # guard for the enumeration of the outcomes of one window


def exact_prob(prob: np.ndarray, n_hist: np.ndarray) -> float:
    """
    Multinomial probability n! * prod(p_k^x_k / x_k!) of the counts n_hist (exact_prob.m)

    Args:
        prob: probabilities of the categories (normalized if they do not sum to 1)
        n_hist: counts of the categories
    """
    from math import factorial
    prob = np.array(prob, dtype=float).flatten()
    n_hist = np.array(n_hist).flatten()
    if len(prob) != len(n_hist):
        raise ValueError("Length of prob must be same as length of n_hist")
    if np.min(n_hist) < 0:
        raise ValueError("Elements of n_hist must be non-negative")
    if not np.isclose(np.sum(prob), 1.0):
        prob = normalize(prob)
    N = int(np.sum(n_hist))
    if N == 0:
        return 1.0
    coeff = factorial(N)
    for count in n_hist:
        coeff //= factorial(int(count))
    result = float(coeff)
    for p, count in zip(prob, n_hist):
        if count > 0:
            result *= p ** int(count)
    return result


def _log_pmf(log_p: np.ndarray, counts: np.ndarray, n: int) -> np.ndarray:
    """log multinomial pmf of the rows of counts (all categories have p > 0)"""
    lg = np.vectorize(lgamma)
    return lgamma(n + 1) - lg(counts + 1).sum(axis=1) + counts @ log_p


@lru_cache(maxsize=None)
def _null_distribution(weights: Tuple[int, ...], n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Probabilities of all outcomes of n draws from categories with the given weights (counts in the catalog),
    sorted, and their cumulative sums (p-value lookup)
    """
    k = len(weights)
    n_outcomes = 1
    for i in range(1, k):
        n_outcomes = n_outcomes * (n + i) // i
    if n_outcomes > _MAX_OUTCOMES:
        raise ValueError(f"{n_outcomes} outcomes for a window of {n} items over {k} categories; "
                         "reduce the largest scale or the number of value combinations")
    # stars and bars: positions of k - 1 bars among n + k - 1 slots
    bars = np.array(list(itertools.combinations(range(n + k - 1), k - 1)), dtype=int).reshape(-1, k - 1)
    edges = np.hstack([np.full((len(bars), 1), -1), bars, np.full((len(bars), 1), n + k - 1)])
    counts = np.diff(edges, axis=1) - 1
    log_p = np.log(np.asarray(weights, dtype=float) / sum(weights))
    pmf = np.sort(np.exp(_log_pmf(log_p, counts, n)))
    return pmf, np.cumsum(pmf)


def _pvalue(weights: Tuple[int, ...], counts: np.ndarray) -> float:
    """exact test: sum of the probabilities of the outcomes not more likely than the observed counts (Eq. 2)"""
    n = int(np.sum(counts))
    pmf, cum = _null_distribution(weights, n)
    log_p = np.log(np.asarray(weights, dtype=float) / sum(weights))
    q = float(np.exp(_log_pmf(log_p, np.asarray(counts, dtype=int)[None, :], n))[0])
    i = np.searchsorted(pmf, q * _TIES, side='right')
    return float(min(cum[i - 1], 1.0)) if i else 0.0


def _significant(p: float, alpha: float) -> bool:
    """p < alpha; a p-value equal to alpha up to rounding (e.g. exactly 0.01 with shares 1/10) is not significant"""
    return p * _TIES < alpha


def _scales(scales: Sequence[int]) -> range:
    lo, hi = int(scales[0]), int(scales[1])
    if not 1 <= lo <= hi:
        raise ValueError(f"scales must satisfy 1 <= min <= max: {scales}")
    return range(lo, hi + 1)


def _categorical(n_values: int, catalog: Sequence[ValueSet], seq: Sequence[ValueSet], alpha: float,
                 scales: Sequence[int]):
    T = len(seq)
    patterns = sorted(set(catalog))  # with one value per item: the values present, in value order
    index = {p: i for i, p in enumerate(patterns)}
    weights = tuple(sum(c == p for c in catalog) for p in patterns)
    try:
        ids = np.array([index[v] for v in seq], dtype=int)
    except KeyError as e:
        raise ValueError(f"browsed value set {e} does not occur among the displayed items") from None
    n_with = np.zeros(n_values)  # theta_v: share of the displayed items having value v
    for c in catalog:
        n_with[list(c)] += 1
    share = n_with / len(catalog)
    ls = _scales(scales)
    pvalues = np.full((len(ls), T), np.nan)
    mask = np.zeros((n_values, T))
    for i, win in enumerate(ls):
        for t in range(win - 1, T):
            pvalues[i, t] = _pvalue(weights, np.bincount(ids[t - win + 1:t + 1], minlength=len(patterns)))
            if _significant(pvalues[i, t], alpha):
                counts = np.zeros(n_values)
                for v in seq[t - win + 1:t + 1]:
                    counts[list(v)] += 1
                mask[counts / win > share, t - win + 1:t + 1] = 1
    return pvalues, mask


def _ordinal(n_values: int, catalog: Sequence[ValueSet], seq: Sequence[ValueSet], alpha: float,
             scales: Sequence[int]):
    if any(len(v) != 1 for v in list(catalog) + list(seq)):
        raise ValueError("an ordinal attribute needs exactly one level per item")
    T = len(seq)
    N = len(catalog)
    levels = np.array([v[0] for v in catalog])
    browsed = np.array([v[0] for v in seq])
    n_up = [int(np.sum(levels > k)) for k in range(n_values - 1)]  # displayed items above cut k
    ls = _scales(scales)
    pvalues = np.full((len(ls), T), np.nan)  # minimum over the cuts
    mask = np.zeros((n_values, T))
    for i, win in enumerate(ls):
        for t in range(win - 1, T):
            window = browsed[t - win + 1:t + 1]
            for k, up in enumerate(n_up):
                if up in (0, N):
                    continue  # all displayed items on one side of the cut: no information
                u = int(np.sum(window > k))
                pk = _pvalue((N - up, up), np.array([win - u, u]))
                pvalues[i, t] = pk if np.isnan(pvalues[i, t]) else min(pvalues[i, t], pk)
                if _significant(pk, alpha) and u * N != up * win:
                    side = slice(k + 1, n_values) if u * N > up * win else slice(0, k + 1)
                    mask[side, t - win + 1:t + 1] = 1
    return pvalues, mask


def attribute_aof(kind: str, n_values: int, catalog: Sequence[ValueSet], seq: Sequence[ValueSet],
                  alpha: float = DEFAULT_ALPHA, scales: Sequence[int] = DEFAULT_SCALES) -> Tuple[np.ndarray, np.ndarray]:
    """
    MSET p-values and AOF mask of one attribute in one session

    Args:
        kind: CATEGORICAL or ORDINAL
        n_values: number of values of the attribute
        catalog: value sets (0-based) of the displayed items
        seq: value sets of the browsed items, one per time point
        alpha: significance level
        scales: (smallest, largest) window length

    Returns:
        (pvalues, mask): pvalues (number of scales x T), row i = window of length scales[0] + i ending at t (NaN where
        the window does not fit or, for an ordinal attribute, no cut is informative); mask (n_values x T) 0/1
    """
    if kind == CATEGORICAL:
        return _categorical(n_values, catalog, seq, alpha, scales)
    if kind == ORDINAL:
        return _ordinal(n_values, catalog, seq, alpha, scales)
    raise ValueError(f"unknown attribute kind {kind}")


def segment_labels(mask: np.ndarray) -> np.ndarray:
    """Segment ID of each time point (calc_segments.m): 0 without AOF, a new ID whenever the AOF change"""
    T = mask.shape[1]
    sl = mask.sum(axis=0)
    labels = np.zeros(T, dtype=int)
    if T == 0:
        return labels
    labels[0] = 1 if sl[0] > 0 else 0
    segc = labels[0]
    for t in range(1, T):
        if sl[t] == 0:
            labels[t] = 0
        elif np.any(mask[:, t] != mask[:, t - 1]):
            segc += 1
            labels[t] = segc
        else:
            labels[t] = segc
    return labels


def session_aof(kinds: Sequence[str], n_values: Sequence[int], catalog: Sequence[Sequence[ValueSet]],
                seq: Sequence[Sequence[ValueSet]], alpha: float = DEFAULT_ALPHA,
                scales: Sequence[int] = DEFAULT_SCALES) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """
    AOF of one session over all attributes

    Args:
        kinds, n_values: kind and number of values of each attribute
        catalog: catalog[item][attribute] value sets of the displayed items
        seq: seq[t][attribute] value sets of the browsed items
        alpha, scales: see attribute_aof

    Returns:
        (pvalues, mask, seglabel): pvalues[attribute] (scales x T), mask (sum(n_values) x T) attribute-major,
        seglabel (T,)
    """
    pvalues, masks = {}, []
    for a, (kind, n) in enumerate(zip(kinds, n_values)):
        pvalues[a], m = attribute_aof(kind, int(n), [c[a] for c in catalog], [s[a] for s in seq], alpha, scales)
        masks.append(m)
    mask = np.vstack(masks)
    return pvalues, mask, segment_labels(mask)


def legacy_value_sets(values: np.ndarray) -> List[List[ValueSet]]:
    """(rows x attributes) integers 1..N -> per row, per attribute 0-based single-value sets"""
    return [[(int(v) - 1,) for v in row] for row in np.asarray(values, dtype=int)]
