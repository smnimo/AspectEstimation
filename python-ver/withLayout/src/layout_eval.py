"""
Prediction of gaze regions and items of interest, leave-one-out evaluation and the comparison method
(Shimonishi et al. 2014, section 4.2-4.4)

Translated from main.m, eval_precision.m, main_comparison.m, makeattr.m, est_cosine.m, bhattacharyya.m and
eval_precision_comparison.m. Precision P@k (k = 1..K) as in the MATLAB code; the paper's Fig. 5-7 show P@1
(the region / item predicted first).
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import layout_data as ld
import layout_model as lm
from layout_data import Session

RATIOS = range(1, 10)  # 10 %, 20 %, ..., 90 % of the session used to estimate the interests


def precision_at_k(scores: np.ndarray, relevant: np.ndarray, K: int, descending: bool = True) -> np.ndarray:
    """P@1..K of the ranking of scores (MATLAB stable sort) against the relevant items"""
    order = np.argsort(-scores if descending else scores, kind='stable')[:K]
    return np.cumsum(relevant[order]) / np.arange(1, K + 1)


def gazed_regions(f: np.ndarray, K: int) -> np.ndarray:
    """0/1 vector of the K most gazed regions (ties in region order, sort(..., 'descend'))"""
    out = np.zeros(ld.N_REGION)
    out[np.argsort(-ld.region_histogram(f), kind='stable')[:K]] = 1
    return out


def selected_items(session: Session) -> np.ndarray:
    """0/1 vector of the items selected in the session"""
    out = np.zeros(ld.N_REGION)
    out[session.answer - 1] = 1
    return out


def eval_precision(f_test: np.ndarray, answer: np.ndarray, prv: np.ndarray, piv: np.ndarray,
                   zs: np.ndarray, vz: np.ndarray, K: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    P@k of the predicted regions (Eq. 8) and items of interest (Eq. 9) (eval_precision.m)

    Regions are compared with the K most gazed regions of the evaluation part f_test, items with the selected items.

    Returns:
        (patk_region, patk_item), each (K,)
    """
    prs = zs @ vz @ prv
    pis = zs @ vz @ piv
    return precision_at_k(prs, gazed_regions(f_test, K), K), precision_at_k(pis, answer, K)


def split_session(f: np.ndarray, ratio: int, overlap: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    First ratio * 10 % for the estimation, the rest for the evaluation (main.m)

    main.m uses f(1 : floor(x)) and f(ceil(x) : T) with x = ratio * T / 10, so one time point is in both parts
    when x is an integer (overlap = True, as MATLAB); overlap = False uses disjoint parts.
    """
    T = len(f)
    x = ratio * T / 10
    start = math.ceil(x) - 1 if overlap else math.floor(x)
    return f[:math.floor(x)], f[start:]


def leave_one_out(sessions: Sequence[Session], content: Dict, attr: int, alpha: float, naspects: Sequence[int],
                  train_num: int = 100, train_lim: float = 1e-2, est_lim: float = 1e-3, K: int = 3,
                  jobs: int = 1, seed: int = 0, overlap: bool = True, log=print) -> Dict:
    """
    Leave-one-out evaluation (main.m): for each session, learn the aspects from the other sessions (layout effect from
    their gaze frequency), estimate the interests from the first part of the session and predict the rest

    Returns:
        {'patk_region' / 'patk_item': {n_aspects: (9 ratios x K x S)}, plus their means over the sessions
         'mpatk_region' / 'mpatk_item': {n_aspects: (9 x K)}}
    """
    S = len(sessions)
    pvis = [ld.item_attributes(content, s, attr) for s in sessions]
    pivs = [ld.item_given_value(p) for p in pvis]
    out = {key: {n: np.zeros((len(RATIOS), K, S)) for n in naspects} for key in ('patk_region', 'patk_item')}
    for s, test in enumerate(sessions):
        layout = ld.region_frequency(sessions, exclude=s)
        prvs = [ld.region_given_value(p, layout, alpha) for p in pvis]
        train_data = [(x.f, prvs[i]) for i, x in enumerate(sessions) if i != s]
        for n in naspects:
            log(f"session {s + 1}/{S}, {n} aspects")
            p = lm.train(train_data, n, train_lim, train_num, jobs=jobs, seed=seed)
            for r_i, ratio in enumerate(RATIOS):
                f_est, f_test = split_session(test.f, ratio, overlap)
                zs = lm.estimate(p['vz'], (f_est, prvs[s]), est_lim)
                out['patk_region'][n][r_i, :, s], out['patk_item'][n][r_i, :, s] = eval_precision(
                    f_test, selected_items(test), prvs[s], pivs[s], zs, p['vz'], K)
    for key in ('region', 'item'):
        out[f'mpatk_{key}'] = {n: v.mean(axis=2) for n, v in out[f'patk_{key}'].items()}
    return out


# ---------------------------------------------------------------- comparison (nearest-neighbor recommendation)
def bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance -log sum sqrt(a_i b_i) (bhattacharyya.m)"""
    with np.errstate(divide='ignore'):
        return float(-np.log(np.sum(np.sqrt(np.asarray(a) * np.asarray(b)))))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(1 - a @ b / (na * nb)) if na > 0 and nb > 0 else 1.0


def interest_vector(pvi: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Attribute values of the gazed items: sum_n n(s, I_n) V_In (paper Eq. 12), normalized to sum 1"""
    return ld.normalize(pvi[np.asarray(f) - 1].sum(axis=0))


def predict_comparison(est_vector: np.ndarray, train_vectors: np.ndarray, pvi: np.ndarray,
                       method: str = 'paper') -> np.ndarray:
    """
    Item scores of the nearest-neighbor recommendation (lower = predicted first)

    method 'paper' (paper 4.3): the training session nearest in cosine distance gives V_near; items ranked by the
        cosine distance between V_near and their attribute vectors.
    method 'matlab' (est_cosine.m, eval_precision_comparison.m): the 3 training sessions nearest in Bhattacharyya
        distance, averaged with weights 1 / distance; items ranked by the Bhattacharyya distance.
    """
    if method == 'paper':
        d = np.array([cosine_distance(est_vector, t) for t in train_vectors])
        near = train_vectors[np.argsort(d, kind='stable')[0]]
        return np.array([cosine_distance(near, v) for v in pvi])
    if method == 'matlab':
        d = np.array([bhattacharyya(est_vector, t) for t in train_vectors])
        order = np.argsort(d, kind='stable')[:3]
        with np.errstate(divide='ignore', invalid='ignore'):
            near = ld.normalize(np.sum(train_vectors[order] / d[order, None], axis=0))
        return np.array([bhattacharyya(near, v) for v in pvi])
    raise ValueError(f"unknown comparison method {method}")


def leave_one_out_comparison(sessions: Sequence[Session], content: Dict, attr: int, method: str = 'paper',
                             K: int = 3, overlap: bool = True) -> Dict:
    """
    Leave-one-out evaluation of the comparison method (main_comparison.m); regions are predicted as the regions of
    the predicted items

    Returns:
        {'patk_region' / 'patk_item': (9 x K x S), 'mpatk_region' / 'mpatk_item': (9 x K)}
    """
    S = len(sessions)
    pvis = [ld.item_attributes(content, s, attr) for s in sessions]
    full = np.vstack([interest_vector(p, s.f) for p, s in zip(pvis, sessions)])
    out = {key: np.zeros((len(RATIOS), K, S)) for key in ('patk_region', 'patk_item')}
    for s, test in enumerate(sessions):
        train_vectors = np.delete(full, s, axis=0)
        for r_i, ratio in enumerate(RATIOS):
            f_est, f_test = split_session(test.f, ratio, overlap)
            scores = predict_comparison(interest_vector(pvis[s], f_est), train_vectors, pvis[s], method)
            out['patk_item'][r_i, :, s] = precision_at_k(scores, selected_items(test), K, descending=False)
            out['patk_region'][r_i, :, s] = precision_at_k(scores, gazed_regions(f_test, K), K, descending=False)
    for key in ('region', 'item'):
        out[f'mpatk_{key}'] = out[f'patk_{key}'].mean(axis=2)
    return out
