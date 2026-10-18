"""
Probabilistic generative model of gaze targets driven by aspects of interest, with the layout effect
(Shimonishi et al. 2014, Eq. 2-9)

    P(r_t = R_n | s) = sum_q sum_k P(R_n | V_q) P(V_q | Z_k) P(Z_k | s)

Translated from plsa.m, estep.m, mstep.m, @prob/prob.m (calcL), est.m and train.m.
A session is given as (f, prv): f the gazed regions (1..12), prv (Q x 12) the matrix P(R | V) of its content.
"""

import os
import sys
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from common.math_utils import normalize  # noqa: E402
from common.restarts import run_restarts  # noqa: E402

SessionData = Tuple[np.ndarray, np.ndarray]  # (f, prv)


def posterior(zs_s: np.ndarray, vz: np.ndarray, prv: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    P(z, v | r_t, s) for every time point (estep.m)

    Returns:
        (K x Q x T) array; each [:, :, t] sums to 1
    """
    g = zs_s[:, None, None] * vz[:, :, None] * prv[None, :, np.asarray(f) - 1]
    return g / g.sum(axis=(0, 1), keepdims=True)


def log_likelihood(zs: np.ndarray, vz: np.ndarray, sessions: Sequence[SessionData]) -> float:
    """Mean over the sessions of sum_t log P(r_t | s) (prob.calcL)"""
    return float(np.mean([np.sum(np.log(zs[s] @ vz @ prv[:, np.asarray(f) - 1]))
                          for s, (f, prv) in enumerate(sessions)]))


def _em_loop(step, loglik, lim: float, max_itr: int, name: str) -> List[float]:
    """Iterate until the increase of the log-likelihood is <= lim (plsa.m / est.m: while cL > lim || cL < 0)"""
    history = [loglik()]
    n_decrease = 0
    for itr in range(max_itr):
        step()
        history.append(loglik())
        inc = history[-1] - history[-2]
        if inc < 0 and itr > 0:
            n_decrease += 1
        if not (inc > lim or inc < 0):  # also stops on NaN, as the MATLAB while condition
            break
    else:
        warnings.warn(f"{name}: no convergence after {max_itr} iterations")
    if n_decrease:
        warnings.warn(f"{name}: log-likelihood decreased {n_decrease} times")
    return history


class Packed:
    """All time points of all sessions in one array: P(r_t | v) as (Q x T_all) and the session of each time point"""

    def __init__(self, sessions: Sequence[SessionData]):
        self.S = len(sessions)
        self.pr = np.hstack([prv[:, np.asarray(f) - 1] for f, prv in sessions])
        self.sid = np.concatenate([np.full(len(f), s) for s, (f, _) in enumerate(sessions)])
        self.T = np.bincount(self.sid, minlength=self.S).astype(float)

    def log_likelihood(self, zs: np.ndarray, vz: np.ndarray) -> float:
        """Mean over the sessions of sum_t log P(r_t | s) (prob.calcL)"""
        p_t = np.einsum('tk,kq,qt->t', zs[self.sid], vz, self.pr)
        return float(np.mean(np.bincount(self.sid, weights=np.log(p_t), minlength=self.S)))


def em_step(data: Packed, zs: np.ndarray, vz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    One EM iteration (estep.m + mstep.m) on packed sessions

    The posteriors P(z, v | r_t, s) = zs[s, z] vz[z, v] P(r_t | v) / P(r_t | s) are only needed as sums, so the
    K x Q x T array of estep.m is not formed.

    Returns:
        (zs, vz) updated
    """
    zt = zs[data.sid].T                      # (K x T_all)
    a = vz @ data.pr                         # (K x T_all): P(r_t | z)
    p_t = np.sum(zt * a, axis=0)             # P(r_t | s)
    pvz = vz * ((zt / p_t) @ data.pr.T)      # sum_t P(z, v | r_t, s)
    pz_t = zt * a / p_t                      # P(z | r_t, s)
    zs_new = np.vstack([np.bincount(data.sid, weights=pz_t[k], minlength=data.S)
                        for k in range(zs.shape[1])]).T / data.T[:, None]
    return zs_new, pvz / pvz.sum(axis=1, keepdims=True)


def fit(sessions: Sequence[SessionData], K: int, lim: float = 1e-2, max_itr: int = 10000) -> Dict:
    """
    Learn the aspects P(V | Z) and the interests P(Z | s) by EM from a random start (plsa.m)

    E-step: P(z, v | r_t, s) (estep.m); M-step: P(V | Z) from the summed posteriors, P(Z | s) as their mean over
    the time points of the session (mstep.m). The loop also continues while the log-likelihood decreases (as plsa.m);
    max_itr only guards against an endless loop (plsa.m has no limit).

    Returns:
        {'vz': (K x Q), 'zs': (S x K), 'L': history of the mean log-likelihood}
    """
    data = Packed(sessions)
    zs = normalize(np.random.rand(data.S, K), 1)
    vz = normalize(np.random.rand(K, data.pr.shape[0]), 1)
    state = {'zs': zs, 'vz': vz}

    def step():
        state['zs'], state['vz'] = em_step(data, state['zs'], state['vz'])

    history = _em_loop(step, lambda: data.log_likelihood(state['zs'], state['vz']), lim, max_itr, 'fit')
    return {'vz': state['vz'], 'zs': state['zs'], 'L': history}


def _fit_once(sessions, K, lim, max_itr):
    return fit(sessions, K, lim, max_itr)


def train(sessions: Sequence[SessionData], K: int, lim: float = 1e-2, train_num: int = 100,
          max_itr: int = 10000, jobs: int = 1, seed: int = 0) -> Dict:
    """Best of train_num random restarts by the final log-likelihood (train.m)"""
    results = run_restarts(_fit_once, (list(sessions), K, lim, max_itr), train_num, jobs, seed)
    return max(results, key=lambda r: r['L'][-1])


def estimate(vz: np.ndarray, session: SessionData, lim: float = 1e-3, smoothing: float = 0.01,
             max_itr: int = 10000) -> np.ndarray:
    """
    Interests P(Z | s) of a new session for given aspects (est.m; paper 3.3.2)

    As est.m, the aspects are smoothed (normalize(vz + 0.01, 2)) and EM starts from uniform interests.

    Returns:
        (K,) interests
    """
    f, prv = session
    vz_s = normalize(np.asarray(vz) + smoothing, 1)
    state = {'zs': np.full((1, vz.shape[0]), 1.0 / vz.shape[0])}

    def step():
        g = posterior(state['zs'][0], vz_s, prv, f)
        state['zs'] = (g.sum(axis=(1, 2)) / g.shape[2])[None, :]

    _em_loop(step, lambda: log_likelihood(state['zs'], vz_s, [session]), lim, max_itr, 'estimate')
    return state['zs'][0]
