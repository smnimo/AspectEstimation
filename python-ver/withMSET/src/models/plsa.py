"""
pLSA on attributes-of-focus (pIAF, paper Eq. 2, 3, 8)

Translated from plsa.m, estep.m, mstep.m (train.m: see main.train).
Each time point t of session s is a multi-hot vector f_t; P(f_t | z) = n_t! prod_k p_{z,k}^{f_t,k}
(exact_prob.m). EM learns P(v|z) (vz, the aspects) and P(z|s) (zs, the interests).

The class keeps its historical name "FourLayerPLSA": the Feature -> Observation layer of the
earlier port was an identity mapping and never entered the computation, so it was removed.
"""

import numpy as np
from typing import List, Dict, Tuple
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.math_utils import normalize
from scipy.special import gammaln


def multinomial_probs(p_v_z, observations: np.ndarray) -> np.ndarray:
    """
    Vectorized exact_prob: n_t! * prod_k(p_{z,k}^f_{t,k} / f_{t,k}!) for every topic z and time t

    Args:
        p_v_z: List of K multinomial parameters (each of length N)
        observations: (N x T) matrix of AOF indicators (or counts)

    Returns:
        (K x T) matrix; equals exact_prob(p_v_z[k], observations[:, t]) element-wise
    """
    P = np.stack([np.asarray(p, dtype=float).ravel() for p in p_v_z])  # (K x N)
    F = np.asarray(observations, dtype=float)                           # (N x T)
    log_coef = gammaln(F.sum(axis=0) + 1) - gammaln(F + 1).sum(axis=0)  # (T,)
    with np.errstate(divide='ignore'):
        log_p = np.log(P)
    is_zero = ~np.isfinite(log_p)
    probs = np.exp(np.where(is_zero, 0.0, log_p) @ F + log_coef)
    probs[(is_zero.astype(float) @ F) > 0] = 0.0  # an observed value with p = 0
    return probs


class FourLayerPLSA:
    """
    pLSA trained by EM (plsa.m); one instance = one random restart

    Use main.train for the best of several restarts (train.m).
    """

    def __init__(self, n_topics: int):
        """
        Args:
            n_topics: Number of latent topics (K)
        """
        self.n_topics = n_topics
        self.p_z_s = None      # P(z|s) - topic distribution per session
        self.p_v_z = None      # P(v|z) - feature distribution per topic
        self.log_likelihood_history = []

    def fit(self, sessions: List[np.ndarray], train_lim: float = 1e-4,
            max_itr: int = 1000) -> Tuple[Dict, List[float]]:
        """
        Train by EM from a random initialization (plsa.m)

        Args:
            sessions: List of session matrices (features x time); a session may have no columns
            train_lim: Convergence threshold for the relative log-likelihood change (MATLAB lim = 1e-4)
            max_itr: Maximum number of iterations (MATLAB itr_MAX = 1000; at most max_itr - 1 EM steps)

        Returns:
            ({'zs': (S x K), 'vz': K arrays of length N}, log-likelihood history)

        Note:
            As in plsa.m, the history ends with the log-likelihood of the parameters before the last
            M-step, and an iteration whose log-likelihood decreases stops the training (its M-step is kept).
        """
        S = len(sessions)
        N = sessions[0].shape[0] if sessions else 0  # Number of feature dimensions
        K = self.n_topics

        # Random initialization (same order as plsa.m: zs, then vz{1..K})
        self.p_z_s = normalize(np.random.rand(S, K), dim=1)
        self.p_v_z = [normalize(np.random.rand(N)) for _ in range(K)]

        self.log_likelihood_history = []
        inc = -1e20
        lkh_prev = -1e20
        iteration = 1

        while abs(inc / lkh_prev) > train_lim and iteration < max_itr:
            gamma, lkh_current = self._e_step_four_layer(sessions)
            self._m_step_four_layer(gamma, sessions)

            inc = lkh_current - lkh_prev
            if inc < 0:
                print("Error: Log-likelihood decreased")
                break

            self.log_likelihood_history.append(lkh_current)
            lkh_prev = lkh_current
            iteration += 1

        params = {
            'zs': self.p_z_s,
            'vz': self.p_v_z
        }
        return params, self.log_likelihood_history

    def _e_step_four_layer(self, sessions: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """
        E-step (estep.m)

        Args:
            sessions: List of session data matrices (features x time)

        Returns:
            Tuple of (gamma values, log-likelihood)
        """
        S = len(sessions)
        K = self.n_topics

        gamma = []  # P(z|f_t, s): posterior topic probabilities
        r_values = []
        log_likelihood = 0.0

        for s in range(S):
            session_data = sessions[s]  # (features x time)
            T = session_data.shape[1]   # Number of time points

            # P(z=k|s) * P(f_t|z=k) for all k, t at once; P(f_t|z=k) is exact_prob(p_v_z[k], f_t)
            pvz_s = self.p_z_s[s, :, np.newaxis] * multinomial_probs(self.p_v_z, session_data)  # (K x T)

            r_s = np.sum(pvz_s, axis=0)
            positive = r_s > 0
            gamma_s = np.full((K, T), 1.0 / K)  # Uniform distribution if denominator is 0
            gamma_s[:, positive] = pvz_s[:, positive] / r_s[positive]

            gamma.append(gamma_s)
            r_values.append(r_s)

        # Calculate log-likelihood
        for r_s in r_values:
            log_likelihood += np.sum(np.log(r_s + 1e-16))  # Add small value to avoid log(0)

        return gamma, log_likelihood

    def _m_step_four_layer(self, gamma: List[np.ndarray], sessions: List[np.ndarray]) -> None:
        """
        M-step (mstep.m): update P(z|s) and P(v|z)

        Args:
            gamma: Posterior probabilities from E-step
            sessions: Session data matrices (features x time)
        """
        S = len(gamma)
        N = sessions[0].shape[0] if sessions else 0  # Number of features
        K = self.n_topics

        # Update P(z|s)
        self.p_z_s = np.zeros((S, K))
        for s in range(S):
            if gamma[s].size > 0:
                self.p_z_s[s, :] = normalize(np.sum(gamma[s], axis=1))
            else:
                self.p_z_s[s, :] = normalize(np.ones(K))

        # Update P(v|z)
        rz = np.zeros((K, N))

        for s in range(S):
            session_data = sessions[s]  # (features x time)
            gamma_s = gamma[s]          # (topics x time)

            if gamma_s.size > 0:
                # Sum of gamma over the time points where feature n is observed (== 1)
                rz += gamma_s @ (session_data == 1).T

        sum_expectations = np.sum(rz, axis=1)
        for k in range(K):
            if sum_expectations[k] > 0:
                self.p_v_z[k] = rz[k, :] / sum_expectations[k]
            else:
                self.p_v_z[k] = normalize(np.ones(N))
