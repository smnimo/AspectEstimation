"""
True Four-layer Probabilistic Latent Semantic Analysis implementation
Session → Topic → Feature → Observation
Feature → Observation mapping is externally provided and fixed
Translated from MATLAB code
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.math_utils import normalize, exact_prob
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
    True Four-layer Probabilistic Latent Semantic Analysis model
    Session → Topic → Feature → Observation
    
    The Feature → Observation mapping is externally provided and fixed (not learned)
    Only Session → Topic and Topic → Feature distributions are learned
    
    Translated from the MATLAB implementation
    """
    
    def __init__(self, n_topics: int):
        """
        Initialize four-layer pLSA model
        
        Args:
            n_topics: Number of latent topics (K)
        """
        self.n_topics = n_topics
        self.p_z_s = None      # P(z|s) - topic distribution per session (LEARNED)
        self.p_v_z = None      # P(v|z) - feature distribution per topic (LEARNED) 
        self.p_o_v = None      # P(o|v) - observation given feature (FIXED, externally provided)
        self.log_likelihood_history = []
        
    def set_observation_model(self, p_o_v: np.ndarray):
        """
        Set the fixed observation model P(o|v)
        
        Args:
            p_o_v: Observation probability matrix (observations x features)
                   This represents the Feature → Observation mapping
        """
        self.p_o_v = p_o_v
        print(f"Set observation model with shape: {p_o_v.shape} (observations x features)")
        
    def fit(self, sessions: List[np.ndarray], segments: List[np.ndarray] = None,
            train_lim: float = 1e-6, max_itr: int = 100) -> Tuple[Dict, List[float]]:
        """
        Train four-layer pLSA model using EM algorithm
        
        Args:
            sessions: List of session data matrices (observations x time)
                     NOTE: These are already transformed through Feature → Observation
            segments: List of segment data matrices (optional, for segment-aware training)
            train_lim: Convergence threshold for relative log-likelihood change
            max_itr: Maximum number of iterations
            
        Returns:
            Tuple of (parameters dict, log-likelihood history)
        """
        if self.p_o_v is None:
            print("Warning: No observation model set. Using identity mapping.")
            # Create identity mapping if none provided
            n_features = sessions[0].shape[0] if sessions else 0
            self.p_o_v = np.eye(n_features)
        
        # Use segments if provided, otherwise use sessions
        data_to_train = segments if segments is not None else sessions
        
        return self._fit_four_layer_plsa(data_to_train, train_lim, max_itr)
    
    def _fit_four_layer_plsa(self, sessions: List[np.ndarray], train_lim: float, 
                            max_itr: int) -> Tuple[Dict, List[float]]:
        """
        Four-layer pLSA training: Session → Topic → Feature → Observation
        
        The Feature → Observation mapping is fixed (self.p_o_v)
        Only Session → Topic (p_z_s) and Topic → Feature (p_v_z) are learned
        
        Args:
            sessions: List of session data matrices (observations x time)
                     Already transformed through the fixed Feature → Observation layer
        """
        S = len(sessions)
        N = sessions[0].shape[0] if sessions else 0  # Number of feature dimensions
        K = self.n_topics
        
        # Initialize learnable parameters randomly
        self.p_z_s = normalize(np.random.rand(S, K), dim=1)  # P(z|s): Session → Topic
        self.p_v_z = []                                      # P(v|z): Topic → Feature
        for k in range(K):
            self.p_v_z.append(normalize(np.random.rand(N)))
        
        self.log_likelihood_history = []
        inc = -1e20
        lkh_prev = -1e20
        iteration = 1
        
        print(f"Training 4-layer pLSA: {S} sessions, {K} topics, {N} features")
        
        while abs(inc / lkh_prev) > train_lim and iteration < max_itr:
            # E-step: compute posterior probabilities
            gamma, lkh_current = self._e_step_four_layer(sessions)
            
            # M-step: update learnable parameters
            self._m_step_four_layer(gamma, sessions)
            
            inc = lkh_current - lkh_prev
            if inc < 0:
                print("Error: Log-likelihood decreased")
                break
                
            self.log_likelihood_history.append(lkh_current)
            lkh_prev = lkh_current
            iteration += 1
            
        # Return parameters in the same format as MATLAB version
        params = {
            'zs': self.p_z_s,  # Session → Topic
            'vz': self.p_v_z   # Topic → Feature
        }
        
        return params, self.log_likelihood_history
    
    def _e_step_four_layer(self, sessions: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """
        E-step for four-layer pLSA: Session → Topic → Feature → Observation
        
        Args:
            sessions: List of session data matrices (observations x time)
            
        Returns:
            Tuple of (gamma values, log-likelihood)
        """
        S = len(sessions)
        K = self.n_topics
        
        gamma = []  # P(z|o_t, s): posterior topic probabilities
        r_values = []
        log_likelihood = 0.0
        
        for s in range(S):
            session_data = sessions[s]  # (observations x time)
            T = session_data.shape[1]   # Number of time points
            
            # P(z=k|s) * P(o_t|z=k) for all k, t at once; P(o_t|z=k) is exact_prob(p_v_z[k], o_t)
            # (observations are multi-hot AOF vectors, so the four-layer model reduces to this)
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
        M-step for four-layer pLSA: Update learnable parameters
        
        Updates:
        - P(z|s): Session → Topic distribution
        - P(v|z): Topic → Feature distribution  
        
        Does NOT update P(o|v): Feature → Observation (fixed externally)
        
        Args:
            gamma: Posterior probabilities from E-step
            sessions: Session data matrices (observations x time)
        """
        S = len(gamma)
        N = sessions[0].shape[0] if sessions else 0  # Number of features
        K = self.n_topics
        
        # Update P(z|s): Session → Topic
        self.p_z_s = np.zeros((S, K))
        for s in range(S):
            if gamma[s].size > 0:
                self.p_z_s[s, :] = normalize(np.sum(gamma[s], axis=1))
            else:
                self.p_z_s[s, :] = normalize(np.ones(K))
        
        # Update P(v|z): Topic → Feature
        rz = np.zeros((K, N))
        
        for s in range(S):
            session_data = sessions[s]  # (observations x time)
            gamma_s = gamma[s]          # (topics x time)
            
            if gamma_s.size > 0:
                # Sum of gamma over the time points where feature n is observed (== 1)
                rz += gamma_s @ (session_data == 1).T
        
        # Normalize to get P(v|z): Topic → Feature
        sum_expectations = np.sum(rz, axis=1)
        for k in range(K):
            if sum_expectations[k] > 0:
                self.p_v_z[k] = rz[k, :] / sum_expectations[k]
            else:
                self.p_v_z[k] = normalize(np.ones(N))


def train_multiple_runs(sessions: List[np.ndarray], n_topics: int, 
                       segments: List[np.ndarray] = None,
                       train_lim: float = 1e-6, train_num: int = 10, 
                       max_itr: int = 100) -> Tuple[Dict, List[float]]:
    """
    Train four-layer pLSA model with multiple random initializations
    
    Args:
        sessions: List of session data matrices
        n_topics: Number of topics
        segments: List of segment data matrices (optional)
        train_lim: Convergence threshold
        train_num: Number of training runs with different initializations
        max_itr: Maximum iterations per run
        
    Returns:
        Best parameters and their log-likelihood history
        
    Note:
        Translated from train.m with four-layer support
    """
    best_params = None
    best_likelihood = -np.inf
    best_history = []
    
    for i in range(train_num):
        print(f"Training run {i+1}/{train_num}")
        
        model = FourLayerPLSA(n_topics)
        params, history = model.fit(sessions, segments, train_lim, max_itr)
        
        if history and history[-1] > best_likelihood:
            best_likelihood = history[-1]
            best_params = params
            best_history = history
    
    return best_params, best_history


# Backward compatibility aliases
PLSA = FourLayerPLSA  # For backward compatibility

def train_standard_plsa(sessions: List[np.ndarray], n_topics: int, 
                       train_lim: float = 1e-6, train_num: int = 10, 
                       max_itr: int = 100) -> Tuple[Dict, List[float]]:
    """
    Train standard 3-layer pLSA model (backward compatibility)
    """
    return train_multiple_runs(sessions, n_topics, None, train_lim, train_num, max_itr)
