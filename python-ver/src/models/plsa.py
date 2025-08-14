"""
Four-layer Probabilistic Latent Semantic Analysis implementation
Custom model with Session -> Segment -> Topic -> Vocabulary layers
Translated from MATLAB code
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.math_utils import normalize, exact_prob


class FourLayerPLSA:
    """
    Four-layer Probabilistic Latent Semantic Analysis model
    Session -> Segment -> Topic -> Vocabulary
    
    This is a custom extension of standard pLSA with an additional segment layer
    Translated from the MATLAB implementation
    """
    
    def __init__(self, n_topics: int):
        """
        Initialize four-layer pLSA model
        
        Args:
            n_topics: Number of latent topics (K)
        """
        self.n_topics = n_topics
        self.p_z_s = None      # P(z|s) - topic distribution per session
        self.p_v_z = None      # P(v|z) - vocabulary distribution per topic
        self.p_seg_s = None    # P(seg|s) - segment distribution per session
        self.p_z_seg = None    # P(z|seg) - topic distribution per segment
        self.log_likelihood_history = []
        
    def fit(self, sessions: List[np.ndarray], segments: List[np.ndarray] = None,
            train_lim: float = 1e-6, max_itr: int = 100) -> Tuple[Dict, List[float]]:
        """
        Train four-layer pLSA model using EM algorithm
        
        Args:
            sessions: List of session data matrices (vocabulary x time)
            segments: List of segment data matrices (optional, for segment-aware training)
            train_lim: Convergence threshold for relative log-likelihood change
            max_itr: Maximum number of iterations
            
        Returns:
            Tuple of (parameters dict, log-likelihood history)
        """
        if segments is None:
            # Fall back to standard 3-layer pLSA if no segments provided
            return self._fit_standard_plsa(sessions, train_lim, max_itr)
        else:
            # Use 4-layer model with segments
            return self._fit_four_layer(sessions, segments, train_lim, max_itr)
    
    def _fit_standard_plsa(self, sessions: List[np.ndarray], train_lim: float, 
                          max_itr: int) -> Tuple[Dict, List[float]]:
        """
        Standard 3-layer pLSA training (backward compatibility)
        """
        S = len(sessions)
        N = sessions[0].shape[0] if sessions else 0  # Number of vocabulary items
        K = self.n_topics
        
        # Initialize parameters randomly
        self.p_z_s = normalize(np.random.rand(S, K), dim=1)  # Normalize rows
        self.p_v_z = []
        for k in range(K):
            self.p_v_z.append(normalize(np.random.rand(N)))
        
        self.log_likelihood_history = []
        inc = -1e20
        lkh_prev = -1e20
        iteration = 1
        
        while abs(inc / lkh_prev) > train_lim and iteration < max_itr:
            # E-step
            gamma, lkh_current = self._e_step_standard(sessions)
            
            # M-step
            self._m_step_standard(gamma, sessions)
            
            inc = lkh_current - lkh_prev
            if inc < 0:
                print("Error: Log-likelihood decreased")
                break
                
            self.log_likelihood_history.append(lkh_current)
            lkh_prev = lkh_current
            iteration += 1
            
        # Return parameters in the same format as MATLAB version
        params = {
            'zs': self.p_z_s,
            'vz': self.p_v_z
        }
        
        return params, self.log_likelihood_history
    
    def _fit_four_layer(self, sessions: List[np.ndarray], segments: List[np.ndarray],
                       train_lim: float, max_itr: int) -> Tuple[Dict, List[float]]:
        """
        Four-layer pLSA training with segment layer
        """
        S = len(sessions)
        N = sessions[0].shape[0] if sessions else 0  # Number of vocabulary items
        K = self.n_topics
        
        # Initialize parameters
        self.p_z_s = normalize(np.random.rand(S, K), dim=1)
        self.p_v_z = []
        for k in range(K):
            self.p_v_z.append(normalize(np.random.rand(N)))
        
        # Additional parameters for segment layer
        max_segments = max(seg.shape[1] for seg in segments if seg.size > 0)
        self.p_seg_s = normalize(np.random.rand(S, max_segments), dim=1)
        self.p_z_seg = normalize(np.random.rand(max_segments, K), dim=1)
        
        self.log_likelihood_history = []
        inc = -1e20
        lkh_prev = -1e20
        iteration = 1
        
        while abs(inc / lkh_prev) > train_lim and iteration < max_itr:
            # E-step for four-layer model
            gamma, lkh_current = self._e_step_four_layer(sessions, segments)
            
            # M-step for four-layer model
            self._m_step_four_layer(gamma, sessions, segments)
            
            inc = lkh_current - lkh_prev
            if inc < 0:
                print("Error: Log-likelihood decreased")
                break
                
            self.log_likelihood_history.append(lkh_current)
            lkh_prev = lkh_current
            iteration += 1
            
        params = {
            'zs': self.p_z_s,
            'vz': self.p_v_z,
            'seg_s': self.p_seg_s,
            'z_seg': self.p_z_seg
        }
        
        return params, self.log_likelihood_history
    
    def _e_step_standard(self, sessions: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """
        E-step for standard 3-layer pLSA
        
        Args:
            sessions: List of session data matrices
            
        Returns:
            Tuple of (gamma values, log-likelihood)
        """
        S = len(sessions)
        K = self.n_topics
        
        gamma = []  # P(z|v_t, s)
        r_values = []
        log_likelihood = 0.0
        
        for s in range(S):
            session_data = sessions[s]
            T = session_data.shape[1]  # Number of time points
            
            gamma_s = np.zeros((K, T))  # P(z|r_t, s)
            r_s = np.zeros(T)
            
            for t in range(T):
                pvz_s = np.zeros(K)
                
                for k in range(K):
                    # Calculate P(z=k, s) * P(v_t | z=k)
                    pvz_s[k] = self.p_z_s[s, k] * exact_prob(self.p_v_z[k], session_data[:, t])
                
                r_s[t] = np.sum(pvz_s)
                if r_s[t] > 0:
                    gamma_s[:, t] = pvz_s / r_s[t]
                else:
                    gamma_s[:, t] = np.ones(K) / K  # Uniform distribution if denominator is 0
            
            gamma.append(gamma_s)
            r_values.append(r_s)
            
        # Calculate log-likelihood
        for r_s in r_values:
            log_likelihood += np.sum(np.log(r_s + 1e-16))  # Add small value to avoid log(0)
            
        return gamma, log_likelihood
    
    def _e_step_four_layer(self, sessions: List[np.ndarray], 
                          segments: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """
        E-step for four-layer pLSA with segments
        
        Args:
            sessions: List of session data matrices
            segments: List of segment data matrices
            
        Returns:
            Tuple of (gamma values, log-likelihood)
        """
        S = len(sessions)
        K = self.n_topics
        
        gamma = []
        log_likelihood = 0.0
        
        for s in range(S):
            if segments[s].size == 0:
                # No segments for this session, use empty gamma
                gamma.append(np.array([]))
                continue
                
            segment_data = segments[s]
            T = segment_data.shape[1]  # Number of time points in segments
            
            gamma_s = np.zeros((K, T))
            r_s = np.zeros(T)
            
            for t in range(T):
                pvz_s = np.zeros(K)
                
                for k in range(K):
                    # Four-layer probability calculation
                    # P(z=k|s) * P(v_t|z=k) considering segment structure
                    pvz_s[k] = self.p_z_s[s, k] * exact_prob(self.p_v_z[k], segment_data[:, t])
                
                r_s[t] = np.sum(pvz_s)
                if r_s[t] > 0:
                    gamma_s[:, t] = pvz_s / r_s[t]
                else:
                    gamma_s[:, t] = np.ones(K) / K
            
            gamma.append(gamma_s)
            
            # Add to log-likelihood
            log_likelihood += np.sum(np.log(r_s + 1e-16))
            
        return gamma, log_likelihood
    
    def _m_step_standard(self, gamma: List[np.ndarray], sessions: List[np.ndarray]) -> None:
        """
        M-step for standard 3-layer pLSA
        
        Args:
            gamma: Posterior probabilities from E-step
            sessions: Session data matrices
        """
        S = len(gamma)
        N = sessions[0].shape[0] if sessions else 0
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
            session_data = sessions[s]
            gamma_s = gamma[s]
            
            if gamma_s.size > 0:
                for n in range(N):
                    # Find time points where vocabulary item n is observed
                    observed_times = np.where(session_data[n, :] == 1)[0]
                    if len(observed_times) > 0:
                        rz[:, n] += np.sum(gamma_s[:, observed_times], axis=1)
        
        # Normalize to get P(v|z)
        sum_expectations = np.sum(rz, axis=1)
        for k in range(K):
            if sum_expectations[k] > 0:
                self.p_v_z[k] = rz[k, :] / sum_expectations[k]
            else:
                self.p_v_z[k] = normalize(np.ones(N))
    
    def _m_step_four_layer(self, gamma: List[np.ndarray], sessions: List[np.ndarray], 
                          segments: List[np.ndarray]) -> None:
        """
        M-step for four-layer pLSA with segments
        
        Args:
            gamma: Posterior probabilities from E-step
            sessions: Session data matrices
            segments: Segment data matrices
        """
        S = len(gamma)
        N = sessions[0].shape[0] if sessions else 0
        K = self.n_topics
        
        # Update P(z|s) - similar to standard pLSA
        self.p_z_s = np.zeros((S, K))
        for s in range(S):
            if gamma[s].size > 0:
                self.p_z_s[s, :] = normalize(np.sum(gamma[s], axis=1))
            else:
                self.p_z_s[s, :] = normalize(np.ones(K))
        
        # Update P(v|z) using segment data
        rz = np.zeros((K, N))
        
        for s in range(S):
            if segments[s].size == 0:
                continue
                
            segment_data = segments[s]
            gamma_s = gamma[s]
            
            if gamma_s.size > 0:
                for n in range(N):
                    # Find time points where vocabulary item n is observed in segments
                    observed_times = np.where(segment_data[n, :] == 1)[0]
                    if len(observed_times) > 0:
                        rz[:, n] += np.sum(gamma_s[:, observed_times], axis=1)
        
        # Normalize to get P(v|z)
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
