"""
Window likelihoods of the browsed attribute values (calc_lkh.m)

gazedata.mat is loaded by main.load_gazedata_mat; raw gaze data by data.gaze_reader.
"""

import os
import sys
from typing import List, Dict

import numpy as np

# normalize is shared with withLayout (python-ver/common)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from common.math_utils import normalize  # noqa: E402


def exact_prob(prob: np.ndarray, n_hist: np.ndarray) -> float:
    """
    Calculate exact probability for multinomial distribution
    
    Args:
        prob: Multinomial probability distribution (should sum to 1)
        n_hist: Histogram of observations (can be counts or binary: 0 or 1)
        
    Returns:
        Exact probability value
        
    Note:
        Translated from exact_prob.m: n! * prod(p_k^x_k / x_k!) for binary and count data
    """
    prob = np.array(prob).flatten()
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
    
    # Check if it's binary data (0s and 1s only)
    is_binary = np.all((n_hist == 0) | (n_hist == 1))
    
    if is_binary:
        # Original binary case: find indices where n_hist == 1
        selected_indices = np.where(n_hist == 1)[0]
        if len(selected_indices) != N:
            raise ValueError("Number of selected items doesn't match histogram sum")
        # n! * prod(p^x / x!) with x in {0, 1}, as in exact_prob.m
        from math import factorial
        result = factorial(N) * np.prod(prob[selected_indices])
    else:
        # Multinomial case: calculate multinomial probability
        # P(n_hist | prob) = N! / (n1! * n2! * ... * nk!) * p1^n1 * p2^n2 * ... * pk^nk
        from math import factorial
        
        # Calculate multinomial coefficient
        multinomial_coeff = factorial(N)
        for count in n_hist:
            if count > 0:
                multinomial_coeff //= factorial(int(count))
        
        # Calculate probability part
        prob_part = 1.0
        for i, count in enumerate(n_hist):
            if count > 0:
                prob_part *= (prob[i] ** count)
        
        result = multinomial_coeff * prob_part
    
    return result


class LikelihoodCalculator:
    """
    Calculate likelihood values for sequences
    
    Translated from calc_lkh.m
    """
    
    def __init__(self, win_size: int = 5):
        """
        Initialize likelihood calculator
        
        Args:
            win_size: Window size for likelihood calculation
        """
        self.win_size = win_size
        
    def generate_all_combinations(self, n_categories: int = 3) -> np.ndarray:
        """
        Generate all possible combinations for the window
        
        Args:
            n_categories: Number of categories
            
        Returns:
            Array of all possible combinations
            
        Note:
            Translated from allpair3 generation in calc_lkh.m
        """
        combinations = []
        
        # Use recursive generation for any number of categories
        def generate_recursive(remaining, current_combo, depth):
            if depth == n_categories - 1:
                current_combo.append(remaining)
                combinations.append(current_combo.copy())
                current_combo.pop()
                return
            
            for i in range(remaining + 1):
                current_combo.append(i)
                generate_recursive(remaining - i, current_combo, depth + 1)
                current_combo.pop()
        
        generate_recursive(self.win_size, [], 0)
        
        return np.array(combinations)
    
    def calculate_session_likelihood(self, attribute_seq: np.ndarray, 
                                   prob_distributions: List[np.ndarray],
                                   ordinal_attributes: List[int] = None) -> Dict[int, np.ndarray]:
        """
        Calculate likelihood for each attribute in a session
        
        Args:
            attribute_seq: Sequence of attribute values
            prob_distributions: Probability distributions for each attribute
            ordinal_attributes: List of ordinal attribute indices (default: all available)
            
        Returns:
            Dictionary mapping attribute index to likelihood array
            
        Note:
            Translated from the main loop in calc_lkh.m
        """
        if ordinal_attributes is None:
            # Use all available attributes (up to 5 or the actual number)
            n_attributes = min(attribute_seq.shape[0], len(prob_distributions))
            ordinal_attributes = list(range(n_attributes))
            
        T = attribute_seq.shape[1]
        likelihood_results = {}
        
        for att in ordinal_attributes:
            if att >= attribute_seq.shape[0] or att >= len(prob_distributions):
                continue  # Skip if attribute doesn't exist
                
            val_seq = attribute_seq[att, :]
            likelihood = np.ones(T)
            
            # Get the number of categories for this attribute
            n_categories = len(prob_distributions[att])
            
            # Generate all possible combinations for this attribute's category count
            all_combinations = self.generate_all_combinations(n_categories)
            
            # Pre-calculate probabilities for all combinations
            prob_set = np.zeros(len(all_combinations))
            for i, combo in enumerate(all_combinations):
                prob_set[i] = self._exact_prob_for_combo(prob_distributions[att], combo)
            
            # Calculate likelihood for each time point
            # MATLAB: for t = win_size:T (1-based), i.e. the first window ends at index win_size-1
            for t in range(self.win_size - 1, T):
                window_vals = val_seq[t - self.win_size + 1:t + 1]
                
                # Calculate histogram with proper range for this attribute
                hist_counts = np.histogram(window_vals, 
                                         bins=n_categories,
                                         range=(1, n_categories + 1))[0]
                
                # Find matching combination
                matching_prob = 0.0
                for i, combo in enumerate(all_combinations):
                    if len(combo) == len(hist_counts) and np.array_equal(hist_counts, combo):
                        matching_prob = prob_set[i]
                        break
                
                likelihood[t] = matching_prob if matching_prob > 0 else 1e-16
            
            likelihood_results[att] = likelihood
            
        return likelihood_results
    
    def _exact_prob_for_combo(self, prob_dist: np.ndarray, combination: np.ndarray) -> float:
        """
        Calculate exact probability for a specific combination
        
        Args:
            prob_dist: Probability distribution
            combination: Histogram combination
            
        Returns:
            Exact probability value
        """
        return exact_prob(prob_dist, combination)
