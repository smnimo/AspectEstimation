"""
Data loading and processing utilities
Translated from MATLAB data loading functions
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import scipy.io
import os
import sys

# Add the parent directory to the path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.math_utils import exact_prob


class GazeDataLoader:
    """
    Data loader for gaze tracking data
    
    Translated from LoadGaze.m and related MATLAB functions
    """
    
    def __init__(self, data_dir: str = "../../"):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = data_dir
        self.semantic_attributes = None
        self.semantic_attribute_seq = None
        self.sessions_data = None
        
    def load_gazedata(self, filename: str = "gazedata.mat") -> Dict[str, Any]:
        """
        Load gaze data from MATLAB .mat file
        
        Args:
            filename: Name of the .mat file to load
            
        Returns:
            Dictionary containing loaded data
            
        Note:
            Equivalent to 'load gazedata' in MATLAB
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        try:
            data = scipy.io.loadmat(filepath)
            
            # Extract relevant variables (adjust based on actual .mat file structure)
            self.semantic_attributes = data.get('semantic_attributes', None)
            self.semantic_attribute_seq = data.get('semantic_attribute_seq', None)
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Error loading .mat file: {e}")
    
    def process_attributes(self) -> Tuple[np.ndarray, List[range]]:
        """
        Process semantic attributes to get value ranges
        
        Returns:
            Tuple of (N_value array, attribute ranges)
            
        Note:
            Translated from the attribute processing in calc_lkh.m
        """
        if self.semantic_attributes is None:
            raise ValueError("Semantic attributes not loaded. Call load_gazedata() first.")
        
        # Convert cell array to matrix if needed
        if isinstance(self.semantic_attributes, np.ndarray):
            attr_matrix = self.semantic_attributes
        else:
            # Handle MATLAB cell array structure
            attr_matrix = np.vstack([np.array(attr).flatten() for attr in self.semantic_attributes])
        
        N_value = np.max(attr_matrix, axis=0)
        N_att = attr_matrix.shape[1]
        
        attrange = []
        for n in range(N_att):
            attrange.append(range(1, N_value[n] + 1))
            
        return N_value, attrange
    
    def calculate_attribute_probabilities(self, session_idx: int) -> List[np.ndarray]:
        """
        Calculate probability distribution for each attribute in a session
        
        Args:
            session_idx: Index of the session
            
        Returns:
            List of probability distributions for each attribute
            
        Note:
            Translated from the prob_att calculation in calc_lkh.m
        """
        if self.semantic_attributes is None:
            raise ValueError("Data not loaded. Call load_gazedata() first.")
        
        # Get attributes for the specific session
        session_attrs = self.semantic_attributes[session_idx]
        
        # Handle different data formats
        if session_attrs.ndim == 1:
            # Single attribute case
            n_attributes = 1
            session_attrs = session_attrs.reshape(-1, 1)
        else:
            n_attributes = session_attrs.shape[1]
        
        # Calculate value ranges for this session
        prob_att = []
        for att_idx in range(n_attributes):
            attr_values = session_attrs[:, att_idx]
            
            # Get unique values and their range
            unique_vals = np.unique(attr_values)
            min_val, max_val = int(np.min(unique_vals)), int(np.max(unique_vals))
            
            # Create histogram
            hist_counts = np.histogram(attr_values, 
                                     bins=max_val - min_val + 1,
                                     range=(min_val - 0.5, max_val + 0.5))[0]
            
            # Convert to probability distribution
            prob_dist = hist_counts / np.sum(hist_counts) if np.sum(hist_counts) > 0 else np.ones_like(hist_counts) / len(hist_counts)
            prob_att.append(prob_dist)
            
        return prob_att


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
            for t in range(self.win_size, T):
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


def save_likelihood_results(likelihood_data: Dict, filename: str = "lkh_win_5.npz"):
    """
    Save likelihood results to file
    
    Args:
        likelihood_data: Dictionary containing likelihood results
        filename: Output filename
        
    Note:
        Equivalent to save('lkh_win_5', 'lkh_ses') in MATLAB
    """
    np.savez(filename, **likelihood_data)


def load_likelihood_results(filename: str = "lkh_win_5.npz") -> Dict:
    """
    Load likelihood results from file
    
    Args:
        filename: Input filename
        
    Returns:
        Dictionary containing likelihood results
    """
    return dict(np.load(filename, allow_pickle=True))
