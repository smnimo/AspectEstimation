"""
Utility functions for AspectEstimation
Translated from MATLAB code
"""

import numpy as np
from typing import Optional


def normalize(N: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
    """
    Normalize matrix along specified dimension
    
    Args:
        N: Input matrix
        dim: Dimension to normalize along (0 for columns, 1 for rows, None for all)
        
    Returns:
        Normalized matrix
        
    Note:
        Translated from normalize.m
        dim=0 corresponds to MATLAB dim=1 (normalize columns)
        dim=1 corresponds to MATLAB dim=2 (normalize rows)
    """
    if dim is None:
        # Normalize entire array to sum to 1
        z = np.sum(N)
        s = z + (z == 0)
        return N / s
    elif dim == 0:
        # Normalize columns (each column sums to 1)
        z = np.sum(N, axis=0)
        s = z + (z == 0)
        return N / s[np.newaxis, :]
    elif dim == 1:
        # Normalize rows (each row sums to 1) 
        z = np.sum(N, axis=1)
        s = z + (z == 0)
        return N / s[:, np.newaxis]
    else:
        # General case for higher dimensions
        z = np.sum(N, axis=dim)
        s = z + (z == 0)
        # Create shape for broadcasting
        shape = list(N.shape)
        shape[dim] = 1
        s_reshaped = s.reshape(shape)
        return N / s_reshaped


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


def homogenous_product(base: int, length: int) -> int:
    """
    Calculate the number of ways to distribute 'length' identical items 
    into 'base' distinct groups (including empty groups)
    
    This is equivalent to MATLAB's HomogenousProduct function
    
    Args:
        base: Number of distinct groups
        length: Total number of items to distribute
        
    Returns:
        Number of possible distributions
        
    Note:
        This is a combinatorial function: C(length + base - 1, base - 1)
    """
    from math import comb
    return comb(length + base - 1, base - 1)
