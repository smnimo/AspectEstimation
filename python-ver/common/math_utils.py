"""
Numerical helpers shared by withMSET and withLayout (normalize.m of both MATLAB versions)
"""

from typing import Optional

import numpy as np


def normalize(N: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
    """
    Normalize an array so that it sums to 1 (normalize.m)

    Args:
        N: Input array
        dim: None (whole array), 0 (each column; MATLAB dim = 1) or 1 (each row; MATLAB dim = 2)

    Returns:
        Normalized array; slices that sum to 0 are left unchanged (division by 1), as in normalize.m
    """
    N = np.asarray(N, dtype=float)
    if dim is None:
        z = np.sum(N)
        return N / (z + (z == 0))
    z = np.sum(N, axis=dim, keepdims=True)
    return N / (z + (z == 0))
