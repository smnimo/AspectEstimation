"""
Session matrices for the four-layer model (learn_aspects_from_mask.m); the AOF masks are made by models.aof
"""

import numpy as np
from typing import List, Dict, Tuple
import sys
import os

# normalize is shared with withLayout (python-ver/common)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from common.math_utils import normalize  # noqa: E402,F401


class SessionProcessor:
    """
    Process sessions for the four-layer model
    
    Translated from learn_aspects_from_mask.m session processing
    """
    
    def __init__(self):
        """Initialize session processor"""
        pass
    
    def create_session_all(self, semantic_attribute_seq: List[np.ndarray], 
                          N_value: np.ndarray) -> List[np.ndarray]:
        """
        Create concatenated session data with proper vocabulary mapping
        
        Args:
            semantic_attribute_seq: List of attribute sequences per session
            N_value: Maximum values for each attribute
            
        Returns:
            List of session matrices with unified vocabulary space
            
        Note:
            Translated from the session_all creation in learn_aspects_from_mask.m:
            value v of attribute a -> feature sum(N_value[:a]) + v (1-based), i.e. offsets cumsum(N_value)
            (the 2019 script hard-coded +3, +6, +9, +12 for 5 attributes x 3 values)
        """
        N_value = np.asarray(N_value, dtype=int)
        offset = np.concatenate([[0], np.cumsum(N_value)[:-1]])
        vocab_size = int(np.sum(N_value))
        session_all = []

        for seq in semantic_attribute_seq:
            seq = np.asarray(seq, dtype=int)  # (attributes x timepoints)
            if seq.shape[0] != len(N_value) or np.any((seq < 1) | (seq > N_value[:, None])):
                raise ValueError(f"attribute values outside 1..N_value = {N_value}")
            onehot = np.zeros((vocab_size, seq.shape[1]))
            onehot[seq + offset[:, None] - 1, np.arange(seq.shape[1])[None, :]] = 1
            session_all.append(onehot)

        return session_all
    
    def remove_empty_columns(self, seqmask: List[np.ndarray]) -> List[np.ndarray]:
        """
        Remove time points without any detected AOF from each session mask

        Args:
            seqmask: List of AOF masks (features x time), as returned by models.aof.session_aof

        Returns:
            List of masks containing only the columns with at least one AOF

        Note:
            Translated from seqmask{s}(:, sum(seqmask{s})==0) = [] in learn_aspects_from_mask.m.
            Must be applied before training; an all-zero column has probability 1 under
            every topic and would only distort P(z|s).
        """
        return [mask[:, np.sum(mask, axis=0) > 0] for mask in seqmask]
