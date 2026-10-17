"""
Segment calculation and mask generation for the four-layer model
Translated from calc_segments.m and related functions
"""

import numpy as np
from typing import List, Dict, Tuple
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.math_utils import normalize


class SegmentCalculator:
    """
    Calculate segments and masks for the four-layer pLSA model
    
    Translated from calc_segments.m
    """
    
    def __init__(self, win_size: int = 5, threshold: float = 0.03):
        """
        Initialize segment calculator
        
        Args:
            win_size: Window size for likelihood calculation
            threshold: Threshold for segment detection
        """
        self.win_size = win_size
        self.threshold = threshold
    
    def calc_segments(self, lkh_ses: List[Dict], semantic_attributes: List, 
                     semantic_attribute_seq: List, win_size: int, 
                     threshold: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Calculate segments and sequence masks
        
        Args:
            lkh_ses: Likelihood values for each session and attribute
            semantic_attributes: Semantic attributes for each session
            semantic_attribute_seq: Sequence of semantic attributes
            win_size: Window size
            threshold: Threshold for detection
            
        Returns:
            Tuple of (segment labels, sequence masks)
            
        Note:
            Translated from calc_segments.m
        """
        S = len(semantic_attribute_seq)
        seglabel = []
        seqmask = []
        
        # Attribute value ranges from the catalogs of all sessions, as calc_lkh.m and the fixed calc_segments.m
        # (the 2019 calc_segments.m used the browsed values of session 1; identical on the paper data)
        if S == 0:
            raise ValueError("no session")
        N_value = np.max(np.vstack(semantic_attributes), axis=0)
        N_att = len(N_value)
            
        attrange = []
        for n in range(N_att):
            attrange.append(list(range(1, N_value[n] + 1)))
        
        for s in range(S):
            attseq = semantic_attribute_seq[s]
            
            # Calculate attribute probabilities for this session
            prob_att = self._calculate_attribute_probabilities(
                semantic_attributes[s], attrange, N_att
            )
            
            # Get session length
            T = attseq.shape[1]  # session length
            
            # Likelihood of this session: {attribute: array (T,)} (main.calc_lkh)
            session_lkh = lkh_ses[s]
            
            looked_att = []
            
            for att in range(N_att):
                valseq = attseq[att, :]
                looked_value = np.zeros((N_value[att], T))
                
                lkh_values = session_lkh[att]
                
                # Find time points where likelihood is below threshold
                tmp = np.where(lkh_values < threshold)[0]
                
                for i in range(len(tmp)):
                    x1 = max(0, tmp[i] - win_size + 1)
                    x2 = tmp[i] + 1
                    
                    # Calculate value flag
                    if x2 <= len(valseq):
                        window_vals = valseq[x1:x2]
                        hist_counts = np.histogram(window_vals, 
                                                 bins=len(prob_att[att]),
                                                 range=(1, len(prob_att[att]) + 1))[0]
                        normalized_hist = normalize(hist_counts.astype(float))
                        # paper: x / l > phi (calc_segments.m used >=; identical when phi = 1/3)
                        valflag = (normalized_hist > prob_att[att])
                        
                        for v_idx, v in enumerate(attrange[att]):
                            if v_idx < len(valflag) and valflag[v_idx]:
                                looked_value[v-1, x1:x2] = 1
                
                looked_att.append(looked_value)
            
            # Combine all attributes
            looked_att_combined = np.vstack(looked_att)
            sl = np.sum(looked_att_combined, axis=0)
            
            # Create segment labels
            seg_labels = np.zeros(T, dtype=int)
            if sl[0] == 0:
                seg_labels[0] = 0
            else:
                seg_labels[0] = 1
            
            segc = seg_labels[0]
            for t in range(1, T):
                if sl[t] == 0:
                    seg_labels[t] = 0
                elif np.sum(np.abs(looked_att_combined[:, t] - looked_att_combined[:, t-1])) > 0:
                    segc += 1
                    seg_labels[t] = segc
                else:
                    seg_labels[t] = segc
            
            seglabel.append(seg_labels)
            seqmask.append(looked_att_combined)
        
        return seglabel, seqmask
    
    def _calculate_attribute_probabilities(self, semantic_attrs: np.ndarray, 
                                         attrange: List[List[int]], 
                                         N_att: int) -> List[np.ndarray]:
        """
        Calculate probability distribution for each attribute
        
        Args:
            semantic_attrs: Semantic attributes matrix
            attrange: Attribute value ranges
            N_att: Number of attributes
            
        Returns:
            List of probability distributions
        """
        prob_att = []
        
        for att in range(N_att):
            attr_values = semantic_attrs[:, att]
            hist_counts = np.histogram(attr_values, 
                                     bins=len(attrange[att]),
                                     range=(attrange[att][0], attrange[att][-1] + 1))[0]
            prob_dist = hist_counts / np.sum(hist_counts) if np.sum(hist_counts) > 0 else np.ones_like(hist_counts) / len(hist_counts)
            prob_att.append(prob_dist)
        
        return prob_att


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
            seqmask: List of AOF masks (features x time), as returned by calc_segments

        Returns:
            List of masks containing only the columns with at least one AOF

        Note:
            Translated from seqmask{s}(:, sum(seqmask{s})==0) = [] in learn_aspects_from_mask.m.
            Must be applied before training; an all-zero column has probability 1 under
            every topic and would only distort P(z|s).
        """
        return [mask[:, np.sum(mask, axis=0) > 0] for mask in seqmask]
