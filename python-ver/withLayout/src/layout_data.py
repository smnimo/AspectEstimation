"""
Data of the 2014 experiment and the observation model P(R|V) with the layout effect

Translated from item_attr.m, reg_attr.m (type 1 = sum, paper Eq. 4 and 10), make_area.m (paper Eq. 11) and
the session loading part of main.m. Layout: 3 x 4 grid of 12 items; regions and items are indexed 1..12 in the
data (row-major) and 0..11 here.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.io

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from common.math_utils import normalize  # noqa: E402

N_REGION = 12
ATTRIBUTE_TYPES = {1: 'SA', 2: 'AA', 3: 'SA+AA'}  # semantic, appearance, both (paper 4.2)
# category order as in Fig. 8 of the paper (content.mat does not name its columns)
SEMANTIC_LABELS = (['Rice', 'Meat', 'Fish', 'Vegetable', 'Pasta'] +
                   ['Calorie Low', 'Calorie Mid_low', 'Calorie Mid_high', 'Calorie High'] +
                   [f'{i} steps' for i in range(1, 6)])


@dataclass
class Session:
    """One browsing session (data/mat/<name>.mat)"""
    name: str
    f: np.ndarray          # gazed regions 1..12, one per gaze-target transition
    index: np.ndarray      # list.index: content group of each item (1..4)
    num: np.ndarray        # list.num: item number in its group (1..15)
    answer: np.ndarray     # selected items (1..12)
    content_id: int
    task_id: int


def load_content(path: str) -> Dict[str, list]:
    """content.mat (makecontent.m): per content group, item attributes (category, calorie, step, colhist)"""
    c = scipy.io.loadmat(path)['content'][0, 0]
    out = {name: [np.asarray(g, dtype=float) for g in c[name].ravel()] for name in ('category', 'calorie', 'step')}
    out['colhist'] = [[np.asarray(item, dtype=float) for item in g.ravel()] for g in c['colhist'].ravel()]
    return out


def load_sessions(data_dir: str, timestamp_list: Optional[str] = None) -> List[Session]:
    """Sessions listed in experiment/timestamp.list, read from mat/<name>.mat (main.m)"""
    timestamp_list = timestamp_list or os.path.join(data_dir, 'experiment', 'timestamp.list')
    with open(timestamp_list) as fp:
        names = [line.strip() for line in fp if line.strip()]
    sessions = []
    for name in names:
        d = scipy.io.loadmat(os.path.join(data_dir, 'mat', f'{name}.mat'))
        lst = d['list'][0, 0]
        sessions.append(Session(name=name, f=d['f'][:, 0].astype(int),
                                index=np.asarray(lst['index']).ravel().astype(int),
                                num=np.asarray(lst['num']).ravel().astype(int),
                                answer=d['answer'].ravel().astype(int),
                                content_id=int(d['content_id'].ravel()[0]), task_id=int(d['task_id'].ravel()[0])))
    return sessions


def attribute_labels(attr: int, n_color: int = 16) -> List[str]:
    """Labels of the attribute values of an attribute type (1: SA, 2: AA, 3: SA+AA)"""
    color = [f'color {i}' for i in range(1, n_color + 1)]
    return {1: SEMANTIC_LABELS, 2: color, 3: SEMANTIC_LABELS + color}[attr]


def item_attributes(content: Dict[str, list], session: Session, attr: int) -> np.ndarray:
    """
    Attribute values of the 12 displayed items (rows of item_attr.m / reg_attr.m / makeattr.m)

    Returns:
        (12 x Q) 0/1 matrix; attr 1: category + calorie + steps, 2: color class, 3: both
    """
    rows = []
    for g, n in zip(session.index - 1, session.num - 1):
        semantic = np.concatenate([content['category'][g][n], content['calorie'][g][n], content['step'][g][n]])
        color = content['colhist'][g][n].sum(axis=0)
        rows.append({1: semantic, 2: color, 3: np.concatenate([semantic, color])}[attr])
    return np.vstack(rows)


def item_given_value(pvi: np.ndarray) -> np.ndarray:
    """
    P(I_n | V_q) = 1 / n_q for the n_q items with value V_q (paper Eq. 10; item_attr.m)

    A value that no displayed item has gets a uniform distribution over the items.

    Returns:
        (Q x 12) matrix, rows sum to 1
    """
    pvi = np.array(pvi, dtype=float)
    pvi[:, pvi.sum(axis=0) == 0] = 1
    return normalize(pvi.T, 1)


def region_given_value(pvi: np.ndarray, layout: np.ndarray, alpha: float) -> np.ndarray:
    """
    P(R_n | V_q) = (1 - alpha) P(I_n | V_q) + alpha P(R_n; phi) (paper Eq. 4; reg_attr.m, type 1 = sum)

    Args:
        pvi: (12 x Q) item attributes
        layout: (12,) layout effect P(R; phi) (normalized here)
        alpha: mixture ratio

    Returns:
        (Q x 12) matrix, rows sum to 1
    """
    effect = np.tile(normalize(np.asarray(layout, dtype=float)), (pvi.shape[1], 1))
    return normalize((1 - alpha) * item_given_value(pvi) + alpha * effect, 1)


def region_histogram(f: np.ndarray) -> np.ndarray:
    """Number of gazes on each region 1..12 (hist(f, 1:12))"""
    return np.bincount(np.asarray(f, dtype=int) - 1, minlength=N_REGION)[:N_REGION].astype(float)


def region_frequency(sessions: Sequence[Session], exclude: Optional[int] = None) -> np.ndarray:
    """
    Relative gaze frequency of each region over the sessions (paper Eq. 11; make_area.m)

    Args:
        exclude: index of a session left out (leave-one-out evaluation: area_s(:, s) of make_area.m)
    """
    counts = sum(region_histogram(s.f) for i, s in enumerate(sessions) if i != exclude)
    return normalize(counts)
