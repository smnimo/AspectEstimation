"""
Figures of Shimonishi et al. (2014): Fig. 5 (gaze regions), Fig. 6 (mixture ratio alpha), Fig. 7 (items of interest),
Fig. 8 (learned aspects), Fig. 9 (estimated interests). Call matplotlib.use('Agg') first without a display.
"""

import os
import sys
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from common.plotting import grouped_bars, plot_likelihood_convergence, plot_matrix_circles  # noqa: E402,F401

PANELS = {1: 'SA', 2: 'AA', 3: 'SA+AA'}


def plot_precision(bars: Dict[int, Dict[str, Sequence[float]]], ratios: Sequence[int], title: str = '') -> plt.Figure:
    """
    Precision of the first prediction (P@1) for each attribute type (paper Fig. 5 / Fig. 7)

    Args:
        bars: bars[attr][label] = precision at each ratio, e.g. label 'aspects: 6, alpha: 0', 'comparison'
        ratios: ratios (%) of the session used to estimate the interests (x axis)
    """
    attrs = sorted(bars)
    fig, axes = plt.subplots(1, len(attrs), figsize=(4 * len(attrs) + 2, 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, attr in zip(axes, attrs):
        labels = list(bars[attr])
        grouped_bars(ax, np.array([bars[attr][label] for label in labels]).T, [str(r) for r in ratios], labels)
        ax.set_xlabel('ratio (%)')
        ax.set_title(PANELS[attr])
        ax.set_ylim(0, 0.7)
    axes[0].set_ylabel('precision')
    axes[-1].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1))
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_alpha(alphas: Sequence[float], curves: Dict[int, Sequence[float]]) -> plt.Figure:
    """
    Precision of the predicted gaze region vs mixture ratio alpha, one line per ratio (paper Fig. 6)

    Args:
        curves: curves[ratio %] = precision for each alpha
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for ratio, values in sorted(curves.items()):
        ax.plot(alphas, values, 'o-', label=str(ratio))
    ax.set_xlabel('alpha')
    ax.set_ylabel('precision')
    ax.legend(title='ratio (%)', fontsize=7)
    fig.tight_layout()
    return fig


def plot_aspects(vz: np.ndarray, labels: Sequence[str], title: str = '') -> plt.Figure:
    """Learned aspects P(V | Z), rows = attribute values, columns = aspects (paper Fig. 8)"""
    return plot_matrix_circles(np.asarray(vz).T, labels, relative=True, title=title)


def plot_interests(zs_by_task: Dict[int, np.ndarray]) -> plt.Figure:
    """
    Estimated interests theta(s) of sessions (paper Fig. 9: sessions of content 1, one panel per task)

    Args:
        zs_by_task: zs_by_task[task] = (sessions x K) interests
    """
    tasks = sorted(zs_by_task)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 4))
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        zs = np.asarray(zs_by_task[task])
        ax.imshow(zs.T, cmap='Blues', vmin=0, aspect='auto', interpolation='nearest')
        ax.set_xticks(range(zs.shape[0]))
        ax.set_xticklabels(range(1, zs.shape[0] + 1))
        ax.set_yticks(range(zs.shape[1]))
        ax.set_yticklabels(range(1, zs.shape[1] + 1))
        ax.set_xlabel('session')
        ax.set_title(f'Task {task}')
    axes[0].set_ylabel('aspect')
    fig.tight_layout()
    return fig
