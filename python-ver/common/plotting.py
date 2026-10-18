"""
Figures shared by withMSET and withLayout
"""

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle


def plot_matrix_circles(N: np.ndarray, row_labels: Sequence[str], col_labels: Optional[Sequence] = None,
                        relative: bool = False, title: str = '') -> plt.Figure:
    """
    Non-negative matrix as circles (aspects: rows = attribute values, columns = aspects)

    relative = False (corrplot of plot_aspect.R, withMSET): colors red-white-blue over [-1, 1] and circle area
    proportional to the value on an absolute scale. relative = True (figures of the 2014 paper): white-blue over
    [0, max] and circle area proportional to value / max. The color bar shows [min, max] or [0, max].

    Args:
        N: (rows x columns) matrix
        row_labels: label of each row
        col_labels: label of each column (default 1, 2, ...)
        relative: scale of colors and circles (see above)
    """
    N = np.asarray(N, dtype=float)
    rows, cols = N.shape
    if relative:
        cmap = LinearSegmentedColormap.from_list('wb', ['white', 'blue'], N=200)
        vmax = N.max() if N.max() > 0 else 1.0
        norm = plt.Normalize(0, vmax)
        radius = lambda v: 0.45 * np.sqrt(max(v, 0) / vmax)  # noqa: E731
        bar_cmap, bar_norm = cmap, norm
    else:
        cmap = LinearSegmentedColormap.from_list('rwb', ['red', 'white', 'blue'], N=200)
        norm = plt.Normalize(-1, 1)
        radius = lambda v: 0.45 * np.sqrt(min(abs(v), 1))  # noqa: E731
        vmin, vmax = N.min(), N.max()
        bar_cmap = LinearSegmentedColormap.from_list('cl', [cmap(norm(vmin)), cmap(norm(vmax))])
        bar_norm = plt.Normalize(vmin, vmax)
    fig, ax = plt.subplots(figsize=(2.5 + 0.6 * cols, 0.5 * rows + 1))
    for (i, j), v in np.ndenumerate(N):
        ax.add_patch(Circle((j, i), radius(v), facecolor=cmap(norm(v)), edgecolor='none'))
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='lightgray')
    ax.tick_params(which='minor', length=0)
    ax.set_xticks(range(cols))
    ax.set_xticklabels(col_labels if col_labels is not None else range(1, cols + 1))
    ax.xaxis.tick_top()
    ax.set_yticks(range(rows))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    fig.colorbar(plt.cm.ScalarMappable(cmap=bar_cmap, norm=bar_norm), ax=ax)
    fig.tight_layout()
    return fig


def grouped_bars(ax, Y: np.ndarray, group_labels: Sequence[str], bar_labels: Sequence[str]):
    """MATLAB bar(Y): one group per row of Y, one bar per column"""
    Y = np.asarray(Y, dtype=float)
    n_group, n_bar = Y.shape
    width = 0.8 / n_bar
    for j in range(n_bar):
        ax.bar(np.arange(n_group) + (j - (n_bar - 1) / 2) * width, Y[:, j], width, label=bar_labels[j])
    ax.set_xticks(range(n_group))
    ax.set_xticklabels(group_labels)


def plot_likelihood_convergence(histories: Dict[int, Sequence[float]], title: str = '') -> plt.Figure:
    """Log-likelihood over EM iterations for each number of aspects"""
    fig, ax = plt.subplots(figsize=(6, 4))
    for K, L in sorted(histories.items()):
        ax.plot(np.arange(1, len(L) + 1), L, label=f'K = {K}')
    ax.set_xlabel('EM iteration')
    ax.set_ylabel('log-likelihood')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
