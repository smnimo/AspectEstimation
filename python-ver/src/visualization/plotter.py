"""
Figures of the MATLAB version (plot_*.m, disp_mask_for_paper.m, plot_aspect.R)

Each function draws what one MATLAB script draws, from data computed elsewhere
(main.learn_aspects, models.evaluation), and returns the matplotlib Figure.
Call matplotlib.use('Agg') before importing this module when no display is available.
"""

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

ATTRIBUTES = ['price', 'screen', 'CPU', 'memory', 'weight']


def plot_L_topic(L_all: Sequence[float], L_spe: Sequence[float]) -> plt.Figure:
    """Final log-likelihood vs number of aspects (plot_L_topic.m, figure 1)"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, L, title in zip(axes, (L_all, L_spe), ('all', 'MSET')):
        ax.plot(np.arange(1, len(L) + 1), L, 'o-')
        ax.set_title(title)
        ax.set_xlabel('number of aspects')
        ax.set_ylabel('log-likelihood')
    fig.tight_layout()
    return fig


def plot_aspects(aspects: Dict[int, np.ndarray], giventask: np.ndarray, name: str) -> plt.Figure:
    """
    Learned aspects for each number of aspects and the given tasks (plot_L_topic.m, figures 'all' / 'MSET')

    Args:
        aspects: aspects[K] = vz (15 x K), as cell2mat(p.vz)
        giventask: (n_task x 15), drawn transposed in the last panel
        name: figure title ('all' or 'MSET')
    """
    Ks = sorted(aspects)
    fig, axes = plt.subplots(1, len(Ks) + 1, figsize=(2 * (len(Ks) + 1), 4))
    for ax, K in zip(axes, Ks):
        ax.imshow(aspects[K], cmap='gray', aspect='auto', interpolation='nearest')  # imagesc: per-panel scaling
        ax.set_title(f'K = {K}')
        ax.set_xticks(range(K))
        ax.set_xticklabels(range(1, K + 1))
    axes[-1].imshow(giventask.T, cmap='gray', aspect='auto', interpolation='nearest')
    axes[-1].set_title('given task')
    axes[-1].set_xticks(range(giventask.shape[0]))
    axes[-1].set_xticklabels(range(1, giventask.shape[0] + 1))
    for ax in axes:
        ax.set_yticks(np.arange(1, 15, 3))
        ax.set_yticklabels(ATTRIBUTES)
    fig.suptitle(name)
    fig.tight_layout()
    return fig


def plot_conf_matrix(conf: np.ndarray, name: str) -> plt.Figure:
    """Task estimation confusion matrix (plot_conf_matrix.m); last column = no comparison"""
    n_task = conf.shape[0]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(conf, cmap='hot', interpolation='nearest')
    for (i, j), v in np.ndenumerate(conf):
        ax.text(j, i, int(v), ha='center', va='center', color='gray')
    ax.set_xticks(range(n_task + 1))
    ax.set_xticklabels([str(t + 1) for t in range(n_task)] + ['no comp.'])
    ax.set_yticks(range(n_task))
    ax.set_yticklabels([str(t + 1) for t in range(n_task)])
    ax.set_xlabel('estimated task')
    ax.set_ylabel('given task')
    ax.set_title(name)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _grouped_bars(ax, Y: np.ndarray, group_labels: List[str], bar_labels: List[str]):
    """MATLAB bar(Y): one group per row of Y, one bar per column"""
    n_group, n_bar = Y.shape
    width = 0.8 / n_bar
    for j in range(n_bar):
        ax.bar(np.arange(n_group) + (j - (n_bar - 1) / 2) * width, Y[:, j], width, label=bar_labels[j])
    ax.set_xticks(range(n_group))
    ax.set_xticklabels(group_labels)


def plot_sim(cor_spe: np.ndarray, cor_all: np.ndarray) -> plt.Figure:
    """
    Correlation between the given tasks and the learned aspects (plot_sim.m, active figure, topic = 3)

    For each task, the best cosine similarity over the aspects; left: per task, right: mean with an
    error bar of the standard deviation (the 2019 plot_sim.m used the variance). The 'aspect to task'
    figure of plot_sim.m is commented out in MATLAB and not drawn.

    Args:
        cor_spe, cor_all: cor_task (n_task x K) of CalcEval for the models with / without MET
    """
    Y = np.vstack([cor_spe.max(axis=1), cor_all.max(axis=1)])  # [cortt; cortt_a]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _grouped_bars(ax1, Y, ['with MET', 'without MET'], [f'task {t + 1}' for t in range(Y.shape[1])])
    ax1.set_title('correlations between given task and learned aspects')
    ax1.set_ylabel('Cosine similarity')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax2.bar(range(2), Y.mean(axis=1))
    ax2.errorbar(range(2), Y.mean(axis=1), yerr=Y.std(axis=1, ddof=1), fmt='.', color='k')
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(['with MET', 'without MET'])
    ax2.set_title('mean of correlations')
    ax2.set_ylabel('Cosine similarity')
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def plot_sim_task_asp(cor_spe: np.ndarray, cor_all: np.ndarray) -> plt.Figure:
    """
    Cosine similarity between task k and the matched aspect k (plot_sim_task_asp.m: bar([diag(cor_all{3}), diag(cor{3})]'))
    """
    Y = np.vstack([np.diag(cor_all), np.diag(cor_spe)])
    fig, ax = plt.subplots(figsize=(6, 4))
    _grouped_bars(ax, Y, ['without MET (all)', 'with MET (spe)'], [f'task {t + 1}' for t in range(Y.shape[1])])
    ax.set_ylabel('Cosine similarity')
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_eval_aof(dists: Dict[str, np.ndarray]) -> plt.Figure:
    """
    Cosine similarity between the given task and the detected attributes-of-focus (plot_eval_AoF.m)

    Box plots of All / Top / Comparison / LD (= MSET) with means as diamonds; sessions without AOF
    (NaN for LD) are excluded from all four, as in MATLAB.

    Args:
        dists: output of evaluation.aof_similarities
    """
    keep = ~np.isnan(dists['cos_dist_mset'])
    data = [dists[k][keep] for k in ('cos_dist_all', 'cos_dist_top', 'cos_dist_c', 'cos_dist_mset')]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data)
    ax.plot(range(1, 5), [np.mean(d) for d in data], 'd')
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(['All', 'Top', 'Comparison', 'LD'])
    ax.set_ylabel('Cosine similarity')
    fig.tight_layout()
    return fig


def plot_dur_usr(dur: np.ndarray, usr_nocomp2: Sequence[int] = (28, 36),
                 usr_nocomp1: Sequence[int] = (1, 3, 6, 18, 23, 35, 37), ylim: float = 160) -> plt.Figure:
    """
    Session durations per participant, sorted by mean and by median (plot_dur_usr.m)

    Args:
        dur: (n_usr x 3) totalduration (s), columns = session_id (a box over the 3 sessions of a participant
             does not depend on the column order)
        usr_nocomp2, usr_nocomp1: participants highlighted in red / yellow (hard-coded lists of plot_dur_usr.m;
             their meaning is not documented in the MATLAB code)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for ax, stat, title in zip(axes, (np.mean, np.median), ('sort by mean', 'sort by median')):
        order = np.argsort(stat(dur, axis=1), kind='stable')  # MATLAB sort (ascending, stable)
        ax.boxplot([dur[i] for i in order])
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([str(i + 1) for i in order])
        for users, color in ((usr_nocomp2, 'r'), (usr_nocomp1, 'y')):
            for u in users:
                pos = int(np.nonzero(order == u - 1)[0][0]) + 1 if u - 1 in order else None
                if pos is not None:
                    ax.axvspan(pos - 0.5, pos + 0.5, color=color, alpha=0.3)
        ax.set_ylim(0, ylim)
        ax.set_ylabel('duration(s)')
        ax.set_xlabel('participant id')
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_org_gaze(gaze_data: np.ndarray, image: Optional[np.ndarray] = None,
                  image_size: Sequence[int] = (1920, 1080)) -> plt.Figure:
    """
    Raw gaze trajectory over the catalog screenshot (plot_org_gaze.m)

    Samples with x < 0 or y < 0 are dropped as in MATLAB; the eyes are averaged with the fixed
    missing-eye handling of data.gaze_reader.gaze_xy (MATLAB drops samples with a missing left eye).

    Args:
        gaze_data: (timepoints x 5) raw gaze CSV
        image: screenshot (H x W [x C]), e.g. matplotlib.pyplot.imread('<session>.jpg'); optional
        image_size: (width, height) used without an image
    """
    from data.gaze_reader import gaze_xy
    x, y = gaze_xy(gaze_data)
    keep = (x >= 0) & (y >= 0)
    width, height = (image.shape[1], image.shape[0]) if image is not None else image_size
    fig, ax = plt.subplots(figsize=(10, 10 * height / width))
    if image is not None:
        ax.imshow(image)
    ax.plot(x[keep] * width, y[keep] * height)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    fig.tight_layout()
    return fig


def plot_mask(session_all: np.ndarray, seqmask: np.ndarray, title: str = '') -> plt.Figure:
    """
    All browsed values (top) and AOF mask (bottom) of one session (disp_mask_for_paper.m)

    Gray rows (0.5) separate the 5 attributes.
    """
    def with_separators(m):
        rows = []
        for a in range(5):
            rows.append(m[a * 3:(a + 1) * 3])
            if a < 4:
                rows.append(0.5 * np.ones((1, m.shape[1])))
        return np.vstack(rows)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for ax, m, name in zip(axes, (session_all, seqmask), ('all values', 'attributes-of-focus')):
        # extent: time 1..T as in imagesc
        ax.imshow(with_separators(m), cmap='gray', aspect='auto', interpolation='nearest',
                  extent=(0.5, m.shape[1] + 0.5, 18.5, -0.5))
        ax.set_yticks(np.arange(1, 19, 4))
        ax.set_yticklabels(ATTRIBUTES)
        ax.set_title(name)
    axes[1].set_xlabel('time (browsed item)')
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_aspect_circles(vz: np.ndarray) -> plt.Figure:
    """
    Aspects as a circle matrix (plot_aspect.R: corrplot(method = 'circle', col = red-white-blue); paper Fig. 9)

    As corrplot with its default is.corr = TRUE: colors span [-1, 1] (non-negative values are white..blue),
    circle area is proportional to |value| on the same absolute scale, the color bar shows [min, max] (cl.lim).
    The values are plotted as they are, as in the paper (Fig. 9 color bars 0..0.2 / 0..0.33 = max of vz,
    refs/results/spe3.pdf). The vz * 2 of the hand-edited Akq_*.csv / *_2.pdf files (for visibility) is not used.

    Args:
        vz: (15 x K) aspects
    """
    N = np.asarray(vz, dtype=float)
    cmap = LinearSegmentedColormap.from_list('rwb', ['red', 'white', 'blue'], N=200)
    norm = plt.Normalize(-1, 1)
    rows, cols = N.shape
    fig, ax = plt.subplots(figsize=(2.5 + 0.6 * cols, 0.5 * rows + 1))
    for (i, j), v in np.ndenumerate(N):
        ax.add_patch(Circle((j, i), 0.45 * np.sqrt(min(abs(v), 1)), facecolor=cmap(norm(v)), edgecolor='none'))
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='lightgray')
    ax.tick_params(which='minor', length=0)
    ax.set_xticks(range(cols))
    ax.set_xticklabels(range(1, cols + 1))
    ax.xaxis.tick_top()
    ax.set_yticks(range(rows))
    ax.set_yticklabels([f'{ATTRIBUTES[i // 3]} {i % 3 + 1}' for i in range(rows)])
    vmin, vmax = N.min(), N.max()
    sm = plt.cm.ScalarMappable(cmap=LinearSegmentedColormap.from_list(
        'cl', [cmap(norm(vmin)), cmap(norm(vmax))]), norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=ax)
    fig.tight_layout()
    return fig


def plot_likelihood_convergence(histories: Dict[int, Sequence[float]], title: str = '') -> plt.Figure:
    """Log-likelihood over EM iterations for each number of aspects (no MATLAB counterpart)"""
    fig, ax = plt.subplots(figsize=(6, 4))
    for K, L in sorted(histories.items()):
        ax.plot(np.arange(1, len(L) + 1), L, label=f'K = {K}')
    ax.set_xlabel('EM iteration')
    ax.set_ylabel('log-likelihood')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
