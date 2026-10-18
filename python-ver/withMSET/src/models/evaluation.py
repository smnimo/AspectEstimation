"""
Evaluation of learned aspects against the given tasks

Translated from matlab-ver/calc_matching.m, CalcEval.m, CalcSim.m and plot_conf_matrix.m
(paper: Eq. 4, Fig. 8, Table 2).

Session order is MATLAB's cell(:) order (column-major): the sessions of task k are
rows (k-1)*n_usr : k*n_usr of p.zs.
"""

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np

# task-related attribute values (paper Table 1), index = attribute*3 + value-1,
# attributes: price, screen size, CPU score, memory capacity, weight
# (identical to giventask in task_att.mat)
GIVENTASK = np.zeros((3, 15))
GIVENTASK[0, [8, 11]] = 1   # task 1: CPU = 3, memory = 3
GIVENTASK[1, [3, 12]] = 1   # task 2: screen = 1, weight = 1
GIVENTASK[2, [0, 11]] = 1   # task 3: price = 1, memory = 3


def load_giventask(path: Optional[str] = None) -> np.ndarray:
    """giventask (n_task x N) from task_att.mat, or the built-in GIVENTASK if path is None"""
    if path is None:
        return GIVENTASK.copy()
    import scipy.io
    return np.asarray(scipy.io.loadmat(path)['giventask'], dtype=float)


def load_matlab_result(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load p_{spe,all}_topicK.mat saved by learn_aspects_from_mask.m

    Returns:
        (zs, vz, L): zs is (S x K), vz is (N x K) as cell2mat(p.vz), L the log-likelihood history
    """
    import scipy.io
    d = scipy.io.loadmat(path)
    p = d['p'][0, 0]
    vz = np.hstack([np.asarray(c, dtype=float) for c in np.asarray(p['vz']).ravel()])
    return np.asarray(p['zs'], dtype=float), vz, np.asarray(d['L'], dtype=float).ravel()


def cosine_task_aspect(giventask: np.ndarray, vz: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between the given tasks and the learned aspects (CalcEval.m)

    1 - squareform(pdist([normalize(giventask, 2); vz'], 'cosine')), rows: tasks, columns: aspects

    Args:
        giventask: (n_task x N)
        vz: (N x K) aspects P(v|z) as cell2mat(p.vz)

    Returns:
        (n_task x K) matrix
    """
    g = giventask / giventask.sum(axis=1, keepdims=True)
    a = np.asarray(vz, dtype=float).T
    dist = 1 - (g @ a.T) / (np.linalg.norm(g, axis=1)[:, None] * np.linalg.norm(a, axis=1)[None, :])
    return 1 - dist


def match_aspects(vz: np.ndarray, zs: np.ndarray,
                  giventask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reorder the aspects so that aspect k corresponds to task k (calc_matching.m)

    The permutation maximizes the sum of the cosine similarities between task k and aspect k
    over the first min(n_task, K) aspects. Permutations are visited in lexicographic order
    (sortrows(perms(1:K)) in the fixed calc_matching.m) and the first maximum is taken, so an
    already matched result keeps its order (idempotent). The 2019 calc_matching.m used perms order
    (reverse lexicographic), which differs only in ties among the unmatched aspects (K > n_task).

    Returns:
        (vz, zs, perm): permuted vz (N x K), zs (S x K), and perm (0-based)
    """
    K = vz.shape[1]
    cos = cosine_task_aspect(giventask, vz)
    k = min(giventask.shape[0], K)
    best, best_score = None, -np.inf
    for perm in itertools.permutations(range(K)):  # lexicographic
        score = np.sum(np.diag(cos[:, list(perm)])[:k])
        if score > best_score:
            best, best_score = np.array(perm), score
    return vz[:, best], zs[:, best], best


def calc_eval(vz: np.ndarray, zs: np.ndarray, tasks: np.ndarray,
              giventask: np.ndarray, no_aof: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Similarity between the task of each session and its top aspect (CalcEval.m)

    Args:
        vz: (N x K) aspects, zs: (S x K) interests, tasks: (S,) task of each session (1-based)
        no_aof: (S,) bool, sessions without AOF (model 'spe'); their cortop is NaN as in the fixed CalcEval.m
            (the 2019 CalcEval.m used aspect 1 of their uniform zs)

    Returns:
        (cor_task, cortop): cor_task (n_task x K), cortop (S,) = cor_task[task(s), argmax_r zs(s, r)]
    """
    cor_task = cosine_task_aspect(giventask, vz)
    cortop = cor_task[np.asarray(tasks) - 1, np.argmax(zs, axis=1)]
    if no_aof is not None:
        cortop = np.where(no_aof, np.nan, cortop)
    return cor_task, cortop


def confusion_matrix(zs: np.ndarray, n_usr: int, n_task: int,
                     no_aof: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """
    Task estimation by argmax zs (plot_conf_matrix.m)

    Sessions without AOF are counted in the last column ("no comparison"): given by no_aof (fixed
    plot_conf_matrix.m), or, if no_aof is None, detected by max zs == 1/n_task exactly (their zs is
    uniform; rule of the 2019 plot_conf_matrix.m). Both agree on the paper data.

    Returns:
        (conf, accuracy): conf (n_task x n_task+1) counts, accuracy = trace / number of sessions
    """
    conf = np.zeros((n_task, n_task + 1), dtype=int)
    for task in range(n_task):
        z = zs[task * n_usr:(task + 1) * n_usr]
        est = np.argmax(z, axis=1)
        if no_aof is None:
            est[np.max(z, axis=1) == 1 / n_task] = n_task
        else:
            est[np.asarray(no_aof)[task * n_usr:(task + 1) * n_usr]] = n_task
        conf[task] = np.bincount(est, minlength=n_task + 1)
    return conf, np.trace(conf[:, :n_task]) / conf.sum()


def session_tasks(n_sessions: int, n_task: int) -> np.ndarray:
    """Task of each session in column-major order (session{s}.task in MATLAB)"""
    if n_sessions % n_task:
        raise ValueError(f"{n_sessions} sessions cannot be split into {n_task} tasks")
    return np.repeat(np.arange(1, n_task + 1), n_sessions // n_task)


def evaluate(models: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
             giventask: np.ndarray, conf_topic: int = 3, no_aof: Optional[np.ndarray] = None) -> Dict:
    """
    Matching + similarity + confusion matrix for all learned models (calc_matching -> CalcSim -> plot_conf_matrix)

    Args:
        models: models[tag][K] = (vz (N x K), zs (S x K)), tag in {'spe', 'all'}
        giventask: (n_task x N)
        conf_topic: number of aspects used for the confusion matrix (MATLAB: 3)
        no_aof: (S,) bool, sessions without AOF; used for the 'spe' models (cortop NaN, "no comparison").
            If None, they are detected by their uniform zs in a 'spe' model with K >= 2 (same sessions on the
            paper data; with K = 1 every zs is 1).

    Returns:
        out[tag][K] = {'vz', 'zs', 'perm', 'cor_task', 'cortop'} (matched),
        plus 'conf' and 'accuracy' for K == conf_topic
    """
    n_task = giventask.shape[0]
    if no_aof is None and any(K >= 2 for K in models.get('spe', {})):
        K = min(k for k in models['spe'] if k >= 2)
        no_aof = np.all(np.isclose(models['spe'][K][1], 1 / K, rtol=0, atol=1e-15), axis=1)
    out = {}
    for tag, by_topic in models.items():
        out[tag] = {}
        for K, (vz, zs) in sorted(by_topic.items()):
            S = zs.shape[0]
            tasks = session_tasks(S, n_task)
            if K >= 2:  # calc_matching.m: topic = 2:5
                vz, zs, perm = match_aspects(vz, zs, giventask)
            else:
                perm = np.arange(K)
            skip = no_aof if tag == 'spe' else None  # 'all': every session has observations
            cor_task, cortop = calc_eval(vz, zs, tasks, giventask, skip)
            res = {'vz': vz, 'zs': zs, 'perm': perm, 'cor_task': cor_task, 'cortop': cortop}
            if K == conf_topic:
                res['conf'], res['accuracy'] = confusion_matrix(
                    zs, S // n_task, n_task, skip if skip is not None else np.zeros(S, dtype=bool))
            out[tag][K] = res
    return out


def format_summary(evaluation: Dict, n_task: int = 3) -> List[str]:
    """Human-readable summary lines of evaluate()"""
    lines = []
    for tag, by_topic in evaluation.items():
        for K, r in sorted(by_topic.items()):
            k = min(n_task, K)
            lines.append(f"[{tag}] topic {K}: cos(task k, aspect k) = "
                         f"{np.round(np.diag(r['cor_task'])[:k], 3).tolist()}, "
                         f"mean cortop = {np.nanmean(r['cortop']):.3f}")
            if 'conf' in r:
                lines.append(f"[{tag}] topic {K}: accuracy = {r['accuracy']:.3f}, confusion matrix "
                             f"(rows: task, last column: no comparison) =\n{r['conf']}")
    return lines


def aof_similarities(f_list: List[np.ndarray], semantic_attributes: List[np.ndarray],
                     session_all: List[np.ndarray], seqmask: List[np.ndarray],
                     tasks: np.ndarray, giventask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Cosine similarity between the task of each session and its detected attributes-of-focus (plot_eval_AoF.m)

    Four detectors (keys as in distances.mat):
      'cos_dist_top':  values of the most browsed item
      'cos_dist_c':    values shared by the two most browsed items ("Comparison")
      'cos_dist_all':  histogram of all browsed values (sum of session_all)
      'cos_dist_mset': histogram of the AOF masks (sum of seqmask, "LD"); NaN for a session without AOF

    Args:
        f_list: browsed grid positions (1..12) of each session (session{s}.f)
        semantic_attributes: (12 x 5) attribute values of the items in grid order
        session_all, seqmask: (15 x T) all values / AOF masks of each session
        tasks: (S,) task of each session (1-based)
        giventask: (n_task x 15)

    Note:
        The most browsed items are ordered by MATLAB's stable sort(..., 'descend').
    """
    out = {k: np.zeros(len(f_list)) for k in ('cos_dist_c', 'cos_dist_mset', 'cos_dist_top', 'cos_dist_all')}

    def cos(g, v, guard=True):
        n = np.linalg.norm(v)
        return g @ v / (n + (n == 0)) if guard else g @ v / n

    for s, f in enumerate(f_list):
        counts = np.bincount(np.asarray(f, dtype=int), minlength=13)[1:13]
        order = np.argsort(-counts, kind='stable')
        aof1, aof2 = semantic_attributes[s][order[0]], semantic_attributes[s][order[1]]
        idx = np.arange(5) * 3 + aof1 - 1  # (attribute, value of item 1) -> 15-dim index
        aof = np.zeros(15)
        aof[idx] = aof1 == aof2
        top = np.zeros(15)
        top[idx] = 1
        g = giventask[tasks[s] - 1] / np.linalg.norm(giventask[tasks[s] - 1])
        out['cos_dist_c'][s] = cos(g, aof)
        out['cos_dist_top'][s] = cos(g, top)
        out['cos_dist_all'][s] = cos(g, np.sum(session_all[s], axis=1))
        with np.errstate(invalid='ignore'):
            out['cos_dist_mset'][s] = cos(g, np.sum(seqmask[s], axis=1), guard=False)
    return out


def aof_ttests(dists: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
    """
    Two-sample t-tests between the detectors (plot_eval_AoF.m, ttest2 with equal variances)

    Sessions with NaN 'cos_dist_mset' (no AOF) are excluded from all detectors, as in plot_eval_AoF.m.

    Returns:
        (names, p): names = ['cos_dist_c', 'cos_dist_top', 'cos_dist_all', 'cos_dist_mset'] (MATLAB datas order),
        p (4 x 4) p-values in the upper triangle (zeros elsewhere)
    """
    from scipy.stats import ttest_ind
    names = ['cos_dist_c', 'cos_dist_top', 'cos_dist_all', 'cos_dist_mset']
    keep = ~np.isnan(dists['cos_dist_mset'])
    data = [dists[n][keep] for n in names]
    p = np.zeros((4, 4))
    for i, j in itertools.combinations(range(4), 2):
        p[i, j] = ttest_ind(data[i], data[j]).pvalue
    return names, p


def aof_temporal_counts(seqmask: List[np.ndarray], n_segments: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    AOF per temporal position in the session (paper Fig. 8, corrected count of the 2019 for_rev2.m)

    Time point l (1-based) of a session with T time points falls into segment floor((l - 1) * n_segments / T).

    Args:
        seqmask: AOF masks (features x T) of all sessions, with the time points without AOF (calc_segments output)

    Returns:
        (n_aof, n_points): per segment, the total number of AOF (sum of the masks) and the
        number of time points with at least one AOF. The 2019 for_rev2.m summed seqmask{s}(l), a linear index into
        the mask, which gives the values shown in the paper ([37 20 21 31 28 30 40 36 36 30] on refs).
    """
    n_aof = np.zeros(n_segments)
    n_points = np.zeros(n_segments)
    for mask in seqmask:
        T = mask.shape[1]
        segment = (np.arange(T) * n_segments) // T
        per_time = np.sum(mask, axis=0)
        np.add.at(n_aof, segment, per_time)
        np.add.at(n_points, segment, per_time > 0)
    return n_aof, n_points
