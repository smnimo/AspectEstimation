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
    over the first min(n_task, K) aspects. Permutations are visited in the order of MATLAB's
    perms(1:K) (reverse lexicographic) and the first maximum is taken, so ties among the
    unmatched aspects (K > n_task) are broken as in MATLAB.

    Note: as in MATLAB, applying this twice is not idempotent for K > n_task + 1
    (the unmatched aspects are reversed again).

    Returns:
        (vz, zs, perm): permuted vz (N x K), zs (S x K), and perm (0-based)
    """
    K = vz.shape[1]
    cos = cosine_task_aspect(giventask, vz)
    k = min(giventask.shape[0], K)
    best, best_score = None, -np.inf
    for perm in reversed(list(itertools.permutations(range(K)))):
        score = np.sum(np.diag(cos[:, list(perm)])[:k])
        if score > best_score:
            best, best_score = np.array(perm), score
    return vz[:, best], zs[:, best], best


def calc_eval(vz: np.ndarray, zs: np.ndarray, tasks: np.ndarray,
              giventask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Similarity between the task of each session and its top aspect (CalcEval.m)

    Args:
        vz: (N x K) aspects, zs: (S x K) interests, tasks: (S,) task of each session (1-based)

    Returns:
        (cor_task, cortop): cor_task (n_task x K), cortop (S,) = cor_task[task(s), argmax_r zs(s, r)]
        (sessions without AOF have uniform zs and therefore use aspect 1, as in MATLAB)
    """
    cor_task = cosine_task_aspect(giventask, vz)
    cortop = cor_task[np.asarray(tasks) - 1, np.argmax(zs, axis=1)]
    return cor_task, cortop


def confusion_matrix(zs: np.ndarray, n_usr: int, n_task: int) -> Tuple[np.ndarray, float]:
    """
    Task estimation by argmax zs (plot_conf_matrix.m)

    A session whose maximum of zs equals 1/n_task exactly (uniform zs, i.e., a session
    without AOF) is counted in the last column ("no comparison"), as in MATLAB.

    Returns:
        (conf, accuracy): conf (n_task x n_task+1) counts, accuracy = trace / number of sessions
    """
    conf = np.zeros((n_task, n_task + 1), dtype=int)
    for task in range(n_task):
        z = zs[task * n_usr:(task + 1) * n_usr]
        est = np.argmax(z, axis=1)
        est[np.max(z, axis=1) == 1 / n_task] = n_task
        conf[task] = np.bincount(est, minlength=n_task + 1)
    return conf, np.trace(conf[:, :n_task]) / conf.sum()


def session_tasks(n_sessions: int, n_task: int) -> np.ndarray:
    """Task of each session in column-major order (session{s}.task in MATLAB)"""
    if n_sessions % n_task:
        raise ValueError(f"{n_sessions} sessions cannot be split into {n_task} tasks")
    return np.repeat(np.arange(1, n_task + 1), n_sessions // n_task)


def evaluate(models: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
             giventask: np.ndarray, conf_topic: int = 3) -> Dict:
    """
    Matching + similarity + confusion matrix for all learned models (calc_matching -> CalcSim -> plot_conf_matrix)

    Args:
        models: models[tag][K] = (vz (N x K), zs (S x K)), tag in {'spe', 'all'}
        giventask: (n_task x N)
        conf_topic: number of aspects used for the confusion matrix (MATLAB: 3)

    Returns:
        out[tag][K] = {'vz', 'zs', 'perm', 'cor_task', 'cortop'} (matched),
        plus 'conf' and 'accuracy' for K == conf_topic
    """
    n_task = giventask.shape[0]
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
            cor_task, cortop = calc_eval(vz, zs, tasks, giventask)
            res = {'vz': vz, 'zs': zs, 'perm': perm, 'cor_task': cor_task, 'cortop': cortop}
            if K == conf_topic:
                res['conf'], res['accuracy'] = confusion_matrix(zs, S // n_task, n_task)
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
                         f"mean cortop = {np.mean(r['cortop']):.3f}")
            if 'conf' in r:
                lines.append(f"[{tag}] topic {K}: accuracy = {r['accuracy']:.3f}, confusion matrix "
                             f"(rows: task, last column: no comparison) =\n{r['conf']}")
    return lines
