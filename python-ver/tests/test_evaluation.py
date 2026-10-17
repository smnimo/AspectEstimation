"""
Equivalence tests of src/models/evaluation.py with calc_matching.m, CalcEval.m and plot_conf_matrix.m.

The references below follow the MATLAB code (pdist 'cosine', perms(1:K), max) with scipy / numpy,
independent of the code under test.

Run from python-ver/:  python -m pytest tests -q
"""

import os
import sys

import numpy as np
from scipy.spatial.distance import pdist, squareform

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from models import evaluation  # noqa: E402

GT = evaluation.GIVENTASK


# ------------------------------------------------------------------ MATLAB references
def ml_perms(K):
    """perms(1:K) (0-based): reverse lexicographic order"""
    if K == 1:
        return [[0]]
    out = []
    for first in range(K - 1, -1, -1):
        rest = [v for v in range(K) if v != first]
        out += [[first] + [rest[i] for i in p] for p in ml_perms(K - 1)]
    return out


def ml_cor_task(giventask, vz):
    """1 - squareform(pdist([normalize(giventask, 2); vz'], 'cosine'))(1:n_task, end-K+1:end)"""
    g = giventask / giventask.sum(1, keepdims=True)
    c = 1 - squareform(pdist(np.vstack([g, vz.T]), 'cosine'))
    return c[:giventask.shape[0], -vz.shape[1]:]


def ml_calc_matching(vz, zs, giventask):
    K = vz.shape[1]
    allpair = ml_perms(K)
    corscore = [np.sum(np.diag(ml_cor_task(giventask, vz[:, p]))) for p in allpair]
    mi = int(np.argmax(corscore))  # first maximum, as MATLAB max
    return vz[:, allpair[mi]], zs[:, allpair[mi]]


def ml_conf(zs, n_usr, n_task):
    conf = np.zeros((n_task, n_task + 1), int)
    for task in range(n_task):
        z = zs[task * n_usr:(task + 1) * n_usr]
        est = z.argmax(1) + 1
        est[z.max(1) == 1 / n_task] = n_task + 1
        conf[task] = np.histogram(est, bins=np.arange(1, n_task + 3) - 0.5)[0]
    return conf


def synthetic_model(K, perm, n_usr=6, seed=0):
    """aspects close to the task vectors (+ noise), stored in the order perm; zs peaked at the task"""
    rng = np.random.default_rng(seed)
    base = [GT[k % 3] + (k // 3) * rng.random(15) for k in range(K)]
    vz = np.stack([b + 0.05 * rng.random(15) for b in base], axis=1)
    vz /= vz.sum(0, keepdims=True)
    zs = rng.random((3 * n_usr, K)) * 0.2
    for s in range(3 * n_usr):
        zs[s, min(s // n_usr, K - 1)] += 1
    zs[1] = 1 / K  # a session without AOF (uniform)
    zs /= zs.sum(1, keepdims=True)
    inv = np.argsort(perm)
    return vz[:, inv], zs[:, inv]


# ------------------------------------------------------------------ tests
def test_perms_order():
    assert ml_perms(3) == [[2, 1, 0], [2, 0, 1], [1, 2, 0], [1, 0, 2], [0, 2, 1], [0, 1, 2]]


def test_cosine_matches_pdist():
    rng = np.random.default_rng(0)
    for K in range(1, 6):
        vz = rng.dirichlet(np.ones(15), K).T
        assert np.allclose(evaluation.cosine_task_aspect(GT, vz), ml_cor_task(GT, vz), atol=1e-12)


def test_matching_recovers_known_permutation():
    for K, perm in ((2, [1, 0]), (3, [2, 0, 1]), (4, [3, 1, 0, 2]), (5, [4, 2, 3, 0, 1])):
        vz, zs = synthetic_model(K, perm)
        mvz, mzs, p = evaluation.match_aspects(vz, zs, GT)
        assert np.array_equal(np.argmax(evaluation.cosine_task_aspect(GT, mvz)[:, :min(3, K)], axis=0),
                              np.arange(min(3, K)))
        rvz, rzs = ml_calc_matching(vz, zs, GT)
        assert np.array_equal(mvz, rvz) and np.array_equal(mzs, rzs)  # incl. MATLAB tie-breaking


def test_calc_eval_and_confusion():
    vz, zs = synthetic_model(3, [0, 1, 2])
    tasks = evaluation.session_tasks(zs.shape[0], 3)
    cor_task, cortop = evaluation.calc_eval(vz, zs, tasks, GT)
    ref = ml_cor_task(GT, vz)
    assert np.allclose(cor_task, ref, atol=1e-12)
    assert np.allclose(cortop, [ref[t - 1, np.argmax(zs[s])] for s, t in enumerate(tasks)], atol=1e-12)
    conf, acc = evaluation.confusion_matrix(zs, 6, 3)
    assert np.array_equal(conf, ml_conf(zs, 6, 3))
    assert conf[0, 3] == 1 and acc == np.trace(conf[:, :3]) / 18


def test_evaluate_layout():
    models = {'spe': {K: synthetic_model(K, list(range(K))[::-1]) for K in (1, 2, 3)}}
    ev = evaluation.evaluate(models, GT)
    assert set(ev['spe']) == {1, 2, 3}
    assert 'conf' in ev['spe'][3] and 'conf' not in ev['spe'][2]
    assert np.array_equal(ev['spe'][3]['perm'], [2, 1, 0])
    assert ev['spe'][3]['cortop'].shape == (18,)
