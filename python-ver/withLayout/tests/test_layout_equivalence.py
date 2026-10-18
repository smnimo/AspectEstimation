"""
Equivalence tests between python-ver/withLayout and matlab-ver/withLayout.

The reference functions below are line-by-line re-implementations of the MATLAB code (item_attr.m, reg_attr.m,
estep.m, mstep.m, prob.calcL, plsa.m, est.m, eval_precision.m, est_cosine.m, eval_precision_comparison.m) with loops,
independent of the code under test. Synthetic data in the format of content.mat / data/mat/*.mat.

Run from python-ver/withLayout:  python -m pytest tests -q
"""

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import layout_data as ld  # noqa: E402
import layout_eval as le  # noqa: E402
import layout_model as lm  # noqa: E402


def norm(a, dim=None):
    """normalize.m"""
    a = np.asarray(a, dtype=float)
    z = a.sum() if dim is None else a.sum(axis=dim, keepdims=True)
    return a / (z + (z == 0))


# ------------------------------------------------------------------ synthetic data
@pytest.fixture(scope='module')
def synthetic():
    rng = np.random.default_rng(0)
    content = {'category': [], 'calorie': [], 'step': [], 'colhist': []}
    for _ in range(4):
        content['category'].append((rng.random((15, 5)) < 0.3).astype(float))
        content['calorie'].append(np.eye(4)[rng.integers(0, 4, 15)])
        content['step'].append(np.eye(5)[rng.integers(0, 5, 15)])
        content['colhist'].append([np.eye(16)[[rng.integers(0, 16)]] for _ in range(15)])
    content['category'][0][0] = 0  # an item without category
    sessions = []
    for s in range(10):
        items = rng.choice(60, 12, replace=False)
        f = [int(rng.integers(1, 13))]
        while len(f) < int(rng.integers(40, 60)):
            nxt = int(rng.integers(1, 13))
            if nxt != f[-1]:
                f.append(nxt)
        sessions.append(ld.Session(name=str(s), f=np.array(f), index=items // 15 + 1, num=items % 15 + 1,
                                   answer=rng.choice(12, 3, replace=False) + 1, content_id=1, task_id=s % 3 + 1))
    return content, sessions


# ------------------------------------------------------------------ MATLAB references
def ml_pvi(content, s, attr):
    rows = []
    for i in range(12):
        g, n = s.index[i] - 1, s.num[i] - 1
        sem = list(content['category'][g][n]) + list(content['calorie'][g][n]) + list(content['step'][g][n])
        col = list(content['colhist'][g][n][0])
        rows.append({1: sem, 2: col, 3: sem + col}[attr])
    return np.array(rows, dtype=float)


def ml_item_attr(pvr):
    pvr = pvr.copy()
    pvr[:, pvr.sum(0) == 0] = 1
    return norm(pvr.T, 1)


def ml_reg_attr(pvi, layout, alpha):
    effect = np.tile(norm(layout), (pvi.shape[1], 1))
    pvi = pvi.copy()
    pvi[:, pvi.sum(0) == 0] = 1
    return norm((1 - alpha) * norm(pvi.T, 1) + alpha * effect, 1)


def ml_estep(sessions, zs, vz):
    gamma = []
    for s, (f, prv) in enumerate(sessions):
        g = np.zeros(vz.shape + (len(f),))
        for t in range(len(f)):
            x = np.outer(zs[s], prv[:, f[t] - 1]) * vz
            g[:, :, t] = x / x.sum()
        gamma.append(g)
    return gamma


def ml_mstep(gamma):
    pvz = sum(g.sum(2) for g in gamma)
    vz = pvz / pvz.sum(1, keepdims=True)
    zs = np.vstack([g.sum(2).sum(1) / g.shape[2] for g in gamma])
    return zs, vz


def ml_calcL(sessions, zs, vz):
    Ls = [sum(math.log(zs[s] @ vz @ prv[:, f[t] - 1]) for t in range(len(f))) for s, (f, prv) in enumerate(sessions)]
    return float(np.mean(Ls))


def ml_loop(zs, vz, sessions, lim, update):
    """while cL > lim || cL < 0 (plsa.m / est.m)"""
    Lt = [ml_calcL(sessions, zs, vz)]
    cL = 1
    while cL > lim or cL < 0:
        zs, vz = update(zs, vz)
        Lt.append(ml_calcL(sessions, zs, vz))
        cL = Lt[-1] - Lt[-2]
    return zs, vz, Lt


def ml_plsa(sessions, K, lim, seed):
    np.random.seed(seed)  # same draws as layout_model.fit: rand(S, K), then rand(K, Q)
    zs = norm(np.random.rand(len(sessions), K), 1)
    vz = norm(np.random.rand(K, sessions[0][1].shape[0]), 1)
    return ml_loop(zs, vz, sessions, lim, lambda z, v: ml_mstep(ml_estep(sessions, z, v)))


def ml_est(vz, session, lim):
    vz_s = norm(vz + 0.01, 1)
    zs0 = np.ones((1, vz.shape[0])) / vz.shape[0]

    def update(z, v):
        g = ml_estep([session], z, v)[0]
        return (g.sum(2).sum(1) / g.shape[2])[None, :], v
    return ml_loop(zs0, vz_s, [session], lim, update)[0][0]


def ml_patk(scores, answer, K, descend=True):
    order = sorted(range(len(scores)), key=lambda i: -scores[i] if descend else scores[i])[:K]
    return np.array([sum(answer[order[:i + 1]]) / (i + 1) for i in range(K)])


def ml_eval_precision(f_test, answer_items, prv, piv, zs, vz, K):
    pis = zs @ vz @ piv
    item = ml_patk(pis, answer_items, K)
    prs = zs @ vz @ prv
    hist = np.array([np.sum(f_test == r) for r in range(1, 13)])
    focused = sorted(range(12), key=lambda i: -hist[i])[:K]
    ans = np.zeros(12)
    ans[focused] = 1
    return ml_patk(prs, ans, K), item


def ml_bhattacharyya(a, b):
    bc = sum(math.sqrt(x * y) for x, y in zip(a, b))
    return -math.log(bc) if bc > 0 else math.inf  # MATLAB: -log(0) = Inf


def ml_comparison_scores(test, train, attr_set):
    """est_cosine.m (3 nearest in Bhattacharyya distance) + eval_precision_comparison.m distances"""
    cdist = [ml_bhattacharyya(test, t) for t in train]
    i = sorted(range(len(cdist)), key=lambda j: cdist[j])
    with np.errstate(divide='ignore', invalid='ignore'):
        attr = norm(sum(train[i[k]] / cdist[i[k]] for k in range(3)))
    return np.array([ml_bhattacharyya(attr, attr_set[n]) for n in range(12)])


# ------------------------------------------------------------------ tests
def test_item_and_region_given_value(synthetic):
    content, sessions = synthetic
    layout = np.random.default_rng(1).random(12)
    for s in sessions:
        for attr in (1, 2, 3):
            pvi = ld.item_attributes(content, s, attr)
            assert np.array_equal(pvi, ml_pvi(content, s, attr))
            assert np.allclose(ld.item_given_value(pvi), ml_item_attr(pvi), atol=1e-15)
            for alpha in (0, 0.3, 1):
                assert np.allclose(ld.region_given_value(pvi, layout, alpha), ml_reg_attr(pvi, layout, alpha), atol=1e-15)


def test_region_frequency_leave_one_out(synthetic):
    _, sessions = synthetic
    counts = np.array([[np.sum(s.f == r) for r in range(1, 13)] for s in sessions], dtype=float)
    for s in range(len(sessions)):
        ref = norm(np.delete(counts, s, axis=0).sum(0))
        assert np.allclose(ld.region_frequency(sessions, exclude=s), ref, atol=1e-15)


@pytest.fixture(scope='module')
def model_data(synthetic):
    content, sessions = synthetic
    layout = ld.region_frequency(sessions)
    return [(s.f, ld.region_given_value(ld.item_attributes(content, s, 1), layout, 0.5)) for s in sessions]


def test_em_step_and_loglik(model_data):
    rng = np.random.default_rng(2)
    zs, vz = rng.dirichlet(np.ones(4), len(model_data)), rng.dirichlet(np.ones(14), 4)
    packed = lm.Packed(model_data)
    zs1, vz1 = lm.em_step(packed, zs, vz)
    rzs, rvz = ml_mstep(ml_estep(model_data, zs, vz))
    assert np.allclose(zs1, rzs, atol=1e-14) and np.allclose(vz1, rvz, atol=1e-14)
    assert packed.log_likelihood(zs, vz) == pytest.approx(ml_calcL(model_data, zs, vz), rel=1e-12)
    assert lm.log_likelihood(zs, vz, model_data) == pytest.approx(ml_calcL(model_data, zs, vz), rel=1e-12)


@pytest.mark.parametrize('K', [1, 3])
def test_fit_matches_plsa_m(model_data, K):
    np.random.seed(7)
    p = lm.fit(model_data, K, 1e-2)
    zs, vz, Lt = ml_plsa(model_data, K, 1e-2, 7)
    assert len(p['L']) == len(Lt)
    assert np.allclose(p['L'], Lt, rtol=1e-12)
    assert np.allclose(p['zs'], zs, atol=1e-12) and np.allclose(p['vz'], vz, atol=1e-12)


def test_estimate_matches_est_m(model_data):
    np.random.seed(3)
    vz = norm(np.random.rand(4, 14), 1)
    f, prv = model_data[0]
    assert np.allclose(lm.estimate(vz, (f[:20], prv), 1e-3), ml_est(vz, (f[:20], prv), 1e-3), atol=1e-13)


def test_split_session():
    f = np.arange(1, 41)  # T = 40: x = ratio * 4 is an integer -> one shared time point (main.m)
    est, test = le.split_session(f, 3)
    assert list(est) == list(range(1, 13)) and test[0] == 12 and test[-1] == 40
    est, test = le.split_session(f, 3, overlap=False)
    assert test[0] == 13
    f = np.arange(1, 44)  # T = 43: x = 12.9
    est, test = le.split_session(f, 3)
    assert len(est) == 12 and test[0] == 13


def test_eval_precision(synthetic, model_data):
    content, sessions = synthetic
    rng = np.random.default_rng(4)
    zs, vz = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(14), 4)
    for s, (f, prv) in zip(sessions, model_data):
        piv = ld.item_given_value(ld.item_attributes(content, s, 1))
        own = le.eval_precision(f[20:], le.selected_items(s), prv, piv, zs, vz, 3)
        ref = ml_eval_precision(f[20:], le.selected_items(s), prv, piv, zs, vz, 3)
        assert np.allclose(own[0], ref[0]) and np.allclose(own[1], ref[1])


def test_comparison_matlab_method(synthetic):
    content, sessions = synthetic
    for attr in (1, 2, 3):
        pvis = [ld.item_attributes(content, s, attr) for s in sessions]
        full = np.vstack([le.interest_vector(p, s.f) for p, s in zip(pvis, sessions)])
        test = le.interest_vector(pvis[0], sessions[0].f[:15])
        own = le.predict_comparison(test, full[1:], pvis[0], 'matlab')
        assert np.allclose(own, ml_comparison_scores(test, full[1:], pvis[0]), atol=1e-12)
        # makeattr + normalize(mean(...)) equals normalize(sum(...))
        assert np.allclose(test, norm(np.mean(pvis[0][sessions[0].f[:15] - 1], axis=0)))


def test_comparison_paper_method(synthetic):
    content, sessions = synthetic
    pvis = [ld.item_attributes(content, s, 1) for s in sessions]
    full = np.vstack([le.interest_vector(p, s.f) for p, s in zip(pvis, sessions)])
    test = le.interest_vector(pvis[0], sessions[0].f[:15])
    cos = lambda a, b: 1 - a @ b / np.linalg.norm(a) / np.linalg.norm(b)  # noqa: E731
    near = full[1:][int(np.argmin([cos(test, t) for t in full[1:]]))]
    assert np.allclose(le.predict_comparison(test, full[1:], pvis[0], 'paper'), [cos(near, v) for v in pvis[0]])


def test_leave_one_out_smoke(synthetic):
    content, sessions = synthetic
    res = le.leave_one_out(sessions, content, 1, 0.5, [2], train_num=2, log=lambda m: None)
    assert res['mpatk_region'][2].shape == (9, 3) and res['patk_item'][2].shape == (9, 3, len(sessions))
    cmp = le.leave_one_out_comparison(sessions, content, 2, 'matlab')
    assert cmp['mpatk_item'].shape == (9, 3)
