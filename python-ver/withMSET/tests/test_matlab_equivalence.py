"""
Equivalence tests between python-ver and matlab-ver.

The reference functions below are line-by-line re-implementations of the MATLAB code
(calc_lkh.m, calc_segments.m, estep.m / mstep.m / plsa.m, exact_prob.m, LoadGaze.m) written
with plain numpy / scipy, independent of the code under test.

Run from python-ver/:  python -m pytest tests -q
"""

import os
import sys
from math import factorial

import numpy as np
import pytest
from scipy.stats import multinomial

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

import main as pipeline  # noqa: E402
from data.data_loader import LikelihoodCalculator, exact_prob  # noqa: E402
from data.gaze_reader import GazeDataReader  # noqa: E402
from models.plsa import FourLayerPLSA, multinomial_probs  # noqa: E402
from models.segments import SegmentCalculator, SessionProcessor  # noqa: E402

SAMPLE_DIR = os.path.abspath(os.path.join(ROOT, '..', '..', 'sample_data'))
REFS = os.path.abspath(os.path.join(ROOT, '..', '..', 'matlab-ver', 'withMSET', 'refs'))  # MATLAB outputs of the paper
WIN, THR = 5, 0.03


# ------------------------------------------------------------------ MATLAB references
def ml_exact_prob(prob, hist):
    """exact_prob.m: n! * prod(p .^ x ./ x!)"""
    x = np.asarray(hist, float)
    return factorial(int(x.sum())) * np.prod(np.asarray(prob) ** x / np.array([factorial(int(v)) for v in x]))


def ml_calc_lkh(attrs, seq, win=WIN):
    """calc_lkh.m for one session: lkh = 1 except for t = win_size:T (1-based)"""
    T = seq.shape[1]
    out = []
    for att in range(5):
        p = np.bincount(attrs[:, att], minlength=4)[1:] / attrs.shape[0]
        lkh = np.ones(T)
        for t in range(win - 1, T):
            h = np.bincount(seq[att, t - win + 1:t + 1], minlength=4)[1:]
            lkh[t] = multinomial.pmf(h, win, p)
        out.append(lkh)
    return out


def ml_calc_segments(attrs, seq, lkh, win=WIN, thr=THR):
    """calc_segments.m (looked_att) for one session"""
    rows = []
    for att in range(5):
        p = np.bincount(attrs[:, att], minlength=4)[1:] / attrs.shape[0]
        looked = np.zeros((3, seq.shape[1]))
        for t in np.where(lkh[att] < thr)[0]:
            h = np.bincount(seq[att, t - win + 1:t + 1], minlength=4)[1:]
            h = h / h.sum()
            for v in range(3):
                if h[v] > p[v]:  # fixed calc_segments.m (paper); was >=
                    looked[v, t - win + 1:t + 1] = 1
        rows.append(looked)
    return np.vstack(rows)


def ml_plsa(sessions, K, lim=1e-4, itr_max=1000, seed=1):
    """plsa.m + estep.m + mstep.m, with the same random initialization order as FourLayerPLSA"""
    rs = np.random.RandomState(seed)
    S, N = len(sessions), sessions[0].shape[0]
    zs = rs.rand(S, K)
    zs /= zs.sum(1, keepdims=True)
    vz = [rs.rand(N) for _ in range(K)]
    vz = [v / v.sum() for v in vz]
    inc, prev, itr, hist = -1e20, -1e20, 1, []
    while abs(inc / prev) > lim and itr < itr_max:
        gamma, ll = [], 0.0
        for s, f in enumerate(sessions):
            g = np.zeros((K, f.shape[1]))
            for t in range(f.shape[1]):
                pv = np.array([zs[s, k] * ml_exact_prob(vz[k], f[:, t]) for k in range(K)])
                g[:, t] = pv / pv.sum()
                ll += np.log(pv.sum())
            gamma.append(g)
        zs = np.array([g.sum(1) / g.sum() if g.size else np.ones(K) / K for g in gamma])
        rz = np.zeros((K, N))
        for s, f in enumerate(sessions):
            for n in range(N):
                rz[:, n] += gamma[s][:, f[n] == 1].sum(1)
        vz = [rz[k] / rz[k].sum() for k in range(K)]
        inc = ll - prev
        if inc < 0:
            break
        hist.append(ll)
        prev = ll
        itr += 1
    return zs, vz, hist


def ml_load_gaze(data, matlab_bug=False):
    """
    LoadGaze.m: returns [target, duration(100 ms)] after the 150 ms filter and merging

    matlab_bug=True reproduces LoadGaze.m exactly (a missing left eye is never filled, x == 1 / y == 1
    leave the grid); the default is the fixed specification implemented in python-ver.
    """
    ts, g = data[:, 0], data[:, 1:]
    x, y = g[:, [0, 2]].copy(), g[:, [1, 3]].copy()
    for v in (x, y):
        i = v[:, 0] == -1
        v[i, 0] = v[i, 1]
        i = (v[:, 0] if matlab_bug else v[:, 1]) == -1
        v[i, 1] = v[i, 0]
    x, y = x.mean(1), y.mean(1)
    if matlab_bug:
        tgt = np.floor(y * 3) * 4 + np.floor(x * 4) + 1
    else:
        tgt = np.minimum(np.floor(y * 3), 2) * 4 + np.minimum(np.floor(x * 4), 3) + 1
    tgt[(x < 0) | (x > 1) | (y < 0) | (y > 1)] = -1
    ts = (ts - ts[0]) / 1000 / 100
    keep = tgt >= 0
    ts, tgt = ts[keep], tgt[keep]
    begin = tgt != np.concatenate([[0], tgt[:-1]])  # diff([0; t]) ~= 0
    end = tgt != np.concatenate([tgt[1:], [0]])     # diff([t; 0]) ~= 0
    tb = np.column_stack([tgt[begin], ts[begin], ts[end]])
    tb = tb[(tb[:, 2] - tb[:, 1]) >= 1.5]
    merged = [tb[0].copy()]
    for row in tb[1:]:
        if row[0] == merged[-1][0]:
            merged[-1][2] = row[2]
        else:
            merged.append(row.copy())
    merged = np.array(merged)
    return np.column_stack([merged[:, 0], merged[:, 2] - merged[:, 1]])


# ------------------------------------------------------------------ synthetic data
@pytest.fixture(scope='module')
def sessions_data():
    """12 items, 5 attributes x 3 values (4 items per value), screening + comparison phases"""
    rng = np.random.default_rng(0)
    kb = np.stack([rng.permutation(np.repeat([1, 2, 3], 4)) for _ in range(5)], axis=1)
    attrs, seqs = [], []
    for _ in range(8):
        a = kb[rng.permutation(12)]
        c1, c2 = rng.choice(12, 2, replace=False)
        f = np.array(list(rng.integers(0, 12, 12)) + [c1, c2] * 6 + list(rng.integers(0, 12, 8)))
        attrs.append(a)
        seqs.append(a[f].T)
    return attrs, seqs


def own_likelihoods(attrs, seqs):
    lkh, _ = pipeline.calc_lkh(attrs, seqs, WIN)
    return lkh


# ------------------------------------------------------------------ tests
def test_exact_prob_matches_exact_prob_m():
    rng = np.random.default_rng(1)
    for _ in range(100):
        p = rng.dirichlet(np.ones(15))
        b = np.zeros(15)
        b[rng.choice(15, rng.integers(1, 6), replace=False)] = 1
        assert np.isclose(exact_prob(p, b), ml_exact_prob(p, b))
        pc = rng.dirichlet(np.ones(3))
        c = rng.multinomial(5, rng.dirichlet(np.ones(3)))
        assert np.isclose(exact_prob(pc, c), ml_exact_prob(pc, c))


def test_multinomial_probs_matches_exact_prob():
    rng = np.random.default_rng(2)
    P = rng.dirichlet(np.ones(15), 4)
    P[1, 3] = 0
    P[2, 0] = 0
    P /= P.sum(1, keepdims=True)
    F = (rng.random((15, 40)) < .2).astype(float)
    F[:, 5] = 0  # no AOF
    F[3, 6] = 1  # value with probability 0 for topic 1
    ref = np.array([[ml_exact_prob(P[k], F[:, t]) for t in range(40)] for k in range(4)])
    assert np.allclose(multinomial_probs(list(P), F), ref, rtol=1e-12, atol=0)


def test_allpair_order():
    # allpair3 of the 2019 calc_lkh.m
    expected = np.array([[i, j, WIN - i - j] for i in range(WIN + 1) for j in range(WIN - i + 1)])
    assert np.array_equal(LikelihoodCalculator(WIN).generate_all_combinations(3), expected)


def test_likelihood_matches_calc_lkh(sessions_data):
    attrs, seqs = sessions_data
    lkh = own_likelihoods(attrs, seqs)
    for s in range(len(seqs)):
        ref = ml_calc_lkh(attrs[s], seqs[s])
        for att in range(5):
            assert np.allclose(lkh[s][att], ref[att], atol=1e-12)
        # the first window (index win_size-1) must be evaluated
        assert lkh[s][0][WIN - 1] == pytest.approx(ref[0][WIN - 1])


def test_aof_mask_matches_calc_segments(sessions_data):
    attrs, seqs = sessions_data
    lkh = own_likelihoods(attrs, seqs)
    _, masks = SegmentCalculator(WIN, THR).calc_segments(lkh, attrs, seqs, WIN, THR)
    for s in range(len(seqs)):
        ref = ml_calc_segments(attrs[s], seqs[s], ml_calc_lkh(attrs[s], seqs[s]))
        assert np.array_equal(masks[s], ref)


def test_remove_empty_columns(sessions_data):
    attrs, seqs = sessions_data
    lkh = own_likelihoods(attrs, seqs)
    _, masks = SegmentCalculator(WIN, THR).calc_segments(lkh, attrs, seqs, WIN, THR)
    cleaned = SessionProcessor().remove_empty_columns(masks)
    assert sum(int((m.sum(0) == 0).sum()) for m in masks) > 0  # the test data has empty columns
    for m, c in zip(masks, cleaned):
        assert np.array_equal(c, m[:, m.sum(0) > 0])
        assert (c.sum(0) > 0).all()


@pytest.mark.parametrize('K', [1, 3])
def test_em_matches_plsa_m(sessions_data, K):
    attrs, seqs = sessions_data
    lkh = own_likelihoods(attrs, seqs)
    _, masks = SegmentCalculator(WIN, THR).calc_segments(lkh, attrs, seqs, WIN, THR)
    data = SessionProcessor().remove_empty_columns(masks)
    np.random.seed(1)
    params, hist = FourLayerPLSA(K).fit(data, train_lim=1e-4, max_itr=1000)
    zs, vz, ref_hist = ml_plsa(data, K)
    assert len(hist) == len(ref_hist)
    assert hist[-1] == pytest.approx(ref_hist[-1], rel=1e-10)
    assert np.allclose(params['zs'], zs, atol=1e-10)
    assert all(np.allclose(a, b, atol=1e-10) for a, b in zip(params['vz'], vz))


@pytest.mark.skipif(not os.path.exists(os.path.join(SAMPLE_DIR, '170116024150092.csv')),
                    reason='sample_data not found')
def test_gaze_reader_matches_loadgaze():
    csv = os.path.join(SAMPLE_DIR, '170116024150092.csv')
    lst = os.path.join(SAMPLE_DIR, '170116024150092.list')
    ref = ml_load_gaze(np.loadtxt(csv, delimiter=','))
    reader = GazeDataReader(base_path='.', knowledge_csv_path=os.path.join(SAMPLE_DIR, 'knowledge.csv'))
    kb = reader.load_knowledge_base()
    session = reader.load_session_from_files(csv, lst, 1, 1)
    f = ref[:, 0].astype(int)
    assert np.array_equal(session['f'], f)
    assert np.allclose(session['dur'], ref[:, 1])
    assert np.array_equal(session['semantic_attribute_seq'], kb[session['list'][f - 1] - 1, :].T)


def test_gaze_missing_eye_and_edges():
    """fixed handling of a missing eye (either side) and of coordinates equal to 1"""
    reader = GazeDataReader(base_path='.')
    #                 ts  rx    ry    lx    ly
    data = np.array([[0, 0.10, 0.10, 0.20, 0.20],   # both eyes          -> mean (0.15, 0.15) -> 1
                     [1, -1,   -1,   0.60, 0.60],   # right eye missing  -> (0.6, 0.6)       -> 7
                     [2, 0.60, 0.90, -1,   -1],     # left eye missing   -> (0.6, 0.9)       -> 11 (MATLAB: dropped)
                     [3, -1,   -1,   -1,   -1],     # both missing       -> invalid
                     [4, 1.00, 1.00, 1.00, 1.00],   # corner x = y = 1   -> 12 (MATLAB: 17)
                     [5, 1.00, 0.10, 1.00, 0.10]])  # right edge x = 1   -> 4 (MATLAB: 5)
    assert reader.gaze_to_items(data).tolist() == [1, 7, 11, -1, 12, 4]


@pytest.mark.skipif(not os.path.exists(os.path.join(REFS, 'gazedata.mat')), reason='refs/ not found')
def test_loadgaze_reference_matches_matlab_output():
    """ml_load_gaze(matlab_bug=True) reproduces gazedata.mat (usr 2, task 1) made by MATLAB from the same raw data"""
    import scipy.io
    session = scipy.io.loadmat(os.path.join(REFS, 'gazedata.mat'))['session'][1, 0]
    ref = ml_load_gaze(np.loadtxt(os.path.join(SAMPLE_DIR, '170116024150092.csv'), delimiter=','), matlab_bug=True)
    assert np.array_equal(ref[:, 0], np.asarray(session['f'][0, 0]).ravel())
    assert np.allclose(ref[:, 1], np.asarray(session['dur'][0, 0]).ravel(), rtol=0, atol=1e-12)
    assert np.array_equal(np.loadtxt(os.path.join(SAMPLE_DIR, '170116024150092.list'), dtype=int) % 100,
                          np.asarray(session['list'][0, 0]).ravel())


def test_load_all_sessions(tmp_path):
    """ReadwithTimestamp.m loop: 4 lines per user, 4th skipped, task from the .list, {usr, task} keys"""
    import shutil
    csv = os.path.join(SAMPLE_DIR, '170116024150092.csv')
    items = np.loadtxt(os.path.join(SAMPLE_DIR, '170116024150092.list'), dtype=int) % 100
    order = [[2, 1, 3, None], [3, 2, 1, None]]  # tasks per user; None = 4th session (no files, must be skipped)
    names = []
    for u, tasks in enumerate(order):
        for i, task in enumerate(tasks):
            name = f'u{u + 1}s{i + 1}'
            names.append(name)
            if task is not None:
                shutil.copy(csv, tmp_path / f'{name}.csv')
                np.savetxt(tmp_path / f'{name}.list', items + task * 100, fmt='%d')
    (tmp_path / 'timestamp.list').write_text('\n'.join(names) + '\n')

    reader = GazeDataReader(base_path='.', knowledge_csv_path=os.path.join(SAMPLE_DIR, 'knowledge.csv'))
    reader.load_knowledge_base()
    data = reader.load_all_sessions(str(tmp_path / 'timestamp.list'))
    assert data['n_usr'] == 2
    assert sorted(data['session']) == [(u, t) for u in (1, 2) for t in (1, 2, 3)]
    assert data['session'][(1, 2)]['name'] == 'u1s1' and data['session'][(2, 1)]['name'] == 'u2s3'
    assert data['session'][(1, 2)]['session_id'] == 1 and data['session'][(2, 1)]['session_id'] == 3
    single = reader.load_session_from_files(csv, os.path.join(SAMPLE_DIR, '170116024150092.list'), 1, 1)
    for s in data['session'].values():
        assert np.array_equal(s['f'], single['f']) and np.allclose(s['dur'], single['dur'])
        assert np.array_equal(s['list'], items)

    out = tmp_path / 'gazedata.mat'
    reader.save_gazedata(data, str(out))
    attrs, seqs = pipeline.load_gazedata_mat(str(out))
    ref_attrs, ref_seqs = pipeline.from_reader_sessions(data)  # column-major order
    assert all(np.array_equal(a, b) for a, b in zip(seqs, ref_seqs)) and len(seqs) == 6
    assert all(np.array_equal(a, b) for a, b in zip(attrs, ref_attrs))


def test_pipeline_smoke(tmp_path):
    attrs, seqs = pipeline.create_demo_data(n_users=3)
    res = pipeline.learn_aspects(attrs, seqs, max_topics=2, train_num=2, max_itr=30,
                                 jobs=1, savedir=str(tmp_path))
    for tag in ('spe', 'all'):
        for k in (1, 2):
            assert len(res[tag][k]['params']['vz']) == k
            assert (tmp_path / f'p_{tag}_topic{k}.npz').exists()
            assert (tmp_path / f'Akq_{tag}_topic{k}.csv').exists()
