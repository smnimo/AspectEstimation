"""
Regression tests against the MATLAB outputs used for the paper (python-ver/refs/, not in git).

Skipped when refs/ is not available.

Run from python-ver/:  python -m pytest tests -q
"""

import contextlib
import io
import os
import sys
from math import factorial

import numpy as np
import pytest
import scipy.io

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

import main as pipeline  # noqa: E402
from models import evaluation  # noqa: E402
from models.plsa import FourLayerPLSA  # noqa: E402
from models.segments import SegmentCalculator, SessionProcessor  # noqa: E402

REFS = os.path.join(ROOT, 'refs')
RESULTS = os.path.join(REFS, 'results')
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(REFS, 'gazedata.mat')),
                                reason='refs/ (MATLAB outputs) not found')

TABLE2 = np.array([[32, 0, 3, 2], [0, 30, 7, 0], [4, 0, 31, 2]])  # paper Table 2


def cell_list(a):
    return [np.asarray(c) for c in a.ravel()]


@pytest.fixture(scope='module')
def gazedata():
    return pipeline.load_gazedata_mat(os.path.join(REFS, 'gazedata.mat'))


@pytest.fixture(scope='module')
def step1(gazedata):
    attrs, seqs = gazedata
    lkh, N_value = pipeline.calc_lkh(attrs, seqs, 5)
    _, mask = SegmentCalculator(5, 0.03).calc_segments(lkh, attrs, seqs, 5, 0.03)
    with contextlib.redirect_stdout(io.StringIO()):
        session_all = SessionProcessor().create_session_all(seqs, N_value)
    return lkh, mask, session_all


def test_giventask_matches_task_att():
    assert np.array_equal(evaluation.load_giventask(os.path.join(REFS, 'task_att.mat')), evaluation.GIVENTASK)


def test_session_order(gazedata):
    session = scipy.io.loadmat(os.path.join(REFS, 'gazedata.mat'))['session']
    tasks = [int(np.asarray(c['task'][0, 0]).ravel()[0]) for c in session.flatten(order='F')]
    assert np.array_equal(tasks, evaluation.session_tasks(len(gazedata[1]), 3))


def test_lkh_matches_lkh_win_5(step1):
    ml = scipy.io.loadmat(os.path.join(REFS, 'lkh_win_5.mat'))['lkh_ses'].ravel()
    lkh = step1[0]
    assert len(lkh) == len(ml) == 111
    for s, cells in enumerate(ml):
        for att, ref in enumerate(np.asarray(cells).ravel()):
            assert np.allclose(lkh[s][att], np.asarray(ref).ravel(), rtol=0, atol=1e-15)


def test_seqmask_and_session_all_match_segsessions(step1):
    ss = scipy.io.loadmat(os.path.join(REFS, 'segsessions.mat'))
    assert ss['thr'].item() == 0.03 and ss['win_size'].item() == 5
    _, mask, session_all = step1
    for py, ml in zip(mask, cell_list(ss['seqmask'])):
        assert np.array_equal(py, ml)
    for py, ml in zip(session_all, cell_list(ss['session_all'])):
        assert np.array_equal(py, ml)
    spe = SessionProcessor().remove_empty_columns(mask)
    assert sum(m.shape[1] == 0 for m in spe) == TABLE2[:, -1].sum()  # 4 sessions without AOF


@pytest.mark.parametrize('tag', ['spe', 'all'])
def test_calc_eval_matches_cor_mat(step1, tag):
    """
    cor.mat (CalcSim.m) vs CalcEval on the saved p, without re-matching.

    cor.mat was computed before the aspects of spe K=5 / all K=4 were reordered by calc_matching.m
    (same values, permuted columns), so the aspect order of cor.mat is recovered first.
    """
    cor = scipy.io.loadmat(os.path.join(RESULTS, 'cor.mat'))
    key = 'cor' if tag == 'spe' else 'cor_all'
    for K in range(1, 6):
        zs, vz, _ = evaluation.load_matlab_result(os.path.join(RESULTS, f'p_{tag}_topic{K}.mat'))
        ref = cor[key].ravel()[K - 1]
        own = evaluation.cosine_task_aspect(evaluation.GIVENTASK, vz)
        perm = [int(np.argmin(np.abs(own - ref[:, [j]]).sum(0))) for j in range(K)]
        assert sorted(perm) == list(range(K))
        no_aof = np.array([not m.any() for m in step1[1]]) if tag == 'spe' else None
        cor_task, cortop = evaluation.calc_eval(vz[:, perm], zs[:, perm],
                                                evaluation.session_tasks(zs.shape[0], 3), evaluation.GIVENTASK, no_aof)
        assert np.allclose(cor_task, ref, atol=1e-12)
        # the 2019 CalcEval.m kept sessions without AOF (aspect 1); the fixed one gives NaN for them
        ref_top = cor[key.replace('cor', 'cortop')].ravel()[K - 1].ravel()
        keep = ~np.isnan(cortop)
        assert np.allclose(cortop[keep], ref_top[keep], atol=1e-12)
        assert (~keep).sum() == (TABLE2[:, -1].sum() if tag == 'spe' else 0)


def test_saved_results_are_matched_and_reproduce_table2(step1):
    models = pipeline.load_matlab_models(RESULTS)
    no_aof = np.array([not m.any() for m in step1[1]])
    ev = evaluation.evaluate(models, evaluation.GIVENTASK, no_aof=no_aof)
    assert np.array_equal(evaluation.evaluate(models, evaluation.GIVENTASK)['spe'][3]['conf'], TABLE2)
    for tag in ('spe', 'all'):
        for K in range(2, 6):
            k = min(3, K)
            assert np.array_equal(ev[tag][K]['perm'][:k], np.arange(k))
    assert np.array_equal(ev['spe'][3]['conf'], TABLE2)
    assert ev['spe'][3]['accuracy'] == pytest.approx(93 / 111)


def test_aof_similarities_match_distances_mat(gazedata, step1):
    """plot_eval_AoF.m: the four detector similarities equal refs/distances.mat"""
    attrs, seqs = gazedata
    _, mask, session_all = step1
    sessions = pipeline.load_gazedata_sessions(os.path.join(REFS, 'gazedata.mat'))
    assert [s['task'] for s in sessions] == list(evaluation.session_tasks(len(seqs), 3))
    assert all(sorted(s['session_id'] for s in sessions[u::37]) == [1, 2, 3] for u in range(37))
    dists = evaluation.aof_similarities([s['f'] for s in sessions], attrs, session_all, mask,
                                        evaluation.session_tasks(len(seqs), 3), evaluation.GIVENTASK)
    ref = scipy.io.loadmat(os.path.join(REFS, 'distances.mat'))
    for name, value in dists.items():
        assert np.allclose(value, ref[name].ravel(), rtol=0, atol=1e-12, equal_nan=True)
    assert np.isnan(dists['cos_dist_mset']).sum() == TABLE2[:, -1].sum()  # sessions without AOF
    names, p = evaluation.aof_ttests(dists)
    assert p[names.index('cos_dist_top'), names.index('cos_dist_mset')] < 0.01


def test_aof_temporal_counts_fig8(step1):
    """paper Fig. 8: the 2019 for_rev2.m (linear index) gives the paper's values; the corrected count differs"""
    import test_evaluation
    masks = cell_list(scipy.io.loadmat(os.path.join(REFS, 'segsessions.mat'))['seqmask'])
    assert np.array_equal(test_evaluation.ml_for_rev2(masks, buggy=True), [37, 20, 21, 31, 28, 30, 40, 36, 36, 30])
    n_aof, n_points = evaluation.aof_temporal_counts(step1[1])
    assert np.array_equal(n_aof, [579, 779, 818, 878, 802, 962, 977, 919, 932, 791])
    assert np.array_equal(n_points, [319, 360, 374, 380, 354, 390, 396, 377, 369, 325])


# ------------------------------------------------------------------ training (T7)
def ml_em_step(sessions, zs, vz):
    """One iteration of plsa.m (estep.m + mstep.m) with loops, independent of FourLayerPLSA"""
    K, N = zs.shape[1], vz.shape[0]
    gamma, ll = [], 0.0
    for s, f in enumerate(sessions):
        g = np.zeros((K, f.shape[1]))
        for t in range(f.shape[1]):
            x = f[:, t]
            pv = zs[s] * np.array([factorial(int(x.sum())) * np.prod(vz[:, k] ** x) for k in range(K)])
            g[:, t] = pv / pv.sum()
            ll += np.log(pv.sum())
        gamma.append(g)
    new_zs = np.array([g.sum(1) / g.sum() if g.size else np.ones(K) / K for g in gamma])
    rz = np.zeros((K, N))
    for g, f in zip(gamma, sessions):
        rz += g @ (f == 1).T
    return new_zs, (rz / rz.sum(1, keepdims=True)).T, ll


@pytest.mark.parametrize('tag', ['spe', 'all'])
def test_em_from_matlab_p(step1, tag):
    """
    Deterministic check on real data: starting from MATLAB's p_{tag}_topic3, FourLayerPLSA's E/M-steps
    equal plsa.m semantics, and the first log-likelihood is >= MATLAB's L(end) (L(end) belongs to the
    parameters before the last M-step, so EM monotonicity requires this).
    """
    _, mask, session_all = step1
    sessions = SessionProcessor().remove_empty_columns(mask) if tag == 'spe' else session_all
    zs, vz, L = evaluation.load_matlab_result(os.path.join(RESULTS, f'p_{tag}_topic3.mat'))
    assert abs((L[-1] - L[-2]) / L[-1]) <= 1e-4  # plsa.m stopping rule (relative to the new value)
    model = FourLayerPLSA(3)
    model.p_z_s, model.p_v_z = zs.copy(), [vz[:, k].copy() for k in range(3)]
    ref_zs, ref_vz = zs, vz
    for i in range(3):
        gamma, ll = model._e_step_four_layer(sessions)
        model._m_step_four_layer(gamma, sessions)
        ref_zs, ref_vz, ref_ll = ml_em_step(sessions, ref_zs, ref_vz)
        assert ll == pytest.approx(ref_ll, rel=1e-12)
        assert np.allclose(model.p_z_s, ref_zs, atol=1e-12)
        assert np.allclose(np.stack(model.p_v_z, axis=1), ref_vz, atol=1e-12)
        if i == 0:
            assert ll >= L[-1]


def test_python_training_close_to_paper(step1):
    """
    Python training (100 restarts, seed 0) vs paper Table 2.

    Bit-identical results are impossible (MATLAB parfor random streams) and the saved MATLAB p is not a
    fixed point (lim = 1e-4 stops early). Over 20 seeds Python gave 92-95 correct sessions (paper: 93) and
    moved <= 2 sessions w.r.t. Table 2; acceptance criterion (TODO D6): <= 2 sessions moved.
    """
    spe = SessionProcessor().remove_empty_columns(step1[1])
    params, _ = pipeline.train(spe, 3, 1e-4, 100, 1000, jobs=os.cpu_count() or 1, seed=0)
    ev = evaluation.evaluate({'spe': {3: (np.stack(params['vz'], axis=1), params['zs'])}},
                             evaluation.GIVENTASK)['spe'][3]
    assert np.abs(ev['conf'] - TABLE2).sum() // 2 <= 2
    assert np.allclose(np.diag(ev['cor_task']), [0.822, 0.856, 0.862], atol=0.01)
