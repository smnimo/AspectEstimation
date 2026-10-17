"""
Regression tests against the MATLAB outputs used for the paper (python-ver/refs/, not in git).

Skipped when refs/ is not available.

Run from python-ver/:  python -m pytest tests -q
"""

import contextlib
import io
import os
import sys

import numpy as np
import pytest
import scipy.io

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

import main as pipeline  # noqa: E402
from models import evaluation  # noqa: E402
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
def test_calc_eval_matches_cor_mat(tag):
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
        cor_task, cortop = evaluation.calc_eval(vz[:, perm], zs[:, perm],
                                                evaluation.session_tasks(zs.shape[0], 3), evaluation.GIVENTASK)
        assert np.allclose(cor_task, ref, atol=1e-12)
        assert np.allclose(cortop, cor[key.replace('cor', 'cortop')].ravel()[K - 1].ravel(), atol=1e-12)


def test_saved_results_are_matched_and_reproduce_table2():
    models = pipeline.load_matlab_models(RESULTS)
    ev = evaluation.evaluate(models, evaluation.GIVENTASK)
    for tag in ('spe', 'all'):
        for K in range(2, 6):
            k = min(3, K)
            assert np.array_equal(ev[tag][K]['perm'][:k], np.arange(k))
    assert np.array_equal(ev['spe'][3]['conf'], TABLE2)
    assert ev['spe'][3]['accuracy'] == pytest.approx(93 / 111)
