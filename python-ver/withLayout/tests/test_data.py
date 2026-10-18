"""
Tests against the 2014 experiment data and the MATLAB results in matlab-ver/withLayout (not in git; skipped without them).

Run from python-ver/withLayout:  python -m pytest tests -q
"""

import os
import sys

import numpy as np
import pytest
import scipy.io

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import layout_data as ld  # noqa: E402
import layout_eval as le  # noqa: E402

MATLAB_DIR = os.path.abspath(os.path.join(ROOT, '..', '..', 'matlab-ver', 'withLayout'))
DATA = os.path.join(MATLAB_DIR, 'data')
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(DATA, 'content.mat')), reason='data not found')


@pytest.fixture(scope='module')
def data():
    return ld.load_sessions(DATA), ld.load_content(os.path.join(DATA, 'content.mat'))


def test_design(data):
    sessions, content = data
    assert len(sessions) == 81 and all(len(s.f) >= 40 for s in sessions)
    pairs = sorted({(s.task_id, s.content_id) for s in sessions})  # paper Table 1
    assert pairs == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 4), (3, 1), (3, 2), (3, 5)]
    assert [ld.item_attributes(content, sessions[0], a).shape[1] for a in (1, 2, 3)] == [14, 16, 30]


@pytest.mark.skipif(not os.path.exists(os.path.join(MATLAB_DIR, 'area_s.mat')), reason='area_s.mat not found')
def test_layout_effect_equals_area_s(data):
    sessions, _ = data
    area_s = scipy.io.loadmat(os.path.join(MATLAB_DIR, 'area_s.mat'))['area_s']
    for s in range(len(sessions)):
        assert np.allclose(ld.region_frequency(sessions, exclude=s), area_s[:, s], atol=1e-15)


@pytest.mark.skipif(not os.path.exists(os.path.join(MATLAB_DIR, 'result', 'item', 'MP_comp_type1.mat')),
                    reason='MATLAB results not found')
@pytest.mark.parametrize('attr', [1, 2, 3])
def test_comparison_reproduces_matlab_results(data, attr):
    """main_comparison.m results (result/{region,item}/MP_comp_type<attr>.mat) are reproduced exactly"""
    sessions, content = data
    res = le.leave_one_out_comparison(sessions, content, attr, 'matlab')
    for kind in ('region', 'item'):
        m = scipy.io.loadmat(os.path.join(MATLAB_DIR, 'result', kind, f'MP_comp_type{attr}.mat'))['mpatk']
        ref = np.vstack([np.asarray(c).ravel() for c in m[0]])
        assert np.allclose(res[f'mpatk_{kind}'], ref, atol=1e-12)
