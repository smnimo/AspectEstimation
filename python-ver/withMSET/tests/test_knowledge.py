"""
knowledge.csv with any number of items / attributes and categorical or ordinal attributes (data.knowledge, models.aof)

The AOF detection is compared with references written from the definitions with scipy.stats; the paper's setting
(headerless knowledge.csv) is covered by test_matlab_equivalence.py / test_refs.py.
"""

import os
import sys
from collections import Counter

import numpy as np
import pytest
from scipy.stats import binom, multinomial

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

import main as pipeline  # noqa: E402
from data.gaze_reader import GazeDataReader  # noqa: E402
from data.knowledge import CATEGORICAL, ORDINAL, load_knowledge  # noqa: E402
from models import aof, evaluation  # noqa: E402

SAMPLE_DIR = os.path.abspath(os.path.join(ROOT, '..', '..', 'sample_data'))
REFS = os.path.abspath(os.path.join(ROOT, '..', '..', 'matlab-ver', 'withLD', 'refs'))
TYPED = os.path.join(ROOT, 'examples', 'knowledge_typed.csv')
TASK_TYPED = os.path.join(ROOT, 'examples', 'task_typed.csv')


# ---------------------------------------------------------------- knowledge.csv
def test_legacy_knowledge():
    k = load_knowledge(os.path.join(SAMPLE_DIR, 'knowledge.csv'))
    assert k.legacy and k.kinds == [CATEGORICAL] * 5 and k.n_values.tolist() == [3] * 5
    assert np.array_equal(k.value_matrix(), np.loadtxt(os.path.join(SAMPLE_DIR, 'knowledge.csv'), delimiter=',', dtype=int))
    assert k.n_features == 15 and np.array_equal(k.features([4]), [[1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]])


def test_typed_knowledge():
    k = load_knowledge(TYPED)
    assert not k.legacy and k.n_items == 12
    assert k.kinds == [ORDINAL] * 5 + [CATEGORICAL] * 2
    assert k.attributes[5].labels == ['silver', 'black', 'white']
    assert k.attributes[6].labels == ['business', 'gaming', 'travel']
    assert k.values[1][6] == (0, 1)  # gaming;business -> sorted value indices
    assert k.features([2])[0, 18:].tolist() == [1, 1, 0]
    with pytest.raises(ValueError):
        k.value_matrix()


@pytest.mark.parametrize('body, message', [
    ('a,b\nordinal,nominal\n1,2\n', 'kind'),
    ('a,b\nordinal,categorical\n1;2,x\n', 'ordinal'),
    ('a,b\nordinal,categorical\n1\n', 'cells'),
])
def test_invalid_knowledge(tmp_path, body, message):
    path = tmp_path / 'k.csv'
    path.write_text(body)
    with pytest.raises(ValueError, match=message):
        load_knowledge(str(path))


# ---------------------------------------------------------------- AOF detection (MSET)
def ref_pvalue(pmf_all, q):
    """exact test: probability of the outcomes not more likely than q"""
    return sum(p for p in pmf_all if p <= q * (1 + 1e-9))


def ref_categorical(n_values, catalog, seq, alpha, scales):
    """MSET for categorical values: multinomial exact test over the value combinations of the catalog for every window
    of every scale; AOF of a distinctive window: share in the window > share in the catalog; union over windows"""
    import itertools
    T = len(seq)
    counts = Counter(catalog)
    patterns = sorted(counts)
    p = [counts[c] / len(catalog) for c in patterns]
    share = np.array([sum(v in c for c in catalog) for v in range(n_values)]) / len(catalog)
    ls = range(scales[0], scales[1] + 1)
    pvalues, mask = np.full((len(ls), T), np.nan), np.zeros((n_values, T))
    for i, l in enumerate(ls):
        pmf_all = [multinomial.pmf(x, l, p) for x in itertools.product(range(l + 1), repeat=len(p)) if sum(x) == l]
        for t in range(l - 1, T):
            w = seq[t - l + 1:t + 1]
            pvalues[i, t] = ref_pvalue(pmf_all, multinomial.pmf([w.count(c) for c in patterns], l, p))
            if pvalues[i, t] * (1 + 1e-9) < alpha:  # p == alpha is not significant
                in_win = np.array([sum(v in c for c in w) for v in range(n_values)]) / l
                mask[in_win > share, t - l + 1:t + 1] = 1
    return pvalues, mask


def ref_ordinal(n_values, catalog, seq, alpha, scales):
    """MSET for ordinal values: exact binomial test per cut; AOF: biased side of every distinctive cut, united"""
    T = len(seq)
    lv = np.array([c[0] for c in catalog])
    sv = np.array([c[0] for c in seq])
    ls = range(scales[0], scales[1] + 1)
    pvalues, mask = np.full((len(ls), T), np.nan), np.zeros((n_values, T))
    for i, l in enumerate(ls):
        for t in range(l - 1, T):
            w = sv[t - l + 1:t + 1]
            for k in range(n_values - 1):
                p = np.mean(lv > k)
                if p in (0, 1):
                    continue
                u = np.sum(w > k)
                pk = ref_pvalue(binom.pmf(range(l + 1), l, p), binom.pmf(u, l, p))
                pvalues[i, t] = np.nanmin([pvalues[i, t], pk])
                if pk * (1 + 1e-9) < alpha and u / l != p:
                    side = range(k + 1, n_values) if u / l > p else range(k + 1)
                    mask[list(side), t - l + 1:t + 1] = 1
    return pvalues, mask


def random_session(rng, values, n_items, T, bias):
    """catalog value sets of all items, and a browsing sequence that prefers the items in bias"""
    order = rng.permutation(n_items)
    catalog = [values[i] for i in order]
    pref = [i for i in range(n_items) if order[i] in bias]
    f = [int(rng.choice(pref)) if rng.random() < 0.7 else int(rng.integers(n_items)) for _ in range(T)]
    return catalog, [catalog[i] for i in f]


@pytest.mark.parametrize('alpha, scales', [(0.05, (1, 10)), (0.01, (3, 6))])
@pytest.mark.parametrize('seed', range(3))
def test_categorical_multivalued_matches_reference(seed, alpha, scales):
    rng = np.random.default_rng(seed)
    n_values = 4
    values = [tuple(sorted(set(rng.choice(n_values, rng.integers(1, 3)).tolist()))) for _ in range(10)]
    catalog, seq = random_session(rng, values, 10, 40, bias={0, 1, 2})
    pv, mask = aof.attribute_aof(CATEGORICAL, n_values, catalog, seq, alpha, scales)
    rpv, rmask = ref_categorical(n_values, catalog, seq, alpha, scales)
    assert np.allclose(pv, rpv, rtol=1e-9, atol=0, equal_nan=True)
    assert np.array_equal(mask, rmask) and mask.any()


@pytest.mark.parametrize('alpha, scales', [(0.05, (1, 10)), (0.01, (3, 6))])
@pytest.mark.parametrize('seed', range(3))
def test_ordinal_matches_reference(seed, alpha, scales):
    rng = np.random.default_rng(seed)
    n_values = 4
    values = [(int(v),) for v in rng.integers(0, n_values, 12)]
    catalog, seq = random_session(rng, values, 12, 40, bias=set(np.argsort([v[0] for v in values])[-3:]))
    pv, mask = aof.attribute_aof(ORDINAL, n_values, catalog, seq, alpha, scales)
    rpv, rmask = ref_ordinal(n_values, catalog, seq, alpha, scales)
    assert np.allclose(pv, rpv, rtol=1e-9, atol=0, equal_nan=True)
    assert np.array_equal(mask, rmask) and mask.any()


def test_pvalue_matches_paper_example():
    """3 values of equal share, window of 5: (5, 0, 0) -> 3 * (1/3)^5; (2, 2, 1) -> 1 (the most likely outcomes)"""
    one = [(0,), (1,), (2,)] * 4
    pv, _ = aof.attribute_aof(CATEGORICAL, 3, one, [(0,)] * 5, 0.05, (5, 5))
    assert pv[0, -1] == pytest.approx(3 / 3 ** 5)
    pv, _ = aof.attribute_aof(CATEGORICAL, 3, one, [(0,), (0,), (1,), (1,), (2,)], 0.05, (5, 5))
    assert pv[0, -1] == pytest.approx(1)


def test_multiscale_union():
    """a run of 4 equal values is distinctive at scale 4 (p = 3 / 81 < 0.05) but not at scales 1..3; the AOF cover
    the 4 time points of the run only"""
    one = [(0,), (1,), (2,)] * 4
    seq = [(1,), (2,), (0,), (0,), (0,), (0,), (1,), (2,)]
    pv, mask = aof.attribute_aof(CATEGORICAL, 3, one, seq, 0.05, (1, 4))
    assert np.nanmin(pv[:3]) > 0.05 and pv[3, 5] == pytest.approx(3 / 81)
    assert mask.tolist() == [[0, 0, 1, 1, 1, 1, 0, 0], [0] * 8, [0] * 8]
    _, mask3 = aof.attribute_aof(CATEGORICAL, 3, one, seq, 0.05, (1, 3))
    assert not mask3.any()


def test_ordinal_sides():
    """window of the highest level: only cut 1|2 is distinctive -> levels above it; an absent level gives no cut"""
    catalog = [(0,), (0,), (1,), (1,), (2,), (2,)]
    pv, mask = aof.attribute_aof(ORDINAL, 4, catalog, [(2,)] * 5, 0.05, (5, 5))
    assert pv[0, -1] == pytest.approx((1 / 3) ** 5)  # cut 1|2 (1/3 above); cut 0|1: p = 0.177; cut 2|3 skipped
    assert mask[:, 0].tolist() == [0, 0, 1, 1]


def test_ordinal_union_of_sides():
    """10 items of the middle level: cut 0|1 (all above) and cut 1|2 (none above) are both distinctive; the union of
    their sides is every level"""
    catalog = [(0,), (1,), (2,)] * 4
    pv, mask = aof.attribute_aof(ORDINAL, 3, catalog, [(1,)] * 10, 0.05, (10, 10))
    assert pv[0, -1] < 0.05
    assert mask[:, 0].tolist() == [1, 1, 1]


@pytest.mark.parametrize('n_values', [3, 4, 6])
def test_false_positive_rate(n_values):
    """neutral browsing: a window is distinctive with probability <= alpha for any number of values"""
    import itertools
    catalog = [(v,) for v in range(n_values)] * 2
    rate = 0.0
    for x in (x for x in itertools.product(range(6), repeat=n_values) if sum(x) == 5):
        w = [(v,) for v in range(n_values) for _ in range(x[v])]
        pv, _ = aof.attribute_aof(CATEGORICAL, n_values, catalog, w, 0.05, (5, 5))
        rate += (pv[0, -1] < 0.05) * multinomial.pmf(x, 5, [1 / n_values] * n_values)
    assert 0 < rate <= 0.05


def test_paper_data_as_typed_categorical(tmp_path):
    """the integer (paper) path and a typed knowledge.csv with the same categorical attributes give the same masks"""
    attrs, seqs = pipeline.create_demo_data(n_users=2)
    knowledge, catalogs, seqsets = pipeline.legacy_sets(attrs, seqs)
    _, mask, label = pipeline.detect_aof(knowledge, catalogs, seqsets)
    for s, (a, q) in enumerate(zip(attrs, seqs)):
        _, m2, s2 = aof.session_aof([CATEGORICAL] * 5, [3] * 5, aof.legacy_value_sets(a), aof.legacy_value_sets(q.T))
        assert np.array_equal(m2, mask[s]) and np.array_equal(s2, label[s])


# ---------------------------------------------------------------- reader and pipeline
def write_sessions(tmp_path, n_users=2, tasks=(1, 2, 3)):
    """raw files (copies of the sample session, items shuffled per session) in ReadwithTimestamp.m order"""
    import shutil
    rng = np.random.default_rng(0)
    names = []
    for u in range(n_users):
        for i, task in enumerate(list(tasks) + [None]):
            name = f'u{u + 1}s{i + 1}'
            names.append(name)
            if task is not None:
                shutil.copy(os.path.join(SAMPLE_DIR, '170116024150092.csv'), tmp_path / f'{name}.csv')
                np.savetxt(tmp_path / f'{name}.list', rng.permutation(12) + 1 + task * 100, fmt='%d')
    (tmp_path / 'timestamp.list').write_text('\n'.join(names) + '\n')
    return str(tmp_path / 'timestamp.list')


@pytest.mark.skipif(not os.path.exists(os.path.join(SAMPLE_DIR, '170116024150092.csv')), reason='sample_data not found')
def test_reader_typed_knowledge(tmp_path):
    reader = GazeDataReader(base_path='.', knowledge_csv_path=TYPED)
    data = reader.load_all_sessions(write_sessions(tmp_path))
    assert data['KB'] is None and data['knowledge'].n_features == 21
    s = data['session'][(1, 2)]
    assert s['semantic_attribute_seq'] is None
    assert np.array_equal(s['feature_seq'], reader.knowledge.features(s['list'][s['f'] - 1]).T)

    out = tmp_path / 'gazedata.mat'
    reader.save_gazedata(data, str(out))
    loaded = pipeline.load_gazedata_sessions(str(out))
    ref = pipeline.reader_session_list(data)
    assert [x['task'] for x in loaded] == [1, 1, 2, 2, 3, 3]
    for a, b in zip(loaded, ref):
        assert np.array_equal(a['f'], b['f']) and np.array_equal(a['list'], b['list'])
        assert a['usr'] == b['usr'] and a['session_id'] == b['session_id']

    reader = GazeDataReader(base_path='.', knowledge_csv_path=TYPED, grid=(2, 4))
    with pytest.raises(ValueError, match='grid'):
        reader.load_all_sessions(str(tmp_path / 'timestamp.list'))


def test_grid_positions():
    reader = GazeDataReader(base_path='.', grid=(2, 3))
    data = np.array([[0, 0.1, 0.1, 0.1, 0.1], [1, 0.9, 0.9, 0.9, 0.9], [2, 0.5, 0.2, 0.5, 0.2]])
    assert reader.gaze_to_items(data).tolist() == [1, 6, 2]


def test_giventask_resolution(tmp_path):
    legacy = load_knowledge(os.path.join(SAMPLE_DIR, 'knowledge.csv'))
    typed = load_knowledge(TYPED)
    assert np.array_equal(pipeline.resolve_giventask(None, legacy), evaluation.GIVENTASK)
    assert pipeline.resolve_giventask(None, typed) is None
    g = pipeline.resolve_giventask(TASK_TYPED, typed)
    assert g.shape == (3, 21) and np.array_equal(g[:, :15], evaluation.GIVENTASK)
    with pytest.raises(ValueError, match='columns'):
        pipeline.resolve_giventask(TASK_TYPED, legacy)


def test_typed_pipeline_smoke(tmp_path):
    """learning, evaluation and figures with 7 attributes (ordinal and multi-valued categorical, 21 features)"""
    import matplotlib
    matplotlib.use('Agg')
    knowledge = load_knowledge(TYPED)
    rng = np.random.default_rng(1)
    task_items = {1: [0, 1], 2: [7, 10], 3: [2, 3]}  # items matching the task-related values
    sessions = []
    for task in (1, 2, 3):
        for usr in (1, 2, 3):
            items = rng.permutation(12) + 1
            pos = [int(np.nonzero(items == i + 1)[0][0]) + 1 for i in task_items[task]]
            f = list(rng.permutation(12) + 1) + [int(rng.choice(pos)) if rng.random() < 0.8
                                                 else int(rng.integers(1, 13)) for _ in range(20)]
            sessions.append({'list': items, 'f': np.array(f), 'task': task, 'usr': usr})
    catalogs, seqsets = pipeline.session_value_sets(knowledge, sessions)
    res = pipeline.learn_aspects_from_sets(knowledge, catalogs, seqsets, max_topics=3, train_num=2, max_itr=50,
                                           savedir=str(tmp_path))
    assert res['seqmask'][0].shape[0] == 21 and any(m.any() for m in res['seqmask'])
    giventask = pipeline.resolve_giventask(TASK_TYPED, knowledge)
    pipeline.check_task_order(sessions, giventask.shape[0])
    models = {tag: {K: (np.stack(r['params']['vz'], axis=1), r['params']['zs']) for K, r in res[tag].items()}
              for tag in ('spe', 'all')}
    aof_sim = evaluation.aof_similarities([s['f'] for s in sessions], [knowledge.features(s['list']) for s in sessions],
                                          res['session_all'], res['seqmask'], np.array([s['task'] for s in sessions]),
                                          giventask)
    no_aof = np.array([not m.any() for m in res['seqmask']])
    ev = pipeline.run_evaluation(models, giventask, str(tmp_path), aof_sim, no_aof, res['seqmask'])
    assert ev['spe'][3]['conf'].sum() == 9
    histories = {tag: {K: r['history'] for K, r in res[tag].items()} for tag in ('spe', 'all')}
    paths = pipeline.save_figures(ev, giventask, histories, str(tmp_path / 'figures'), aof_sim, None,
                                  res['session_all'], res['seqmask'], knowledge.attributes)
    assert len(paths) > 5
