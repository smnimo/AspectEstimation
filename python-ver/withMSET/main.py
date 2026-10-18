"""
Main pipeline of AspectEstimation (two-step approach)

Based on matlab-ver/main_mask.m (learn_aspects_from_mask.m), with step 1 replaced by MSET:

  1st step: multi-scale exact test (MSET, AAAI 2020; models.aof) -> AOF masks
            (matlab-ver: likelihood of windows of 5 items < 0.03, calc_lkh.m + calc_segments.m; not ported)
  2nd step: pLSA on the AOF masks ("spe") and, for comparison, on all attribute values ("all")

Evaluation (calc_matching.m, CalcSim.m, plot_conf_matrix.m) with --eval, or on MATLAB results with --eval-matlab.

The number of items, the attributes and their kinds (categorical / ordinal) are read from knowledge.csv
(data.knowledge); without --knowledge, the integer attributes of gazedata.mat are used as in the paper.

Usage:
  python main.py --mat ../gazedata.mat          # gazedata.mat made by ReadwithTimestamp.m
  python main.py --mat ../gazedata.mat --eval   # + matching, cosine similarity, confusion matrix
  python main.py --raw ../experiment/timestamp.list --knowledge knowledge.csv --grid 3 4 \
      --task-att task.csv --eval                # raw gaze data (gaze_reader) with any knowledge.csv
  python main.py --mat gazedata.mat --knowledge knowledge.csv   # session structs (f, list) of gazedata.mat
  python main.py --eval-matlab ../../matlab-ver/withLD/refs/results   # evaluate MATLAB p_{spe,all}_topicK.mat
  python main.py --mat ../gazedata.mat --plot   # + figures of the plot_*.m scripts in results/figures
  python main.py --demo --train-num 5           # synthetic data (smoke test)
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python-ver (common)

from data import knowledge as kb
from models import evaluation
from models.plsa import FourLayerPLSA
from models import aof as aof_detection
from models.segments import SessionProcessor
from common.restarts import run_restarts


# ---------------------------------------------------------------- data input
def load_gazedata_mat(path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load semantic_attributes / semantic_attribute_seq from a MATLAB gazedata.mat

    The cell arrays are indexed {usr, task} in MATLAB and traversed with (:) there,
    i.e., in column-major order, so the same order is used here.
    """
    import scipy.io
    data = scipy.io.loadmat(path)

    def to_list(cells: np.ndarray) -> List[np.ndarray]:
        out = [np.asarray(c).astype(int) for c in cells.flatten(order='F')]
        if any(c.size == 0 for c in out):
            raise ValueError("gazedata contains empty sessions")
        return out

    return to_list(data['semantic_attributes']), to_list(data['semantic_attribute_seq'])


def load_gazedata_sessions(path: str) -> List[Dict]:
    """
    Load the session structs of a MATLAB gazedata.mat in cell(:) order (column-major)

    Returns:
        list of dicts with the fields present in the file (f, dur, list, task, usr, totalduration,
        session_id, candidate, selected; scalars as int / float, vectors as 1-D arrays)
    """
    import scipy.io
    cells = scipy.io.loadmat(path)['session'].flatten(order='F')
    out = []
    for c in cells:
        d = {}
        for name in c.dtype.names:
            v = np.asarray(c[name][0, 0]).ravel()
            d[name] = v[0].item() if v.size == 1 and name not in ('f', 'dur', 'list') else v
        for name in ('f', 'list', 'task', 'usr', 'session_id'):
            if name in d:
                d[name] = np.asarray(d[name]).astype(int) if np.ndim(d[name]) else int(d[name])
        out.append(d)
    return out


def reader_session_list(data: Dict) -> List[Dict]:
    """
    Sessions of GazeDataReader output (dicts keyed by (usr, task)) ordered like MATLAB's cell(:)
    (column-major: by task, then by user)
    """
    return [data['session'][k] for k in sorted(data['session'].keys(), key=lambda k: (k[1], k[0]))]


def from_reader_sessions(data: Dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Integer attributes (semantic_attributes, semantic_attribute_seq) of GazeDataReader output in cell(:) order"""
    sessions = reader_session_list(data)
    attrs = [np.asarray(s['semantic_attributes']).astype(int) for s in sessions]
    seqs = [np.asarray(s['semantic_attribute_seq']).astype(int) for s in sessions]
    return attrs, seqs


def session_value_sets(knowledge: 'kb.Knowledge', sessions: List[Dict]):
    """
    Value sets of the displayed and browsed items from the session structs (list: item ids of the grid cells,
    f: browsed grid positions, both 1-based)

    Returns:
        (catalogs, seqsets): catalogs[s][item][attribute], seqsets[s][t][attribute] (0-based value sets)
    """
    catalogs, seqsets = [], []
    for s in sessions:
        items = np.asarray(s['list'], dtype=int).ravel()
        f = np.asarray(s['f'], dtype=int).ravel()
        if np.any((items < 1) | (items > knowledge.n_items)):
            raise ValueError(f"item ids out of 1..{knowledge.n_items} (knowledge.csv): {items}")
        if np.any((f < 1) | (f > len(items))):
            raise ValueError(f"browsed positions out of 1..{len(items)}")
        catalog = knowledge.item_values(items)
        catalogs.append(catalog)
        seqsets.append([catalog[i - 1] for i in f])
    return catalogs, seqsets


def create_demo_data(n_users: int = 8, seed: int = 0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Synthetic sessions in the setting of the paper (12 items, 5 attributes x 3 values, 3 tasks)"""
    rng = np.random.default_rng(seed)
    # attributes: price, screen size, CPU score, memory capacity, weight (each value has 4 items)
    kbm = np.stack([rng.permutation(np.repeat([1, 2, 3], 4)) for _ in range(5)], axis=1)
    task_values = {1: [(2, 3), (3, 3)], 2: [(1, 1), (4, 1)], 3: [(3, 3), (0, 1)]}
    attrs, seqs = [], []
    for task in (1, 2, 3):  # column-major order: task, then user
        for _ in range(n_users):
            a = kbm[rng.permutation(12)]
            score = sum((a[:, k] == v).astype(int) for k, v in task_values[task])
            cand = np.argsort(-(score + rng.random(12) * 0.1))[:2]
            f = list(rng.permutation(12))
            f += [int(cand[i % 2]) if rng.random() > 0.15 else int(rng.integers(12)) for i in range(14)]
            f += list(rng.integers(0, 12, 6))
            attrs.append(a)
            seqs.append(a[np.array(f)].T)
    return attrs, seqs


# ---------------------------------------------------------------- 1st step
def legacy_sets(semantic_attributes: List[np.ndarray], semantic_attribute_seq: List[np.ndarray]):
    """
    Integer inputs of the paper (gazedata.mat) as value sets: every attribute single-valued categorical, with the
    number of values taken from the catalogs of all sessions (as calc_lkh.m)

    Returns:
        (knowledge, catalogs, seqsets): catalogs[s][item][attribute], seqsets[s][t][attribute] value sets
    """
    n_values = np.max(np.vstack(semantic_attributes), axis=0)
    knowledge = kb.from_value_matrix(np.vstack(semantic_attributes), n_values=n_values)
    catalogs = [aof_detection.legacy_value_sets(a) for a in semantic_attributes]
    seqsets = [aof_detection.legacy_value_sets(np.asarray(s).T) for s in semantic_attribute_seq]
    return knowledge, catalogs, seqsets


def detect_aof(knowledge: 'kb.Knowledge', catalogs, seqsets, alpha: float = aof_detection.DEFAULT_ALPHA,
               scales: Tuple[int, int] = aof_detection.DEFAULT_SCALES):
    """
    Step 1 (MSET, models.aof) for all sessions

    Args:
        alpha: significance level; scales: (smallest, largest) window length

    Returns:
        (pvalues, seqmask, seglabel): per session pvalues[attribute] (scales x T), mask (n_features x T),
        segment labels (T,)
    """
    out = [aof_detection.session_aof(knowledge.kinds, knowledge.n_values, c, q, alpha, scales)
           for c, q in zip(catalogs, seqsets)]
    return [o[0] for o in out], [o[1] for o in out], [o[2] for o in out]


# ---------------------------------------------------------------- 2nd step
def _train_once(sessions, n_topics, lim, max_itr):
    return FourLayerPLSA(n_topics).fit(sessions, lim, max_itr)


def train(sessions: List[np.ndarray], n_topics: int, lim: float, train_num: int,
          max_itr: int, jobs: int = 1, seed: int = 0) -> Tuple[Dict, List[float]]:
    """Best of train_num random restarts by final log-likelihood (train.m; parfor -> processes)"""
    results = run_restarts(_train_once, (sessions, n_topics, lim, max_itr), train_num, jobs, seed)
    results = [r for r in results if r[1]]
    if not results:
        raise RuntimeError("All training runs failed (log-likelihood decreased at the first iteration)")
    return max(results, key=lambda r: r[1][-1])


def _object_array(arrays: List[np.ndarray]) -> np.ndarray:
    """1-D object array of per-session arrays (like a MATLAB cell); np.array(..., dtype=object)
    fails or stacks when the arrays share their first dimension"""
    out = np.empty(len(arrays), dtype=object)
    out[:] = arrays
    return out


def learn_aspects(semantic_attributes: List[np.ndarray], semantic_attribute_seq: List[np.ndarray],
                  alpha: float = aof_detection.DEFAULT_ALPHA, scales: Tuple[int, int] = aof_detection.DEFAULT_SCALES,
                  max_topics: int = 5, train_num: int = 100, lim: float = 1e-4, max_itr: int = 1000,
                  jobs: int = 1, seed: int = 0, savedir: str = 'results') -> Dict:
    """Run the whole pipeline on the integer inputs of the paper (gazedata.mat; every attribute categorical)"""
    knowledge, catalogs, seqsets = legacy_sets(semantic_attributes, semantic_attribute_seq)
    return learn_aspects_from_sets(knowledge, catalogs, seqsets, alpha, scales, max_topics, train_num, lim,
                                   max_itr, jobs, seed, savedir)


def learn_aspects_from_sets(knowledge: 'kb.Knowledge', catalogs, seqsets, alpha: float = aof_detection.DEFAULT_ALPHA,
                            scales: Tuple[int, int] = aof_detection.DEFAULT_SCALES, max_topics: int = 5,
                            train_num: int = 100, lim: float = 1e-4, max_itr: int = 1000,
                            jobs: int = 1, seed: int = 0, savedir: str = 'results') -> Dict:
    """
    Run the whole pipeline on value sets (any number of attributes / values, categorical or ordinal attributes)

    Args:
        knowledge: attributes (data.knowledge.Knowledge)
        catalogs: catalogs[s][item][attribute] value sets of the displayed items of session s
        seqsets: seqsets[s][t][attribute] value sets of the browsed items of session s
        alpha, scales: step 1 (MSET) significance level and window lengths (see detect_aof)
    """
    S = len(seqsets)
    print(f"step 1: MSET, windows of {scales[0]}..{scales[1]} items, p < {alpha}")
    pvalues, seqmask, seglabel = detect_aof(knowledge, catalogs, seqsets, alpha, scales)
    session_all = [knowledge.set_features(q).T for q in seqsets]
    session_spe = SessionProcessor().remove_empty_columns(seqmask)

    n_aof = sum(m.shape[1] for m in session_spe)
    n_no_aof = sum(1 for m in session_spe if m.shape[1] == 0)
    print(f"{S} sessions, {n_aof} time points with AOF, {n_no_aof} sessions without any AOF")

    os.makedirs(savedir, exist_ok=True)
    np.savez(os.path.join(savedir, 'segsessions.npz'),
             seglabel=_object_array(seglabel), seqmask=_object_array(seqmask),
             session_all=_object_array(session_all), alpha=alpha, scales=np.array(scales))

    results = {'seglabel': seglabel, 'seqmask': seqmask, 'session_all': session_all, 'knowledge': knowledge,
               'spe': {}, 'all': {}}
    for topic in range(1, max_topics + 1):
        for tag, sessions in (('spe', session_spe), ('all', session_all)):
            t0 = time.time()
            params, history = train(sessions, topic, lim, train_num, max_itr, jobs, seed)
            vz = np.stack(params['vz'], axis=1)  # (features x topics) as cell2mat(p.vz)
            np.savez(os.path.join(savedir, f'p_{tag}_topic{topic}.npz'),
                     zs=params['zs'], vz=vz, L=np.array(history))
            np.savetxt(os.path.join(savedir, f'Akq_{tag}_topic{topic}.csv'), vz, delimiter=',')
            results[tag][topic] = {'params': params, 'history': history}
            print(f"topic {topic} [{tag}]: log-likelihood {history[-1]:.4f}, "
                  f"{len(history)} iterations, {time.time() - t0:.1f}s")
    return results


# ---------------------------------------------------------------- evaluation
def run_evaluation(models: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
                   giventask: np.ndarray, savedir: str, aof: Dict[str, np.ndarray] = None,
                   no_aof: np.ndarray = None, seqmask: List[np.ndarray] = None) -> Dict:
    """
    Evaluate models[tag][K] = (vz, zs) and save the matched results to savedir/evaluation.npz

    Unlike calc_matching.m, the training outputs (p_*_topicK) are not overwritten.
    aof (evaluation.aof_similarities, plot_eval_AoF.m) is printed and saved as well when given.
    no_aof: sessions without AOF (see evaluation.evaluate); detected from the uniform zs if None.
    seqmask: AOF masks; the AOF per temporal position (paper Fig. 8) is printed and saved when given.
    """
    ev = evaluation.evaluate(models, giventask, no_aof=no_aof)
    for line in evaluation.format_summary(ev, giventask.shape[0]):
        print(line)
    arrays = {f'{tag}_topic{K}_{name}': value
              for tag, by_topic in ev.items() for K, r in by_topic.items() for name, value in r.items()}
    if aof is not None:
        names, p = evaluation.aof_ttests(aof)
        print("AOF detection, mean cosine similarity (sessions with AOF): " +
              ", ".join(f"{n} = {np.mean(aof[n][~np.isnan(aof['cos_dist_mset'])]):.3f}" for n in names))
        print(f"  t-test p-values (upper triangle, order {names}):\n{np.round(p, 4)}")
        arrays.update(aof)
        arrays['aof_ttest_p'] = p
    if seqmask is not None:
        arrays['aof_temporal_n_aof'], arrays['aof_temporal_n_points'] = evaluation.aof_temporal_counts(seqmask)
        print(f"AOF per temporal position (10 segments, paper Fig. 8): {arrays['aof_temporal_n_aof'].astype(int).tolist()}")
    os.makedirs(savedir, exist_ok=True)
    np.savez(os.path.join(savedir, 'evaluation.npz'), **arrays)
    return ev


def save_figures(ev: Dict, giventask: np.ndarray, histories: Dict[str, Dict[int, List[float]]], figdir: str,
                 aof: Dict[str, np.ndarray] = None, sessions: List[Dict] = None,
                 session_all: List[np.ndarray] = None, seqmask: List[np.ndarray] = None,
                 attributes: List['kb.Attribute'] = None) -> List[str]:
    """
    Figures of plot_L_topic.m, plot_conf_matrix.m, plot_sim.m, plot_sim_task_asp.m, plot_aspect.R and,
    with session data, plot_eval_AoF.m, plot_dur_usr.m, paper Fig. 8 (corrected count), plot_mask_for_paper.m (sessions 17, 34, 46)

    Args:
        ev: output of evaluation.evaluate (matched aspects)
        histories: histories[tag][K] = log-likelihood history
        aof, sessions, session_all, seqmask: optional session data (sessions: load_gazedata_sessions)
        attributes: attributes of the features (Knowledge.attributes; default: the paper's names)

    Returns:
        paths of the saved PNG files
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from visualization import plotter

    os.makedirs(figdir, exist_ok=True)
    figs = {}
    Ks = sorted(set(ev['spe']) & set(ev['all']))
    figs['L_topic'] = plotter.plot_L_topic([histories['all'][K][-1] for K in Ks],
                                           [histories['spe'][K][-1] for K in Ks])
    for tag, name in (('all', 'all'), ('spe', 'MSET')):
        figs[f'aspects_{tag}'] = plotter.plot_aspects({K: ev[tag][K]['vz'] for K in Ks}, giventask, name, attributes)
        figs[f'likelihood_convergence_{tag}'] = plotter.plot_likelihood_convergence(histories[tag], name)
        for K in Ks:
            if K >= 2:  # paper Fig. 9
                figs[f'aspect_circles_{tag}_topic{K}'] = plotter.plot_aspect_circles(ev[tag][K]['vz'], attributes)
    if 3 in Ks:
        figs['conf_spe'] = plotter.plot_conf_matrix(ev['spe'][3]['conf'], 'Task estimation MSET: confusion matrix')
        figs['conf_all'] = plotter.plot_conf_matrix(ev['all'][3]['conf'], 'Task estimation w/o MSET: confusion matrix')
        figs['sim'] = plotter.plot_sim(ev['spe'][3]['cor_task'], ev['all'][3]['cor_task'])
        figs['sim_task_asp'] = plotter.plot_sim_task_asp(ev['spe'][3]['cor_task'], ev['all'][3]['cor_task'])
    if aof is not None:
        figs['eval_aof'] = plotter.plot_eval_aof(aof)
    if sessions is not None and all('session_id' in s and 'totalduration' in s for s in sessions):
        dur = np.zeros((max(s['usr'] for s in sessions), max(s['session_id'] for s in sessions)))
        for s in sessions:
            dur[s['usr'] - 1, s['session_id'] - 1] = s['totalduration']
        # participants highlighted by plot_dur_usr.m: only for the paper's data
        figs['dur_usr'] = (plotter.plot_dur_usr(dur) if attributes is None else
                           plotter.plot_dur_usr(dur, (), (), ylim=max(1.1 * dur.max(), 1)))
    if seqmask is not None:
        figs['aof_temporal'] = plotter.plot_aof_temporal(evaluation.aof_temporal_counts(seqmask)[0])
    if session_all is not None and seqmask is not None:
        for s in (17, 34, 46):  # MATLAB indices of plot_mask_for_paper.m
            if s <= len(seqmask):
                title = (f"participant:{sessions[s - 1]['usr']}, task:{sessions[s - 1]['task']}"
                         if sessions is not None else f'session {s}')
                figs[f'mask_session{s}'] = plotter.plot_mask(session_all[s - 1], seqmask[s - 1], title, attributes)

    paths = []
    for name, fig in figs.items():
        path = os.path.join(figdir, f'{name}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    print(f"Saved {len(paths)} figures to {figdir}")
    return paths


def load_matlab_models(loaddir: str, max_topics: int = 5) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """Load p_{spe,all}_topicK.mat (K = 1..max_topics) saved by learn_aspects_from_mask.m"""
    models = {}
    for tag in ('spe', 'all'):
        models[tag] = {}
        for K in range(1, max_topics + 1):
            path = os.path.join(loaddir, f'p_{tag}_topic{K}.mat')
            if os.path.exists(path):
                zs, vz, _ = evaluation.load_matlab_result(path)
                models[tag][K] = (vz, zs)
    return models


def load_matlab_histories(loaddir: str, max_topics: int = 5) -> Dict[str, Dict[int, List[float]]]:
    """Log-likelihood histories L of p_{spe,all}_topicK.mat"""
    return {tag: {K: list(evaluation.load_matlab_result(os.path.join(loaddir, f'p_{tag}_topic{K}.mat'))[2])
                  for K in range(1, max_topics + 1)
                  if os.path.exists(os.path.join(loaddir, f'p_{tag}_topic{K}.mat'))}
            for tag in ('spe', 'all')}


def resolve_giventask(path: str, knowledge: 'kb.Knowledge' = None):
    """
    Task-related values: from path (task_att.mat or CSV, checked against the features of knowledge), else the
    built-in values of the paper when the features are those of the paper (headerless knowledge.csv with
    5 attributes x 3 values), else None
    """
    if path is not None:
        return evaluation.load_giventask(path, None if knowledge is None else knowledge.n_features)
    if knowledge is None or (knowledge.legacy and knowledge.n_features == evaluation.GIVENTASK.shape[1]):
        return evaluation.load_giventask()
    return None


def check_task_order(sessions: List[Dict], n_task: int):
    """The evaluation expects the sessions in cell(:) order: equal numbers of users, grouped by task 1..n_task"""
    tasks = np.array([int(s['task']) for s in sessions])
    if len(tasks) % n_task or not np.array_equal(tasks, evaluation.session_tasks(len(tasks), n_task)):
        raise ValueError(f"the evaluation needs every user to have one session of each of the {n_task} tasks "
                         f"(tasks of the sessions: {tasks.tolist()})")


def main():
    parser = argparse.ArgumentParser(description="AspectEstimation pipeline (main_mask.m)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--mat', type=str, help="gazedata.mat created by ReadwithTimestamp.m (or data/gaze_reader.py)")
    src.add_argument('--raw', type=str, metavar='TIMESTAMP_LIST',
                     help="raw gaze data: timestamp.list with <name>.csv / <name>.list (needs --knowledge)")
    src.add_argument('--demo', action='store_true', help="use synthetic data")
    src.add_argument('--eval-matlab', type=str, metavar='DIR',
                     help="only evaluate MATLAB results (p_{spe,all}_topicK.mat) in DIR")
    parser.add_argument('--knowledge', type=str, default=None,
                        help="knowledge.csv: item attributes and their kinds (see data/knowledge.py); with --mat, "
                             "the session structs (f, list) are used instead of the integer attributes")
    parser.add_argument('--grid', type=int, nargs=2, default=(3, 4), metavar=('ROWS', 'COLS'),
                        help="item grid on the screen for --raw (default: 3 4)")
    parser.add_argument('--experiment-dir', type=str, default=None,
                        help="directory of the raw files for --raw (default: directory of timestamp.list)")
    parser.add_argument('--eval', action='store_true', help="evaluate the learned aspects")
    parser.add_argument('--plot', action='store_true', help="evaluate and save figures to SAVEDIR/figures")
    parser.add_argument('--task-att', type=str, default=None,
                        help="task-related values: task_att.mat (giventask) or CSV (one row per task, one 0/1 column "
                             "per feature); default: built-in values of the paper (paper features only)")
    parser.add_argument('--alpha', type=float, default=aof_detection.DEFAULT_ALPHA,
                        help=f"step 1 (MSET) significance level (default: {aof_detection.DEFAULT_ALPHA})")
    parser.add_argument('--scales', type=int, nargs=2, default=aof_detection.DEFAULT_SCALES, metavar=('MIN', 'MAX'),
                        help="step 1 (MSET) window lengths (default: %(default)s)")
    parser.add_argument('--max-topics', type=int, default=5)
    parser.add_argument('--train-num', type=int, default=100, help="number of random restarts")
    parser.add_argument('--lim', type=float, default=1e-4, help="EM convergence threshold")
    parser.add_argument('--max-itr', type=int, default=1000)
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--savedir', type=str, default='results')
    args = parser.parse_args()
    if args.raw and not args.knowledge:
        parser.error("--raw needs --knowledge")
    figdir = os.path.join(args.savedir, 'figures')

    if args.eval_matlab:
        giventask = resolve_giventask(args.task_att)
        ev = run_evaluation(load_matlab_models(args.eval_matlab, args.max_topics), giventask, args.savedir)
        if args.plot:
            save_figures(ev, giventask, load_matlab_histories(args.eval_matlab, args.max_topics), figdir)
        return

    sessions, item_features = None, None
    if args.knowledge:
        if args.raw:
            from data.gaze_reader import GazeDataReader
            reader = GazeDataReader(base_path='.', knowledge_csv_path=args.knowledge, grid=args.grid)
            sessions = reader_session_list(reader.load_all_sessions(args.raw, args.experiment_dir))
            knowledge = reader.knowledge
        else:
            knowledge = kb.load_knowledge(args.knowledge)
            sessions = load_gazedata_sessions(args.mat)
        catalogs, seqsets = session_value_sets(knowledge, sessions)
        item_features = [knowledge.features(np.asarray(s['list'], dtype=int).ravel()) for s in sessions]
    else:
        if args.demo:
            attrs, seqs = create_demo_data(seed=args.seed)
        else:
            attrs, seqs = load_gazedata_mat(args.mat)
            sessions = load_gazedata_sessions(args.mat)  # f, totalduration, session_id for plot_eval_AoF.m etc.
            item_features = evaluation.legacy_item_features(attrs)
        knowledge, catalogs, seqsets = legacy_sets(attrs, seqs)

    giventask = None
    if args.eval or args.plot:  # checked before the (long) training
        giventask = resolve_giventask(args.task_att, knowledge)
        if giventask is None:
            parser.error(f"--eval / --plot need --task-att for these attributes ({knowledge.n_features} features: "
                         f"{', '.join(knowledge.feature_labels())})")
        if sessions is not None:
            check_task_order(sessions, giventask.shape[0])

    results = learn_aspects_from_sets(knowledge, catalogs, seqsets, args.alpha, tuple(args.scales), args.max_topics,
                                      args.train_num, args.lim, args.max_itr, args.jobs, args.seed, args.savedir)
    if args.eval or args.plot:
        models = {tag: {K: (np.stack(r['params']['vz'], axis=1), r['params']['zs'])
                        for K, r in results[tag].items()} for tag in ('spe', 'all')}
        aof = None
        if sessions is not None:  # plot_eval_AoF.m
            aof = evaluation.aof_similarities([s['f'] for s in sessions], item_features, results['session_all'],
                                              results['seqmask'], np.array([int(s['task']) for s in sessions]),
                                              giventask)
        no_aof = np.array([not m.any() for m in results['seqmask']])
        ev = run_evaluation(models, giventask, args.savedir, aof, no_aof, results['seqmask'])
        if args.plot:
            histories = {tag: {K: r['history'] for K, r in results[tag].items()} for tag in ('spe', 'all')}
            attributes = None if knowledge.legacy and knowledge.n_features == 15 else knowledge.attributes
            save_figures(ev, giventask, histories, figdir, aof, sessions, results['session_all'], results['seqmask'],
                         attributes)


if __name__ == "__main__":
    main()
