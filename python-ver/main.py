"""
Main pipeline of AspectEstimation (two-step approach)

Translated from matlab-ver/main_mask.m (calc_lkh.m + learn_aspects_from_mask.m):

  1st step: likelihood-based short-term analysis -> AOF masks   (calc_lkh, calc_segments)
  2nd step: pLSA on the AOF masks ("spe") and, for comparison, on all attribute values ("all")

Evaluation (calc_matching.m, CalcSim.m, plot_conf_matrix.m) with --eval, or on MATLAB results with --eval-matlab.

Usage:
  python main.py --mat ../gazedata.mat          # gazedata.mat made by ReadwithTimestamp.m
  python main.py --mat ../gazedata.mat --eval   # + matching, cosine similarity, confusion matrix
  python main.py --eval-matlab refs/results     # evaluate MATLAB p_{spe,all}_topicK.mat
  python main.py --mat ../gazedata.mat --plot   # + figures of the plot_*.m scripts in results/figures
  python main.py --demo --train-num 5           # synthetic data (smoke test)
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from data.data_loader import LikelihoodCalculator
from models import evaluation
from models.plsa import FourLayerPLSA
from models.segments import SegmentCalculator, SessionProcessor


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


def from_reader_sessions(data: Dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Convert GazeDataReader output (dicts keyed by (usr, task)) to session lists,
    ordered like MATLAB's cell(:) (column-major: by task, then by user)
    """
    keys = sorted(data['session'].keys(), key=lambda k: (k[1], k[0]))
    attrs = [np.asarray(data['session'][k]['semantic_attributes']).astype(int) for k in keys]
    seqs = [np.asarray(data['session'][k]['semantic_attribute_seq']).astype(int) for k in keys]
    return attrs, seqs


def create_demo_data(n_users: int = 8, seed: int = 0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Synthetic sessions in the setting of the paper (12 items, 5 attributes x 3 values, 3 tasks)"""
    rng = np.random.default_rng(seed)
    # attributes: price, screen size, CPU score, memory capacity, weight (each value has 4 items)
    kb = np.stack([rng.permutation(np.repeat([1, 2, 3], 4)) for _ in range(5)], axis=1)
    task_values = {1: [(2, 3), (3, 3)], 2: [(1, 1), (4, 1)], 3: [(3, 3), (0, 1)]}
    attrs, seqs = [], []
    for task in (1, 2, 3):  # column-major order: task, then user
        for _ in range(n_users):
            a = kb[rng.permutation(12)]
            score = sum((a[:, k] == v).astype(int) for k, v in task_values[task])
            cand = np.argsort(-(score + rng.random(12) * 0.1))[:2]
            f = list(rng.permutation(12))
            f += [int(cand[i % 2]) if rng.random() > 0.15 else int(rng.integers(12)) for i in range(14)]
            f += list(rng.integers(0, 12, 6))
            attrs.append(a)
            seqs.append(a[np.array(f)].T)
    return attrs, seqs


# ---------------------------------------------------------------- 1st step
def calc_lkh(semantic_attributes: List[np.ndarray], semantic_attribute_seq: List[np.ndarray],
             win_size: int = 5) -> Tuple[List[Dict[int, np.ndarray]], np.ndarray]:
    """
    Likelihood of the observed-attribute distribution in each window (calc_lkh.m)

    Returns:
        (lkh_ses, N_value): lkh_ses[s][att] is the likelihood sequence, N_value the number of
        values of each attribute type (max over all sessions)
    """
    N_value = np.max(np.vstack(semantic_attributes), axis=0)
    calc = LikelihoodCalculator(win_size)
    lkh_ses = []
    for attrs, seq in zip(semantic_attributes, semantic_attribute_seq):
        # phi_k = N_k / N: relative frequency of each value in the catalog of the session
        prob_att = [np.bincount(attrs[:, a], minlength=N_value[a] + 1)[1:] / attrs.shape[0]
                    for a in range(len(N_value))]
        lkh_ses.append(calc.calculate_session_likelihood(seq, prob_att))
    return lkh_ses, N_value


# ---------------------------------------------------------------- 2nd step
def _train_once(args):
    sessions, n_topics, lim, max_itr, seed = args
    np.random.seed(seed)
    return FourLayerPLSA(n_topics).fit(sessions, lim, max_itr)


def train(sessions: List[np.ndarray], n_topics: int, lim: float, train_num: int,
          max_itr: int, jobs: int = 1, seed: int = 0) -> Tuple[Dict, List[float]]:
    """Best of train_num random restarts by final log-likelihood (train.m; parfor -> processes)"""
    tasks = [(sessions, n_topics, lim, max_itr, seed + i) for i in range(train_num)]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_train_once, tasks))
    else:
        results = [_train_once(t) for t in tasks]
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
                  win_size: int = 5, thr: float = 0.03, max_topics: int = 5,
                  train_num: int = 100, lim: float = 1e-4, max_itr: int = 1000,
                  jobs: int = 1, seed: int = 0, savedir: str = 'results') -> Dict:
    """Run the whole pipeline (main_mask.m: calc_lkh -> learn_aspects_from_mask)"""
    S = len(semantic_attribute_seq)
    lkh_ses, N_value = calc_lkh(semantic_attributes, semantic_attribute_seq, win_size)
    if not np.all(N_value == 3):
        # the offsets of session_all (+3, +6, ...) are fixed to 3 values per attribute, as in MATLAB
        raise ValueError(f"3 values per attribute type are expected, got {N_value}")

    seglabel, seqmask = SegmentCalculator(win_size, thr).calc_segments(
        lkh_ses, semantic_attributes, semantic_attribute_seq, win_size, thr)
    processor = SessionProcessor()
    session_all = processor.create_session_all(semantic_attribute_seq, N_value)
    session_spe = processor.remove_empty_columns(seqmask)

    n_aof = sum(m.shape[1] for m in session_spe)
    n_no_aof = sum(1 for m in session_spe if m.shape[1] == 0)
    print(f"{S} sessions, {n_aof} time points with AOF, {n_no_aof} sessions without any AOF")

    os.makedirs(savedir, exist_ok=True)
    np.savez(os.path.join(savedir, 'segsessions.npz'),
             seglabel=_object_array(seglabel), seqmask=_object_array(seqmask),
             session_all=_object_array(session_all), win_size=win_size, thr=thr)

    results = {'seglabel': seglabel, 'seqmask': seqmask, 'session_all': session_all,
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
                   giventask: np.ndarray, savedir: str, aof: Dict[str, np.ndarray] = None) -> Dict:
    """
    Evaluate models[tag][K] = (vz, zs) and save the matched results to savedir/evaluation.npz

    Unlike calc_matching.m, the training outputs (p_*_topicK) are not overwritten.
    aof (evaluation.aof_similarities, plot_eval_AoF.m) is printed and saved as well when given.
    """
    ev = evaluation.evaluate(models, giventask)
    for line in evaluation.format_summary(ev, giventask.shape[0]):
        print(line)
    arrays = {f'{tag}_topic{K}_{name}': value
              for tag, by_topic in ev.items() for K, r in by_topic.items() for name, value in r.items()}
    if aof is not None:
        names, p = evaluation.aof_ttests(aof)
        print("AOF detection, mean cosine similarity (sessions with AOF): " +
              ", ".join(f"{n} = {np.nanmean(aof[n][aof['cos_dist_mset'] >= 0]):.3f}" for n in names))
        print(f"  t-test p-values (upper triangle, order {names}):\n{np.round(p, 4)}")
        arrays.update(aof)
        arrays['aof_ttest_p'] = p
    os.makedirs(savedir, exist_ok=True)
    np.savez(os.path.join(savedir, 'evaluation.npz'), **arrays)
    return ev


def save_figures(ev: Dict, giventask: np.ndarray, histories: Dict[str, Dict[int, List[float]]], figdir: str,
                 aof: Dict[str, np.ndarray] = None, sessions: List[Dict] = None,
                 session_all: List[np.ndarray] = None, seqmask: List[np.ndarray] = None) -> List[str]:
    """
    Figures of plot_L_topic.m, plot_conf_matrix.m, plot_sim.m, plot_sim_task_asp.m, plot_aspect.R and,
    with session data, plot_eval_AoF.m, plot_dur_usr.m, plot_mask_for_paper.m (sessions 17, 34, 46)

    Args:
        ev: output of evaluation.evaluate (matched aspects)
        histories: histories[tag][K] = log-likelihood history
        aof, sessions, session_all, seqmask: optional session data (sessions: load_gazedata_sessions)

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
        figs[f'aspects_{tag}'] = plotter.plot_aspects({K: ev[tag][K]['vz'] for K in Ks}, giventask, name)
        figs[f'likelihood_convergence_{tag}'] = plotter.plot_likelihood_convergence(histories[tag], name)
        for K in Ks:
            if K >= 2:  # refs/results/{spe,all}K.pdf (vz) and {spe,all}K_2.pdf (vz * 2)
                figs[f'aspect_circles_{tag}_topic{K}'] = plotter.plot_aspect_circles(ev[tag][K]['vz'])
                figs[f'aspect_circles_{tag}_topic{K}_x2'] = plotter.plot_aspect_circles(ev[tag][K]['vz'], scale=2)
    if 3 in Ks:
        figs['conf_spe'] = plotter.plot_conf_matrix(ev['spe'][3]['conf'], 'Task estimation MSET: confusion matrix')
        figs['conf_all'] = plotter.plot_conf_matrix(ev['all'][3]['conf'], 'Task estimation w/o MSET: confusion matrix')
        figs['sim'] = plotter.plot_sim(ev['spe'][3]['cor_task'], ev['all'][3]['cor_task'])
        figs['sim_task_asp'] = plotter.plot_sim_task_asp(ev['spe'][3]['cor_task'], ev['all'][3]['cor_task'])
    if aof is not None:
        figs['eval_aof'] = plotter.plot_eval_aof(aof)
    if sessions is not None and all('session_id' in s and 'totalduration' in s for s in sessions):
        dur = np.zeros((max(s['usr'] for s in sessions), 3))
        for s in sessions:
            dur[s['usr'] - 1, s['session_id'] - 1] = s['totalduration']
        figs['dur_usr'] = plotter.plot_dur_usr(dur)
    if session_all is not None and seqmask is not None:
        for s in (17, 34, 46):  # MATLAB indices of plot_mask_for_paper.m
            if s <= len(seqmask):
                title = (f"participant:{sessions[s - 1]['usr']}, task:{sessions[s - 1]['task']}"
                         if sessions is not None else f'session {s}')
                figs[f'mask_session{s}'] = plotter.plot_mask(session_all[s - 1], seqmask[s - 1], title)

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


def main():
    parser = argparse.ArgumentParser(description="AspectEstimation pipeline (main_mask.m)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--mat', type=str, help="gazedata.mat created by ReadwithTimestamp.m")
    src.add_argument('--demo', action='store_true', help="use synthetic data")
    src.add_argument('--eval-matlab', type=str, metavar='DIR',
                     help="only evaluate MATLAB results (p_{spe,all}_topicK.mat) in DIR")
    parser.add_argument('--eval', action='store_true', help="evaluate the learned aspects")
    parser.add_argument('--plot', action='store_true', help="evaluate and save figures to SAVEDIR/figures")
    parser.add_argument('--task-att', type=str, default=None,
                        help="task_att.mat with giventask (default: built-in task-related values)")
    parser.add_argument('--win-size', type=int, default=5)
    parser.add_argument('--thr', type=float, default=0.03)
    parser.add_argument('--max-topics', type=int, default=5)
    parser.add_argument('--train-num', type=int, default=100, help="number of random restarts")
    parser.add_argument('--lim', type=float, default=1e-4, help="EM convergence threshold")
    parser.add_argument('--max-itr', type=int, default=1000)
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--savedir', type=str, default='results')
    args = parser.parse_args()
    giventask = evaluation.load_giventask(args.task_att)
    figdir = os.path.join(args.savedir, 'figures')

    if args.eval_matlab:
        ev = run_evaluation(load_matlab_models(args.eval_matlab, args.max_topics), giventask, args.savedir)
        if args.plot:
            save_figures(ev, giventask, load_matlab_histories(args.eval_matlab, args.max_topics), figdir)
        return

    if args.demo:
        attrs, seqs = create_demo_data(seed=args.seed)
    else:
        attrs, seqs = load_gazedata_mat(args.mat)

    results = learn_aspects(attrs, seqs, args.win_size, args.thr, args.max_topics, args.train_num,
                            args.lim, args.max_itr, args.jobs, args.seed, args.savedir)
    if args.eval or args.plot:
        models = {tag: {K: (np.stack(r['params']['vz'], axis=1), r['params']['zs'])
                        for K, r in results[tag].items()} for tag in ('spe', 'all')}
        sessions, aof = None, None
        if args.mat:  # session structs (f, totalduration, session_id) for plot_eval_AoF.m / plot_dur_usr.m
            sessions = load_gazedata_sessions(args.mat)
            aof = evaluation.aof_similarities([s['f'] for s in sessions], attrs, results['session_all'],
                                              results['seqmask'], evaluation.session_tasks(len(seqs), giventask.shape[0]),
                                              giventask)
        ev = run_evaluation(models, giventask, args.savedir, aof)
        if args.plot:
            histories = {tag: {K: r['history'] for K, r in results[tag].items()} for tag in ('spe', 'all')}
            save_figures(ev, giventask, histories, figdir, aof, sessions, results['session_all'], results['seqmask'])


if __name__ == "__main__":
    main()
