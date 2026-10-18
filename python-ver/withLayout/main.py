"""
Aspects of interest from gaze with the layout effect (Shimonishi, Ishikawa, Yonetani, Kawashima & Matsuyama 2014,
"Learning Aspects of Interest from Gaze", Human Interface Society 16(2))

Translated from matlab-ver/withLayout (main_learning_aspects.m, main.m, main_comparison.m).

Usage (from python-ver/withLayout):
  python main.py learn --attr 1 --alpha 0.5 --aspects 3 12            # learn aspects from all sessions (Fig. 8, 9)
  python main.py evaluate --attr 1 2 3 --alpha 0 0.5 --aspects 6 12   # leave-one-out prediction (Fig. 5, 7)
  python main.py evaluate --attr 1 --alpha 0 0.1 0.2 ... 1 --aspects 9   # mixture ratio (Fig. 6)
  python main.py compare --attr 1 2 3 --method paper matlab           # nearest-neighbor recommendation
  python main.py plot                                                 # figures from the saved results
  python main.py plot --matlab-results ../../matlab-ver/withLayout/result   # Fig. 5, 7 from the MATLAB results
"""

import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')  # one BLAS thread per process (the restarts run in parallel)

import argparse  # noqa: E402
import glob  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Dict  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import layout_data as ld  # noqa: E402
import layout_eval as le  # noqa: E402
import layout_model as lm  # noqa: E402

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'matlab-ver', 'withLayout', 'data')


def _pct(alpha: float) -> int:
    return int(round(alpha * 100))


def learn(args, sessions, content):
    """Aspects and interests from all sessions (main_learning_aspects.m), layout effect from all sessions (Eq. 11)"""
    layout = ld.region_frequency(sessions)
    for attr in args.attr:
        for alpha in args.alpha:
            data = [(s.f, ld.region_given_value(ld.item_attributes(content, s, attr), layout, alpha)) for s in sessions]
            out = {'task_id': [s.task_id for s in sessions], 'content_id': [s.content_id for s in sessions],
                   'layout': layout}
            for K in args.aspects:
                t0 = time.time()
                p = lm.train(data, K, args.train_lim, args.train_num, jobs=args.jobs, seed=args.seed)
                out.update({f'vz_K{K}': p['vz'], f'zs_K{K}': p['zs'], f'L_K{K}': np.array(p['L'])})
                print(f"learn attr {attr} alpha {alpha} K {K}: log-likelihood {p['L'][-1]:.4f} ({time.time() - t0:.1f} s)")
            np.savez(os.path.join(args.savedir, f'learn_attr{attr}_alpha{_pct(alpha)}.npz'), **out)


def evaluate(args, sessions, content):
    """Leave-one-out prediction of gaze regions and items of interest (main.m)"""
    for attr in args.attr:
        for alpha in args.alpha:
            t0 = time.time()
            res = le.leave_one_out(sessions, content, attr, alpha, args.aspects, args.train_num, args.train_lim,
                                   args.est_lim, jobs=args.jobs, seed=args.seed, overlap=not args.disjoint,
                                   log=lambda msg: None)
            out = {f'{key}_K{K}': v for key in ('patk_region', 'patk_item', 'mpatk_region', 'mpatk_item')
                   for K, v in res[key].items()}
            np.savez(os.path.join(args.savedir, f'loo_attr{attr}_alpha{_pct(alpha)}.npz'), **out)
            for K in args.aspects:
                print(f"evaluate attr {attr} alpha {alpha} K {K}: P@1 region / item at 30 % "
                      f"{res['mpatk_region'][K][2, 0]:.3f} / {res['mpatk_item'][K][2, 0]:.3f}, at 80 % "
                      f"{res['mpatk_region'][K][7, 0]:.3f} / {res['mpatk_item'][K][7, 0]:.3f} ({time.time() - t0:.0f} s)")


def compare(args, sessions, content):
    """Nearest-neighbor recommendation (main_comparison.m; method 'paper' or 'matlab', see layout_eval)"""
    for method in args.method:
        for attr in args.attr:
            res = le.leave_one_out_comparison(sessions, content, attr, method, overlap=not args.disjoint)
            np.savez(os.path.join(args.savedir, f'comparison_{method}_attr{attr}.npz'), **res)
            print(f"compare {method} attr {attr}: P@1 region / item at 30 % {res['mpatk_region'][2, 0]:.3f} / "
                  f"{res['mpatk_item'][2, 0]:.3f}, at 80 % {res['mpatk_region'][7, 0]:.3f} / {res['mpatk_item'][7, 0]:.3f}")


# ---------------------------------------------------------------- figures
def load_matlab_mpatk(result_dir: str, kind: str, name: str) -> np.ndarray:
    """mpatk of the MATLAB results (result/<kind>/MP_<name>.mat): cell (aspects x 9) of (K x 1), or (1 x 9)"""
    import scipy.io
    m = scipy.io.loadmat(os.path.join(result_dir, kind, f'MP_{name}.mat'))['mpatk']
    return m


def _precision_bars(get, attrs, aspects, alphas, comparison) -> Dict[int, Dict[str, list]]:
    """P@1 at 30 % and 80 % (ratio index 2 and 7) for each (aspects, alpha) and the comparison method"""
    bars = {}
    for attr in attrs:
        bars[attr] = {}
        for K in aspects:
            for alpha in alphas:
                v = get('model', attr, K, alpha)
                if v is not None:
                    bars[attr][f'aspects: {K}, alpha: {alpha:g}'] = [v[2][0], v[7][0]]
        v = get('comparison', attr, None, None) if comparison else None
        if v is not None:
            bars[attr]['comparison'] = [v[2][0], v[7][0]]
    return bars


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import layout_plot as lp

    figdir = os.path.join(args.savedir, 'figures')
    os.makedirs(figdir, exist_ok=True)
    figs = {}
    for kind, fig_id in (('region', 5), ('item', 7)):
        if args.matlab_results:
            def get(what, attr, K, alpha, kind=kind):
                # main.m: MP_attr<a>_alpha<100 alpha>.mat (results of the 2014 scripts: MP_attr<a>_type1_alpha...)
                names = ([f'attr{attr}_alpha{_pct(alpha)}', f'attr{attr}_type1_alpha{_pct(alpha)}'] if what == 'model'
                         else [f'comp_type{attr}'])
                names = [n for n in names if os.path.exists(os.path.join(args.matlab_results, kind, f'MP_{n}.mat'))]
                if not names:
                    return None
                m = load_matlab_mpatk(args.matlab_results, kind, names[0])
                row = m[K - 1] if what == 'model' else m[0]
                return [np.asarray(c).ravel() for c in row]
        else:
            def get(what, attr, K, alpha, kind=kind):
                path = (os.path.join(args.savedir, f'loo_attr{attr}_alpha{_pct(alpha)}.npz') if what == 'model' else
                        os.path.join(args.savedir, f'comparison_{args.comparison}_attr{attr}.npz'))
                if not os.path.exists(path):
                    return None
                d = np.load(path)
                key = f'mpatk_{kind}_K{K}' if what == 'model' else f'mpatk_{kind}'
                return d[key] if key in d else None
        bars = _precision_bars(get, [1, 2, 3], args.plot_aspects, args.plot_alpha, True)
        if any(bars.values()):
            figs[f'fig{fig_id}_{kind}'] = lp.plot_precision(bars, [30, 80], f'Prediction of {kind}s (P@1)')
    # Fig. 6: alpha sweep for K = 9, SA
    files = sorted(glob.glob(os.path.join(args.savedir, 'loo_attr1_alpha*.npz')),
                   key=lambda p: int(re.search(r'alpha(\d+)', p).group(1)))
    alphas, curves = [], {r: [] for r in (10, 30, 50, 70, 90)}
    for path in files:
        d = np.load(path)
        if 'mpatk_region_K9' in d:
            alphas.append(int(re.search(r'alpha(\d+)', path).group(1)) / 100)
            for r in curves:
                curves[r].append(d['mpatk_region_K9'][r // 10 - 1, 0])
    if len(alphas) > 1:
        figs['fig6_alpha'] = lp.plot_alpha(alphas, curves)
    # Fig. 8, 9: aspects / interests learned from all sessions (SA, alpha 0.5)
    path = os.path.join(args.savedir, 'learn_attr1_alpha50.npz')
    if os.path.exists(path):
        d = np.load(path)
        labels = ld.attribute_labels(1)
        for key in [k for k in d if k.startswith('vz_K')]:
            figs[f'fig8_aspects_{key[3:]}'] = lp.plot_aspects(d[key], labels, f'aspects ({key[3:]})')
        histories = {int(k[3:]): d[k] for k in d if k.startswith('L_K')}
        if histories:
            figs['likelihood_convergence'] = lp.plot_likelihood_convergence(histories, 'learning (SA, alpha 0.5)')
        if 'zs_K12' in d:
            sel = d['content_id'] == 1
            figs['fig9_interests'] = lp.plot_interests({t: d['zs_K12'][sel & (d['task_id'] == t)] for t in (1, 2, 3)})
    for name, fig in figs.items():
        fig.savefig(os.path.join(figdir, f'{name}.png'), dpi=150)
        plt.close(fig)
    print(f"Saved {len(figs)} figures to {figdir}")


def main():
    parser = argparse.ArgumentParser(description="Aspects of interest with the layout effect (2014)")
    parser.add_argument('command', choices=['learn', 'evaluate', 'compare', 'plot'])
    parser.add_argument('--data', default=DEFAULT_DATA, help="data directory (mat/, experiment/timestamp.list, content.mat)")
    parser.add_argument('--savedir', default='results')
    parser.add_argument('--attr', type=int, nargs='+', default=[1], help="attribute types: 1 SA, 2 AA, 3 SA+AA")
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.5], help="mixture ratio(s) of the layout effect")
    parser.add_argument('--aspects', type=int, nargs='+', default=[3, 6, 9, 12], help="numbers of aspects")
    parser.add_argument('--method', nargs='+', default=['paper', 'matlab'], help="comparison methods")
    parser.add_argument('--train-num', type=int, default=100, help="random restarts (train.m)")
    parser.add_argument('--train-lim', type=float, default=1e-2, help="EM threshold for learning (plsa.m)")
    parser.add_argument('--est-lim', type=float, default=1e-3, help="EM threshold for estimating interests (est.m)")
    parser.add_argument('--disjoint', action='store_true',
                        help="disjoint estimation / evaluation parts (main.m shares one time point)")
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--matlab-results', default=None, help="plot: MATLAB result/ directory instead of savedir")
    parser.add_argument('--comparison', default='paper', help="plot: comparison method shown in Fig. 5, 7")
    parser.add_argument('--plot-aspects', type=int, nargs='+', default=[6, 12])
    parser.add_argument('--plot-alpha', type=float, nargs='+', default=[0, 0.5])
    args = parser.parse_args()
    os.makedirs(args.savedir, exist_ok=True)
    if args.command == 'plot':
        plot(args)
        return
    sessions = ld.load_sessions(args.data)
    content = ld.load_content(os.path.join(args.data, 'content.mat'))
    {'learn': learn, 'evaluate': evaluate, 'compare': compare}[args.command](args, sessions, content)


if __name__ == '__main__':
    main()
