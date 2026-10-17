"""
Smoke tests of src/visualization/plotter.py and main.save_figures (figures are not compared with MATLAB;
the numbers they show are tested in test_evaluation.py / test_refs.py).

Run from python-ver/:  python -m pytest tests -q
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

import main as pipeline  # noqa: E402
from models import evaluation  # noqa: E402
from visualization import plotter  # noqa: E402

SAMPLE_DIR = os.path.abspath(os.path.join(ROOT, '..', 'sample_data'))


def test_plot_functions():
    rng = np.random.default_rng(0)
    vz = {K: rng.dirichlet(np.ones(15), K).T for K in (1, 2, 3)}
    cor = evaluation.cosine_task_aspect(evaluation.GIVENTASK, vz[3])
    dists = {k: rng.random(20) for k in ('cos_dist_c', 'cos_dist_top', 'cos_dist_all', 'cos_dist_mset')}
    dists['cos_dist_mset'][3] = np.nan
    mask = (rng.random((15, 30)) < 0.2).astype(float)
    figs = [plotter.plot_L_topic([-3, -2, -1], [-2, -1.5, -1]),
            plotter.plot_aspects(vz, evaluation.GIVENTASK, 'MSET'),
            plotter.plot_conf_matrix(np.array([[32, 0, 3, 2], [0, 30, 7, 0], [4, 0, 31, 2]]), 'conf'),
            plotter.plot_sim(cor, cor * 0.8),
            plotter.plot_sim_task_asp(cor, cor * 0.8),
            plotter.plot_eval_aof(dists),
            plotter.plot_dur_usr(rng.random((37, 3)) * 100),
            plotter.plot_mask(mask, mask * (rng.random((15, 30)) < 0.5), 'session'),
            plotter.plot_aspect_circles(vz[3]),
            plotter.plot_likelihood_convergence({1: [-3, -2], 2: [-4, -2, -1.9]})]
    gaze = np.loadtxt(os.path.join(SAMPLE_DIR, '170116024150092.csv'), delimiter=',')
    figs.append(plotter.plot_org_gaze(gaze, plt.imread(os.path.join(SAMPLE_DIR, '170116024150092.jpg'))))
    figs.append(plotter.plot_org_gaze(gaze))
    assert all(isinstance(f, plt.Figure) for f in figs)
    for f in figs:
        plt.close(f)


def test_save_figures_demo(tmp_path):
    attrs, seqs = pipeline.create_demo_data(n_users=6)  # 18 sessions: mask figure of session 17
    res = pipeline.learn_aspects(attrs, seqs, max_topics=3, train_num=2, max_itr=30, jobs=1, savedir=str(tmp_path))
    models = {tag: {K: (np.stack(r['params']['vz'], axis=1), r['params']['zs']) for K, r in res[tag].items()}
              for tag in ('spe', 'all')}
    ev = pipeline.run_evaluation(models, evaluation.GIVENTASK, str(tmp_path))
    histories = {tag: {K: r['history'] for K, r in res[tag].items()} for tag in ('spe', 'all')}
    paths = pipeline.save_figures(ev, evaluation.GIVENTASK, histories, str(tmp_path / 'figures'),
                                  session_all=res['session_all'], seqmask=res['seqmask'])
    names = {os.path.basename(p) for p in paths}
    assert {'L_topic.png', 'conf_spe.png', 'sim.png', 'sim_task_asp.png', 'mask_session17.png'} <= names
    assert all(os.path.getsize(p) > 0 for p in paths)
