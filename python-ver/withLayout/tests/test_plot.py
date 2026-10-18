"""Smoke tests of src/layout_plot.py"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import layout_data as ld  # noqa: E402
import layout_plot as lp  # noqa: E402


def test_plots():
    rng = np.random.default_rng(0)
    bars = {a: {'aspects: 6, alpha: 0': [0.4, 0.5], 'comparison': [0.2, 0.3]} for a in (1, 2, 3)}
    figs = [lp.plot_precision(bars, [30, 80], 'test'),
            lp.plot_alpha([0, 0.5, 1], {10: [0.3, 0.4, 0.35], 90: [0.4, 0.38, 0.3]}),
            lp.plot_aspects(rng.dirichlet(np.ones(14), 3), ld.attribute_labels(1)),
            lp.plot_interests({t: rng.dirichlet(np.ones(12), 3) for t in (1, 2, 3)})]
    assert all(isinstance(f, plt.Figure) for f in figs)
    for f in figs:
        plt.close(f)
