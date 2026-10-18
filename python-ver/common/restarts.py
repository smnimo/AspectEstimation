"""
Random restarts of an EM algorithm (train.m of both MATLAB versions: parfor over restarts, keep the best)
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, List, Sequence

import numpy as np


def _seeded_call(job):
    func, seed, args = job
    np.random.seed(seed)  # the fits draw their initial values from numpy's global generator
    return func(*args)


def run_restarts(func: Callable, args: Sequence[Any], train_num: int, jobs: int = 1, seed: int = 0) -> List[Any]:
    """
    Run func(*args) train_num times; restart i seeds numpy's global generator with seed + i

    Args:
        func: module-level function (picklable for jobs > 1)
        args: arguments of func
        train_num: number of restarts
        jobs: number of processes (1 = sequential)
        seed: seed of the first restart

    Returns:
        list of the results in restart order
    """
    tasks = [(func, seed + i, tuple(args)) for i in range(train_num)]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(_seeded_call, tasks))
    return [_seeded_call(t) for t in tasks]
