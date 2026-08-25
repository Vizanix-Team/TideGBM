from __future__ import annotations

import numpy as np

from tidegbm.utils import parse_horizon, parse_timestamps


class WalkForwardSplit:
    """Expanding-window walk-forward with purge and embargo.

    Parameters
    ----------
    n_splits : number of test folds
    embargo : gap after train (same units as timestamps, or timedelta)
    purge : extra past dropped from the *end* of train so overlapping
        labels cannot leak into the test fold
    min_train_fraction : skip folds whose train window is too short
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo=None,
        purge=None,
        min_train_fraction: float = 0.3,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = int(n_splits)
        self.embargo = embargo
        self.purge = purge
        self.min_train_fraction = min_train_fraction

    def split(self, X, y=None, timestamp=None, groups=None):
        if timestamp is None:
            raise ValueError("WalkForwardSplit requires timestamp")
        ts = parse_timestamps(timestamp)
        n = ts.shape[0]
        embargo = parse_horizon(self.embargo, ts)
        purge = parse_horizon(self.purge, ts)
        uniq = np.unique(ts)
        if uniq.size < self.n_splits + 2:
            raise ValueError("not enough unique timestamps for n_splits")
        # first chunk is history-only; remaining n_splits chunks are successive tests
        bounds = np.linspace(0, uniq.size, self.n_splits + 2, dtype=int)
        min_train = int(self.min_train_fraction * n)
        for i in range(1, self.n_splits + 1):
            test_lo = bounds[i]
            test_hi = bounds[i + 1]
            t_test0 = uniq[test_lo]
            t_test1 = uniq[max(test_hi - 1, test_lo)]
            train_end_t = t_test0 - embargo - purge
            train_idx = np.where(ts < train_end_t)[0]
            test_idx = np.where((ts >= t_test0) & (ts <= t_test1))[0]
            if train_idx.size < max(min_train, 10) or test_idx.size == 0:
                continue
            yield train_idx.astype(np.int32), test_idx.astype(np.int32)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def combinatorial_purged_cv(
    timestamp,
    n_splits: int = 6,
    n_test_folds: int = 2,
    embargo=None,
):
    """Combinatorial purged CV (López de Prado).

    Yields (train_idx, test_idx) for every combination of ``n_test_folds``
    among ``n_splits`` contiguous time blocks, with embargo around tests.
    """
    from itertools import combinations

    ts = parse_timestamps(timestamp)
    embargo = parse_horizon(embargo, ts)
    uniq = np.unique(ts)
    if uniq.size < n_splits:
        raise ValueError("not enough unique timestamps")
    bounds = np.linspace(0, uniq.size, n_splits + 1, dtype=int)
    blocks = []
    for i in range(n_splits):
        t0 = uniq[bounds[i]]
        t1 = uniq[min(bounds[i + 1], uniq.size) - 1]
        blocks.append((t0, t1))
    order = np.arange(ts.shape[0])
    for test_ids in combinations(range(n_splits), n_test_folds):
        test_mask = np.zeros(ts.shape[0], dtype=bool)
        banned = np.zeros(ts.shape[0], dtype=bool)
        for k in test_ids:
            t0, t1 = blocks[k]
            test_mask |= (ts >= t0) & (ts <= t1)
            banned |= (ts >= t0 - embargo) & (ts <= t1 + embargo)
        train_mask = ~banned
        train_idx = order[train_mask]
        test_idx = order[test_mask]
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        yield train_idx.astype(np.int32), test_idx.astype(np.int32)
