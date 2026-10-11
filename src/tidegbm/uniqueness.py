"""Average-uniqueness sample weights for overlapping labels.

A label built from an ``h``-step forward return that is sampled every step overlaps its
``h-1`` neighbours, so consecutive rows carry far less than one observation of independent
information each. Purge and embargo keep the *evaluation* honest; they do nothing about the
*training* set, where the booster still counts every overlapping row as a full sample.

Following Lopez de Prado, each label is weighted by its average uniqueness: the mean of
``1 / concurrency`` over the span the label covers, where concurrency counts how many labels
are open at that moment. Concurrency is computed per entity, since one instrument's labels
overlap its own future labels and not those of other instruments.
"""

from __future__ import annotations

import numpy as np

from tidegbm.utils import parse_horizon, parse_timestamps


def _uniqueness_one(ts_sorted: np.ndarray, horizon: int) -> np.ndarray:
    """Average uniqueness for one entity's sorted observation times."""
    m = ts_sorted.shape[0]
    if m == 0:
        return np.empty(0, dtype=np.float64)
    if m == 1 or horizon <= 0:
        return np.ones(m, dtype=np.float64)

    # Label i occupies observation indices [i, end_i): every own-grid point it spans.
    end = np.searchsorted(ts_sorted, ts_sorted + horizon, side="left")
    end = np.maximum(end, np.arange(m) + 1)  # a label always covers its own point

    # Concurrency per observation via a difference array over label spans.
    diff = np.zeros(m + 1, dtype=np.int64)
    np.add.at(diff, np.arange(m), 1)
    np.add.at(diff, end, -1)
    conc = np.cumsum(diff[:m])
    conc = np.maximum(conc, 1)

    # u_i = mean of 1/conc over [i, end_i) -> prefix sums make it O(1) per label.
    inv = 1.0 / conc
    csum = np.concatenate(([0.0], np.cumsum(inv)))
    return (csum[end] - csum[np.arange(m)]) / (end - np.arange(m))


def average_uniqueness(timestamp, horizon, entity=None) -> np.ndarray:
    """Weight in ``(0, 1]`` per row: 1.0 means the label overlaps nothing.

    Parameters
    ----------
    timestamp : array-like
        Observation times, any form ``parse_timestamps`` accepts.
    horizon : int or timedelta
        Label length in timestamp units — the same value passed to ``fit``.
    entity : array-like, optional
        Panel key. Concurrency is counted within each entity separately.

    Returns
    -------
    np.ndarray
        Uniqueness per row, aligned with the input order.
    """
    ts = parse_timestamps(timestamp)
    h = int(parse_horizon(horizon, ts))
    n = ts.shape[0]
    out = np.ones(n, dtype=np.float64)
    if h <= 0 or n == 0:
        return out

    if entity is None:
        keys = np.zeros(n, dtype=np.int64)
    else:
        _, keys = np.unique(np.asarray(entity), return_inverse=True)
        keys = keys.astype(np.int64)

    order = np.lexsort((ts, keys))
    ks = keys[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    bounds = np.r_[starts, n]
    for a, b in zip(bounds[:-1], bounds[1:]):
        sl = order[a:b]
        out[sl] = _uniqueness_one(ts[sl], h)
    return out


def uniqueness_weights(timestamp, horizon, entity=None, sample_weight=None) -> np.ndarray:
    """``average_uniqueness`` rescaled to mean 1 and folded into any existing weights.

    Rescaling keeps the effective learning rate comparable to an unweighted fit, so the
    same ``n_estimators`` and ``learning_rate`` stay in range when the option is switched on.
    """
    u = average_uniqueness(timestamp, horizon, entity)
    mean = float(u.mean())
    if mean > 0:
        u = u / mean
    if sample_weight is None:
        return u
    return np.asarray(sample_weight, dtype=np.float64) * u
