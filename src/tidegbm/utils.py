from __future__ import annotations

import numpy as np
import pandas as pd


def as_1d(a) -> np.ndarray:
    arr = np.asarray(a)
    return np.ravel(arr)


def check_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        n_bad = int(np.size(arr) - np.isfinite(arr).sum())
        raise ValueError(f"{name} contains {n_bad} non-finite values")


def parse_timestamps(ts) -> np.ndarray:
    """Convert timestamps to int64 in a comparable unit.

    Datetime-like values become nanoseconds. Integer/float bar indices stay as int64.
    """
    if ts is None:
        raise ValueError("timestamp is required unless iid_mode=True")
    arr = as_1d(ts)
    if arr.size == 0:
        raise ValueError("timestamp is empty")
    if np.issubdtype(arr.dtype, np.datetime64) or isinstance(arr[0], (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(arr).view("int64")
    if arr.dtype == object:
        try:
            converted = pd.to_datetime(arr, utc=False)
            if not pd.isna(converted).all():
                return converted.view("int64")
        except (ValueError, TypeError):
            pass
    if np.issubdtype(arr.dtype, np.number):
        if np.issubdtype(arr.dtype, np.floating):
            if not np.allclose(arr, np.round(arr)):
                # treat as epoch seconds/ms if large, else as ordered floats
                scale = 1e9 if np.nanmax(np.abs(arr)) < 1e12 else 1.0
                return np.round(arr * scale).astype(np.int64)
        return arr.astype(np.int64)
    return pd.to_datetime(arr).view("int64")


def parse_horizon(value, timestamps: np.ndarray) -> int:
    """Horizon/embargo in the same int64 units as ``timestamps``."""
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        if np.issubdtype(timestamps.dtype, np.integer):
            med = _median_step(timestamps)
            return int(value) * max(med, 1)
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return parse_horizon(int(value), timestamps)
    td = pd.to_timedelta(value)
    return int(td.to_timedelta64().view("int64"))


def _median_step(ts: np.ndarray) -> int:
    uniq = np.unique(ts)
    if uniq.size < 2:
        return 1
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1
    return int(np.median(diffs))


def time_buckets(timestamps: np.ndarray, n_buckets: int) -> np.ndarray:
    """Assign expanding-style equal-count time buckets in [0, n_buckets)."""
    n_buckets = max(2, int(n_buckets))
    order = np.argsort(timestamps, kind="mergesort")
    ranks = np.empty(timestamps.shape[0], dtype=np.int64)
    ranks[order] = np.arange(timestamps.shape[0])
    # equal count buckets by time order, not by raw timestamp gaps
    edges = np.linspace(0, timestamps.shape[0], n_buckets + 1)
    buckets = np.digitize(ranks, edges[1:-1], right=False).astype(np.int32)
    buckets = np.clip(buckets, 0, n_buckets - 1)
    return buckets


def recency_weights(timestamps: np.ndarray, decay: float) -> np.ndarray:
    if decay >= 1.0:
        return np.ones(timestamps.shape[0], dtype=np.float64)
    t_max = timestamps.max()
    step = max(_median_step(timestamps), 1)
    age = (t_max - timestamps).astype(np.float64) / float(step)
    w = np.power(decay, age)
    return np.clip(w, 1e-8, 1.0)


def softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    z = (x - np.max(x)) / max(tau, 1e-12)
    e = np.exp(np.clip(z, -50, 50))
    s = e.sum()
    if s <= 0:
        return np.full_like(e, 1.0 / max(e.size, 1))
    return e / s
