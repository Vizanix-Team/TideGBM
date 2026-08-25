from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def resolve_feature_groups(
    feature_names: list[str],
    feature_groups: Mapping[str, Sequence[str]] | Sequence[int] | None,
    n_features: int,
) -> np.ndarray:
    """Return group id per feature, -1 if ungrouped.

    Mapping form: {"momentum": ["rsi_14", "rsi_21"], ...}
    Sequence form: group id per column.
    """
    out = np.full(n_features, -1, dtype=np.int32)
    if feature_groups is None:
        return out
    if isinstance(feature_groups, Mapping):
        name_to_i = {n: i for i, n in enumerate(feature_names)}
        gid = 0
        for _, cols in feature_groups.items():
            idxs = []
            for c in cols:
                if c not in name_to_i:
                    raise ValueError(f"feature_groups column {c!r} is not in X")
                idxs.append(name_to_i[c])
            for i in idxs:
                out[i] = gid
            gid += 1
        return out
    arr = np.asarray(list(feature_groups), dtype=np.int32)
    if arr.shape[0] != n_features:
        raise ValueError("feature_groups sequence must have n_features entries")
    return arr


def auto_groups(
    X: np.ndarray,
    names: list[str],
    max_corr: float = 0.92,
    sample_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Causal-friendly grouping: cluster highly correlated columns on a prefix mask."""
    n_f = X.shape[1]
    out = np.full(n_f, -1, dtype=np.int32)
    if n_f < 2:
        return out
    mask = np.ones(X.shape[0], dtype=bool) if sample_mask is None else sample_mask
    data = X[mask]
    if data.shape[0] < 10:
        return out
    # rank-corr approximation via numpy
    ranks = np.zeros_like(data, dtype=np.float64)
    for j in range(n_f):
        col = data[:, j]
        finite = np.isfinite(col)
        r = np.zeros(col.size)
        r[finite] = col[finite].argsort().argsort()
        ranks[:, j] = r
    ranks = ranks - ranks.mean(axis=0)
    std = ranks.std(axis=0)
    std[std == 0] = 1.0
    corr = (ranks.T @ ranks) / (ranks.shape[0] * np.outer(std, std))
    np.fill_diagonal(corr, 0.0)
    parent = np.arange(n_f)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n_f):
        for j in range(i + 1, n_f):
            if abs(corr[i, j]) >= max_corr:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pj] = pi
    clusters: dict[int, list[int]] = {}
    for i in range(n_f):
        clusters.setdefault(find(i), []).append(i)
    gid = 0
    for members in clusters.values():
        if len(members) < 2:
            continue
        for i in members:
            out[i] = gid
        gid += 1
    return out


def monotone_vector(
    feature_names: list[str],
    monotone_constraints,
    n_features: int,
) -> np.ndarray:
    out = np.zeros(n_features, dtype=np.int8)
    if monotone_constraints is None:
        return out
    if isinstance(monotone_constraints, dict):
        name_to_i = {n: i for i, n in enumerate(feature_names)}
        for k, v in monotone_constraints.items():
            if k not in name_to_i:
                raise ValueError(f"unknown monotone feature {k!r}")
            out[name_to_i[k]] = int(np.sign(v))
        return out
    arr = np.asarray(list(monotone_constraints), dtype=np.int8)
    if arr.shape[0] != n_features:
        raise ValueError("monotone_constraints length mismatch")
    return np.sign(arr).astype(np.int8)
