from __future__ import annotations

import numpy as np
import pandas as pd


def as_feature_matrix(X, feature_names: list[str]) -> np.ndarray:
    """Slice user input down to the feature columns used at fit time."""
    if isinstance(X, pd.DataFrame):
        missing = [c for c in feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"X is missing training columns: {missing}")
        df = X.loc[:, feature_names].copy()
        for c in df.columns:
            if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == object:
                df[c] = pd.Categorical(df[c]).codes.astype(np.float64)
                df.loc[df[c] < 0, c] = np.nan
        return df.to_numpy(dtype=np.float64, copy=True)
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if arr.shape[1] != len(feature_names):
        raise ValueError(f"expected {len(feature_names)} features, got {arr.shape[1]}")
    return arr
