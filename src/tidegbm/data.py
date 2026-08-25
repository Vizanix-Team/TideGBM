from __future__ import annotations

import numpy as np
import pandas as pd

from tidegbm.utils import as_1d, check_finite


def _extract_frame(X, y, timestamp, entity, sample_weight, feature_name, categorical_feature, regime):
    feature_names: list[str]
    ts_arr = None
    ent_arr = None
    regime_arr = None
    cat_cols: set[str] = set()

    if isinstance(X, pd.DataFrame):
        df = X.copy()
        if timestamp is not None and isinstance(timestamp, str):
            if timestamp not in df.columns:
                raise ValueError(f"timestamp column {timestamp!r} not in X")
            ts_arr = df.pop(timestamp).to_numpy()
        if entity is not None and isinstance(entity, str):
            if entity not in df.columns:
                raise ValueError(f"entity column {entity!r} not in X")
            ent_arr = df.pop(entity).to_numpy()
        if regime is not None and isinstance(regime, str):
            if regime not in df.columns:
                raise ValueError(f"regime column {regime!r} not in X")
            regime_arr = df[regime].to_numpy()
        for c in df.columns:
            if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == object:
                cat_cols.add(str(c))
                df[c] = pd.Categorical(df[c]).codes.astype(np.float64)
                df.loc[df[c] < 0, c] = np.nan
        feature_names = [str(c) for c in df.columns]
        X_num = df.to_numpy(dtype=np.float64, copy=True)
    else:
        X_num = np.asarray(X, dtype=np.float64)
        if X_num.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if feature_name is not None:
            feature_names = list(feature_name)
        else:
            feature_names = [f"f{i}" for i in range(X_num.shape[1])]

    y_arr = as_1d(y).astype(np.float64)
    if y_arr.shape[0] != X_num.shape[0]:
        raise ValueError("X and y length mismatch")
    check_finite("y", y_arr)

    if timestamp is not None and not isinstance(timestamp, str):
        ts_arr = as_1d(timestamp)
    if entity is not None and not isinstance(entity, str):
        ent_arr = as_1d(entity)
    if regime is not None and not isinstance(regime, str):
        regime_arr = as_1d(regime)

    if sample_weight is None:
        w = np.ones(y_arr.shape[0], dtype=np.float64)
    else:
        w = as_1d(sample_weight).astype(np.float64)
        if w.shape[0] != y_arr.shape[0]:
            raise ValueError("sample_weight length mismatch")

    is_cat = np.zeros(X_num.shape[1], dtype=bool)
    if categorical_feature == "auto" or categorical_feature is True:
        for i, name in enumerate(feature_names):
            if name in cat_cols:
                is_cat[i] = True
    elif categorical_feature:
        name_to_i = {n: i for i, n in enumerate(feature_names)}
        for c in categorical_feature:
            if isinstance(c, (int, np.integer)):
                is_cat[int(c)] = True
            else:
                if c not in name_to_i:
                    raise ValueError(f"categorical feature {c!r} not in X")
                is_cat[name_to_i[c]] = True

    return X_num, y_arr, feature_names, ts_arr, ent_arr, w, is_cat, regime_arr
