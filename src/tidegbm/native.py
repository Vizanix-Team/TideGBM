from __future__ import annotations

import numpy as np

from tidegbm import _core
from tidegbm.data import _extract_frame
from tidegbm.groups import auto_groups, monotone_vector, resolve_feature_groups
from tidegbm.io import as_feature_matrix
from tidegbm.lint import lint
from tidegbm.utils import as_1d, parse_horizon, parse_timestamps


def _obj_for_task(objective: str, task: str) -> str:
    if objective not in {"auto", None}:
        return objective
    if task == "classifier":
        return "logloss"
    if task == "ranker":
        return "rank"
    return "mse"


class NativeBooster:
    def __init__(self, **params) -> None:
        self.params = dict(params)
        self._b = _core.Booster()
        self.feature_names: list[str] = []
        self.is_fitted_ = False

    def fit(
        self,
        X,
        y,
        *,
        timestamp=None,
        entity=None,
        sample_weight=None,
        eval_set=None,
        eval_timestamp=None,
        feature_name=None,
        categorical_feature="auto",
        feature_groups=None,
        regime=None,
        auto_group: bool = False,
        task: str = "regressor",
        lint_mode: str = "strict",
        **kwargs,
    ) -> "NativeBooster":
        params = {**self.params, **kwargs}
        iid = bool(params.get("iid_mode", False))
        if not iid and timestamp is None:
            raise ValueError(
                "timestamp is required. Pass timestamp=... or set iid_mode=True "
                "(iid_mode disables TORB and is unsafe for market data)."
            )
        X_num, y_arr, names, ts_arr, ent_arr, w, is_cat, _regime = _extract_frame(
            X, y, timestamp, entity, sample_weight, feature_name, categorical_feature, regime
        )
        if _regime is not None and "regime_feature" not in params:
            if isinstance(regime, str) and regime in names:
                params["regime_feature"] = names.index(regime)
                is_cat = np.asarray(is_cat, dtype=bool).copy()
                is_cat[params["regime_feature"]] = True
            else:
                _, codes = np.unique(np.asarray(_regime), return_inverse=True)
                X_num = np.column_stack([X_num, codes.astype(np.float64)])
                names = list(names) + ["__regime__"]
                is_cat = np.append(np.asarray(is_cat, dtype=bool), True)
                params["regime_feature"] = int(X_num.shape[1] - 1)
        self.feature_names = names
        ts_i64 = None
        horizon_u = -1
        embargo_u = 0
        entity_i64 = None
        if not iid:
            ts_i64 = parse_timestamps(ts_arr)
            if params.get("horizon") is None:
                raise ValueError(
                    "horizon is required in time mode (label length in timestamp units)."
                )
            horizon_u = parse_horizon(params.get("horizon"), ts_i64)
            embargo_u = parse_horizon(params.get("embargo"), ts_i64)
            if lint_mode == "strict":
                report = lint(X_num, y_arr, ts_i64, feature_names=names)
                fatal = [
                    f
                    for f in report.errors
                    if f.code in {"lookahead", "timestamp_feature", "bad_timestamp", "length"}
                ]
                if fatal:
                    raise ValueError("TideGBM leakage lint failed: " + "; ".join(f.message for f in fatal))
            ent_key = np.zeros(len(y_arr), dtype=np.int64)
            if ent_arr is not None:
                _, ent_key = np.unique(np.asarray(ent_arr), return_inverse=True)
                ent_key = ent_key.astype(np.int64)
            order = np.lexsort((ent_key, ts_i64))
            if not np.all(order == np.arange(order.size)):
                X_num = np.ascontiguousarray(X_num[order])
                y_arr = np.ascontiguousarray(y_arr[order])
                ts_i64 = np.ascontiguousarray(ts_i64[order])
                w = np.ascontiguousarray(w[order])
                ent_key = np.ascontiguousarray(ent_key[order])
            entity_i64 = ent_key
        cw = params.get("class_weight")
        if cw is not None and task == "classifier":
            y_int = y_arr.astype(int)
            if isinstance(cw, dict):
                ww = np.array([float(cw.get(int(v), 1.0)) for v in y_int], dtype=np.float64)
            elif cw == "balanced":
                _, inv, cnt = np.unique(y_int, return_inverse=True, return_counts=True)
                ww = (len(y_int) / (cnt.size * cnt[inv])).astype(np.float64)
            else:
                ww = np.ones_like(w)
            w = w * ww
        group_of = resolve_feature_groups(names, feature_groups, X_num.shape[1])
        if auto_group and feature_groups is None:
            mask = np.ones(y_arr.shape[0], dtype=bool)
            if ts_i64 is not None:
                order = np.argsort(ts_i64)
                mask[:] = False
                mask[order[: max(10, y_arr.shape[0] // 2)]] = True
            group_of = auto_groups(X_num, names, sample_mask=mask)
        mon = monotone_vector(names, params.get("monotone_constraints"), X_num.shape[1])
        rank_groups = None
        obj = _obj_for_task(str(params.get("objective", "auto")), task)
        if params.get("custom_objective") is not None:
            obj = "custom"
        if obj in {"rank", "listnet", "pairwise", "lambdarank"}:
            if ts_i64 is None:
                raise ValueError("rank objective requires timestamps")
            rank_groups = ts_i64

        Xe = ye = te = None
        if eval_set is not None:
            Xv, yv = eval_set
            ye = as_1d(yv).astype(np.float64)
            Xe = as_feature_matrix(Xv, names)
            if not iid:
                if eval_timestamp is None:
                    raise ValueError("eval_timestamp is required when iid_mode=False")
                import pandas as pd

                if isinstance(eval_timestamp, str) and isinstance(Xv, pd.DataFrame):
                    te = parse_timestamps(Xv[eval_timestamp].to_numpy())
                else:
                    te = parse_timestamps(eval_timestamp)

        fit_kw = dict(params)
        fit_kw["objective"] = obj
        fit_kw["iid_mode"] = iid
        if not iid:
            fit_kw["horizon"] = int(horizon_u)
            fit_kw["embargo"] = int(embargo_u)
        ix = params.get("interaction_constraints")
        if ix is not None:
            if isinstance(ix, (list, tuple)) and ix and isinstance(ix[0], (list, tuple, set)):
                ix = {str(i): list(g) for i, g in enumerate(ix)}
            g = resolve_feature_groups(names, ix, X_num.shape[1])
            fit_kw["interaction_group"] = [int(v) for v in g.tolist()]
        pairs = params.get("interaction_pairs")
        if pairs is not None:
            name_to_i = {n: i for i, n in enumerate(names)}
            fp = []
            for a, b in pairs:
                ia = name_to_i[a] if isinstance(a, str) else int(a)
                ib = name_to_i[b] if isinstance(b, str) else int(b)
                fp.append((ia, ib))
            fit_kw["forced_pairs"] = fp
        ncl = params.get("n_classes")
        if ncl is not None:
            fit_kw["n_classes"] = int(ncl)
        CFG = {
            "n_estimators", "learning_rate", "max_depth", "max_leaves", "min_samples_leaf",
            "min_child_weight", "l2", "min_split_gain", "max_bin", "subsample", "colsample",
            "n_time_buckets", "horizon", "embargo", "iid_mode", "halo", "halo_gamma",
            "halo_candidates", "pair_splits", "pair_trigger", "pair_top_k",
            "max_pair_splits_per_tree", "pair_l2", "time_decay", "tail_buffer", "tree_decay",
            "objective", "quantile", "huber_delta", "rank_temperature", "n_jobs", "random_state",
            "verbose", "early_stopping_rounds", "causal_structure", "finer_torb", "device",
            "rank_pairwise", "cost_threshold", "n_classes", "n_horizons", "regime_feature",
            "regime_priority", "init_score", "quantiles", "interaction_group", "forced_pairs",
            "l1", "max_delta_step", "goss", "goss_top", "goss_other", "structure_frac",
            "causal_bins", "torb_min_cover",
            "cat_smooth", "col_major_hist", "cat_min_count", "colsample_bynode",
            "bin_sample", "entity_subsample", "eval_metric",
        }
        cfg_kw = {k: v for k, v in fit_kw.items() if k in CFG and v is not None}
        self._b.set_config(**cfg_kw)
        custom = params.get("custom_objective")
        if custom is not None:
            self._b.set_custom_objective(custom)
        cost = kwargs.get("cost", params.get("cost"))
        try:
            self._b.fit(
                np.ascontiguousarray(X_num, dtype=np.float64),
                np.ascontiguousarray(y_arr, dtype=np.float64),
                timestamps=None if ts_i64 is None else np.ascontiguousarray(ts_i64, dtype=np.int64),
                weight=np.ascontiguousarray(w, dtype=np.float64),
                is_cat=np.ascontiguousarray(is_cat.astype(np.int32)),
                group_of=np.ascontiguousarray(group_of.astype(np.int32)),
                monotone=np.ascontiguousarray(mon.astype(np.int8)),
                rank_groups=None if rank_groups is None else np.ascontiguousarray(rank_groups, dtype=np.int64),
                X_eval=None if Xe is None else np.ascontiguousarray(Xe, dtype=np.float64),
                y_eval=None if ye is None else np.ascontiguousarray(ye, dtype=np.float64),
                ts_eval=None if te is None else np.ascontiguousarray(te, dtype=np.int64),
                entity=None if entity_i64 is None else np.ascontiguousarray(entity_i64, dtype=np.int64),
                cost=None if cost is None else np.ascontiguousarray(as_1d(cost), dtype=np.float64),
                **cfg_kw,
            )
        except _core.TideError as exc:
            raise ValueError(str(exc)) from exc
        self._b.feature_names = names
        self.is_fitted_ = True
        return self

    def _X(self, X) -> np.ndarray:
        return np.ascontiguousarray(as_feature_matrix(X, self.feature_names), dtype=np.float64)

    def predict(self, X):
        return np.asarray(self._b.predict(self._X(X)))

    def raw_predict(self, X):
        return np.asarray(self._b.raw_predict(self._X(X)))

    def predict_proba(self, X):
        p = self.predict(X)
        if int(self._b.n_heads()) > 1 and str(self.params.get("objective", "")) == "multiclass":
            return np.asarray(self._b.predict_multi(self._X(X)))
        return np.column_stack([1.0 - p, p])

    def predict_multi(self, X):
        return np.asarray(self._b.predict_multi(self._X(X)))

    def feature_importances(self, kind: str = "gain"):
        return np.asarray(self._b.feature_importance(kind))

    def pred_contrib(self, X):
        return np.asarray(self._b.pred_contrib(self._X(X)))

    def bin_transform(self, X):
        return np.asarray(self._b.bin_transform(self._X(X)))

    def update(self, X, y, n_estimators: int = 10):
        Xm = self._X(X)
        self._b.update(Xm, np.ascontiguousarray(as_1d(y), dtype=np.float64), int(n_estimators))
        return self

    def save(self) -> bytes:
        return bytes(self._b.save())

    def save_json(self) -> str:
        return str(self._b.save_json())

    @classmethod
    def load(cls, blob: bytes) -> "NativeBooster":
        obj = cls()
        obj._b = _core.Booster.load(blob)
        obj.feature_names = list(obj._b.feature_names)
        obj.is_fitted_ = True
        return obj

    def to_onnx_json(self) -> str:
        return str(self._b.to_onnx_json())

    @property
    def evals_result_(self):
        return dict(self._b.evals_result())

    @property
    def best_iteration_(self):
        return int(self._b.best_iteration)
