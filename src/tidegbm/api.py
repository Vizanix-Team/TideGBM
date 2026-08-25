from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

try:
    from sklearn.base import RankerMixin
except ImportError:  # pragma: no cover - sklearn < 1.4

    class RankerMixin:  # type: ignore[no-redef]
        _estimator_type = "ranker"

from tidegbm.native import NativeBooster


def _params(est: "TideEstimator") -> dict:
    p = {k: v for k, v in est.get_params().items() if k != "monotone_constraints"}
    p["monotone_constraints"] = est.monotone_constraints
    if p.get("random_state") is None:
        p["random_state"] = 0
    if p.get("early_stopping_rounds") is None:
        p["early_stopping_rounds"] = 0
    return p


class TideEstimator(BaseEstimator):
    """Base gradient boosting estimator (histogram GBDT + TORB).

    Use :class:`TideRegressor`, :class:`TideClassifier`, or :class:`TideRanker`
    instead of instantiating this class directly.

    See the API reference for the full parameter list and ``fit()`` kwargs:
    https://github.com/Vizanix/TideGBM/blob/main/docs/API.md
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        max_leaves: int = 31,
        min_samples_leaf: int = 20,
        min_child_weight: float = 1e-3,
        l2: float = 1.0,
        min_split_gain: float = 0.0,
        max_bin: int = 255,
        subsample: float = 1.0,
        colsample: float = 1.0,
        n_time_buckets: int = 0,
        horizon: Any = None,
        embargo: Any = None,
        iid_mode: bool = False,
        halo: bool = False,
        halo_gamma: float = 0.4,
        halo_candidates: int = 3,
        pair_splits: bool = True,
        pair_trigger: float = 0.25,
        pair_top_k: int = 6,
        max_pair_splits_per_tree: int = 3,
        pair_l2: float = 2.0,
        time_decay: float = 1.0,
        tail_buffer: float = 0.02,
        tree_decay: float = 1.0,
        objective: str = "auto",
        quantile: float = 0.5,
        huber_delta: float = 1.0,
        rank_temperature: float = 1.0,
        n_jobs: int = 0,
        random_state: int | None = 0,
        verbose: int = 0,
        early_stopping_rounds: int | None = None,
        regime_priority: bool = True,
        cat_min_count: int = 5,
        monotone_constraints=None,
        causal_structure: bool = False,
        finer_torb: bool = True,
        device: str = "cpu",
        lint_mode: str = "strict",
        rank_pairwise: bool = False,
        cost_threshold: float = 0.0,
        quantiles=None,
        n_horizons: int = 1,
        n_classes: int = 1,
        class_weight=None,
        init_score: float | None = None,
        interaction_constraints=None,
        interaction_pairs=None,
        custom_objective=None,
        l1: float = 0.0,
        max_delta_step: float = 0.0,
        goss: bool | None = None,
        goss_top: float = 0.2,
        goss_other: float = 0.1,
        structure_frac: float = 1.0,
        causal_bins: bool = False,
        torb_min_cover: float = 0.0,
        cat_smooth: float = 1.0,
        col_major_hist: bool = True,
        colsample_bynode: float = 1.0,
        bin_sample: int = 262144,
        entity_subsample: float = 1.0,
        eval_metric: str = "auto",
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.l2 = l2
        self.min_split_gain = min_split_gain
        self.max_bin = max_bin
        self.subsample = subsample
        self.colsample = colsample
        self.n_time_buckets = n_time_buckets
        self.horizon = horizon
        self.embargo = embargo
        self.iid_mode = iid_mode
        self.halo = halo
        self.halo_gamma = halo_gamma
        self.halo_candidates = halo_candidates
        self.pair_splits = pair_splits
        self.pair_trigger = pair_trigger
        self.pair_top_k = pair_top_k
        self.max_pair_splits_per_tree = max_pair_splits_per_tree
        self.pair_l2 = pair_l2
        self.time_decay = time_decay
        self.tail_buffer = tail_buffer
        self.tree_decay = tree_decay
        self.objective = objective
        self.quantile = quantile
        self.huber_delta = huber_delta
        self.rank_temperature = rank_temperature
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.regime_priority = regime_priority
        self.cat_min_count = cat_min_count
        self.monotone_constraints = monotone_constraints
        self.causal_structure = causal_structure
        self.finer_torb = finer_torb
        self.device = device
        self.lint_mode = lint_mode
        self.rank_pairwise = rank_pairwise
        self.cost_threshold = cost_threshold
        self.quantiles = quantiles
        self.n_horizons = n_horizons
        self.n_classes = n_classes
        self.class_weight = class_weight
        self.init_score = init_score
        self.interaction_constraints = interaction_constraints
        self.interaction_pairs = interaction_pairs
        self.custom_objective = custom_objective
        self.l1 = l1
        self.max_delta_step = max_delta_step
        self.goss = goss
        self.goss_top = goss_top
        self.goss_other = goss_other
        self.structure_frac = structure_frac
        self.causal_bins = causal_bins
        self.torb_min_cover = torb_min_cover
        self.cat_smooth = cat_smooth
        self.col_major_hist = col_major_hist
        self.colsample_bynode = colsample_bynode
        self.bin_sample = bin_sample
        self.entity_subsample = entity_subsample
        self.eval_metric = eval_metric
        self._booster: NativeBooster | None = None
        self._calibrator = None

    def _task(self) -> str:
        return "regressor"

    def fit(self, X, y, **fit_params):
        p = _params(self)
        p.update(fit_params)
        lint_mode = p.pop("lint_mode", "strict")
        self._booster = NativeBooster(**{k: v for k, v in p.items() if k != "lint_mode"})
        self._booster.fit(X, y, task=self._task(), lint_mode=lint_mode, **fit_params)
        return self

    def predict(self, X):
        if self._booster is None:
            raise RuntimeError("not fitted")
        return self._booster.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("not fitted")
        return self._booster.feature_importances()

    @property
    def evals_result_(self) -> dict:
        if self._booster is None:
            return {}
        return self._booster.evals_result_

    @property
    def best_iteration_(self) -> int | None:
        if self._booster is None:
            return None
        return self._booster.best_iteration_

    def save_model(self, path: str) -> None:
        if self._booster is None:
            raise RuntimeError("not fitted")
        path = str(path)
        if path.endswith(".json"):
            import base64
            import json

            payload = {
                "format": "tidegbm",
                "json": json.loads(self._booster.save_json()),
                "artifact_b64": base64.b64encode(self._booster.save()).decode("ascii"),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            return
        with open(path, "wb") as f:
            f.write(self._booster.save())

    @classmethod
    def load_model(cls, path: str):
        path = str(path)
        if path.endswith(".json"):
            import base64
            import json

            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            blob = base64.b64decode(payload["artifact_b64"])
        else:
            with open(path, "rb") as f:
                blob = f.read()
        obj = cls()
        obj._booster = NativeBooster.load(blob)
        return obj

    def predict_multi(self, X):
        if self._booster is None:
            raise RuntimeError("not fitted")
        return self._booster.predict_multi(X)

    def feature_importances(self, kind: str = "gain") -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("not fitted")
        return self._booster.feature_importances(kind)

    def shap_values(self, X):
        if self._booster is None:
            raise RuntimeError("not fitted")
        return self._booster.pred_contrib(X)

    def to_onnx_json(self) -> str:
        if self._booster is None:
            raise RuntimeError("not fitted")
        return self._booster.to_onnx_json()

    def to_onnx(self, path: str) -> None:
        """Write an ONNX protobuf. Requires ``pip install onnx`` (or ``tidegbm[onnx]``).

        Axis-split thresholds use frozen bin edges. Categorical and PAIR nodes are
        not faithful to native scoring on raw X. Prefer ``predict`` in production.
        """
        if self._booster is None:
            raise RuntimeError("not fitted")
        from tidegbm.onnx_export import booster_to_onnx_bytes

        blob = booster_to_onnx_bytes(self._booster.save_json())
        with open(path, "wb") as f:
            f.write(blob)

    def update(self, X, y, n_estimators: int = 10):
        if self._booster is None:
            raise RuntimeError("not fitted")
        self._booster.update(X, y, n_estimators=n_estimators)
        return self

    def __getstate__(self):
        st = {k: getattr(self, k) for k in self.get_params()}
        st["_blob"] = None if self._booster is None else self._booster.save()
        st["_fitted_names"] = None if self._booster is None else self._booster.feature_names
        return st

    def __setstate__(self, st):
        blob = st.pop("_blob", None)
        names = st.pop("_fitted_names", None)
        self.__dict__.update(st)
        self._booster = None
        if blob is not None:
            self._booster = NativeBooster.load(blob)
            if names:
                self._booster.feature_names = names


class TideRegressor(TideEstimator, RegressorMixin):
    """Gradient boosting regressor with temporal ordered residuals (TORB).

    Parameters
    ----------
    n_estimators, learning_rate, max_depth, ...
        See ``TideEstimator`` and ``docs/API.md`` on GitHub.

    Examples
    --------
    >>> from tidegbm import TideRegressor
    >>> model = TideRegressor(horizon=1, objective="pairwise")
    >>> model.fit(X, y, timestamp="ts", entity="symbol")
    >>> model.predict(X_new)
    """

    def _task(self) -> str:
        return "regressor"


class TideClassifier(TideEstimator, ClassifierMixin):
    """Binary or multiclass gradient boosting classifier with TORB.

    Binary problems use ``logloss`` when ``objective="auto"``.
    Multiclass uses ``softmax`` over ``n_classes``.
    """

    def _task(self) -> str:
        return "classifier"

    def fit(self, X, y, **fit_params):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        mapping = {c: i for i, c in enumerate(self.classes_)}
        y01 = np.vectorize(mapping.get)(y).astype(np.float64)
        p = _params(self)
        p.update(fit_params)
        lint_mode = p.pop("lint_mode", "strict")
        if self.classes_.size == 2:
            p["objective"] = "logloss" if self.objective in {"auto", "mse"} else self.objective
        else:
            p["objective"] = "multiclass"
            p["n_classes"] = int(self.classes_.size)
        self._booster = NativeBooster(**{k: v for k, v in p.items() if k != "lint_mode"})
        self._booster.fit(X, y01, task="classifier", lint_mode=lint_mode, **fit_params)
        return self

    def predict_proba(self, X):
        if self._booster is None:
            raise RuntimeError("not fitted")
        if len(self.classes_) == 2:
            return self._booster.predict_proba(X)
        raw = self._booster.predict_multi(X)
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


class TideRanker(TideEstimator, RankerMixin):
    """Cross-sectional learning-to-rank within timestamp groups.

    Groups are formed by ``timestamp`` (all entities at time *t*).
    Default objective is ``"rank"`` when ``objective`` is ``"auto"`` or ``"mse"``.
    """

    def _task(self) -> str:
        return "ranker"

    def fit(self, X, y, **fit_params):
        p = _params(self)
        p.update(fit_params)
        if p.get("objective") in {"auto", "mse"}:
            p["objective"] = "rank"
        lint_mode = p.pop("lint_mode", "strict")
        self._booster = NativeBooster(**{k: v for k, v in p.items() if k != "lint_mode"})
        self._booster.fit(X, y, task="ranker", lint_mode=lint_mode, **fit_params)
        return self
