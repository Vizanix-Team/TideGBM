# Python API reference

TideGBM exposes a **scikit-learn–compatible** API on top of a C++ histogram gradient boosting core. All estimators share the same hyperparameters and `fit` kwargs unless noted.

```python
import tidegbm
from tidegbm import TideRegressor, TideClassifier, TideRanker, TideConfig
from tidegbm import WalkForwardSplit, combinatorial_purged_cv, lint, calibrate_oof
```

---

## Table of contents

1. [Data conventions](#data-conventions)
2. [Estimators](#estimators)
3. [Constructor parameters](#constructor-parameters)
4. [`fit()` parameters](#fit-parameters)
5. [Prediction & inference](#prediction--inference)
6. [Model persistence](#model-persistence)
7. [Cross-validation & lint](#cross-validation--lint)
8. [`TideConfig`](#tideconfig)
9. [Optional extras](#optional-extras)
10. [Constants & build info](#constants--build-info)

---

## Data conventions

### Input types

| Argument | Accepted types |
|---|---|
| `X` | `pandas.DataFrame`, `numpy.ndarray` (float), column names become feature names |
| `y` | 1-D array-like, same length as `X` |
| `timestamp` | `datetime64`, `int` (epoch or bar index), `str` parseable dates, or a column name if `X` is a DataFrame |
| `entity` | int/str labels per row (symbol, user, SKU, …) for panel structure |
| `sample_weight` | per-row weights, same length as `y` |

Rows should be sortable by `(timestamp, entity)`. TideGBM re-sorts internally if needed.

### Time mode vs IID mode

| Mode | Requirements | Use case |
|---|---|---|
| **Time mode** (default) | `timestamp` + `horizon` required | Markets, panels, any ordered data |
| **`iid_mode=True`** | No timestamp; TORB disabled | Debugging only — unsafe for production panels |

In time mode, `eval_set` timestamps must be **strictly after** all training timestamps.

### Categorical features

Pass `categorical_feature="auto"` (default) to treat low-cardinality string/object columns as categorical, or pass an explicit list of column names / indices.

---

## Estimators

### `TideRegressor`

Gradient boosting **regressor** with temporal ordered residual boosting (TORB).

```python
from tidegbm import TideRegressor

model = TideRegressor(n_estimators=200, max_depth=6, horizon=1, objective="pairwise")
model.fit(X, y, timestamp="ts", entity="symbol")
pred = model.predict(X_new)
```

**Task:** `"regressor"`. Default objective when `objective="auto"`: `"mse"`.

---

### `TideClassifier`

Binary or multiclass **classifier**.

```python
from tidegbm import TideClassifier

clf = TideClassifier(n_estimators=100, horizon=5)
clf.fit(X, y, timestamp="ts", entity="id")
proba = clf.predict_proba(X_new)
labels = clf.predict(X_new)
```

| Classes | Objective used |
|---|---|
| 2 | `"logloss"` when `objective` is `"auto"` or `"mse"` |
| >2 | `"multiclass"`; set `n_classes` automatically |

---

### `TideRanker`

**Learning-to-rank** within cross-sections grouped by timestamp.

```python
from tidegbm import TideRanker

ranker = TideRanker(n_estimators=80, horizon=1)
ranker.fit(X, y, timestamp="ts", entity="symbol")
scores = ranker.predict(X_new)
```

Default objective when `objective` is `"auto"` or `"mse"`: `"rank"`.

---

## Constructor parameters

All three estimators accept the same keyword arguments (sklearn `get_params` / `set_params` compatible).

### Tree ensemble

| Parameter | Default | Description |
|---|---:|---|
| `n_estimators` | `100` | Number of boosting trees |
| `learning_rate` | `0.1` | Shrinkage per tree |
| `max_depth` | `6` | Maximum tree depth |
| `max_leaves` | `31` | Leaf cap (depth also applies) |
| `min_samples_leaf` | `20` | Minimum rows per leaf |
| `min_child_weight` | `1e-3` | Minimum sum of Hessian in a leaf |
| `min_split_gain` | `0.0` | Minimum loss reduction to split |
| `subsample` | `1.0` | Row subsample fraction per tree |
| `colsample` | `1.0` | Column subsample fraction per tree |
| `colsample_bynode` | `1.0` | Column subsample per split node |
| `l2` | `1.0` | L2 leaf regularization |
| `l1` | `0.0` | L1 leaf regularization |
| `max_delta_step` | `0.0` | Max leaf weight update (0 = unlimited) |

### Histograms & binning

| Parameter | Default | Description |
|---|---:|---|
| `max_bin` | `255` | Histogram bins per numeric feature |
| `bin_sample` | `262144` | Max rows for quantile edge estimation |
| `causal_bins` | `False` | If `True`, bin edges from causal prefix only |
| `cat_min_count` | `5` | Min count to keep a categorical level |
| `cat_smooth` | `1.0` | Laplace smoothing for categorical splits |
| `col_major_hist` | `True` | Column-major histogram layout (fast path) |

### Time / panel (TORB)

| Parameter | Default | Description |
|---|---:|---|
| `horizon` | `None` | Label horizon in `timestamp` units (**required** in time mode) |
| `embargo` | `None` | Extra gap after train end (defaults to 0) |
| `n_time_buckets` | `0` | TORB time buckets; `0` = auto (32 or 64 from horizon) |
| `causal_structure` | `False` | Hold out tail of train for split structure (legacy; default off) |
| `structure_frac` | `1.0` | Fraction of train used for structure when causal |
| `finer_torb` | `True` | Finer-grained TORB buckets |
| `torb_min_cover` | `0.0` | Blend sparse TORB prefixes toward full-train leaves |
| `time_decay` | `1.0` | Exponential decay weight by recency |
| `tail_buffer` | `0.02` | Fraction of newest rows down-weighted |
| `tree_decay` | `1.0` | Per-tree weight decay |
| `entity_subsample` | `1.0` | Subsample whole entities, not random rows |
| `uniqueness_weight` | `False` | Down-weight overlapping labels by average uniqueness |

#### `uniqueness_weight`

An `h`-step label sampled every step overlaps its `h-1` neighbours, so consecutive rows are
far from independent observations. Purge and embargo keep *evaluation* honest; they do not
touch the training set, where the booster still counts every overlapping row as a full
sample. With `uniqueness_weight=True` each row is weighted by the mean of `1 / concurrency`
over the span its label covers, following Lopez de Prado. Concurrency is counted **per
entity**, since an instrument's labels overlap its own future labels and not other
instruments'. Weights are rescaled to mean 1, so `learning_rate` and `n_estimators` keep
their usual range.

At `horizon=1` there is no overlap and the weights are exactly 1.0 — the option is inert,
not merely small. It only does work when labels overlap.

The weights are also available directly:

```python
from tidegbm import average_uniqueness, uniqueness_weights

u = average_uniqueness(timestamp, horizon=5, entity=symbols)   # in (0, 1]
w = uniqueness_weights(timestamp, horizon=5, entity=symbols)   # rescaled to mean 1
```

### Objectives & metrics

| Parameter | Default | Description |
|---|---:|---|
| `objective` | `"auto"` | See [objectives table](#objectives) |
| `quantile` | `0.5` | Target quantile for `"quantile"` |
| `quantiles` | `None` | Multi-quantile heads |
| `huber_delta` | `1.0` | Huber delta |
| `rank_temperature` | `1.0` | Softmax temperature for rank objective |
| `eval_metric` | `"auto"` | `"auto"` → cross-sectional IC in time/rank mode |
| `cost_threshold` | `0.0` | Cost-aware residual threshold |
| `init_score` | `None` | Initial prediction bias |

#### Objectives

| Value | Meaning |
|---|---|
| `"auto"` | MSE for regressor, logloss for binary classifier, rank for ranker |
| `"mse"` | Squared error |
| `"pairwise"` | Within-date LambdaMART-style ranking (cross-sectional IC) |
| `"rank"` | Learning-to-rank |
| `"quantile"` | Quantile regression |
| `"huber"` | Huber loss |
| `"logloss"` | Binary log loss |
| `"multiclass"` | Softmax multiclass |
| custom callback | Pass `custom_objective` callable |

> **Note:** Use `"pairwise"` for cross-sectional IC / ranking. Keep `"mse"` or `"auto"` when you need level fit (R² on panels).

### Sampling & speed

| Parameter | Default | Description |
|---|---:|---|
| `goss` | `None` | Gradient-based One-Side Sampling; `None` = auto on at ≥250k rows |
| `goss_top` | `0.2` | Top gradient fraction kept |
| `goss_other` | `0.1` | Random sample from remainder |
| `n_jobs` | `0` | OpenMP threads (`0` = hardware concurrency) |
| `device` | `"cpu"` | `"gpu"` only if wheel built with `TIDEGBM_CUDA=1` |

### Splits & constraints

| Parameter | Default | Description |
|---|---:|---|
| `monotone_constraints` | `None` | Per-feature −1 / 0 / +1 |
| `interaction_constraints` | `None` | Groups of features that may interact |
| `interaction_pairs` | `None` | Allowed interaction pairs |
| `pair_splits` | `True` | Enable PAIR split type |
| `pair_trigger` | `0.25` | Activation threshold |
| `pair_top_k` | `6` | Top feature pairs to consider |
| `max_pair_splits_per_tree` | `3` | Cap PAIR splits per tree |
| `pair_l2` | `2.0` | L2 on pair coefficients |
| `regime_priority` | `True` | Prefer regime-aware splits when regime feature set |

### HALO (optional, expensive)

| Parameter | Default | Description |
|---|---:|---|
| `halo` | `False` | HALO candidate search (off by default) |
| `halo_gamma` | `0.4` | HALO mixing |
| `halo_candidates` | `3` | Candidates per level |

### Training control

| Parameter | Default | Description |
|---|---:|---|
| `random_state` | `0` | RNG seed |
| `verbose` | `0` | Log level |
| `early_stopping_rounds` | `None` | Stop after N rounds without eval improvement |
| `lint_mode` | `"strict"` | `"strict"` \| `"warn"` \| `"off"` for leakage lint at fit |
| `iid_mode` | `False` | Disable time semantics |
| `n_horizons` | `1` | Multi-horizon heads |
| `n_classes` | `1` | Multiclass class count |
| `class_weight` | `None` | `"balanced"` or dict for classifier |

---

## `fit()` parameters

Passed to `TideRegressor.fit`, `TideClassifier.fit`, or `TideRanker.fit`:

| Parameter | Description |
|---|---|
| `timestamp` | **Required** in time mode. Per-row time index |
| `entity` | Panel entity id per row (recommended for markets) |
| `sample_weight` | Row weights |
| `eval_set` | `(X_val, y_val)` tuple — validation must be in the **future** |
| `eval_timestamp` | Timestamps for eval rows |
| `feature_name` | Override feature names |
| `categorical_feature` | `"auto"` or list of cat column names/indices |
| `feature_groups` | Dict mapping group name → feature list (TORB grouping) |
| `regime` | Regime column name or array for regime splits |
| `auto_group` | Infer feature groups from correlations |

Example:

```python
model.fit(
    X_train,
    y_train,
    timestamp=df["ts"],
    entity=df["symbol"],
    sample_weight=df["w"],
    eval_set=(X_val, y_val),
    eval_timestamp=df_val["ts"],
    categorical_feature=["sector"],
    feature_groups={"risk": ["vol"], "alpha": ["mom", "rev"]},
    lint_mode="strict",
)
```

---

## Prediction & inference

| Method | Returns |
|---|---|
| `predict(X)` | Point predictions (regressor/ranker) or class labels (classifier) |
| `predict_proba(X)` | Class probabilities (`TideClassifier` only) |
| `predict_multi(X)` | Multi-head / multi-quantile output |
| `feature_importances_` | Gain-based importances (property) |
| `feature_importances(kind="gain"\|"split")` | Importance by kind |
| `shap_values(X)` | Saabas path contributions (additive, not Independent TreeSHAP) |
| `evals_result_` | Eval metric history when early stopping used |
| `best_iteration_` | Best tree index or `None` |

---

## Model persistence

| Method | Description |
|---|---|
| `save_model(path)` | Binary `.tgbm` (TGBM v3) or `.json` (JSON + base64 artifact) |
| `load_model(path)` | Class method; returns fitted estimator |
| `to_onnx_json()` | ONNX-compatible JSON (thresholds from frozen bins) |
| `to_onnx(path)` | ONNX protobuf; requires `pip install tidegbm[onnx]` |
| `update(X, y, n_estimators=10)` | Append trees to an existing model |

Pickle works via `__getstate__` / `__setstate__` (embeds binary blob).

---

## Cross-validation & lint

### `WalkForwardSplit`

Expanding-window walk-forward with purge and embargo.

```python
from tidegbm import WalkForwardSplit

wf = WalkForwardSplit(n_splits=5, embargo=1, purge=1, min_train_fraction=0.3)
for train_idx, test_idx in wf.split(X, timestamp=ts):
    ...
```

| Parameter | Default | Description |
|---|---:|---|
| `n_splits` | `5` | Number of test folds |
| `embargo` | `None` | Gap after train (timestamp units) |
| `purge` | `None` | Extra past dropped from train end (label overlap) |
| `min_train_fraction` | `0.3` | Skip folds with too little train data |

### `combinatorial_purged_cv`

Combinatorial purged CV (López de Prado style):

```python
from tidegbm import combinatorial_purged_cv

for train_idx, test_idx in combinatorial_purged_cv(ts, n_splits=6, n_test_folds=2, embargo=1):
    ...
```

### `lint`

Pre-fit leakage checks:

```python
from tidegbm import lint, LintReport

report: LintReport = lint(X, y, timestamp, feature_names=cols, raise_on_error=False)
print(report)           # human-readable
report.raise_if_errors()
```

Checks: missing/unsorted timestamps, duplicate chaos, suspicious feature names, feature–target correlation spikes, lookahead patterns.

### `calibrate_oof`

Purged out-of-fold isotonic calibration:

```python
from tidegbm import calibrate_oof

iso = calibrate_oof(model, X, y, timestamp, n_splits=4, embargo=1)
calibrated = iso.transform(model.predict(X))
```

Fit only on OOF scores — never on in-sample train predictions.

---

## `TideConfig`

Dataclass mirror of constructor kwargs for scripts and configs:

```python
from tidegbm import TideConfig

cfg = TideConfig(n_estimators=200, max_depth=6, horizon=1, objective="pairwise")
cfg.validate()
params = cfg.__dict__  # pass to TideRegressor(**{k: v for k, v in params.items() if k != "extra"})
```

Call `cfg.validate()` before training to catch invalid combinations early.

---

## Optional extras

Install with bracket notation:

```bash
pip install tidegbm[onnx]    # ONNX export
pip install tidegbm[bench]   # LightGBM + CatBoost benchmarks
pip install tidegbm[fast]    # numba accelerators where applicable
pip install tidegbm[dev]     # pytest, ruff, coverage
```

---

## Constants & build info

```python
import tidegbm

tidegbm.__version__      # "0.4.0"
tidegbm.cuda_compiled    # True if GPU histograms were compiled in
```

---

## Related documents

| Document | Contents |
|---|---|
| [`QUICKSTART.md`](QUICKSTART.md) | Install & first model |
| [`DESIGN.md`](DESIGN.md) | Algorithm & TORB internals |
| [`TIDEBENCH.md`](TIDEBENCH.md) | Benchmark protocol |
| [`LEAKAGE.md`](LEAKAGE.md) | Common leakage mistakes |
| [`COMPATIBILITY.md`](COMPATIBILITY.md) | Semver & model format |

---

<div align="center">

**Maintained by [Vizanix](https://github.com/Vizanix)** · [vizanix.tech](https://vizanix.tech)

</div>
