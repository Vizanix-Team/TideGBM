# Changelog

## 0.4.0

Speed and quality on the C++ trainer; remaining packaging holes.

- Column-major histograms with sorted leaf indices and prefetch; OpenMP bin fit/transform
- L1-consistent split gain and leaf weights; `max_delta_step`; Laplace `cat_smooth`
- GOSS: unbiased hessian scaling; auto-on at ≥250k rows (`goss=None`); `goss=False` keeps full data
- Frequency categorical vocab with `cat_min_count` (no longer keeps the first N uniques)
- `structure_frac` / `colsample_bynode`; PAIR/HALO still auto-off at 1M rows
- Deleted dead Python trainer modules; sklearn wrappers talk only to `_core`
- Optional ONNX protobuf (`to_onnx`, extra `onnx`); cibuildwheel workflow
- Scale smoke `benchmarks/bench_scale.py`; TideBench markets stand vs LightGBM/CatBoost
- SHAP remains Saabas (additive). TreeSHAP Independent is not claimed.
- GPU histograms: real CUDA kernel + `TIDEGBM_CUDA=1` wheel (`scripts/build_gpu_wheel.sh`, `.github/workflows/gpu-wheel.yml`). Default wheel is CPU; `device="gpu"` raises there. `tidegbm.cuda_compiled` reports the build.

- Leaf split cache (unchanged leaves are not rescanned); row-wise forest inference
- `eval_metric="auto"` uses cross-sectional IC for early stopping in time/rank mode
- Quantile edges from `bin_sample` (default 262144) rows; `entity_subsample` keeps whole entities
- Pairwise rank: |Δy| LambdaMART-style weights; large groups use strided pairs
- Defaults: `max_bin=255`, `causal_structure=False`, `structure_frac=1`, `causal_bins=False`. TORB still prefixes leaf residuals; sparse prefixes blend toward full-train leaves.
- `n_time_buckets=0` (default) auto-selects 32 or 64 from label horizon length
- TideBench markets: date-demeaned `y`, horizons 1/3/5/10, depth 6; Tide default on the stand is `objective="pairwise"` (MSE ablation kept as `tide-mse`)

## 0.3.0

Production C++ core.

- Histogram GBDT with OpenMP feature-parallel histograms, subtraction trick, optional mmap bins
- TORB (causal residuals + causal structure), finer time buckets, required `horizon`
- Frozen binner / categorical vocab, missing-bin NA direction, monotone and interaction constraints
- Serialization: binary `TGBM` v3, JSON (+ embedded artifact), C API, ONNX-JSON with raw thresholds
- SHAP-style path contributions (additive), gain/split importance
- Multi-quantile heads, multiclass softmax, pairwise ranking, custom objective callback
- Incremental `update`, cost-aware residual, regime split type, lint-in-fit
- `device="gpu"` requires `TIDEGBM_CUDA`; CPU build raises instead of faking GPU
- Leakage property tests, synthetic TideBench protocol, infer latency smoke test

## 0.2.0

First C++/pybind11 trainer (superseded).

## 0.1.0

Python prototype (no longer the production trainer).
