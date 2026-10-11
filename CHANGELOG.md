# Changelog

## 0.5.0

Hot-path rewrite. Every change below is semantics-preserving: fold ICs on the market stand
are identical to 0.4.0 to five decimals, and the 39 existing tests pass unchanged.

### Speed

- **TORB prefixes no longer rescan the data per bucket.** Rows are sorted by timestamp, so a
  bucket's causal past is the contiguous prefix `[0, cut_b)`. The cuts are precomputed once
  per fit and the per-tree loop became a single sweep with running per-node sums, replacing
  `O(n_buckets * n)` with `O(n + n_buckets * nodes)`. Accumulation order is unchanged, so leaf
  values are bit-identical.
- **Rank objectives stopped rebuilding a hash map every iteration.** Timestamp groups are
  contiguous after the sort, so group boundaries come from one scan; unsorted input falls back
  to a single stable sort. The per-group loop is now parallel — groups own disjoint rows.
- **PAIR split evaluation fused and parallelised.** Hashing the pair bin and accumulating it
  are one pass instead of two, scratch histograms are reused per thread, and candidate pairs
  are evaluated in parallel.
- **Histogram bins are interleaved.** Gradient, hessian and count for a bin now live together,
  turning three cache streams into one in both the accumulate loop and the split scan.
- **Histograms are zeroed once, not three times.** `resize` allocated *and* zeroed, then
  `fill_hist` resized again, then `build_hist` zeroed again. `resize` now only allocates.
- **One histogram allocation per split instead of three.** Only the smaller child is built;
  the parent's own buffer is subtracted in place to become the larger child.

### Quality

- `uniqueness_weight=True` (default off): weight each row by the average uniqueness of its
  label, following Lopez de Prado. An `h`-step label sampled every step overlaps its `h-1`
  neighbours, so overlapping rows are not independent observations — purge and embargo fix the
  evaluation, not the training set. Concurrency is counted per entity. Exposed directly as
  `average_uniqueness` and `uniqueness_weights`.
  On the 30-name market stand the effect is not measurable (see `docs/TIDEBENCH.md`): exactly
  zero at `horizon=1` where labels do not overlap, and within noise at h=5 and h=10. It is
  shipped opt-in for that reason.
- **Ranking objectives report a ranking loss.** `evals_result_["train"]` returned squared error
  for `rank`/`pairwise`, which is meaningless for a ranking fit. It now reports negative mean
  within-group correlation.

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
