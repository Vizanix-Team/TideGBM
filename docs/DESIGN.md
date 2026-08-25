# TideGBM design

## Thesis

Generic GBDT treats rows as i.i.d. Quant datasets are a panel `(time × entity × feature)` with overlapping labels, collinear user features, and regime shifts. TideGBM is a histogram, leaf-wise booster whose **training-time residuals are causal**. Users supply features; the library supplies the prior.

## Invariant

For an observation at time `t`, no gradient, bin edge, categorical statistic, or early-stopping score may be computed from rows with time `≥ t − embargo`, where `embargo = horizon + extra`.

## TORB

Time is quantized into `n_time_buckets` (or unique timestamps when `finer_torb=True`).

- Prediction used to form the gradient of bucket `b` uses leaf values fit only on timestamps strictly before bucket `b` (minus embargo).
- The serialized inference model uses leaf values fit on all training data.
- Split **structure** uses the full train window by default (`causal_structure=False`, `structure_frac=1`). Holding out the newest bucket is optional; it is usually harmful in walk-forward because the next test fold looks like the most recent train days.
- Bin edges use the full purged train window by default (`causal_bins=False`). Optional HAB-C (`causal_bins=True`) fits edges on the older half only.
- Sparse TORB prefixes blend toward full-train leaf values (`torb_min_cover`, default `2 * min_samples_leaf`) so early buckets are not zeroed.
- `n_time_buckets=0` (default) picks **32** when the label horizon is shorter than ~8 median time steps, else **64**. Long overlapping labels need finer TORB; short nearly-iid labels over-fragment.
- For **cross-sectional IC** on multi-name panels, `objective="pairwise"` (LambdaMART-style within timestamp) beats MSE. Keep MSE for absolute-level regression.

This is CatBoost Ordered with the permutation replaced by calendar order on **residuals**, not by deleting the newest data.

## Other modules

| Module | Role |
|---|---|
| HAB-C | Quantile bins on a causal prefix; edges frozen in the artifact |
| IFB | One candidate feature per user group per split |
| GIS-T | Sample time blocks; keep a tail-buffer of extreme blocks |
| HALO | Optional: score a split as `gain + γ (child gains)` (default off) |
| PAIR | Hash of two bin ids, capped per tree (off on ≥1M rows unless forced) |
| Rank | ListNet or pairwise LambdaMART-style within timestamp |
| Regime | User column, `SplitType::Regime`, never derived from `y` |

## Non-goals

No bundled market data, no indicator formulas, no execution. Auto-regime from `y` is forbidden. Do not claim SOTA without TideBench on real panels.

## Implementation

The trainer is C++ (`cpp/`) with a pybind11 module `tidegbm._core`. Python owns lint, CV, calibration, and the sklearn wrapper.
