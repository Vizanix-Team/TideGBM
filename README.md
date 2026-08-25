# TideGBM

Temporal ordered gradient boosting. **C++ core**, Python API.

You bring `X` and `y`. TideGBM does not compute indicators. It treats **time and panel** as first-class (TORB residuals, purge/embargo, leak lint).

```python
from tidegbm import TideRegressor

model = TideRegressor(horizon=1, n_estimators=200)
model.fit(X, y, timestamp="ts", entity="symbol")
pred = model.predict(X_live)
model.save_model("alpha.tgbm")
```

v0.4: column-major histograms, L1 / GOSS (auto at 250k rows), frequency categorical bins, optional ONNX protobuf (`tidegbm[onnx]`), cibuildwheel + CUDA wheel workflow.

This is not “better than neural nets.” Target is a stronger GBDT for time/panel tables under purged evaluation. Do not call it SOTA. On the public 30-name stand (`benchmarks/markets.py --horizons 1,3,5,10`, depth 6, date-demeaned labels) TideGBM with `objective="pairwise"` beats LightGBM mean IC on all four horizons; use MSE when you care about levels, not ranks. LightGBM is still faster. SHAP-style values are Saabas path attributions (additive), not Independent TreeSHAP. Default wheels are CPU; `device="gpu"` raises unless the wheel was built with `TIDEGBM_CUDA=1`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Requires a C++17 compiler and OpenMP (`g++`). Optional: `pip install -e ".[onnx]"` for `to_onnx()`, `".[bench]"` for LightGBM + CatBoost.

```bash
python benchmarks/markets.py --quick    # 8 public names, purged WF
TIDEGBM_CUDA=1 bash scripts/build_gpu_wheel.sh   # needs nvcc, not a GPU
```

Scale smoke: `python benchmarks/bench_scale.py`.

## Defaults that match production

- `timestamp` required (unless `iid_mode=True`)
- `horizon` required in time mode
- `lint_mode="strict"` refuses lookahead features
- `halo=False` (enable only if you can afford it)
- `goss=None` auto-enables GOSS at ≥250k rows; set `goss=False` to keep every row
- `eval_metric="auto"` monitors cross-sectional IC when time or rank is on
- `max_bin=255`; split structure uses the full train window (`causal_structure=False`, `structure_frac=1`). TORB still prefixes residuals.
- `n_time_buckets=0` auto: 32 buckets for short horizons, 64 when the label spans ≥8 median time steps
- `causal_bins=False` (full-train edges); sparse TORB prefixes blend to full leaf values
- For cross-sectional IC on panels, prefer `objective="pairwise"` (MSE remains `objective="auto"` / `"mse"`)
- eval set must be strictly in the future

## Layout

- `cpp/` — library, C API (`include/tidegbm/c_api.h`), optional `histogram_cuda.cu`
- `src/tidegbm/` — sklearn wrappers
- `benchmarks/markets.py` — 30 public names vs LightGBM/CatBoost
- `benchmarks/tidebench.py` — synthetic multi-panel protocol
- `benchmarks/bench_scale.py` — 100k-row CPU histogram smoke
- `docs/TIDEBENCH.md` — stand protocol
- `docs/DESIGN.md` — algorithm RFC
- `docs/LEAKAGE.md` — anti-examples
- `docs/COMPATIBILITY.md` — semver / model format

Apache-2.0
