# TideBench

Two protocols live under `benchmarks/`.

## Synthetic (`tidebench.py`)

Controlled panels (linear / regime / noise) vs sklearn HistGBDT and optional LightGBM.
Use this to debug leakage, not to rank libraries.

```bash
python benchmarks/tidebench.py --quick
```

## Markets (`markets.py`) — 30 public names

Purged expanding walk-forward on **Yahoo daily bars** (adjusted close). Default universe is 30 liquid names: 12 US stocks, 10 ETFs, 4 international listings, 2 FX, 2 crypto.

| | |
|---|---|
| Labels | forward return over `horizon` trading days; **demeaned within date** (`xs_center=True`) so MSE matches cross-sectional IC. Same `y` for every model. |
| Features | own-lags `ret_1, ret_5, ret_21, vol_21, mom_63, amihud, log_dvol` plus same-date ranks `xs_mom, xs_vol` |
| Split | `WalkForwardSplit` with `embargo=purge=horizon` |
| HPs | 80 trees, **depth 6**, lr 0.08, `max_bin=255` (matched across libraries) |
| TideGBM | `timestamp` + `entity` + TORB; **`objective="pairwise"`** (within-date LambdaMART-style); `n_time_buckets=0` auto |
| `tide-mse` | same TORB defaults with `objective="mse"` |
| `tide-holdout` | old MSE defaults: structure holdout + `max_bin=64` + `causal_bins=True` |
| LightGBM / CatBoost | same `X, y`, no time prior |

```bash
pip install -e ".[bench]"
python benchmarks/markets.py --quick
python benchmarks/markets.py --horizons 1,3,5,10   # default
python benchmarks/markets.py --depth 4             # older matched depth
python benchmarks/markets.py --no-xs-center
python benchmarks/markets.py --refresh
```

### Snapshot (2026-08-25, this repo) — best quality recipe

Yahoo daily adj-close, ~2021-10 → 2026-08, 30 names, 80 trees, **depth 6**, 3 purged folds, `xs_center=True`.

| model | h=1 | h=3 | h=5 | h=10 | mean |
|---|---|---|---|---|---|
| **TideGBM pairwise** | **0.026** | **0.020** | **0.050** | 0.027 | **0.031** |
| TideGBM MSE | 0.026 | 0.016 | 0.026 | **0.039** | 0.027 |
| LightGBM 4.7 | 0.023 | 0.008 | 0.024 | 0.024 | 0.019 |
| CatBoost 1.2 | 0.021 | 0.005 | 0.024 | 0.025 | 0.019 |
| tide-holdout | 0.013 | 0.010 | 0.016 | 0.025 | 0.016 |

**Tide pairwise > LightGBM on 4/4 horizons.** Mean IC +60% vs LightGBM (0.031 vs 0.019).  
`tide-mse` still wins h=10 alone on this snapshot — keep both in the table.

Fold ICs (Tide pairwise):  
1d 0.007, 0.048, 0.024 · 3d 0.020, −0.001, 0.042 · 5d 0.055, 0.042, 0.054 · 10d 0.004, 0.043, 0.035.

LightGBM remains ~5–10× faster to fit. Three folds, IC noisy. Not SOTA, not a desk.

### Quality search notes (what actually helped)

Ablation + bakeoff on this stand (same folds, date-demeaned `y`):

| Change | Effect |
|---|---|
| Drop structure/bin holdout; `max_bin=255`; TORB prefix blend; auto buckets | Closed the old 1d gap under MSE |
| **`objective="pairwise"`** | Largest IC lift, especially h=5 |
| **`max_depth=6`** (matched for LGB/CatBoost) | Pairwise likes depth; LGB did not gain |
| More trees / lower lr / time_decay / seed bagging / early-stop on train tail | Neutral or worse mean IC |
| Extra xs ranks (`xs_ret1`, …) | Hurt MSE Tide; mixed for pairwise |
| Vol-scale `y` | Bigger gap vs LGB, **lower absolute** IC — not used |
| Rank-transform `y` | Helped mid horizons, hurt 1d |

For **absolute-level** regression (R² on a linear panel), keep `objective="mse"`. Pairwise optimizes within-date ranking, not level fit.

### What does not generalize

Hyperparameter fishing on this CSV, Tide-only features, or claiming desk alpha.

## GPU wheel

Default `pip install` is CPU. GPU histograms compile when `TIDEGBM_CUDA=1` and `nvcc` is on `PATH`:

```bash
TIDEGBM_CUDA=1 pip wheel . -w dist
# or
bash scripts/build_gpu_wheel.sh
```

GitHub: `.github/workflows/gpu-wheel.yml`. `device="gpu"` needs a CUDA driver at fit time.
