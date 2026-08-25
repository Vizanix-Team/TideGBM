"""TideBench markets: 30 public names vs LightGBM / CatBoost, purged walk-forward.

This is a stand, not a SOTA claim. Prices come from Yahoo's public chart API.
Features are causal lags of each name's own close/volume. The library does not
build indicators; this script does, then hands ``X, y, timestamp, entity`` to
each model. Competitors see the same X; TideGBM additionally uses TORB.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from tidegbm import TideRegressor, WalkForwardSplit

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "XOM", "JNJ", "TSLA", "UNH", "V",
    "SPY", "QQQ", "IWM", "TLT", "GLD", "EEM", "HYG", "XLF", "EFA", "LQD",
    "ASML", "BABA", "TM", "SONY",
    "EURUSD=X", "USDJPY=X", "BTC-USD", "ETH-USD",
]
QUICK = ["AAPL", "MSFT", "JPM", "XOM", "SPY", "TLT", "GLD", "BTC-USD"]
FEATS = ["ret_1", "ret_5", "ret_21", "vol_21", "mom_63", "amihud", "log_dvol", "xs_mom", "xs_vol"]
DATA = Path(__file__).resolve().parent / "data" / "markets.csv.gz"
CACHE = Path(__file__).resolve().parent / "data" / "yahoo"


def _yahoo_chart(symbol: str, range_: str = "5y") -> pd.DataFrame:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1d&range={range_}&includeAdjustedClose=true"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "TideGBM-TideBench/0.4"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        raise RuntimeError(f"empty chart for {symbol}: {payload.get('chart', {}).get('error')}")
    r = result[0]
    ts = r.get("timestamp") or []
    quote = ((r.get("indicators") or {}).get("quote") or [{}])[0]
    adj = ((r.get("indicators") or {}).get("adjclose") or [{}])
    close = (adj[0].get("adjclose") if adj else None) or quote.get("close")
    vol = quote.get("volume")
    if not ts or close is None:
        raise RuntimeError(f"no bars for {symbol}")
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(ts, unit="s", utc=True).tz_localize(None).normalize(),
            "close": close,
            "volume": vol if vol is not None else np.nan,
        }
    )
    df["symbol"] = symbol
    return df.dropna(subset=["close"])


def download(symbols: list[str], range_: str = "5y", pause: float = 0.25) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, sym in enumerate(symbols):
        path = CACHE / f"{sym.replace('/', '_').replace('=', '_')}.csv"
        try:
            if path.exists():
                part = pd.read_csv(path, parse_dates=["ts"])
            else:
                part = _yahoo_chart(sym, range_=range_)
                part.to_csv(path, index=False)
                time.sleep(pause)
            if part.empty:
                print(f"skip empty {sym}")
                continue
            frames.append(part)
            print(f"[{i + 1}/{len(symbols)}] {sym} {len(part)} bars")
        except (urllib.error.URLError, RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"skip {sym}: {exc}")
    if not frames:
        raise RuntimeError("downloaded zero symbols")
    return pd.concat(frames, ignore_index=True)


def make_panel(raw: pd.DataFrame, horizon: int = 1, xs_center: bool = True) -> pd.DataFrame:
    """Causal own-lags + cross-sectional ranks. Label = forward ``horizon``-day return.

    When ``xs_center`` is True (default), demean ``y`` within each date so MSE matches
    cross-sectional IC. Applied to every model on the stand.
    """
    h = max(int(horizon), 1)
    parts = []
    for sym, g in raw.groupby("symbol", sort=False):
        g = g.sort_values("ts").drop_duplicates("ts").copy()
        px = g["close"].astype(float)
        vol = g["volume"].astype(float)
        if vol.isna().all() or float(vol.fillna(0).max()) <= 0:
            vol = pd.Series(1.0, index=g.index)
        vol = vol.replace(0, np.nan).ffill()
        ret = px.pct_change()
        g["ret_1"] = ret
        g["ret_5"] = px.pct_change(5)
        g["ret_21"] = px.pct_change(21)
        g["vol_21"] = ret.rolling(21).std()
        g["mom_63"] = px.pct_change(63)
        dvol = (vol * px).clip(lower=1.0)
        g["log_dvol"] = np.log(dvol)
        g["amihud"] = (ret.abs() / dvol).rolling(21).mean()
        g["y"] = px.shift(-h) / px - 1.0
        g["symbol"] = sym
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    df["xs_mom"] = df.groupby("ts")["ret_21"].rank(pct=True)
    df["xs_vol"] = df.groupby("ts")["vol_21"].rank(pct=True)
    df = df.dropna(subset=FEATS + ["y", "ts"]).copy()
    if xs_center:
        df["y"] = df["y"] - df.groupby("ts")["y"].transform("mean")
    df["e"] = df["symbol"].astype("category").cat.codes.astype(np.int32)
    return df


def ic(pred, y, ts) -> float:
    out = []
    ts = np.asarray(ts)
    pred = np.asarray(pred)
    y = np.asarray(y)
    for t in np.unique(ts):
        m = ts == t
        if m.sum() < 3:
            continue
        c = np.corrcoef(pred[m], y[m])[0, 1]
        if np.isfinite(c):
            out.append(c)
    return float(np.mean(out)) if out else 0.0


def _models(n_estimators: int, max_depth: int, seed: int, horizon: int):
    base = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.08,
        horizon=horizon,
        verbose=0,
        random_state=seed,
        pair_splits=False,
        eval_metric="ic",
        max_bin=255,
    )
    out = [
        # Pairwise within-date ranking is the IC-optimal default on this panel stand.
        ("tide", "tide", dict(base, objective="pairwise")),
        ("tide-mse", "tide", dict(base, objective="mse")),
        (
            "tide-holdout",
            "tide",
            dict(
                base,
                objective="mse",
                causal_structure=True,
                structure_frac=0.85,
                max_bin=64,
                causal_bins=True,
            ),
        ),
    ]
    try:
        import lightgbm as lgb

        out.append(
            (
                "lightgbm",
                "sklearn",
                dict(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=0.08,
                    random_state=seed,
                    verbosity=-1,
                    force_col_wise=True,
                    max_bin=255,
                ),
                lgb.LGBMRegressor,
            )
        )
    except ImportError:
        print("lightgbm not installed (pip install lightgbm)")
    try:
        from catboost import CatBoostRegressor

        out.append(
            (
                "catboost",
                "sklearn",
                dict(
                    iterations=n_estimators,
                    depth=max_depth,
                    learning_rate=0.08,
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                ),
                CatBoostRegressor,
            )
        )
    except ImportError:
        print("catboost not installed (pip install catboost)")
    return out


def run(df: pd.DataFrame, n_splits: int, n_estimators: int, max_depth: int, seed: int = 0,
        horizon: int = 1, names: set[str] | None = None):
    wf = WalkForwardSplit(n_splits=n_splits, embargo=horizon, purge=horizon, min_train_fraction=0.2)
    rows = []
    factories = _models(n_estimators, max_depth, seed, horizon)
    if names is not None:
        factories = [f for f in factories if f[0] in names]
    for spec in factories:
        name, kind = spec[0], spec[1]
        ics, times = [], []
        for tr, te in wf.split(df, timestamp=df["ts"]):
            train, test = df.iloc[tr], df.iloc[te]
            t0 = time.perf_counter()
            if kind == "tide":
                m = TideRegressor(**spec[2])
                m.fit(train[FEATS], train["y"], timestamp=train["ts"], entity=train["symbol"])
            else:
                cls = spec[3]
                m = cls(**spec[2])
                m.fit(train[FEATS], train["y"])
            times.append(time.perf_counter() - t0)
            ics.append(ic(m.predict(test[FEATS]), test["y"].to_numpy(), test["ts"].to_numpy()))
        rows.append(
            {
                "model": name,
                "n_rows": int(len(df)),
                "n_names": int(df["symbol"].nunique()),
                "ic": float(np.mean(ics)) if ics else float("nan"),
                "ic_std": float(np.std(ics)) if ics else float("nan"),
                "fit_s": float(np.mean(times)) if times else float("nan"),
                "folds": len(ics),
                "fold_ic": ",".join(f"{x:.3f}" for x in ics),
            }
        )
    return pd.DataFrame(rows)


def load_or_download(symbols: list[str], refresh: bool) -> pd.DataFrame:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if DATA.exists() and not refresh:
        raw = pd.read_csv(DATA, parse_dates=["ts"])
        have = set(raw["symbol"].unique())
        if set(symbols).issubset(have):
            return raw[raw["symbol"].isin(symbols)].copy()
    raw = download(symbols)
    if DATA.exists() and not refresh:
        prev = pd.read_csv(DATA, parse_dates=["ts"])
        raw = pd.concat([prev, raw], ignore_index=True).drop_duplicates(["symbol", "ts"])
    DATA.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(DATA, index=False, compression="gzip")
    return raw[raw["symbol"].isin(symbols)].copy()


def main():
    p = argparse.ArgumentParser(description="Purged 30-name market stand vs LightGBM/CatBoost")
    p.add_argument("--quick", action="store_true", help="8 names, 2 folds, 40 trees")
    p.add_argument("--refresh", action="store_true", help="re-download Yahoo bars")
    p.add_argument("--n-splits", type=int, default=4)
    p.add_argument("--trees", type=int, default=80)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--horizon", type=int, default=None, help="single forward-return horizon")
    p.add_argument(
        "--horizons",
        type=str,
        default="1,3,5,10",
        help="comma-separated horizons (ignored if --horizon is set)",
    )
    p.add_argument("--no-xs-center", action="store_true", help="train on raw returns (not demeaned)")
    args = p.parse_args()
    symbols = QUICK if args.quick else UNIVERSE
    n_splits = 2 if args.quick else args.n_splits
    trees = 40 if args.quick else args.trees
    raw = load_or_download(symbols, refresh=args.refresh)
    horizons = [args.horizon] if args.horizon is not None else [int(x) for x in args.horizons.split(",") if x.strip()]
    xs_center = not args.no_xs_center
    frames = []
    for h in horizons:
        df = make_panel(raw, horizon=h, xs_center=xs_center)
        print(
            f"horizon={h}d xs_center={xs_center}  panel {df['symbol'].nunique()} names, "
            f"{df['ts'].nunique()} dates, {len(df)} rows, "
            f"{df['ts'].min().date()} → {df['ts'].max().date()}"
        )
        out = run(
            df,
            n_splits=n_splits,
            n_estimators=trees,
            max_depth=args.depth,
            horizon=h,
        )
        out.insert(0, "horizon", h)
        frames.append(out)
        print(out.drop(columns=["fold_ic"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("fold ICs:")
        for _, r in out.iterrows():
            print(f"  {r['model']}: {r['fold_ic']}")
        print()
    if len(frames) > 1:
        all_ = pd.concat(frames, ignore_index=True)
        wide = all_.pivot_table(index="model", columns="horizon", values="ic")
        print("mean IC by horizon:")
        print(wide.to_string(float_format=lambda x: f"{x:.4f}"))
        if "tide" in wide.index and "lightgbm" in wide.index:
            beat = int((wide.loc["tide"] > wide.loc["lightgbm"]).sum())
            print(f"tide > lightgbm on {beat}/{len(wide.columns)} horizons")
        if "tide" in wide.index:
            wins = int((wide.idxmax() == "tide").sum())
            print(f"tide is argmax on {wins}/{len(wide.columns)} horizons")
    print(
        "Public daily bars, causal lags, purged walk-forward. "
        "Not a proprietary desk, not SOTA. Protocol: docs/TIDEBENCH.md"
    )


if __name__ == "__main__":
    main()
