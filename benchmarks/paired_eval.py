"""Paired A/B evaluation with error bars.

A single walk-forward split on a 30-name stand has no power to separate two configurations:
changing only the number of folds can reverse the ranking. This harness therefore

* evaluates over **several fold counts**, pooling every (fold count, fold) cell,
* compares configurations **on the same cells**, so the fold-to-fold noise cancels,
* reports the paired difference with a standard error instead of two bare means.

Read the result as: a difference under ~2 SE is not evidence of anything on this stand.

Usage
-----
    python benchmarks/paired_eval.py --a "{}" --b '{"uniqueness_weight": true}'
    python benchmarks/paired_eval.py --a '{"pair_splits": true}' --b '{"pair_splits": false}'
    python benchmarks/paired_eval.py --horizons 1,5 --objectives pairwise --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from markets import FEATS, make_panel  # noqa: E402

from tidegbm import TideRegressor, WalkForwardSplit  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "markets.csv.gz"


def rank_ic(y: np.ndarray, pred: np.ndarray, ts: np.ndarray) -> float:
    """Mean within-date Spearman correlation."""
    d = pd.DataFrame({"y": y, "p": pred, "ts": ts})
    per_date = d.groupby("ts").apply(
        lambda g: g["y"].corr(g["p"], method="spearman") if len(g) >= 3 else np.nan
    )
    return float(np.nanmean(per_date))


def cells(X, y, ts, ent, horizon, objective, params, fold_counts, seeds):
    """One rank-IC per (fold count, fold), averaging predictions over seeds."""
    out = []
    for n_splits in fold_counts:
        splitter = WalkForwardSplit(n_splits=n_splits, embargo=horizon, purge=horizon)
        for tr, te in splitter.split(X, y, timestamp=ts):
            preds = []
            for seed in range(seeds):
                model = TideRegressor(
                    horizon=horizon,
                    objective=objective,
                    lint_mode="off",
                    random_state=seed,
                    **params,
                )
                model.fit(X[tr], y[tr], timestamp=ts[tr], entity=ent[tr])
                preds.append(model.predict(X[te]))
            out.append(rank_ic(y[te], np.mean(preds, axis=0), ts[te]))
    return np.asarray(out, dtype=np.float64)


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired difference b - a over shared cells."""
    d = b - a
    se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
    return {
        "a": float(a.mean()),
        "b": float(b.mean()),
        "delta": float(d.mean()),
        "se": se,
        "t": float(d.mean() / se) if se > 0 else 0.0,
        "n": int(d.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="{}", help="JSON of estimator params for arm A (baseline)")
    ap.add_argument("--b", default="{}", help="JSON of estimator params for arm B")
    ap.add_argument("--horizons", default="1,5,10")
    ap.add_argument("--objectives", default="mse,pairwise")
    ap.add_argument("--folds", default="3,4,5,6,8", help="fold counts to pool over")
    ap.add_argument("--seeds", type=int, default=1, help="average predictions over N seeds")
    ap.add_argument("--n-estimators", type=int, default=80)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.08)
    ap.add_argument("--n-jobs", type=int, default=0)
    args = ap.parse_args()

    common = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_jobs=args.n_jobs,
    )
    arm_a = {**common, **json.loads(args.a)}
    arm_b = {**common, **json.loads(args.b)}
    fold_counts = [int(v) for v in args.folds.split(",")]
    horizons = [int(v) for v in args.horizons.split(",")]
    objectives = args.objectives.split(",")

    if not DATA.exists():
        print(f"missing {DATA} — run `python benchmarks/markets.py --refresh` first")
        return 1
    raw = pd.read_csv(DATA, parse_dates=["ts"])

    print(f"A = {json.loads(args.a) or 'defaults'}")
    print(f"B = {json.loads(args.b) or 'defaults'}")
    print(f"folds pooled: {fold_counts}   seeds: {args.seeds}\n")
    print(f"{'horizon':>7} {'objective':>9} {'A':>9} {'B':>9} {'B-A':>9} {'SE':>8} {'t':>6} {'n':>4}")
    print("-" * 68)

    rows = []
    for h in horizons:
        df = make_panel(raw, horizon=h)
        ts = df["ts"].values.astype("datetime64[D]").astype(np.int64)
        ent = pd.factorize(df["symbol"])[0].astype(np.int64)
        y = df["y"].values.astype(np.float64)
        X = df[FEATS].values.astype(np.float64)
        for obj in objectives:
            t0 = time.perf_counter()
            ca = cells(X, y, ts, ent, h, obj, arm_a, fold_counts, args.seeds)
            cb = cells(X, y, ts, ent, h, obj, arm_b, fold_counts, args.seeds)
            r = paired(ca, cb)
            rows.append({"horizon": h, "objective": obj, **r})
            print(f"{h:>7} {obj:>9} {r['a']:>+9.5f} {r['b']:>+9.5f} "
                  f"{r['delta']:>+9.5f} {r['se']:>8.5f} {r['t']:>+6.2f} {r['n']:>4}"
                  f"   ({time.perf_counter() - t0:.0f}s)", flush=True)

    pooled = np.array([r["t"] for r in rows])
    print("-" * 68)
    print(f"cells with |t| >= 2: {int((np.abs(pooled) >= 2).sum())} of {len(rows)}")
    if not (np.abs(pooled) >= 2).any():
        print("No cell separates the two arms on this stand. Treat B as unproven, not better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
