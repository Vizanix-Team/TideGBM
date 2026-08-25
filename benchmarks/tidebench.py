"""Synthetic TideBench: purged walk-forward protocol, not a public SOTA claim."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from tidegbm import TideRegressor, WalkForwardSplit


def make_panel(kind: str, n_time=180, n_ent=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_time):
        f = rng.normal()
        regime = int(t > n_time // 2)
        for e in range(n_ent):
            mom = rng.normal()
            vol = abs(rng.normal()) + 0.05
            if kind == "linear":
                y = 0.4 * mom - 0.25 * vol + 0.3 * f + 0.35 * rng.normal()
            elif kind == "noise":
                y = rng.normal()
            else:
                y = 0.25 * mom * (1 - regime) + 0.4 * f + 0.4 * rng.normal()
            rows.append({"ts": t, "e": e, "mom": mom, "vol": vol, "f": f, "y": y, "kind": kind})
    return pd.DataFrame(rows)


def ic(pred, y, ts):
    out = []
    for t in np.unique(ts):
        m = ts == t
        if m.sum() < 3:
            continue
        c = np.corrcoef(pred[m], y[m])[0, 1]
        if np.isfinite(c):
            out.append(c)
    return float(np.mean(out)) if out else 0.0


def run_panel(df, label, ablate=None):
    feats = ["mom", "vol", "f"]
    wf = WalkForwardSplit(n_splits=3, embargo=1, min_train_fraction=0.25)
    kw = dict(n_estimators=40, max_depth=3, horizon=1, verbose=0, random_state=0, pair_splits=False)
    if ablate == "no_torb":
        kw["iid_mode"] = True
        kw.pop("horizon", None)
    rows = []
    factories = [
        ("tide", lambda: TideRegressor(**kw)),
        ("histgbdt", lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=40, random_state=0)),
        ("ridge", lambda: Ridge(alpha=1.0)),
    ]
    try:
        import lightgbm as lgb

        factories.append(
            (
                "lightgbm",
                lambda: lgb.LGBMRegressor(n_estimators=40, max_depth=3, random_state=0, verbosity=-1),
            )
        )
    except ImportError:
        pass
    for name, factory in factories:
        ics, times = [], []
        for tr, te in wf.split(df, timestamp=df["ts"]):
            train, test = df.iloc[tr], df.iloc[te]
            m = factory()
            t0 = time.perf_counter()
            if name == "tide":
                if ablate == "no_torb":
                    m.fit(train[feats], train["y"])
                else:
                    m.fit(train[feats], train["y"], timestamp=train["ts"], entity=train["e"])
            else:
                m.fit(train[feats], train["y"])
            times.append(time.perf_counter() - t0)
            ics.append(ic(m.predict(test[feats]), test["y"].to_numpy(), test["ts"].to_numpy()))
        rows.append(
            {
                "panel": label,
                "model": name,
                "ablate": ablate or "full",
                "ic": float(np.mean(ics)),
                "fit_s": float(np.mean(times)),
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    kinds = ["linear", "regime"] if args.quick else ["linear", "regime", "noise"]
    all_rows = []
    for k in kinds:
        df = make_panel(k, n_time=120 if args.quick else 180, n_ent=8 if args.quick else 12)
        all_rows.extend(run_panel(df, k))
        if k == "linear":
            all_rows.extend(run_panel(df, k, ablate="no_torb"))
    out = pd.DataFrame(all_rows)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("This is a synthetic protocol. Do not call it SOTA.")


if __name__ == "__main__":
    main()
