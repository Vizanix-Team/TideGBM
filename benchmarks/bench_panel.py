"""Synthetic panel benchmark: TideGBM vs sklearn HistGradientBoosting."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from tidegbm import TideRegressor, WalkForwardSplit


def make_panel(n_time=400, n_ent=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_time):
        f = rng.normal()
        for e in range(n_ent):
            mom = rng.normal()
            vol = abs(rng.normal()) + 0.05
            y = 0.35 * mom - 0.2 * vol + 0.4 * f + 0.4 * rng.normal()
            rows.append({"ts": t, "e": e, "mom": mom, "vol": vol, "f": f, "y": y})
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


def main():
    df = make_panel()
    feats = ["mom", "vol", "f"]
    wf = WalkForwardSplit(n_splits=4, embargo=1, min_train_fraction=0.2)
    tide_ics, sk_ics = [], []
    tide_t, sk_t = [], []
    for tr, te in wf.split(df, timestamp=df["ts"]):
        train, test = df.iloc[tr], df.iloc[te]
        m = TideRegressor(
            n_estimators=80,
            max_depth=4,
            horizon=1,
            verbose=0,
            random_state=0,
            pair_splits=False,
        )
        t0 = time.perf_counter()
        m.fit(train[feats], train["y"], timestamp=train["ts"], entity=train["e"])
        tide_t.append(time.perf_counter() - t0)
        p = m.predict(test[feats])
        tide_ics.append(ic(p, test["y"].to_numpy(), test["ts"].to_numpy()))

        h = HistGradientBoostingRegressor(max_depth=4, max_iter=80, random_state=0)
        t0 = time.perf_counter()
        h.fit(train[feats], train["y"])
        sk_t.append(time.perf_counter() - t0)
        p2 = h.predict(test[feats])
        sk_ics.append(ic(p2, test["y"].to_numpy(), test["ts"].to_numpy()))

    print("TideGBM  mean IC", round(float(np.mean(tide_ics)), 4), "mean fit s", round(float(np.mean(tide_t)), 3))
    print("HistGBDT mean IC", round(float(np.mean(sk_ics)), 4), "mean fit s", round(float(np.mean(sk_t)), 3))
    print("folds Tide", [round(x, 3) for x in tide_ics])
    print("folds Hist", [round(x, 3) for x in sk_ics])


if __name__ == "__main__":
    main()
