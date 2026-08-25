"""Fit a panel regressor the way a quant would: you bring the features."""

import numpy as np
import pandas as pd

from tidegbm import TideRegressor, WalkForwardSplit, lint

rng = np.random.default_rng(42)
n_time, n_ent = 250, 12
rows = []
for t in range(n_time):
    factor = 0.2 * rng.normal()
    for e in range(n_ent):
        mom = rng.normal()
        vol = abs(rng.normal()) + 0.1
        y = 0.4 * mom - 0.2 * vol + factor + 0.25 * rng.normal()
        rows.append(
            {
                "ts": t,
                "symbol": e,
                "momentum": mom,
                "volatility": vol,
                "btc_lead": factor,
                "y": y,
            }
        )
df = pd.DataFrame(rows)

report = lint(df[["momentum", "volatility", "btc_lead"]], df["y"], df["ts"])
print(report)

features = ["momentum", "volatility", "btc_lead"]
wf = WalkForwardSplit(n_splits=4, embargo=1, min_train_fraction=0.2)
ics = []
for train_idx, test_idx in wf.split(df, timestamp=df["ts"]):
    model = TideRegressor(
        n_estimators=40,
        learning_rate=0.1,
        max_depth=4,
        horizon=1,
        objective="pairwise",  # within-date ranking; use "mse" for level fit
        verbose=0,
        random_state=0,
    )
    tr, te = df.iloc[train_idx], df.iloc[test_idx]
    model.fit(
        tr[features],
        tr["y"],
        timestamp=tr["ts"],
        entity=tr["symbol"],
        feature_groups={"risk": ["volatility"], "alpha": ["momentum", "btc_lead"]},
    )
    pred = model.predict(te[features])
    ics.append(float(np.corrcoef(pred, te["y"])[0, 1]))

print("walk-forward IC per fold:", [round(x, 3) for x in ics])
print("mean IC:", round(float(np.nanmean(ics)), 3))
