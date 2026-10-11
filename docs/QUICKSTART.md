# Quick start

Install from PyPI (Linux/macOS; requires a C++17 compiler with OpenMP):

```bash
pip install tidegbm
```

Development install:

```bash
git clone https://github.com/Vizanix-Team/TideGBM.git && cd TideGBM
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Minimal panel example

```python
import pandas as pd
from tidegbm import TideRegressor, WalkForwardSplit, lint

df = pd.read_csv("your_panel.csv")  # columns: ts, symbol, features..., y

report = lint(df[["feat_a", "feat_b"]], df["y"], df["ts"])
report.raise_if_errors()  # optional but recommended

model = TideRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.08,
    horizon=1,              # label horizon in timestamp units
    objective="pairwise",   # cross-sectional ranking / IC
)

for train_idx, test_idx in WalkForwardSplit(n_splits=4, embargo=1).split(df, timestamp=df["ts"]):
    tr, te = df.iloc[train_idx], df.iloc[test_idx]
    model.fit(
        tr[["feat_a", "feat_b"]],
        tr["y"],
        timestamp=tr["ts"],
        entity=tr["symbol"],
    )
    pred = model.predict(te[["feat_a", "feat_b"]])
```

## When to use which objective

| Goal | `objective` |
|---|---|
| Cross-sectional IC / ranking within date | `"pairwise"` |
| Level fit, R², absolute error | `"mse"` or `"auto"` |
| Binary classification | `"logloss"` (classifier auto) |
| Learning-to-rank groups | `"rank"` (`TideRanker`) |

## Early stopping

Pass a **future** validation slice only:

```python
model.fit(
    X_train, y_train,
    timestamp=ts_train,
    entity=ent_train,
    eval_set=(X_val, y_val),
    eval_timestamp=ts_val,
    early_stopping_rounds=20,
)
print(model.best_iteration_, model.evals_result_)
```

## Save / load

```python
model.save_model("model.tgbm")          # binary TGBM v3
model.save_model("model.json")          # JSON + embedded artifact
loaded = TideRegressor.load_model("model.tgbm")
```

Full API reference: [`docs/API.md`](API.md).
