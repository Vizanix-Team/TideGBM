# Leakage anti-examples

These are the bugs TideGBM is built to refuse. Copying them into `fit` is how a GBM
looks brilliant in a notebook and dies in production.

## Lookahead feature

```python
# BAD: feature at t is tomorrow's return
X["future_ret"] = y.shift(-1)
model.fit(X, y, timestamp=ts, horizon=1)  # lint_mode="strict" raises
```

Fix: only features known at decision time. `lint` / `lint_mode="strict"` (default).

## Overlapping labels without horizon

```python
# BAD: 5-bar return, no horizon → TORB still sees future overlap
model = TideRegressor()  # missing horizon
model.fit(X, y_5bar, timestamp=ts)  # raises
```

Fix: `horizon=5` in the same units as `timestamp`.

## Eval in the past

```python
model.fit(X_train, y_train, timestamp=ts_train, eval_set=(X_old, y_old), eval_timestamp=ts_old)
# raises: eval timestamps overlap the training future
```

Fix: validation strictly after train minus embargo.

## Refitting bins on live data

Bin edges are frozen in the model artifact. Do not recompute quantiles on the
scoring window. `predict` uses the fit-time binner.

## Factorizing categoricals on predict

Category vocab is stored at fit. Unseen levels go to the missing bin, they are
not remapped.

## Sorting / shuffling

Rows are sorted `(timestamp, entity)` inside `fit`. Do not shuffle a panel and
expect TORB to treat it as i.i.d. unless `iid_mode=True`.

## Pickle from the internet

`pickle.loads` of an untrusted blob can execute code. Prefer `save_model` /
`load_model` (binary `.tgbm` or JSON with an embedded artifact). JSON without
the artifact is inspectable but not executable.

## Regime from y

Forbidden. Pass a user column (`regime="session"`), never a label-derived state.
