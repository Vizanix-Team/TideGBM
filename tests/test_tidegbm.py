from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from tidegbm import TideClassifier, TideRanker, TideRegressor, WalkForwardSplit, lint
from tidegbm.cv import combinatorial_purged_cv


def _panel(n_time=120, n_ent=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_time):
        btc = rng.normal()
        for e in range(n_ent):
            x0 = btc + 0.3 * rng.normal()
            x1 = rng.normal()
            y = 0.8 * x0 - 0.4 * x1 + 0.15 * rng.normal()
            rows.append({"ts": t, "symbol": e, "x0": x0, "x1": x1, "y": y})
    return pd.DataFrame(rows)


def test_requires_timestamp():
    df = _panel()
    model = TideRegressor(n_estimators=5, verbose=0)
    with pytest.raises(ValueError, match="timestamp"):
        model.fit(df[["x0", "x1"]], df["y"])


def test_iid_mode_allows_no_time():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] - 0.5 * X[:, 1]
    model = TideRegressor(n_estimators=40, iid_mode=True, verbose=0, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)
    r2 = 1 - np.var(y - pred) / np.var(y)
    assert r2 > 0.85


def test_learns_linear_panel():
    df = _panel(n_time=150, n_ent=6, seed=2)
    model = TideRegressor(
        n_estimators=60,
        learning_rate=0.1,
        max_depth=4,
        n_time_buckets=6,
        verbose=0,
        random_state=0,
        halo=False,
        pair_splits=False,
        horizon=1,
    )
    model.fit(df[["x0", "x1"]], df["y"], timestamp=df["ts"], entity=df["symbol"])
    pred = model.predict(df[["x0", "x1"]])
    r2 = 1 - np.var(df["y"].to_numpy() - pred) / np.var(df["y"].to_numpy())
    assert r2 > 0.7


def test_eval_in_the_past_raises():
    df = _panel(n_time=80, n_ent=4, seed=3)
    cut = 50
    train = df[df["ts"] < cut]
    valid = df[df["ts"] < 20]
    model = TideRegressor(n_estimators=5, verbose=0, early_stopping_rounds=2, horizon=1)
    with pytest.raises(ValueError, match="overlap"):
        model.fit(
            train[["x0", "x1"]],
            train["y"],
            timestamp=train["ts"],
            eval_set=(valid[["x0", "x1"]], valid["y"]),
            eval_timestamp=valid["ts"],
        )


def test_future_eval_ok():
    df = _panel(n_time=100, n_ent=4, seed=4)
    train = df[df["ts"] < 70]
    valid = df[df["ts"] >= 80]
    model = TideRegressor(
        n_estimators=20, verbose=0, early_stopping_rounds=5, random_state=0, halo=False, horizon=1
    )
    model.fit(
        train[["x0", "x1"]],
        train["y"],
        timestamp=train["ts"],
        eval_set=(valid[["x0", "x1"]], valid["y"]),
        eval_timestamp=valid["ts"],
        horizon=1,
    )
    assert len(model.evals_result_["valid"]) >= 1


def test_classifier():
    df = _panel(n_time=100, n_ent=5, seed=5)
    y = (df["y"] > 0).astype(int)
    model = TideClassifier(
        n_estimators=40, verbose=0, random_state=0, halo=False, pair_splits=False, horizon=1
    )
    model.fit(df[["x0", "x1"]], y, timestamp=df["ts"])
    acc = (model.predict(df[["x0", "x1"]]) == y.to_numpy()).mean()
    assert acc > 0.7


def test_ranker_orders_cross_section():
    rng = np.random.default_rng(6)
    rows = []
    for t in range(40):
        for e in range(10):
            strength = rng.normal()
            rows.append({"ts": t, "symbol": e, "x": strength, "y": strength + 0.05 * rng.normal()})
    df = pd.DataFrame(rows)
    model = TideRanker(n_estimators=30, verbose=0, random_state=0, halo=False, pair_splits=False, horizon=1)
    model.fit(df[["x"]], df["y"], timestamp=df["ts"], entity=df["symbol"])
    pred = model.predict(df[["x"]])
    ics = []
    for t in range(40):
        m = df["ts"] == t
        ics.append(np.corrcoef(pred[m], df.loc[m, "y"])[0, 1])
    assert np.nanmean(ics) > 0.4


def test_feature_groups_runs():
    df = _panel(n_time=60, n_ent=4, seed=7)
    df["x0_dup"] = df["x0"] + 1e-6
    model = TideRegressor(n_estimators=15, verbose=0, random_state=0, halo=False, horizon=1)
    model.fit(
        df[["x0", "x0_dup", "x1"]],
        df["y"],
        timestamp=df["ts"],
        feature_groups={"a": ["x0", "x0_dup"], "b": ["x1"]},
    )
    assert model.predict(df[["x0", "x0_dup", "x1"]]).shape[0] == len(df)


def test_lint_lookahead():
    n = 200
    y = np.random.default_rng(0).normal(size=n)
    x_future = np.empty(n)
    x_future[:-1] = y[1:]
    x_future[-1] = 0
    X = np.column_stack([x_future, np.random.default_rng(1).normal(size=n)])
    ts = np.arange(n)
    report = lint(X, y, ts, feature_names=["leak", "noise"])
    codes = {f.code for f in report.errors}
    assert "lookahead" in codes


def test_lint_label_copy():
    n = 100
    y = np.arange(n, dtype=float)
    X = np.column_stack([y, np.zeros(n)])
    report = lint(X, y, np.arange(n), feature_names=["copy", "z"])
    assert any(f.code == "label_in_features" for f in report.errors)


def test_walk_forward_is_causal():
    ts = np.repeat(np.arange(100), 3)
    wf = WalkForwardSplit(n_splits=4, embargo=2, purge=1, min_train_fraction=0.1)
    folds = list(wf.split(None, timestamp=ts))
    assert len(folds) >= 2
    for train, test in folds:
        assert ts[train].max() < ts[test].min()


def test_cpcv_yields_folds():
    ts = np.arange(60)
    folds = list(combinatorial_purged_cv(ts, n_splits=5, n_test_folds=2, embargo=1))
    assert len(folds) >= 3


def test_pair_split_sum_signal():
    rng = np.random.default_rng(8)
    n = 800
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    y = ((x0 + x1) > 0).astype(float) + 0.05 * rng.normal(size=n)
    ts = np.arange(n)
    X = np.column_stack([x0, x1, rng.normal(size=n)])
    model = TideRegressor(
        n_estimators=50,
        max_depth=3,
        pair_splits=True,
        pair_trigger=1.0,
        verbose=0,
        random_state=0,
        min_samples_leaf=10,
        halo=False,
        horizon=1,
    )
    model.fit(X, y, timestamp=ts)
    pred = model.predict(X)
    r2 = 1 - np.var(y - pred) / np.var(y)
    assert r2 > 0.3


def test_horizon_required():
    df = _panel(n_time=40, n_ent=3, seed=9)
    model = TideRegressor(n_estimators=3, verbose=0)
    with pytest.raises(ValueError, match="horizon"):
        model.fit(df[["x0", "x1"]], df["y"], timestamp=df["ts"])


def test_lookahead_refused_in_fit():
    n = 120
    y = np.random.default_rng(0).normal(size=n)
    x = np.empty(n)
    x[:-1] = y[1:]
    x[-1] = 0.0
    X = np.column_stack([x, np.random.default_rng(1).normal(size=n)])
    model = TideRegressor(n_estimators=5, verbose=0, horizon=1, lint_mode="strict")
    with pytest.raises(ValueError, match="lookahead"):
        model.fit(X, y, timestamp=np.arange(n))


def test_save_load_roundtrip(tmp_path):
    df = _panel(n_time=80, n_ent=4, seed=10)
    model = TideRegressor(n_estimators=25, verbose=0, random_state=1, horizon=1, pair_splits=False)
    model.fit(df[["x0", "x1"]], df["y"], timestamp=df["ts"])
    p1 = model.predict(df[["x0", "x1"]])
    path = tmp_path / "m.tgbm"
    model.save_model(path)
    loaded = TideRegressor.load_model(path)
    p2 = loaded.predict(df[["x0", "x1"]])
    np.testing.assert_allclose(p1, p2, rtol=1e-9, atol=1e-9)


def test_pickle_roundtrip():
    import pickle

    df = _panel(n_time=50, n_ent=3, seed=11)
    model = TideRegressor(n_estimators=15, verbose=0, random_state=2, horizon=1, pair_splits=False)
    model.fit(df[["x0", "x1"]], df["y"], timestamp=df["ts"])
    p1 = model.predict(df[["x0", "x1"]])
    model2 = pickle.loads(pickle.dumps(model))
    p2 = model2.predict(df[["x0", "x1"]])
    np.testing.assert_allclose(p1, p2, rtol=1e-9, atol=1e-9)


def test_shap_adds_up():
    df = _panel(n_time=60, n_ent=4, seed=12)
    model = TideRegressor(n_estimators=20, verbose=0, random_state=0, horizon=1, pair_splits=False)
    model.fit(df[["x0", "x1"]], df["y"], timestamp=df["ts"])
    contrib = model.shap_values(df[["x0", "x1"]])
    pred = model.predict(df[["x0", "x1"]])
    np.testing.assert_allclose(contrib.sum(axis=1), pred, rtol=1e-5, atol=1e-5)


def test_onnx_json_exports():
    df = _panel(n_time=40, n_ent=3, seed=13)
    model = TideRegressor(n_estimators=8, verbose=0, random_state=0, horizon=1, pair_splits=False)
    model.fit(df[["x0", "x1"]], df["y"], timestamp=df["ts"])
    js = model.to_onnx_json()
    assert "TreeEnsembleRegressor" in js
    assert "tidegbm" in js


def test_future_label_no_signal_in_x():
    """Property: if y is the future of independent noise and X is current noise, IC ~ 0."""
    rng = np.random.default_rng(21)
    n, e = 80, 6
    noise = rng.normal(size=(n, e))
    rows = []
    for t in range(n - 1):
        for i in range(e):
            rows.append({"ts": t, "e": i, "x": noise[t, i], "y": noise[t + 1, i]})
    df = pd.DataFrame(rows)
    model = TideRegressor(
        n_estimators=30, max_depth=3, horizon=1, verbose=0, random_state=0, pair_splits=False
    )
    cut = 50
    tr, te = df[df["ts"] < cut], df[df["ts"] >= cut + 1]
    model.fit(tr[["x"]], tr["y"], timestamp=tr["ts"], entity=tr["e"])
    pred = model.predict(te[["x"]])
    ic = np.corrcoef(pred, te["y"])[0, 1]
    assert abs(ic) < 0.35


def test_unsorted_rows_match_sorted():
    df = _panel(n_time=40, n_ent=4, seed=22)
    shuf = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    kw = dict(n_estimators=12, verbose=0, random_state=0, horizon=1, pair_splits=False, n_jobs=1)
    a = TideRegressor(**kw)
    b = TideRegressor(**kw)
    a.fit(df[["x0", "x1"]], df["y"], timestamp=df["ts"], entity=df["symbol"])
    b.fit(shuf[["x0", "x1"]], shuf["y"], timestamp=shuf["ts"], entity=shuf["symbol"])
    xa = df.sort_values(["ts", "symbol"])[["x0", "x1"]]
    np.testing.assert_allclose(a.predict(xa), b.predict(xa), rtol=1e-8, atol=1e-8)


def test_json_save_roundtrip(tmp_path):
    df = _panel(n_time=40, n_ent=3, seed=23)
    model = TideRegressor(n_estimators=10, verbose=0, random_state=0, horizon=1, pair_splits=False)
    model.fit(df[["x0", "x1"]], df["y"], timestamp=df["ts"])
    p1 = model.predict(df[["x0", "x1"]])
    path = tmp_path / "m.json"
    model.save_model(path)
    loaded = TideRegressor.load_model(path)
    np.testing.assert_allclose(p1, loaded.predict(df[["x0", "x1"]]), rtol=1e-9, atol=1e-9)


def test_missing_and_seed():
    rng = np.random.default_rng(24)
    X = rng.normal(size=(200, 3))
    X[::7, 0] = np.nan
    y = np.where(np.isnan(X[:, 0]), X[:, 1], X[:, 0]) - 0.2 * X[:, 2]
    a = TideRegressor(n_estimators=20, iid_mode=True, verbose=0, random_state=7, n_jobs=1)
    b = TideRegressor(n_estimators=20, iid_mode=True, verbose=0, random_state=7, n_jobs=1)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_allclose(a.predict(X), b.predict(X), rtol=1e-10, atol=1e-10)


def test_custom_objective_mse():
    rng = np.random.default_rng(25)
    X = rng.normal(size=(120, 2))
    y = X[:, 0] - X[:, 1]

    def obj(y, pred, w):
        g = w * (pred - y)
        h = w
        return g, h

    m = TideRegressor(
        n_estimators=15, iid_mode=True, verbose=0, random_state=0, custom_objective=obj
    )
    m.fit(X, y)
    r2 = 1 - np.var(y - m.predict(X)) / np.var(y)
    assert r2 > 0.7


def test_multiclass():
    rng = np.random.default_rng(26)
    X = rng.normal(size=(180, 3))
    y = np.argmax(np.column_stack([X[:, 0], X[:, 1] - 0.2, -X[:, 0]]), axis=1)
    m = TideClassifier(n_estimators=25, iid_mode=True, verbose=0, random_state=0, pair_splits=False)
    m.fit(X, y)
    acc = (m.predict(X) == y).mean()
    assert acc > 0.7
    assert m.predict_proba(X).shape == (180, 3)


def test_quantiles_multi_head():
    rng = np.random.default_rng(27)
    X = rng.normal(size=(150, 2))
    y = X[:, 0] + 0.5 * rng.normal(size=150)
    m = TideRegressor(
        n_estimators=20,
        iid_mode=True,
        verbose=0,
        random_state=0,
        quantiles=[0.1, 0.5, 0.9],
        pair_splits=False,
    )
    m.fit(X, y)
    q = m.predict_multi(X)
    assert q.shape == (150, 3)
    assert np.mean(q[:, 0] < q[:, 2]) > 0.8


def test_infer_latency_budget():
    rng = np.random.default_rng(28)
    X = rng.normal(size=(2000, 8))
    y = X[:, 0] - 0.3 * X[:, 1]
    m = TideRegressor(n_estimators=30, iid_mode=True, verbose=0, random_state=0, pair_splits=False)
    m.fit(X, y)
    import time

    t0 = time.perf_counter()
    m.predict(X[:1000])
    elapsed = time.perf_counter() - t0
    budget = 1.0 if os.environ.get("CI") else 0.25
    assert elapsed < budget


def test_split_importance():
    df = _panel(n_time=50, n_ent=4, seed=29)
    m = TideRegressor(n_estimators=15, verbose=0, random_state=0, horizon=1, pair_splits=False)
    m.fit(df[["x0", "x1"]], df["y"], timestamp=df["ts"])
    g = m.feature_importances("gain")
    s = m.feature_importances("split")
    assert g.shape == s.shape == (2,)
    assert np.isclose(g.sum(), 1.0)


def test_col_major_matches_row_major():
    rng = np.random.default_rng(30)
    X = rng.normal(size=(400, 6))
    y = X[:, 0] - 0.4 * X[:, 2] + 0.05 * rng.normal(size=400)
    kw = dict(
        n_estimators=20,
        iid_mode=True,
        verbose=0,
        random_state=0,
        pair_splits=False,
        n_jobs=1,
        goss=False,
        halo=False,
    )
    a = TideRegressor(**kw, col_major_hist=True)
    b = TideRegressor(**kw, col_major_hist=False)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_allclose(a.predict(X), b.predict(X), rtol=1e-10, atol=1e-10)


def test_l1_and_goss_run():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(180, 4))
    y = X[:, 0] + 0.2 * rng.normal(size=180)
    m = TideRegressor(
        n_estimators=15,
        iid_mode=True,
        verbose=0,
        random_state=0,
        pair_splits=False,
        l1=0.5,
        goss=True,
        goss_top=0.3,
        goss_other=0.2,
    )
    m.fit(X, y)
    assert m.predict(X).shape == (180,)
    r2 = 1 - np.var(y - m.predict(X)) / np.var(y)
    assert r2 > 0.5


def test_dead_python_trainer_gone():
    import importlib

    for name in ("tidegbm.booster", "tidegbm.tree", "tidegbm.histogram", "tidegbm.binning"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_update_adds_trees():
    rng = np.random.default_rng(32)
    X = rng.normal(size=(150, 3))
    y = X[:, 0] - X[:, 1]
    m = TideRegressor(n_estimators=8, iid_mode=True, verbose=0, random_state=0, pair_splits=False)
    m.fit(X, y)
    p1 = m.predict(X)
    m.update(X, y, n_estimators=6)
    p2 = m.predict(X)
    assert not np.allclose(p1, p2)
    r2 = 1 - np.var(y - p2) / np.var(y)
    assert r2 > 0.7


def test_sklearn_clone_params():
    from sklearn.base import clone

    m = TideRegressor(iid_mode=True, n_estimators=7, l1=0.1, goss=False)
    c = clone(m)
    assert c.get_params()["n_estimators"] == 7
    assert c.get_params()["l1"] == 0.1
    assert c.get_params()["goss"] is False
    rng = np.random.default_rng(34)
    X = rng.normal(size=(80, 2))
    y = X[:, 0]
    c.fit(X, y)
    assert c.predict(X).shape == (80,)


def test_eval_metric_ic():
    df = _panel(n_time=90, n_ent=6, seed=40)
    train = df[df["ts"] < 60]
    valid = df[df["ts"] >= 70]
    model = TideRegressor(
        n_estimators=25,
        verbose=0,
        random_state=0,
        horizon=1,
        pair_splits=False,
        early_stopping_rounds=8,
        eval_metric="ic",
    )
    model.fit(
        train[["x0", "x1"]],
        train["y"],
        timestamp=train["ts"],
        eval_set=(valid[["x0", "x1"]], valid["y"]),
        eval_timestamp=valid["ts"],
    )
    v = np.asarray(model.evals_result_["valid"])
    assert len(v) >= 1
    assert np.all(np.abs(v) <= 1.0 + 1e-9)
    assert model.best_iteration_ is not None


def test_entity_subsample_runs():
    df = _panel(n_time=50, n_ent=8, seed=41)
    m = TideRegressor(
        n_estimators=12,
        verbose=0,
        random_state=0,
        horizon=1,
        pair_splits=False,
        entity_subsample=0.5,
    )
    m.fit(df[["x0", "x1"]], df["y"], timestamp=df["ts"], entity=df["symbol"])
    assert m.predict(df[["x0", "x1"]]).shape[0] == len(df)


def test_onnx_protobuf(tmp_path):
    onnx = pytest.importorskip("onnx")
    rng = np.random.default_rng(33)
    X = rng.normal(size=(80, 2))
    y = X[:, 0] - 0.2 * X[:, 1]
    m = TideRegressor(n_estimators=6, iid_mode=True, verbose=0, random_state=0, pair_splits=False)
    m.fit(X, y)
    path = tmp_path / "m.onnx"
    m.to_onnx(path)
    model = onnx.load(str(path))
    assert model.graph.node[0].op_type == "TreeEnsembleRegressor"
    assert path.stat().st_size > 32


def test_uniqueness_no_overlap_is_one():
    from tidegbm import average_uniqueness

    ts = np.arange(20)
    assert np.allclose(average_uniqueness(ts, 1), 1.0)


def test_uniqueness_matches_concurrency():
    from tidegbm import average_uniqueness

    # A label spanning h steps on a dense daily grid overlaps h-1 neighbours, so in the
    # steady state each observation is worth 1/h of an independent sample.
    ts = np.arange(60)
    u = average_uniqueness(ts, 5)
    assert np.allclose(u[10:], 0.2, atol=1e-9)
    assert u[0] > u[1] > u[2]  # the first labels overlap fewer predecessors


def test_uniqueness_is_per_entity():
    from tidegbm import average_uniqueness

    ts_single = np.arange(30)
    solo = average_uniqueness(ts_single, 5)
    ts_panel = np.repeat(np.arange(30), 3)
    ent = np.tile([0, 1, 2], 30)
    panel = average_uniqueness(ts_panel, 5, ent)
    # Entities must not inflate each other's concurrency.
    for e in (0, 1, 2):
        assert np.allclose(panel[ent == e], solo)


def test_uniqueness_weight_changes_fit():
    df = _panel(n_time=150, n_ent=6)
    X, y = df[["x0", "x1"]], df["y"]
    kw = dict(n_estimators=25, horizon=5, random_state=0, verbose=0, lint_mode="off")
    base = TideRegressor(**kw).fit(X, y, timestamp=df["ts"], entity=df["symbol"])
    weighted = TideRegressor(uniqueness_weight=True, **kw).fit(
        X, y, timestamp=df["ts"], entity=df["symbol"]
    )
    assert not np.allclose(base.predict(X), weighted.predict(X))


def test_rank_loss_is_not_squared_error():
    df = _panel(n_time=120, n_ent=10)
    model = TideRanker(n_estimators=20, horizon=1, random_state=0, verbose=0, lint_mode="off")
    model.fit(df[["x0", "x1"]], df["y"], timestamp=df["ts"], entity=df["symbol"])
    curve = model.evals_result_["train"]
    # Ranking reports negative mean within-date correlation, so a fitting model goes
    # negative -- a squared-error fallback could never do that.
    assert min(curve) < 0.0
