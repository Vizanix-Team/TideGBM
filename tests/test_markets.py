from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tidegbm import TideRegressor, cuda_compiled


def _load_markets():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "markets.py"
    spec = importlib.util.spec_from_file_location("tide_markets", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_cuda_compiled_matches_device_gpu():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = X[:, 0]
    if cuda_compiled:
        m = TideRegressor(n_estimators=5, iid_mode=True, verbose=0, device="gpu", pair_splits=False)
        m.fit(X, y)
        assert m.predict(X).shape == (40,)
        return
    m = TideRegressor(n_estimators=5, iid_mode=True, verbose=0, device="gpu")
    with pytest.raises(ValueError, match="GPU"):
        m.fit(X, y)


def test_market_features_are_causal():
    markets = _load_markets()
    dates = pd.date_range("2020-01-02", periods=90, freq="B")
    rng = np.random.default_rng(1)
    frames = []
    for sym in ["AAA", "BBB", "CCC", "DDD", "EEE"]:
        px = 80.0 * np.exp(np.cumsum(0.0005 + 0.015 * rng.normal(size=len(dates))))
        frames.append(pd.DataFrame({"ts": dates, "close": px, "volume": 1e6, "symbol": sym}))
    raw = pd.concat(frames, ignore_index=True)
    df = markets.make_panel(raw)
    assert set(markets.FEATS).issubset(df.columns)
    aaa = df[df["symbol"] == "AAA"].sort_values("ts").set_index("ts")
    px = raw[raw["symbol"] == "AAA"].sort_values("ts").set_index("ts")["close"]
    ret = px.pct_change()
    # Default panel demeans y within date; recover raw forward return for causality check.
    raw_panel = markets.make_panel(raw, xs_center=False)
    aaa_raw = raw_panel[raw_panel["symbol"] == "AAA"].sort_values("ts").set_index("ts")
    common = aaa_raw.index.intersection(ret.shift(-1).dropna().index)
    np.testing.assert_allclose(aaa_raw.loc[common, "y"], ret.shift(-1).loc[common], rtol=1e-10)
    np.testing.assert_allclose(aaa.loc[common, "ret_1"], ret.loc[common], rtol=1e-10)
    df5 = markets.make_panel(raw, horizon=5, xs_center=False)
    aaa5 = df5[df5["symbol"] == "AAA"].sort_values("ts").set_index("ts")
    y5 = px.shift(-5) / px - 1.0
    common5 = aaa5.index.intersection(y5.dropna().index)
    np.testing.assert_allclose(aaa5.loc[common5, "y"], y5.loc[common5], rtol=1e-10)


def test_market_stand_smoke():
    markets = _load_markets()
    dates = pd.date_range("2019-01-02", periods=260, freq="B")
    rng = np.random.default_rng(2)
    frames = []
    for i, sym in enumerate(["A", "B", "C", "D", "E", "F"]):
        shock = rng.normal(size=len(dates))
        px = 50.0 * np.exp(np.cumsum(0.0004 * (i + 1) + 0.012 * shock))
        frames.append(pd.DataFrame({"ts": dates, "close": px, "volume": 1e6 + i, "symbol": sym}))
    df = markets.make_panel(pd.concat(frames, ignore_index=True))
    out = markets.run(df, n_splits=2, n_estimators=15, max_depth=3, seed=0, names={"tide"})
    assert "tide" in set(out["model"])
    assert out.loc[out["model"] == "tide", "folds"].iloc[0] >= 1
    assert np.isfinite(out.loc[out["model"] == "tide", "ic"].iloc[0])
