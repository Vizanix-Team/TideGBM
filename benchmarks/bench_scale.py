"""Scale smoke: 100k rows × 32 features, CPU histogram path."""

from __future__ import annotations

import time

import numpy as np

from tidegbm import TideRegressor


def main(n=100_000, n_f=32, n_estimators=40):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, n_f))
    y = X[:, 0] - 0.3 * X[:, 1] + 0.1 * rng.normal(size=n)
    m = TideRegressor(
        n_estimators=n_estimators,
        max_depth=5,
        iid_mode=True,
        verbose=0,
        random_state=0,
        pair_splits=False,
        n_jobs=0,
    )
    t0 = time.perf_counter()
    m.fit(X, y)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = m.predict(X[:10_000])
    inf_s = time.perf_counter() - t0
    r2 = 1 - np.var(y[:10_000] - m.predict(X[:10_000])) / np.var(y[:10_000])
    print(f"n={n} f={n_f} trees={n_estimators}")
    print(f"fit {fit_s:.3f}s  infer 10k {inf_s*1e3:.2f}ms  r2 {r2:.3f}")
    print(f"infer per 1k rows {inf_s / 10 * 1e3:.3f}ms")
    return fit_s, inf_s, pred


if __name__ == "__main__":
    main()
