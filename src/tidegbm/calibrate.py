"""Purged out-of-fold isotonic calibration. Fit on OOF scores, never on in-sample train."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from tidegbm.cv import WalkForwardSplit


def calibrate_oof(estimator, X, y, timestamp, *, n_splits: int = 4, embargo: int = 1):
    """Return isotonic map fitted on purged walk-forward OOF predictions.

    The calibrator is fit on out-of-fold scores only. Applying it on the training
    window without OOF would leak.
    """
    wf = WalkForwardSplit(n_splits=n_splits, embargo=embargo, min_train_fraction=0.15)
    oof_p, oof_y = [], []
    ts = np.asarray(timestamp)
    y = np.asarray(y, dtype=np.float64)
    for tr, te in wf.split(X, timestamp=ts):
        est = estimator.__class__(**estimator.get_params())
        Xt, Xe = (X.iloc[tr], X.iloc[te]) if hasattr(X, "iloc") else (X[tr], X[te])
        est.fit(Xt, y[tr], timestamp=ts[tr])
        pred = est.predict(Xe)
        oof_p.append(np.asarray(pred).ravel())
        oof_y.append(y[te])
    p = np.concatenate(oof_p)
    t = np.concatenate(oof_y)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, t)
    return iso
