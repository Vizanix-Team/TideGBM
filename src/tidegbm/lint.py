from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tidegbm.utils import as_1d, parse_timestamps


@dataclass
class LintFinding:
    level: str
    code: str
    message: str
    detail: dict = field(default_factory=dict)


@dataclass
class LintReport:
    findings: list[LintFinding] = field(default_factory=list)

    @property
    def errors(self) -> list[LintFinding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[LintFinding]:
        return [f for f in self.findings if f.level == "warning"]

    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            msgs = "; ".join(e.message for e in self.errors)
            raise ValueError(f"TideGBM leakage lint failed: {msgs}")

    def __str__(self) -> str:
        if not self.findings:
            return "TideGBM lint: no issues"
        lines = [f"[{f.level}] {f.code}: {f.message}" for f in self.findings]
        return "\n".join(lines)


def lint(
    X,
    y,
    timestamp,
    *,
    feature_names: list[str] | None = None,
    corr_threshold: float = 0.95,
    lookahead_threshold: float = 0.85,
    raise_on_error: bool = False,
) -> LintReport:
    """Catch the failure modes that make market GBMs look brilliant in-sample.

    Checks
    ------
    * timestamp missing / unsorted / duplicated chaos
    * a feature that *is* the timestamp
    * a feature nearly equal to y (label leaked into X)
    * lookahead: feature[t] aligns with y[t+1] much better than with y[t]
    """
    report = LintReport()
    y = as_1d(y).astype(np.float64)
    try:
        ts = parse_timestamps(timestamp)
    except Exception as exc:
        report.findings.append(
            LintFinding("error", "bad_timestamp", f"cannot parse timestamp: {exc}")
        )
        if raise_on_error:
            report.raise_if_errors()
        return report

    if ts.shape[0] != y.shape[0]:
        report.findings.append(
            LintFinding("error", "length", "timestamp length does not match y")
        )

    if isinstance(X, pd.DataFrame):
        names = list(X.columns) if feature_names is None else list(feature_names)
        # drop timestamp column from features if present
        mats = []
        use_names = []
        for c in names:
            if isinstance(timestamp, str) and c == timestamp:
                continue
            mats.append(pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=np.float64))
            use_names.append(str(c))
        Xn = np.column_stack(mats) if mats else np.zeros((y.shape[0], 0))
        names = use_names
    else:
        Xn = np.asarray(X, dtype=np.float64)
        names = feature_names or [f"f{i}" for i in range(Xn.shape[1])]

    if Xn.shape[0] != y.shape[0]:
        report.findings.append(LintFinding("error", "length", "X and y length mismatch"))
        if raise_on_error:
            report.raise_if_errors()
        return report

    if not np.all(ts[:-1] <= ts[1:]):
        report.findings.append(
            LintFinding(
                "warning",
                "unsorted",
                "timestamps are not sorted; TideGBM sorts internally, "
                "but unsorted input often means the pipeline already shuffled rows",
            )
        )

    # timestamp-as-feature
    ts_f = ts.astype(np.float64)
    ts_std = ts_f.std()
    if ts_std > 0:
        zts = (ts_f - ts_f.mean()) / ts_std
        for j, name in enumerate(names):
            col = Xn[:, j]
            m = np.isfinite(col)
            if m.sum() < 10:
                continue
            if np.std(col[m]) == 0:
                continue
            zc = (col[m] - col[m].mean()) / (col[m].std() + 1e-12)
            zt = zts[m]
            corr = float(np.corrcoef(zc, zt)[0, 1])
            if abs(corr) > 0.999:
                report.findings.append(
                    LintFinding(
                        "error",
                        "timestamp_feature",
                        f"{name!r} is (almost) the timestamp; the model will memorize the calendar",
                        {"feature": name, "corr": corr},
                    )
                )

    # label in X
    for j, name in enumerate(names):
        col = Xn[:, j]
        m = np.isfinite(col) & np.isfinite(y)
        if m.sum() < 10 or np.std(col[m]) == 0 or np.std(y[m]) == 0:
            continue
        corr = float(np.corrcoef(col[m], y[m])[0, 1])
        if abs(corr) >= corr_threshold:
            report.findings.append(
                LintFinding(
                    "error" if abs(corr) > 0.999 else "warning",
                    "label_in_features",
                    f"{name!r} correlates {corr:.3f} with y — possible label leakage",
                    {"feature": name, "corr": corr},
                )
            )

    # lookahead: corr(x[t], y[t+1]) vs corr(x[t], y[t])
    order = np.argsort(ts, kind="mergesort")
    ys = y[order]
    Xs = Xn[order]
    # only consecutive unique steps of the same delta (single entity or aligned panel)
    for j, name in enumerate(names):
        col = Xs[:, j]
        m = np.isfinite(col) & np.isfinite(ys)
        if m.sum() < 30:
            continue
        c0 = _safe_corr(col[m], ys[m])
        c1 = _safe_corr(col[m][:-1], ys[m][1:])
        cm1 = _safe_corr(col[m][1:], ys[m][:-1])
        if c1 is None or c0 is None:
            continue
        # feature looks like future y: x[t] ≈ y[t+1]
        if abs(c1) >= lookahead_threshold and abs(c1) > abs(c0) + 0.15:
            report.findings.append(
                LintFinding(
                    "error",
                    "lookahead",
                    f"{name!r} aligns with y shifted forward (corr={c1:.3f}) more than "
                    f"contemporaneous y (corr={c0:.3f}). This is classic lookahead bias.",
                    {"feature": name, "corr_y": c0, "corr_y_next": c1, "corr_y_prev": cm1},
                )
            )

    uniq_frac = np.unique(ts).size / max(ts.size, 1)
    if uniq_frac < 0.02:
        report.findings.append(
            LintFinding(
                "warning",
                "panel_heavy",
                "very few unique timestamps relative to rows; pass entity= so ranking "
                "and embargo treat this as a panel",
            )
        )

    if raise_on_error:
        report.raise_if_errors()
    return report


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 10 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])
