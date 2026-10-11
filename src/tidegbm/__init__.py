"""TideGBM: temporal ordered gradient boosting for panel and market data.

Public API
----------
TideRegressor, TideClassifier, TideRanker
    scikit-learn estimators with TORB time/panel inductive bias.

WalkForwardSplit, combinatorial_purged_cv
    Purged walk-forward and combinatorial CV for time series.

lint, LintReport
    Leakage checks before training.

TideConfig
    Hyperparameter dataclass with validation.

calibrate_oof
    Purged out-of-fold isotonic calibration.

cuda_compiled
    Whether this wheel was built with CUDA histograms.

Documentation: https://github.com/Vizanix-Team/TideGBM/blob/main/docs/API.md
"""

from tidegbm._core import cuda_compiled as cuda_compiled
from tidegbm.api import TideClassifier, TideRanker, TideRegressor
from tidegbm.calibrate import calibrate_oof
from tidegbm.config import TideConfig
from tidegbm.cv import WalkForwardSplit, combinatorial_purged_cv
from tidegbm.lint import LintReport, lint
from tidegbm.uniqueness import average_uniqueness, uniqueness_weights

__version__ = "0.4.0"
__all__ = [
    "TideRegressor",
    "TideClassifier",
    "TideRanker",
    "TideConfig",
    "WalkForwardSplit",
    "combinatorial_purged_cv",
    "lint",
    "LintReport",
    "calibrate_oof",
    "average_uniqueness",
    "uniqueness_weights",
    "cuda_compiled",
    "__version__",
]
