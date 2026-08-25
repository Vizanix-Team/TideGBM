"""TideGBM: temporal ordered gradient boosting for panel and market data."""

from tidegbm._core import cuda_compiled as cuda_compiled
from tidegbm.api import TideClassifier, TideRanker, TideRegressor
from tidegbm.calibrate import calibrate_oof
from tidegbm.config import TideConfig
from tidegbm.cv import WalkForwardSplit, combinatorial_purged_cv
from tidegbm.lint import LintReport, lint

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
    "cuda_compiled",
    "__version__",
]
