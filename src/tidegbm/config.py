from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class TideConfig:
    """Hyperparameters. Training itself is the C++ core; this is the documented surface."""

    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 6
    max_leaves: int = 31
    min_samples_leaf: int = 20
    min_child_weight: float = 1e-3
    l2: float = 1.0
    l1: float = 0.0
    min_split_gain: float = 0.0
    max_bin: int = 255
    subsample: float = 1.0
    colsample: float = 1.0
    n_time_buckets: int = 0
    horizon: Any = None
    embargo: Any = None
    iid_mode: bool = False
    halo: bool = False
    halo_gamma: float = 0.4
    halo_candidates: int = 3
    pair_splits: bool = True
    pair_trigger: float = 0.25
    pair_top_k: int = 6
    max_pair_splits_per_tree: int = 3
    pair_l2: float = 2.0
    time_decay: float = 1.0
    tail_buffer: float = 0.02
    tree_decay: float = 1.0
    objective: str = "auto"
    quantile: float = 0.5
    huber_delta: float = 1.0
    rank_temperature: float = 1.0
    n_jobs: int = 0
    random_state: int | None = 0
    verbose: int = 0
    early_stopping_rounds: int | None = None
    causal_structure: bool = False
    regime_priority: bool = True
    cat_min_count: int = 5
    monotone_constraints: Mapping[str, int] | Sequence[int] | None = None
    goss: bool | None = None
    structure_frac: float = 1.0
    causal_bins: bool = False
    torb_min_cover: float = 0.0
    cat_smooth: float = 1.0
    col_major_hist: bool = True
    colsample_bynode: float = 1.0
    bin_sample: int = 262144
    entity_subsample: float = 1.0
    eval_metric: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if not 0 < self.learning_rate <= 1.5:
            raise ValueError("learning_rate must be in (0, 1.5]")
        if self.max_bin < 8:
            raise ValueError("max_bin must be >= 8")
        if not self.iid_mode and self.n_time_buckets == 1:
            raise ValueError("n_time_buckets must be 0 (auto) or >= 2")
        if not self.iid_mode and self.n_time_buckets < 0:
            raise ValueError("n_time_buckets must be 0 (auto) or >= 2")
