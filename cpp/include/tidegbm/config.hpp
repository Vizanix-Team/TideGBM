#pragma once

#include "tidegbm/types.hpp"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace tidegbm {

struct Config {
    int n_estimators = 100;
    f64 learning_rate = 0.1;
    int max_depth = 6;
    int max_leaves = 31;
    int min_samples_leaf = 20;
    f64 min_child_weight = 1e-3;
    f64 l2 = 1.0;
    f64 min_split_gain = 0.0;
    int max_bin = 255;
    f64 subsample = 1.0;
    f64 colsample = 1.0;
    int n_time_buckets = 0;  // 0 = auto from horizon length (32 short / 64 long)
    i64 horizon = -1;  // required in time mode, in timestamp units
    i64 embargo = 0;
    bool iid_mode = false;
    bool halo = false;
    f64 halo_gamma = 0.4;
    int halo_candidates = 3;
    bool pair_splits = true;
    f64 pair_trigger = 0.25;
    int pair_top_k = 6;
    int max_pair_splits_per_tree = 3;
    f64 pair_l2 = 2.0;
    f64 time_decay = 1.0;
    f64 tail_buffer = 0.02;
    f64 tree_decay = 1.0;
    ObjectiveKind objective = ObjectiveKind::MSE;
    f64 quantile = 0.5;
    f64 huber_delta = 1.0;
    f64 rank_temperature = 1.0;
    int n_jobs = 0;  // 0 => all cores
    int random_state = 0;
    int verbose = 0;
    int early_stopping_rounds = 0;
    bool causal_structure = false;
    bool finer_torb = true;
    std::string device = "cpu";
    int n_classes = 1;
    std::vector<f64> quantiles;
    int n_horizons = 1;
    f64 cost_threshold = 0.0;
    bool rank_pairwise = false;
    int regime_feature = -1;
    bool regime_priority = true;
    f64 init_score_given = std::nan("");
    int n_targets = 1;
    std::vector<int> interaction_group;          // per feature, -1 = unconstrained
    std::vector<std::pair<int, int>> forced_pairs;  // PAIR only these if non-empty
    size_t ooc_bytes = 256ull << 20;             // mmap bins above this
    f64 l1 = 0.0;
    f64 max_delta_step = 0.0;
    int goss = -1;  // -1 auto (>=250k rows), 0 off, 1 on
    f64 goss_top = 0.2;
    f64 goss_other = 0.1;
    f64 structure_frac = 1.0;  // 1 = use all train times for splits; TORB still prefixes leaf values
    bool causal_bins = false;  // true = HAB-C: edges from older half of train only
    f64 torb_min_cover = 0.0;  // 0 = auto (2 * min_samples_leaf); blend prefix→full below this
    f64 cat_smooth = 1.0;       // Laplace on categorical G/H ranking
    bool col_major_hist = true;
    int cat_min_count = 5;
    f64 colsample_bynode = 1.0;
    int bin_sample = 262144;          // 0 = use every row for quantile edges
    f64 entity_subsample = 1.0;       // keep this fraction of entities when entity is passed
    std::string eval_metric = "auto"; // auto | loss | ic

    void validate() const {
        if (n_estimators < 1) throw TideError("n_estimators must be >= 1");
        if (learning_rate <= 0 || learning_rate > 1.5) throw TideError("learning_rate out of range");
        if (max_bin < 8 || max_bin > kMaxBinCap) throw TideError("max_bin must be in [8, 256]");
        if (!iid_mode && n_time_buckets == 1) throw TideError("n_time_buckets must be 0 (auto) or >= 2");
        if (!iid_mode && n_time_buckets < 0) throw TideError("n_time_buckets must be 0 (auto) or >= 2");
        if (!iid_mode && horizon < 0) {
            throw TideError(
                "horizon is required in time mode (label length in timestamp units). "
                "Overlapping returns without horizon leak into TORB.");
        }
        if (device != "cpu" && device != "gpu") throw TideError("device must be cpu or gpu");
#ifndef TIDEGBM_CUDA
        if (device == "gpu") {
            throw TideError("GPU histograms are not compiled in this build (TIDEGBM_CUDA unset)");
        }
#endif
        if (n_classes < 1) throw TideError("n_classes must be >= 1");
        if (n_targets < 1) throw TideError("n_targets must be >= 1");
    }
};

}  // namespace tidegbm
