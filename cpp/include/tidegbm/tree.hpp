#pragma once

#include "tidegbm/config.hpp"
#include "tidegbm/histogram.hpp"
#include "tidegbm/types.hpp"

#include <vector>

namespace tidegbm {

struct Node {
    bool is_leaf = true;
    SplitType split_type = SplitType::Axis;
    i32 feature = -1;
    i32 feature2 = -1;
    i32 threshold_bin = -1;
    i32 left = -1;
    i32 right = -1;
    i32 missing_go_left = 1;
    f64 value = 0.0;
    f64 gain = 0.0;
    i32 n_samples = 0;
    i32 depth = 0;
    uint64_t cat_left[4] = {0, 0, 0, 0};
};

inline bool cat_goes_left(const Node& n, u16 b) {
    if (b >= 256) return n.missing_go_left != 0;
    return (n.cat_left[b >> 6] >> (b & 63)) & 1ull;
}

struct Tree {
    std::vector<Node> nodes;
    std::vector<f64> values;          // inference leaf values per node
    std::vector<f64> prefix_values;   // n_buckets * n_nodes
    int n_buckets = 1;
    std::vector<i32> cover;           // samples reaching node (for SHAP)
    int target = 0;                   // quantile / class / horizon head
    int n_targets = 1;
    std::vector<f64> multi_values;    // n_targets * n_nodes when n_targets > 1
};

inline f64 leaf_gain_score(f64 G, f64 H, f64 l2, f64 l1 = 0.0) {
    f64 t = G;
    if (l1 > 0) {
        if (G > l1) t = G - l1;
        else if (G < -l1) t = G + l1;
        else t = 0.0;
    }
    return (t * t) / (H + l2);
}

inline f64 leaf_value(f64 G, f64 H, f64 l2, f64 l1 = 0.0, f64 max_delta = 0.0) {
    f64 v;
    if (l1 > 0) {
        if (G > l1) v = -(G - l1) / (H + l2);
        else if (G < -l1) v = -(G + l1) / (H + l2);
        else v = 0.0;
    } else {
        v = -G / (H + l2);
    }
    if (max_delta > 0) {
        if (v > max_delta) v = max_delta;
        if (v < -max_delta) v = -max_delta;
    }
    return v;
}

int apply_one(const Tree& t, const u16* row, int n_features, int n_bins);
void apply_all(const Tree& t, const u16* xb, int n, int n_features, int n_bins, i32* node_id);
void predict_tree(const Tree& t, const u16* xb, int n, int n_features, int n_bins, const f64* values,
                  f64* out);

Tree grow_tree(const u16* xb_row, const u16* xb_col, int n_rows, const f64* g, const f64* h,
               const i32* sample_idx, int n_idx, int n_features, int n_bins, const Config& cfg,
               const int* is_cat, const i32* group_of, const int8_t* monotone, uint32_t seed);

}  // namespace tidegbm
