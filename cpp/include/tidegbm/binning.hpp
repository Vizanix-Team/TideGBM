#pragma once

#include "tidegbm/config.hpp"
#include "tidegbm/types.hpp"

#include <cmath>
#include <vector>

namespace tidegbm {

struct Binner {
    int max_bin = 64;
    int n_features = 0;
    std::vector<int> is_cat;  // 0/1
    std::vector<std::vector<f64>> edges;           // numeric right-edges
    std::vector<std::vector<f64>> cat_values;      // vocab, index+1 = bin
    std::vector<int> n_bins_col;

    void fit(const f64* X, int n, int n_f, const int* is_cat_in, const u8* mask, int max_bin_,
             int cat_min_count = 5, int bin_sample = 0, uint32_t seed = 0);
    void transform(const f64* X, int n, int n_f, u16* out) const;
    int max_bins() const;
};

inline bool is_nan(f64 v) { return !(v == v); }

}  // namespace tidegbm
