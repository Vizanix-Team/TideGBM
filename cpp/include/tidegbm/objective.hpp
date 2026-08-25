#pragma once

#include "tidegbm/config.hpp"
#include "tidegbm/types.hpp"

#include <vector>

namespace tidegbm {

struct Objective {
    ObjectiveKind kind = ObjectiveKind::MSE;
    f64 huber_delta = 1.0;
    f64 quantile = 0.5;
    f64 rank_temperature = 1.0;
    f64 cost_threshold = 0.0;
    int n_classes = 1;
    const f64* row_cost = nullptr;

    f64 init_score(const f64* y, const f64* w, int n) const;
    void grad_hess(const f64* y, const f64* pred, const f64* w, const i64* groups, int n, f64* g,
                   f64* h) const;
    void grad_hess_multiclass(const f64* y, const f64* pred, const f64* w, int n, int K, f64* g,
                              f64* h) const;
    f64 loss(const f64* y, const f64* pred, const f64* w, const i64* groups, int n) const;
    void finalize(const f64* raw, int n, f64* out) const;
    void softmax_rows(const f64* raw, int n, int K, f64* out) const;
};

Objective make_objective(const Config& cfg);

}  // namespace tidegbm
