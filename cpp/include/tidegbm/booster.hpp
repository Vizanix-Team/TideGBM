#pragma once

#include "tidegbm/binning.hpp"
#include "tidegbm/config.hpp"
#include "tidegbm/objective.hpp"
#include "tidegbm/tree.hpp"

#include <functional>
#include <string>
#include <vector>

namespace tidegbm {

struct EvalResult {
    std::vector<f64> train;
    std::vector<f64> valid;
};

using CustomGradFn = std::function<void(const f64* y, const f64* pred, const f64* w, int n, f64* g, f64* h)>;
using LogFn = std::function<void(int iter, f64 train, f64 valid)>;

class Booster {
public:
    Config cfg;
    Objective obj;
    Binner binner;
    std::vector<Tree> trees;
    std::vector<std::string> feature_names;
    std::vector<int> is_cat;
    std::vector<i32> group_of;
    std::vector<int8_t> monotone;
    f64 init_score = 0.0;
    std::vector<f64> init_prefix;
    std::vector<f64> init_scores;
    int n_features = 0;
    int n_bins = 64;
    int n_time_buckets = 1;
    int best_iteration = 0;
    EvalResult evals;
    bool fitted = false;
    CustomGradFn custom_grad;
    LogFn log_fn;
    std::vector<f64> split_count;

    void fit(const f64* X, const f64* y, int n, int n_f, const i64* timestamps, const f64* weight,
             const int* is_cat_in, const i32* group_of_in, const int8_t* monotone_in,
             const i64* rank_groups, const f64* X_eval, const f64* y_eval, const i64* ts_eval,
             int n_eval, const i64* entity = nullptr, const f64* row_cost = nullptr);

    void predict_raw(const f64* X, int n, int n_f, f64* out) const;
    void predict(const f64* X, int n, int n_f, f64* out) const;
    void predict_binned(const u16* xb, int n, f64* out) const;
    void predict_multi(const f64* X, int n, int n_f, f64* out) const;  // n * n_heads

    void update(const f64* X, const f64* y, int n, int n_f, const f64* weight, const i64* rank_groups,
                int extra_trees);

    void feature_importance(f64* out, const char* kind = "gain") const;
    void pred_contrib(const f64* X, int n, int n_f, f64* out) const;  // (n, n_f+1) path attrib + base

    std::string save_binary() const;
    static Booster load_binary(const std::string& blob);
    std::string save_json() const;
    std::string to_onnx_json() const;
    int n_heads() const;

    void bin_transform(const f64* X, int n, int n_f, std::vector<u16>& xb) const;

private:
    void predict_raw_binned(const u16* xb, int n, f64* out) const;
    void predict_multi_binned(const u16* xb, int n, f64* out) const;
};

}  // namespace tidegbm
