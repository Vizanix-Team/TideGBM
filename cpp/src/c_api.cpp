#include "tidegbm/c_api.h"

#include "tidegbm/booster.hpp"

#include <fstream>
#include <memory>
#include <string>

using tidegbm::Booster;
using tidegbm::TideError;

static thread_local std::string g_err;

static void set_err(const std::string& s) { g_err = s; }

extern "C" {

const char* tide_last_error(void) { return g_err.c_str(); }

int tide_create(TideHandle* out) {
    try {
        *out = new Booster();
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        return -1;
    }
}

void tide_free(TideHandle h) { delete static_cast<Booster*>(h); }

int tide_set_int(TideHandle h, const char* key, int value) {
    auto* b = static_cast<Booster*>(h);
    std::string k(key);
    if (k == "n_estimators") b->cfg.n_estimators = value;
    else if (k == "max_depth") b->cfg.max_depth = value;
    else if (k == "max_leaves") b->cfg.max_leaves = value;
    else if (k == "max_bin") b->cfg.max_bin = value;
    else if (k == "n_time_buckets") b->cfg.n_time_buckets = value;
    else if (k == "random_state") b->cfg.random_state = value;
    else if (k == "n_jobs") b->cfg.n_jobs = value;
    else if (k == "iid_mode") b->cfg.iid_mode = value != 0;
    else if (k == "halo") b->cfg.halo = value != 0;
    else if (k == "verbose") b->cfg.verbose = value;
    else if (k == "horizon") b->cfg.horizon = value;
    else if (k == "embargo") b->cfg.embargo = value;
    else if (k == "goss") b->cfg.goss = value;
    else if (k == "cat_min_count") b->cfg.cat_min_count = value;
    else if (k == "col_major_hist") b->cfg.col_major_hist = value != 0;
    else {
        set_err("unknown int key");
        return -1;
    }
    return 0;
}

int tide_set_double(TideHandle h, const char* key, double value) {
    auto* b = static_cast<Booster*>(h);
    std::string k(key);
    if (k == "learning_rate") b->cfg.learning_rate = value;
    else if (k == "l2") b->cfg.l2 = value;
    else if (k == "l1") b->cfg.l1 = value;
    else if (k == "subsample") b->cfg.subsample = value;
    else if (k == "colsample") b->cfg.colsample = value;
    else {
        set_err("unknown double key");
        return -1;
    }
    return 0;
}

int tide_fit(TideHandle h, const double* X, const double* y, int n, int n_features,
             const long long* timestamps, const double* weight, const int* is_cat, int iid,
             int horizon) {
    auto* b = static_cast<Booster*>(h);
    try {
        b->cfg.iid_mode = iid != 0;
        if (!b->cfg.iid_mode) b->cfg.horizon = horizon;
        b->fit(X, y, n, n_features, reinterpret_cast<const int64_t*>(timestamps), weight, is_cat,
               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        return -1;
    }
}

int tide_predict(TideHandle h, const double* X, int n, int n_features, double* out) {
    auto* b = static_cast<Booster*>(h);
    try {
        b->predict(X, n, n_features, out);
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        return -1;
    }
}

int tide_predict_raw(TideHandle h, const double* X, int n, int n_features, double* out) {
    auto* b = static_cast<Booster*>(h);
    try {
        b->predict_raw(X, n, n_features, out);
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        return -1;
    }
}

int tide_predict_binned(TideHandle h, const unsigned short* xb, int n, double* out) {
    auto* b = static_cast<Booster*>(h);
    try {
        b->predict_binned(xb, n, out);
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        return -1;
    }
}

int tide_save_json(TideHandle h, const char* path) {
    auto* b = static_cast<Booster*>(h);
    try {
        std::string js = b->save_json();
        std::ofstream f(path);
        f << js;
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        return -1;
    }
}

int tide_save(TideHandle h, const char* path) {
    auto* b = static_cast<Booster*>(h);
    try {
        std::string blob = b->save_binary();
        std::ofstream f(path, std::ios::binary);
        f.write(blob.data(), static_cast<std::streamsize>(blob.size()));
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        return -1;
    }
}

int tide_load(TideHandle* out, const char* path) {
    try {
        std::ifstream f(path, std::ios::binary);
        std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        auto b = Booster::load_binary(blob);
        *out = new Booster(std::move(b));
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        return -1;
    }
}

}  // extern C
