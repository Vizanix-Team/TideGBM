#pragma once

#include "tidegbm/types.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace tidegbm {

struct Hist {
    int n_features = 0;
    int n_bins = 0;
    std::vector<f64> G, H;
    std::vector<i64> C;

    void resize(int nf, int nb) {
        n_features = nf;
        n_bins = nb;
        G.assign(static_cast<size_t>(nf) * nb, 0.0);
        H.assign(static_cast<size_t>(nf) * nb, 0.0);
        C.assign(static_cast<size_t>(nf) * nb, 0);
    }
    void zero() {
        std::fill(G.begin(), G.end(), 0.0);
        std::fill(H.begin(), H.end(), 0.0);
        std::fill(C.begin(), C.end(), 0);
    }
    f64& g(int f, int b) { return G[static_cast<size_t>(f) * n_bins + b]; }
    f64& h(int f, int b) { return H[static_cast<size_t>(f) * n_bins + b]; }
    i64& c(int f, int b) { return C[static_cast<size_t>(f) * n_bins + b]; }
    const f64& g(int f, int b) const { return G[static_cast<size_t>(f) * n_bins + b]; }
    const f64& h(int f, int b) const { return H[static_cast<size_t>(f) * n_bins + b]; }
    const i64& c(int f, int b) const { return C[static_cast<size_t>(f) * n_bins + b]; }
};

void build_hist(const u16* xb, int n_features, const i32* idx, int n_idx, const f64* g,
                const f64* h, Hist& out, int n_jobs);

void build_hist_col(const u16* xcol, int n_rows, int n_features, const i32* idx, int n_idx,
                    const f64* g, const f64* h, Hist& out, int n_jobs);

void transpose_bins(const u16* row, u16* col, int n, int n_features, int n_jobs);

/* n_rows lets the CUDA path keep xb/g/h on device across leaves. */
void build_hist_gpu(const u16* xb, int n_features, const i32* idx, int n_idx, const f64* g,
                    const f64* h, Hist& out, int n_rows);

void subtract_hist(const Hist& parent, const Hist& child, Hist& other);

struct SplitCand {
    f64 gain = -1e300;
    f64 halo = -1e300;
    int feature = -1;
    int feature2 = -1;
    int threshold_bin = -1;
    SplitType type = SplitType::Axis;
    int missing_go_left = 1;
    uint64_t cat_left[4] = {0, 0, 0, 0};
};

SplitCand best_numeric_missing(const Hist& hist, int f, f64 lam, f64 gamma, int min_leaf,
                               f64 min_hess, int monotone, f64 l1 = 0.0);
SplitCand best_categorical(const Hist& hist, int f, f64 lam, f64 gamma, int min_leaf,
                           f64 min_hess, f64 cat_smooth = 1.0, f64 l1 = 0.0);

inline u16 pair_bin(u16 b1, u16 b2, int n_bins) {
    return static_cast<u16>((static_cast<uint32_t>(b1) * 2654435761u +
                             static_cast<uint32_t>(b2) * 40503u) %
                            static_cast<uint32_t>(n_bins));
}

}  // namespace tidegbm
