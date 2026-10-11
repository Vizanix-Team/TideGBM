#pragma once

#include "tidegbm/types.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace tidegbm {

/* One bin's gradient, hessian and count live together: the accumulate loop and the split
 * scan both touch all three per bin, so interleaving turns three cache streams into one. */
struct HBin {
    f64 g = 0.0;
    f64 h = 0.0;
    i64 c = 0;
};

struct Hist {
    int n_features = 0;
    int n_bins = 0;
    std::vector<HBin> B;

    // Allocate only. Every caller either fills the buffer (build_hist* zeroes first) or
    // overwrites it wholesale (subtract_hist), so zeroing here just repeated the same
    // memset two extra times per histogram.
    void resize(int nf, int nb) {
        n_features = nf;
        n_bins = nb;
        B.resize(static_cast<size_t>(nf) * nb);
    }
    // All-zero bytes is the correct value for both f64 0.0 and i64 0, and memset vectorises
    // where std::fill over a 24-byte struct degrades to three scalar stores per bin.
    void zero() {
        if (!B.empty()) std::memset(static_cast<void*>(B.data()), 0, B.size() * sizeof(HBin));
    }
    HBin* row(int f) { return B.data() + static_cast<size_t>(f) * n_bins; }
    const HBin* row(int f) const { return B.data() + static_cast<size_t>(f) * n_bins; }
    f64& g(int f, int b) { return B[static_cast<size_t>(f) * n_bins + b].g; }
    f64& h(int f, int b) { return B[static_cast<size_t>(f) * n_bins + b].h; }
    i64& c(int f, int b) { return B[static_cast<size_t>(f) * n_bins + b].c; }
    const f64& g(int f, int b) const { return B[static_cast<size_t>(f) * n_bins + b].g; }
    const f64& h(int f, int b) const { return B[static_cast<size_t>(f) * n_bins + b].h; }
    const i64& c(int f, int b) const { return B[static_cast<size_t>(f) * n_bins + b].c; }
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

/* parent -= child, in place: parent becomes the sibling of child. */
void subtract_hist_inplace(Hist& parent, const Hist& child);

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
