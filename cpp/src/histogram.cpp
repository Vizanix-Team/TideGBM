#include "tidegbm/histogram.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tidegbm {

static int hist_threads(int n_jobs, int n_features) {
#ifdef _OPENMP
    if (n_jobs > 0) return std::min(n_jobs, std::max(n_features, 1));
    return std::min(omp_get_max_threads(), std::max(n_features, 1));
#else
    (void)n_jobs;
    (void)n_features;
    return 1;
#endif
}

void build_hist(const u16* xb, int n_features, const i32* idx, int n_idx, const f64* g,
                const f64* h, Hist& out, int n_jobs) {
    out.zero();
    if (n_idx <= 0) return;
    const int nth = hist_threads(n_jobs, n_features);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nth) if (n_features > 1 && nth > 1)
#endif
    for (int f = 0; f < n_features; ++f) {
        f64* G = &out.G[static_cast<size_t>(f) * out.n_bins];
        f64* H = &out.H[static_cast<size_t>(f) * out.n_bins];
        i64* C = &out.C[static_cast<size_t>(f) * out.n_bins];
        for (int k = 0; k < n_idx; ++k) {
            const i32 i = idx[k];
            const u16 b = xb[static_cast<size_t>(i) * n_features + f];
            G[b] += g[i];
            H[b] += h[i];
            C[b] += 1;
        }
    }
}

void build_hist_col(const u16* xcol, int n_rows, int n_features, const i32* idx, int n_idx,
                    const f64* g, const f64* h, Hist& out, int n_jobs) {
    out.zero();
    if (n_idx <= 0) return;
    const int nth = hist_threads(n_jobs, n_features);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nth) if (n_features > 1 && nth > 1)
#endif
    for (int f = 0; f < n_features; ++f) {
        const u16* col = xcol + static_cast<size_t>(f) * n_rows;
        f64* G = &out.G[static_cast<size_t>(f) * out.n_bins];
        f64* H = &out.H[static_cast<size_t>(f) * out.n_bins];
        i64* C = &out.C[static_cast<size_t>(f) * out.n_bins];
        int k = 0;
        for (; k + 8 < n_idx; ++k) {
#if defined(__GNUC__)
            __builtin_prefetch(&col[idx[k + 8]], 0, 1);
            __builtin_prefetch(&g[idx[k + 8]], 0, 1);
            __builtin_prefetch(&h[idx[k + 8]], 0, 1);
#endif
            const i32 i = idx[k];
            const u16 b = col[i];
            G[b] += g[i];
            H[b] += h[i];
            C[b] += 1;
        }
        for (; k < n_idx; ++k) {
            const i32 i = idx[k];
            const u16 b = col[i];
            G[b] += g[i];
            H[b] += h[i];
            C[b] += 1;
        }
    }
}

void transpose_bins(const u16* row, u16* col, int n, int n_features, int n_jobs) {
    const int nth = hist_threads(n_jobs, n_features);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nth) if (n_features > 1 && nth > 1)
#endif
    for (int f = 0; f < n_features; ++f) {
        u16* dst = col + static_cast<size_t>(f) * n;
        for (int i = 0; i < n; ++i) dst[i] = row[static_cast<size_t>(i) * n_features + f];
    }
}

#ifndef TIDEGBM_CUDA
void build_hist_gpu(const u16*, int, const i32*, int, const f64*, const f64*, Hist&, int) {
    throw TideError("GPU histograms are not compiled in this build (TIDEGBM_CUDA unset)");
}
#endif

void subtract_hist(const Hist& parent, const Hist& child, Hist& other) {
    other.resize(parent.n_features, parent.n_bins);
    const size_t n = parent.G.size();
    for (size_t i = 0; i < n; ++i) {
        other.G[i] = parent.G[i] - child.G[i];
        other.H[i] = parent.H[i] - child.H[i];
        other.C[i] = parent.C[i] - child.C[i];
    }
}

static inline f64 thresh_l1(f64 g, f64 l1) {
    if (l1 <= 0) return g;
    if (g > l1) return g - l1;
    if (g < -l1) return g + l1;
    return 0.0;
}

static inline f64 leaf_score(f64 G, f64 H, f64 lam, f64 l1) {
    const f64 t = thresh_l1(G, l1);
    return (t * t) / (H + lam);
}

static inline f64 split_gain(f64 GL, f64 HL, f64 GR, f64 HR, f64 parent, f64 lam, f64 gamma,
                             f64 l1) {
    return 0.5 * (leaf_score(GL, HL, lam, l1) + leaf_score(GR, HR, lam, l1) - parent) - gamma;
}

static inline f64 weight_of(f64 G, f64 H, f64 lam, f64 l1) {
    const f64 den = H + lam;
    if (l1 <= 0) return -G / den;
    if (G > l1) return -(G - l1) / den;
    if (G < -l1) return -(G + l1) / den;
    return 0.0;
}

SplitCand best_numeric_missing(const Hist& hist, int f, f64 lam, f64 gamma, int min_leaf,
                               f64 min_hess, int monotone, f64 l1) {
    SplitCand best;
    const int nb = hist.n_bins;
    f64 totG = 0, totH = 0;
    i64 totC = 0;
    for (int b = 0; b < nb; ++b) {
        totG += hist.g(f, b);
        totH += hist.h(f, b);
        totC += hist.c(f, b);
    }
    const f64 missG = hist.g(f, 0);
    const f64 missH = hist.h(f, 0);
    const i64 missC = hist.c(f, 0);
    const f64 parent = leaf_score(totG, totH, lam, l1);

    f64 GL = 0, HL = 0;
    i64 CL = 0;
    for (int b = 1; b < nb - 1; ++b) {
        const i64 cb = hist.c(f, b);
        GL += hist.g(f, b);
        HL += hist.h(f, b);
        CL += cb;
        if (cb == 0) continue;
        for (int miss_left = 0; miss_left <= 1; ++miss_left) {
            f64 gL = GL + (miss_left ? missG : 0);
            f64 hL = HL + (miss_left ? missH : 0);
            i64 cL = CL + (miss_left ? missC : 0);
            f64 gR = totG - gL;
            f64 hR = totH - hL;
            i64 cR = totC - cL;
            if (cL < min_leaf || cR < min_leaf) continue;
            if (hL < min_hess || hR < min_hess) continue;
            const f64 wL = weight_of(gL, hL, lam, l1);
            const f64 wR = weight_of(gR, hR, lam, l1);
            if (monotone > 0 && wL < wR) continue;
            if (monotone < 0 && wL > wR) continue;
            const f64 gain = split_gain(gL, hL, gR, hR, parent, lam, gamma, l1);
            if (gain > best.gain) {
                best.gain = gain;
                best.halo = gain;
                best.feature = f;
                best.threshold_bin = b;
                best.type = SplitType::Axis;
                best.missing_go_left = miss_left;
            }
        }
    }
    return best;
}

SplitCand best_categorical(const Hist& hist, int f, f64 lam, f64 gamma, int min_leaf,
                           f64 min_hess, f64 cat_smooth, f64 l1) {
    SplitCand best;
    const int nb = hist.n_bins;
    std::vector<int> order;
    order.reserve(nb);
    f64 totG = 0, totH = 0;
    i64 totC = 0;
    for (int b = 0; b < nb; ++b) {
        totG += hist.g(f, b);
        totH += hist.h(f, b);
        totC += hist.c(f, b);
        if (hist.c(f, b) > 0) order.push_back(b);
    }
    if (order.size() < 2) return best;
    const f64 sm = std::max(cat_smooth, 0.0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return hist.g(f, a) / (hist.h(f, a) + sm) < hist.g(f, b) / (hist.h(f, b) + sm);
    });
    const f64 parent = leaf_score(totG, totH, lam, l1);
    f64 GL = 0, HL = 0;
    i64 CL = 0;
    uint64_t mask[4] = {0, 0, 0, 0};
    auto setbit = [&](int b) {
        if (b < 0 || b >= 256) return;
        mask[b >> 6] |= (uint64_t(1) << (b & 63));
    };
    for (size_t i = 0; i + 1 < order.size(); ++i) {
        const int b = order[i];
        GL += hist.g(f, b);
        HL += hist.h(f, b);
        CL += hist.c(f, b);
        setbit(b);
        const f64 GR = totG - GL, HR = totH - HL;
        const i64 CR = totC - CL;
        if (CL < min_leaf || CR < min_leaf) continue;
        if (HL < min_hess || HR < min_hess) continue;
        const f64 gain = split_gain(GL, HL, GR, HR, parent, lam, gamma, l1);
        if (gain > best.gain) {
            best.gain = gain;
            best.halo = gain;
            best.feature = f;
            best.type = SplitType::Cat;
            best.threshold_bin = -1;
            best.missing_go_left = (mask[0] & 1ull) ? 1 : 0;
            for (int k = 0; k < 4; ++k) best.cat_left[k] = mask[k];
        }
    }
    return best;
}

}  // namespace tidegbm
