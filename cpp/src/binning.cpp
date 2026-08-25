#include "tidegbm/binning.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tidegbm {

static std::vector<f64> quantile_edges(std::vector<f64>& vals, int n_bins) {
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    if (vals.empty()) return {0.0};
    if (static_cast<int>(vals.size()) <= n_bins) return vals;
    std::vector<f64> edges;
    edges.reserve(n_bins - 1);
    for (int i = 1; i < n_bins; ++i) {
        const f64 q = static_cast<f64>(i) / n_bins;
        const f64 pos = q * (vals.size() - 1);
        const size_t lo = static_cast<size_t>(pos);
        const size_t hi = std::min(lo + 1, vals.size() - 1);
        const f64 frac = pos - lo;
        edges.push_back(vals[lo] * (1 - frac) + vals[hi] * frac);
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    return edges;
}

void Binner::fit(const f64* X, int n, int n_f, const int* is_cat_in, const u8* mask, int max_bin_,
                 int cat_min_count, int bin_sample, uint32_t seed) {
    max_bin = max_bin_;
    n_features = n_f;
    is_cat.assign(is_cat_in, is_cat_in + n_f);
    edges.assign(n_f, {});
    cat_values.assign(n_f, {});
    n_bins_col.assign(n_f, 2);
    const int numeric_bins = max_bin - 1;
    const int min_c = std::max(1, cat_min_count);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if (n_f > 1)
#endif
    for (int f = 0; f < n_f; ++f) {
        if (is_cat[f]) {
            std::vector<f64> raw;
            raw.reserve(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                if (mask && !mask[i]) continue;
                const f64 v = X[static_cast<size_t>(i) * n_f + f];
                if (is_nan(v)) continue;
                raw.push_back(v);
            }
            std::sort(raw.begin(), raw.end());
            std::vector<std::pair<int, f64>> freq;
            for (size_t i = 0; i < raw.size();) {
                size_t j = i + 1;
                while (j < raw.size() && raw[j] == raw[i]) ++j;
                freq.emplace_back(static_cast<int>(j - i), raw[i]);
                i = j;
            }
            std::vector<std::pair<int, f64>> kept_f;
            for (const auto& p : freq)
                if (p.first >= min_c) kept_f.push_back(p);
            if (kept_f.empty() && !freq.empty()) {
                auto it = std::max_element(freq.begin(), freq.end(),
                                           [](const auto& a, const auto& b) { return a.first < b.first; });
                kept_f.push_back(*it);
            }
            if (static_cast<int>(kept_f.size()) > numeric_bins) {
                std::partial_sort(kept_f.begin(), kept_f.begin() + numeric_bins, kept_f.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                kept_f.resize(numeric_bins);
            }
            std::vector<f64> uniq;
            uniq.reserve(kept_f.size());
            for (const auto& p : kept_f) uniq.push_back(p.second);
            std::sort(uniq.begin(), uniq.end());
            cat_values[f] = std::move(uniq);
            n_bins_col[f] = static_cast<int>(cat_values[f].size()) + 1;
        } else {
            std::vector<f64> vals;
            vals.reserve(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                if (mask && !mask[i]) continue;
                const f64 v = X[static_cast<size_t>(i) * n_f + f];
                if (is_nan(v)) continue;
                vals.push_back(v);
            }
            if (bin_sample > 0 && static_cast<int>(vals.size()) > bin_sample) {
                std::mt19937 rng(seed + static_cast<uint32_t>(f) * 2654435761u);
                std::shuffle(vals.begin(), vals.end(), rng);
                vals.resize(static_cast<size_t>(bin_sample));
            }
            edges[f] = quantile_edges(vals, numeric_bins);
            n_bins_col[f] = static_cast<int>(edges[f].size()) + 2;
        }
    }
}

void Binner::transform(const f64* X, int n, int n_f, u16* out) const {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 4096)
#endif
    for (int i = 0; i < n; ++i) {
        for (int f = 0; f < n_f; ++f) {
            const f64 v = X[static_cast<size_t>(i) * n_f + f];
            u16 b = kMissingBin;
            if (!is_nan(v)) {
                if (is_cat[f]) {
                    const auto& voc = cat_values[f];
                    auto it = std::lower_bound(voc.begin(), voc.end(), v);
                    if (it != voc.end() && *it == v) {
                        b = static_cast<u16>((it - voc.begin()) + 1);
                    }
                } else {
                    const auto& e = edges[f];
                    const auto it = std::upper_bound(e.begin(), e.end(), v);
                    b = static_cast<u16>((it - e.begin()) + 1);
                }
            }
            out[static_cast<size_t>(i) * n_f + f] = b;
        }
    }
}

int Binner::max_bins() const {
    int m = 2;
    for (int v : n_bins_col) m = std::max(m, v);
    return std::max(m, max_bin);
}

}  // namespace tidegbm
