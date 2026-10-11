#include "tidegbm/objective.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tidegbm {

/* Rank groups are timestamps, and fit() sorts rows by timestamp, so groups are almost
 * always contiguous runs. Detecting that avoids rebuilding a hash map of vectors on every
 * boosting iteration; the unsorted case falls back to one stable sort instead. */
static void group_runs(const i64* groups, int n, std::vector<int>& idx,
                       std::vector<std::pair<int, int>>& runs) {
    idx.resize(n);
    std::iota(idx.begin(), idx.end(), 0);
    bool sorted = true;
    for (int i = 1; i < n; ++i)
        if (groups[i] < groups[i - 1]) {
            sorted = false;
            break;
        }
    if (!sorted)
        std::stable_sort(idx.begin(), idx.end(),
                         [&](int a, int b) { return groups[a] < groups[b]; });
    runs.clear();
    int i = 0;
    while (i < n) {
        int j = i + 1;
        while (j < n && groups[idx[j]] == groups[idx[i]]) ++j;
        runs.emplace_back(i, j - i);
        i = j;
    }
}

static f64 wavg(const f64* y, const f64* w, int n) {
    f64 sw = 0, sy = 0;
    for (int i = 0; i < n; ++i) {
        const f64 wi = w ? w[i] : 1.0;
        sw += wi;
        sy += wi * y[i];
    }
    return sw > 0 ? sy / sw : 0;
}

static void softmax_in(const f64* x, int n, f64 tau, std::vector<f64>& out) {
    out.resize(n);
    f64 m = x[0];
    for (int i = 1; i < n; ++i) m = std::max(m, x[i]);
    f64 s = 0;
    for (int i = 0; i < n; ++i) {
        out[i] = std::exp(std::clamp((x[i] - m) / std::max(tau, 1e-12), -50.0, 50.0));
        s += out[i];
    }
    if (s <= 0) {
        std::fill(out.begin(), out.end(), 1.0 / n);
        return;
    }
    for (int i = 0; i < n; ++i) out[i] /= s;
}

Objective make_objective(const Config& cfg) {
    Objective o;
    o.kind = cfg.objective;
    o.huber_delta = cfg.huber_delta;
    o.quantile = cfg.quantile;
    o.rank_temperature = cfg.rank_temperature;
    o.cost_threshold = cfg.cost_threshold;
    o.n_classes = cfg.n_classes;
    o.row_cost = nullptr;
    if (cfg.rank_pairwise && (o.kind == ObjectiveKind::RankListNet)) o.kind = ObjectiveKind::RankPairwise;
    return o;
}

f64 Objective::init_score(const f64* y, const f64* w, int n) const {
    if (n <= 0) return 0;
    if (kind == ObjectiveKind::Logloss) {
        const f64 p = std::clamp(wavg(y, w, n), 1e-5, 1 - 1e-5);
        return std::log(p / (1 - p));
    }
    if (kind == ObjectiveKind::Quantile) {
        std::vector<f64> v(y, y + n);
        std::sort(v.begin(), v.end());
        return v[std::clamp(static_cast<int>(quantile * (n - 1)), 0, n - 1)];
    }
    if (kind == ObjectiveKind::Huber) {
        std::vector<f64> v(y, y + n);
        std::nth_element(v.begin(), v.begin() + n / 2, v.end());
        return v[n / 2];
    }
    if (kind == ObjectiveKind::RankListNet || kind == ObjectiveKind::RankPairwise ||
        kind == ObjectiveKind::Multiclass)
        return 0;
    return wavg(y, w, n);
}

void Objective::grad_hess(const f64* y, const f64* pred, const f64* w, const i64* groups, int n,
                          f64* g, f64* h) const {
    auto wi = [&](int i) { return w ? w[i] : 1.0; };
    if (kind == ObjectiveKind::MSE) {
        for (int i = 0; i < n; ++i) {
            g[i] = wi(i) * (pred[i] - y[i]);
            h[i] = wi(i);
            if (cost_threshold > 0 && std::abs(pred[i]) < cost_threshold) {
                g[i] += pred[i];
                h[i] += 1.0;
            }
            if (row_cost && row_cost[i] > 0 && std::abs(pred[i]) < row_cost[i]) {
                g[i] += pred[i];
                h[i] += 1.0;
            }
        }
        return;
    }
    if (kind == ObjectiveKind::Huber) {
        for (int i = 0; i < n; ++i) {
            const f64 r = pred[i] - y[i];
            const f64 ar = std::abs(r);
            if (ar <= huber_delta) {
                g[i] = wi(i) * r;
                h[i] = wi(i);
            } else {
                g[i] = wi(i) * huber_delta * (r >= 0 ? 1 : -1);
                h[i] = wi(i) * 1e-3;
            }
        }
        return;
    }
    if (kind == ObjectiveKind::Quantile) {
        for (int i = 0; i < n; ++i) {
            const f64 r = y[i] - pred[i];
            g[i] = wi(i) * (r >= 0 ? -quantile : 1.0 - quantile);
            h[i] = wi(i);
        }
        return;
    }
    if (kind == ObjectiveKind::Logloss) {
        for (int i = 0; i < n; ++i) {
            const f64 p = 1.0 / (1.0 + std::exp(-std::clamp(pred[i], -30.0, 30.0)));
            g[i] = wi(i) * (p - y[i]);
            h[i] = wi(i) * std::max(p * (1 - p), 1e-6);
        }
        return;
    }
    if (kind == ObjectiveKind::RankListNet) {
        std::fill(g, g + n, 0.0);
        std::fill(h, h + n, 0.0);
        if (!groups) return;
        std::vector<int> idx;
        std::vector<std::pair<int, int>> runs;
        group_runs(groups, n, idx, runs);
        const int n_runs = static_cast<int>(runs.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8) if (n_runs > 4)
#endif
        for (int r = 0; r < n_runs; ++r) {
            const int* ix = idx.data() + runs[r].first;
            const int m = runs[r].second;
            if (m < 2) continue;
            std::vector<f64> yy(m), pp(m), t, p;
            for (int k = 0; k < m; ++k) {
                yy[k] = y[ix[k]];
                pp[k] = pred[ix[k]];
            }
            softmax_in(yy.data(), m, rank_temperature, t);
            softmax_in(pp.data(), m, 1.0, p);
            for (int k = 0; k < m; ++k) {
                g[ix[k]] = wi(ix[k]) * (p[k] - t[k]);
                h[ix[k]] = wi(ix[k]) * std::max(p[k] * (1 - p[k]), 1e-6);
            }
        }
        return;
    }
    if (kind == ObjectiveKind::RankPairwise) {
        std::fill(g, g + n, 0.0);
        std::fill(h, h + n, 1e-6);
        if (!groups) return;
        std::vector<int> idx;
        std::vector<std::pair<int, int>> runs;
        group_runs(groups, n, idx, runs);
        const int n_runs = static_cast<int>(runs.size());
        const f64 sigma = 1.0;
        // Groups own disjoint row sets, so the accumulations below never race.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8) if (n_runs > 4)
#endif
        for (int r = 0; r < n_runs; ++r) {
            const int* ix = idx.data() + runs[r].first;
            const int m = runs[r].second;
            if (m < 2) continue;
            const int step = m > 48 ? std::max(1, m / 24) : 1;
            for (int a = 0; a < m; ++a) {
                for (int b = a + step; b < m; b += step) {
                    const int i = ix[a], j = ix[b];
                    if (y[i] == y[j]) continue;
                    int hi = i, lo = j;
                    if (y[j] > y[i]) {
                        hi = j;
                        lo = i;
                    }
                    const f64 s = pred[hi] - pred[lo];
                    const f64 e = std::exp(std::clamp(sigma * s, -30.0, 30.0));
                    const f64 dyw = std::max(std::abs(y[hi] - y[lo]), 1e-6);
                    const f64 lam = -sigma / (1.0 + e) * dyw;
                    const f64 d2 = sigma * sigma * e / ((1 + e) * (1 + e)) * dyw;
                    g[hi] += wi(hi) * lam;
                    g[lo] -= wi(lo) * lam;
                    h[hi] += wi(hi) * d2;
                    h[lo] += wi(lo) * d2;
                }
            }
        }
        return;
    }
    // fallback MSE
    for (int i = 0; i < n; ++i) {
        g[i] = wi(i) * (pred[i] - y[i]);
        h[i] = wi(i);
    }
}

void Objective::grad_hess_multiclass(const f64* y, const f64* pred, const f64* w, int n, int K,
                                     f64* g, f64* h) const {
    auto wi = [&](int i) { return w ? w[i] : 1.0; };
    std::vector<f64> p(K);
    for (int i = 0; i < n; ++i) {
        f64 m = pred[static_cast<size_t>(i) * K];
        for (int c = 1; c < K; ++c) m = std::max(m, pred[static_cast<size_t>(i) * K + c]);
        f64 s = 0;
        for (int c = 0; c < K; ++c) {
            p[c] = std::exp(std::clamp(pred[static_cast<size_t>(i) * K + c] - m, -50.0, 50.0));
            s += p[c];
        }
        const int yi = static_cast<int>(std::lround(y[i]));
        for (int c = 0; c < K; ++c) {
            const f64 pc = p[c] / std::max(s, 1e-12);
            g[static_cast<size_t>(i) * K + c] = wi(i) * (pc - (yi == c ? 1.0 : 0.0));
            h[static_cast<size_t>(i) * K + c] = wi(i) * std::max(pc * (1.0 - pc), 1e-6);
        }
    }
}

void Objective::softmax_rows(const f64* raw, int n, int K, f64* out) const {
    for (int i = 0; i < n; ++i) {
        f64 m = raw[static_cast<size_t>(i) * K];
        for (int c = 1; c < K; ++c) m = std::max(m, raw[static_cast<size_t>(i) * K + c]);
        f64 s = 0;
        for (int c = 0; c < K; ++c) {
            out[static_cast<size_t>(i) * K + c] =
                std::exp(std::clamp(raw[static_cast<size_t>(i) * K + c] - m, -50.0, 50.0));
            s += out[static_cast<size_t>(i) * K + c];
        }
        for (int c = 0; c < K; ++c) out[static_cast<size_t>(i) * K + c] /= std::max(s, 1e-12);
    }
}

f64 Objective::loss(const f64* y, const f64* pred, const f64* w, const i64* groups, int n) const {
    auto wi = [&](int i) { return w ? w[i] : 1.0; };
    f64 s = 0, sw = 0;
    if (kind == ObjectiveKind::Logloss) {
        for (int i = 0; i < n; ++i) {
            const f64 p = std::clamp(1.0 / (1.0 + std::exp(-std::clamp(pred[i], -30.0, 30.0))),
                                     1e-15, 1 - 1e-15);
            s += wi(i) * -(y[i] * std::log(p) + (1 - y[i]) * std::log(1 - p));
            sw += wi(i);
        }
        return sw > 0 ? s / sw : 0;
    }
    if (kind == ObjectiveKind::Huber) {
        for (int i = 0; i < n; ++i) {
            const f64 r = std::abs(pred[i] - y[i]);
            s += wi(i) * (r <= huber_delta ? 0.5 * r * r : huber_delta * (r - 0.5 * huber_delta));
            sw += wi(i);
        }
        return sw > 0 ? s / sw : 0;
    }
    if (kind == ObjectiveKind::Quantile) {
        for (int i = 0; i < n; ++i) {
            const f64 r = y[i] - pred[i];
            s += wi(i) * (r >= 0 ? quantile * r : (quantile - 1) * r);
            sw += wi(i);
        }
        return sw > 0 ? s / sw : 0;
    }
    if ((kind == ObjectiveKind::RankListNet || kind == ObjectiveKind::RankPairwise) && groups) {
        // Squared error says nothing about a ranking fit. Report negative mean within-group
        // correlation instead, so the reported curve still decreases as the fit improves.
        std::vector<int> idx;
        std::vector<std::pair<int, int>> runs;
        group_runs(groups, n, idx, runs);
        f64 acc = 0;
        int ng = 0;
        for (const auto& rn : runs) {
            const int* ix = idx.data() + rn.first;
            const int m = rn.second;
            if (m < 3) continue;
            f64 my = 0, mp = 0;
            for (int k = 0; k < m; ++k) {
                my += y[ix[k]];
                mp += pred[ix[k]];
            }
            my /= m;
            mp /= m;
            f64 vy = 0, vp = 0, c = 0;
            for (int k = 0; k < m; ++k) {
                const f64 dy = y[ix[k]] - my, dp = pred[ix[k]] - mp;
                vy += dy * dy;
                vp += dp * dp;
                c += dy * dp;
            }
            if (vy > 1e-18 && vp > 1e-18) {
                acc += c / std::sqrt(vy * vp);
                ng += 1;
            }
        }
        return ng > 0 ? -acc / ng : 0.0;
    }
    for (int i = 0; i < n; ++i) {
        const f64 d = pred[i] - y[i];
        s += wi(i) * d * d;
        sw += wi(i);
    }
    return sw > 0 ? s / sw : 0;
}

void Objective::finalize(const f64* raw, int n, f64* out) const {
    if (kind == ObjectiveKind::Logloss) {
        for (int i = 0; i < n; ++i) out[i] = 1.0 / (1.0 + std::exp(-std::clamp(raw[i], -30.0, 30.0)));
        return;
    }
    if (kind == ObjectiveKind::Multiclass && n_classes > 1) {
        softmax_rows(raw, n, n_classes, out);
        return;
    }
    std::copy(raw, raw + n, out);
}

}  // namespace tidegbm
