#include "tidegbm/booster.hpp"

#include "tidegbm/histogram.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace tidegbm {

static int threads_of(const Config& cfg) {
    if (cfg.n_jobs > 0) return cfg.n_jobs;
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static i64 median_ts_step(const i64* ts, int n) {
    std::vector<i64> uniq(ts, ts + n);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    if (uniq.size() < 2) return 1;
    std::vector<i64> d;
    d.reserve(uniq.size() - 1);
    for (size_t i = 1; i < uniq.size(); ++i) {
        const i64 gap = uniq[i] - uniq[i - 1];
        if (gap > 0) d.push_back(gap);
    }
    if (d.empty()) return 1;
    std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
    return std::max<i64>(d[d.size() / 2], 1);
}

static int auto_time_buckets(const i64* ts, int n, i64 horizon_units, int uniq_cap) {
    // Short labels: coarser TORB (32). Long overlapping labels: finer (64).
    const i64 step = median_ts_step(ts, n);
    const int h_steps = static_cast<int>(std::max<i64>(horizon_units, 0) / step);
    const int want = (h_steps >= 8) ? 64 : 32;
    return std::max(2, std::min(want, std::max(uniq_cap, 2)));
}

static void time_buckets(const i64* ts, int n, int n_buckets, std::vector<i32>& buckets) {
    buckets.resize(n);
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return ts[a] < ts[b]; });
    std::vector<int> rank(n);
    for (int i = 0; i < n; ++i) rank[order[i]] = i;
    for (int i = 0; i < n; ++i) {
        int b = static_cast<int>(static_cast<f64>(rank[i]) / std::max(n, 1) * n_buckets);
        if (b >= n_buckets) b = n_buckets - 1;
        buckets[i] = b;
    }
}

static void recency_weights(const i64* ts, int n, f64 decay, std::vector<f64>& w) {
    if (decay >= 1.0) return;
    i64 tmax = ts[0], tmin = ts[0];
    for (int i = 1; i < n; ++i) {
        tmax = std::max(tmax, ts[i]);
        tmin = std::min(tmin, ts[i]);
    }
    f64 step = 1.0;
    std::vector<i64> uniq(ts, ts + n);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    if (uniq.size() >= 2) {
        std::vector<i64> d;
        for (size_t i = 1; i < uniq.size(); ++i) d.push_back(uniq[i] - uniq[i - 1]);
        std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
        step = std::max<i64>(d[d.size() / 2], 1);
    }
    for (int i = 0; i < n; ++i) {
        const f64 age = static_cast<f64>(tmax - ts[i]) / step;
        w[i] *= std::pow(decay, age);
        w[i] = std::clamp(w[i], 1e-8, 1.0);
    }
}

static std::vector<i32> sample_blocks(const std::vector<i32>& buckets, const f64* y, int n,
                                      f64 subsample, f64 tail, std::mt19937& rng) {
    std::vector<i32> all(n);
    std::iota(all.begin(), all.end(), 0);
    if (subsample >= 1.0) return all;
    i32 mx = 0;
    for (int i = 0; i < n; ++i) mx = std::max(mx, buckets[i]);
    const int nb = mx + 1;
    std::vector<f64> score(nb, 0);
    std::vector<int> cnt(nb, 0);
    for (int i = 0; i < n; ++i) {
        score[buckets[i]] += std::abs(y[i]);
        cnt[buckets[i]] += 1;
    }
    std::vector<int> ids(nb);
    std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(), [&](int a, int b) {
        return score[a] / std::max(cnt[a], 1) > score[b] / std::max(cnt[b], 1);
    });
    const int n_tail = std::max(1, static_cast<int>(std::ceil(tail * nb)));
    std::vector<char> keep(nb, 0);
    for (int i = 0; i < n_tail && i < nb; ++i) keep[ids[i]] = 1;
    std::vector<int> rest;
    for (int b = 0; b < nb; ++b)
        if (!keep[b]) rest.push_back(b);
    int n_draw = std::max(0, static_cast<int>(std::ceil(subsample * nb)) - n_tail);
    n_draw = std::min(n_draw, static_cast<int>(rest.size()));
    std::shuffle(rest.begin(), rest.end(), rng);
    for (int i = 0; i < n_draw; ++i) keep[rest[i]] = 1;
    std::vector<i32> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i)
        if (keep[buckets[i]]) out.push_back(i);
    return out;
}

static f64 pearson_ix(const f64* y, const f64* p, const i32* ix, int m) {
    if (m < 3) return 0.0;
    f64 my = 0, mp = 0;
    for (int k = 0; k < m; ++k) {
        my += y[ix[k]];
        mp += p[ix[k]];
    }
    my /= m;
    mp /= m;
    f64 vy = 0, vp = 0, c = 0;
    for (int k = 0; k < m; ++k) {
        const f64 dy = y[ix[k]] - my;
        const f64 dp = p[ix[k]] - mp;
        vy += dy * dy;
        vp += dp * dp;
        c += dy * dp;
    }
    if (vy < 1e-18 || vp < 1e-18) return 0.0;
    return c / std::sqrt(vy * vp);
}

static f64 grouped_ic(const f64* y, const f64* pred, const i64* ts, int n) {
    if (n < 3) return 0.0;
    std::vector<i32> order(n);
    std::iota(order.begin(), order.end(), 0);
    if (!ts) return pearson_ix(y, pred, order.data(), n);
    std::sort(order.begin(), order.end(), [&](i32 a, i32 b) { return ts[a] < ts[b]; });
    f64 s = 0;
    int ng = 0;
    int i = 0;
    while (i < n) {
        int j = i + 1;
        while (j < n && ts[order[j]] == ts[order[i]]) ++j;
        if (j - i >= 3) {
            s += pearson_ix(y, pred, order.data() + i, j - i);
            ng += 1;
        }
        i = j;
    }
    return ng > 0 ? s / ng : 0.0;
}

void Booster::bin_transform(const f64* X, int n, int n_f, std::vector<u16>& xb) const {
    xb.resize(static_cast<size_t>(n) * n_f);
    binner.transform(X, n, n_f, xb.data());
}

int Booster::n_heads() const {
    if (cfg.n_classes > 1) return cfg.n_classes;
    if (!cfg.quantiles.empty()) return static_cast<int>(cfg.quantiles.size());
    return std::max(1, cfg.n_horizons);
}

struct BinStore {
    std::vector<u16> ram;
    u16* ptr = nullptr;
#if defined(__linux__)
    int fd = -1;
    size_t bytes = 0;
    void* map = nullptr;
    ~BinStore() {
        if (map) munmap(map, bytes);
        if (fd >= 0) close(fd);
    }
#endif
};

static u16* alloc_bins(BinStore& st, int n, int n_f, size_t ooc_bytes) {
    const size_t nbytes = static_cast<size_t>(n) * static_cast<size_t>(n_f) * sizeof(u16);
#if defined(__linux__)
    if (nbytes >= ooc_bytes && n > 0) {
        char tmpl[] = "/tmp/tidegbm_bins_XXXXXX";
        st.fd = mkstemp(tmpl);
        if (st.fd >= 0) {
            if (ftruncate(st.fd, static_cast<off_t>(nbytes)) == 0) {
                st.map = mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, st.fd, 0);
                unlink(tmpl);
                if (st.map != MAP_FAILED) {
                    st.bytes = nbytes;
                    st.ptr = static_cast<u16*>(st.map);
                    return st.ptr;
                }
                st.map = nullptr;
            }
            close(st.fd);
            st.fd = -1;
            unlink(tmpl);
        }
    }
#else
    (void)ooc_bytes;
#endif
    st.ram.resize(static_cast<size_t>(n) * n_f);
    st.ptr = st.ram.data();
    return st.ptr;
}

void Booster::fit(const f64* X, const f64* y, int n, int n_f, const i64* timestamps, const f64* weight,
                  const int* is_cat_in, const i32* group_of_in, const int8_t* monotone_in,
                  const i64* rank_groups, const f64* X_eval, const f64* y_eval, const i64* ts_eval,
                  int n_eval, const i64* entity, const f64* row_cost) {
    cfg.n_jobs = threads_of(cfg);
    cfg.validate();
    if (n >= 1000000) {
        cfg.halo = false;
        if (cfg.forced_pairs.empty()) cfg.pair_splits = false;
    }
    obj = make_objective(cfg);
    obj.row_cost = row_cost;
    n_features = n_f;
    feature_names.resize(n_f);
    for (int i = 0; i < n_f; ++i) feature_names[i] = "f" + std::to_string(i);
    if (is_cat_in) is_cat.assign(is_cat_in, is_cat_in + n_f);
    else is_cat.assign(n_f, 0);
    if (group_of_in) group_of.assign(group_of_in, group_of_in + n_f);
    else group_of.assign(n_f, -1);
    if (monotone_in) monotone.assign(monotone_in, monotone_in + n_f);
    else monotone.assign(n_f, 0);

    std::vector<f64> X_owned, y_owned, w_owned, cost_owned;
    std::vector<i64> ts_owned, ent_owned, rg_owned;
    const f64* Xp = X;
    const f64* yp = y;
    const i64* tsp = timestamps;
    const i64* entp = entity;
    const i64* rgp = rank_groups;
    const f64* costp = row_cost;
    std::vector<f64> w(n, 1.0);
    if (weight) std::copy(weight, weight + n, w.begin());

    if (!cfg.iid_mode && timestamps && n > 1) {
        std::vector<int> ord(n);
        std::iota(ord.begin(), ord.end(), 0);
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
            if (timestamps[a] != timestamps[b]) return timestamps[a] < timestamps[b];
            if (entity) return entity[a] < entity[b];
            return a < b;
        });
        bool perm = false;
        for (int i = 0; i < n; ++i)
            if (ord[i] != i) {
                perm = true;
                break;
            }
        if (perm) {
            X_owned.resize(static_cast<size_t>(n) * n_f);
            y_owned.resize(n);
            w_owned.resize(n);
            ts_owned.resize(n);
            if (entity) ent_owned.resize(n);
            if (rank_groups) rg_owned.resize(n);
            if (row_cost) cost_owned.resize(n);
            for (int i = 0; i < n; ++i) {
                const int s = ord[i];
                std::memcpy(&X_owned[static_cast<size_t>(i) * n_f], X + static_cast<size_t>(s) * n_f,
                            sizeof(f64) * n_f);
                y_owned[i] = y[s];
                w_owned[i] = w[s];
                ts_owned[i] = timestamps[s];
                if (entity) ent_owned[i] = entity[s];
                if (rank_groups) rg_owned[i] = rank_groups[s];
                if (row_cost) cost_owned[i] = row_cost[s];
            }
            Xp = X_owned.data();
            yp = y_owned.data();
            w.swap(w_owned);
            tsp = ts_owned.data();
            if (entity) entp = ent_owned.data();
            if (rank_groups) rgp = rg_owned.data();
            if (row_cost) {
                costp = cost_owned.data();
                obj.row_cost = costp;
            }
            (void)entp;
        }
    }

    std::vector<i32> buckets(n, 0);
    int n_buckets = 1;
    i64 embargo = cfg.embargo;
    if (!cfg.iid_mode) {
        if (!tsp) throw TideError("timestamp is required unless iid_mode");
        std::vector<i64> uniq(tsp, tsp + n);
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        const int uniq_n = std::max(2, static_cast<int>(uniq.size()));
        int want = cfg.n_time_buckets;
        if (want <= 0) want = auto_time_buckets(tsp, n, std::max<i64>(cfg.horizon, 0), uniq_n);
        if (cfg.finer_torb) {
            n_buckets = std::max(2, std::min(want, uniq_n));
        } else {
            n_buckets = std::max(2, want);
        }
        time_buckets(tsp, n, n_buckets, buckets);
        recency_weights(tsp, n, cfg.time_decay, w);
        embargo += std::max<i64>(cfg.horizon, 0);
    }
    n_time_buckets = n_buckets;

    // TORB prefix bounds. Rows are sorted by timestamp above, and time_buckets assigns
    // buckets by time rank, so bucket b owns a contiguous row range and its causal past
    // {i : ts[i] < t0_b - embargo} is exactly the prefix [0, cut_b). Precomputing the cuts
    // once turns the per-tree O(n_buckets * n) rescan into a single O(n) pass.
    std::vector<int> cut_of_bucket(n_buckets, 0);
    bool torb_fast = false;
    if (!cfg.iid_mode && n > 0) {
        bool sorted = true;
        for (int i = 1; i < n; ++i)
            if (tsp[i] < tsp[i - 1]) {
                sorted = false;
                break;
            }
        if (sorted) {
            torb_fast = true;
            std::vector<int> first(n_buckets, -1);
            for (int i = 0; i < n; ++i) {
                const i32 b = buckets[i];
                if (b >= 0 && b < n_buckets && first[b] < 0) first[b] = i;
            }
            int run = 0;
            for (int b = 0; b < n_buckets; ++b) {
                int cut = 0;
                if (first[b] >= 0) {
                    const i64 lim = tsp[first[b]] - embargo;
                    cut = static_cast<int>(std::lower_bound(tsp, tsp + n, lim) - tsp);
                }
                run = std::max(run, cut);  // cuts are non-decreasing in b
                cut_of_bucket[b] = run;
            }
        }
    }

    std::vector<u8> bin_mask(n, 1);
    // Default: edges from the full purged train window (still causal vs the test fold).
    // Optional HAB-C holdout of the newest half is usually harmful in walk-forward.
    if (cfg.causal_bins && !cfg.iid_mode) {
        i32 med = n_buckets / 2;
        int kept = 0;
        for (int i = 0; i < n; ++i) {
            bin_mask[i] = buckets[i] <= med ? 1 : 0;
            kept += bin_mask[i];
        }
        if (kept < std::max(50, n / 20)) std::fill(bin_mask.begin(), bin_mask.end(), 1);
    }
    binner.fit(Xp, n, n_f, is_cat.data(), bin_mask.data(), cfg.max_bin, cfg.cat_min_count,
               cfg.bin_sample, static_cast<uint32_t>(cfg.random_state));
    n_bins = binner.max_bins();
    BinStore store;
    u16* xb = alloc_bins(store, n, n_f, cfg.ooc_bytes);
    binner.transform(Xp, n, n_f, xb);
    std::vector<u16> xcol;
    const u16* xb_col = nullptr;
    if (cfg.col_major_hist && !store.ram.empty()) {
        xcol.resize(static_cast<size_t>(n) * n_f);
        transpose_bins(xb, xcol.data(), n, n_f, cfg.n_jobs);
        xb_col = xcol.data();
    }

    const int heads = n_heads();
    const bool y_wide = (cfg.n_horizons > 1 && cfg.n_classes <= 1 && cfg.quantiles.empty());
    auto y_at = [&](int i, int head) -> f64 {
        if (y_wide) return yp[static_cast<size_t>(i) * heads + head];
        return yp[i];
    };

    std::vector<f64> pred(static_cast<size_t>(n) * heads, 0.0);
    init_prefix.assign(n_buckets, 0.0);
    init_scores.assign(heads, 0.0);
    if (std::isfinite(cfg.init_score_given)) {
        init_score = cfg.init_score_given;
        std::fill(init_scores.begin(), init_scores.end(), init_score);
        std::fill(pred.begin(), pred.end(), init_score);
    } else if (cfg.iid_mode) {
        if (heads == 1) {
            init_score = obj.init_score(yp, w.data(), n);
            init_scores[0] = init_score;
            std::fill(pred.begin(), pred.end(), init_score);
        } else {
            for (int hd = 0; hd < heads; ++hd) {
                std::vector<f64> ys(n);
                for (int i = 0; i < n; ++i) ys[i] = y_at(i, hd);
                Objective o = obj;
                if (!cfg.quantiles.empty()) o.quantile = cfg.quantiles[hd];
                init_scores[hd] = o.init_score(ys.data(), w.data(), n);
                for (int i = 0; i < n; ++i) pred[static_cast<size_t>(i) * heads + hd] = init_scores[hd];
            }
            init_score = init_scores[0];
        }
    } else {
        for (int b = 0; b < n_buckets; ++b) {
            if (torb_fast) {
                // Causal past is the contiguous prefix [0, cut) — score it in place, no copy.
                const int cut = cut_of_bucket[b];
                init_prefix[b] = cut > 0 ? obj.init_score(yp, w.data(), cut) : std::nan("");
                continue;
            }
            std::vector<f64> ys, ws;
            i64 t0 = 0;
            bool any = false;
            for (int i = 0; i < n; ++i)
                if (buckets[i] == b) {
                    if (!any) {
                        t0 = tsp[i];
                        any = true;
                    } else
                        t0 = std::min(t0, tsp[i]);
                }
            if (!any) continue;
            for (int i = 0; i < n; ++i)
                if (tsp[i] < t0 - embargo) {
                    ys.push_back(yp[i]);
                    ws.push_back(w[i]);
                }
            if (!ys.empty()) init_prefix[b] = obj.init_score(ys.data(), ws.data(), static_cast<int>(ys.size()));
            else init_prefix[b] = std::nan("");
        }
        init_score = obj.init_score(yp, w.data(), n);
        init_scores[0] = init_score;
        // Early buckets have no causal past — use the global train prior (same prior LGBM uses).
        for (int b = 0; b < n_buckets; ++b)
            if (!std::isfinite(init_prefix[b])) init_prefix[b] = init_score;
        for (int i = 0; i < n; ++i)
            for (int hd = 0; hd < heads; ++hd) pred[static_cast<size_t>(i) * heads + hd] = init_prefix[buckets[i]];
    }

    std::vector<u16> xb_eval;
    std::vector<f64> yev, wev;
    if (X_eval && n_eval > 0) {
        if (!cfg.iid_mode) {
            if (!ts_eval) throw TideError("eval_timestamp is required when iid_mode=false");
            i64 tmax = tsp[0];
            for (int i = 1; i < n; ++i) tmax = std::max(tmax, tsp[i]);
            i64 tmin_e = ts_eval[0];
            for (int i = 1; i < n_eval; ++i) tmin_e = std::min(tmin_e, ts_eval[i]);
            if (tmin_e < tmax - embargo) {
                throw TideError(
                    "eval timestamps overlap the training future. "
                    "Validation must be strictly later than train (minus embargo/horizon).");
            }
        }
        xb_eval.resize(static_cast<size_t>(n_eval) * n_f);
        binner.transform(X_eval, n_eval, n_f, xb_eval.data());
        yev.assign(y_eval, y_eval + n_eval);
        wev.assign(n_eval, 1.0);
    }

    trees.clear();
    evals = {};
    split_count.assign(n_f, 0.0);
    std::mt19937 rng(static_cast<uint32_t>(cfg.random_state));
    std::vector<f64> g(n), h(n), g_all, h_all;
    if (cfg.n_classes > 1) {
        g_all.resize(static_cast<size_t>(n) * heads);
        h_all.resize(static_cast<size_t>(n) * heads);
    }
    f64 best_valid = 1e300;
    int stall = 0;
    best_iteration = 0;
    const bool eval_ic = n_eval > 0 && (cfg.eval_metric == "ic" ||
                                        (cfg.eval_metric == "auto" &&
                                         (cfg.objective == ObjectiveKind::RankListNet ||
                                          cfg.objective == ObjectiveKind::RankPairwise || !cfg.iid_mode)));
    if (eval_ic) best_valid = -1e300;

    auto prefix_mask = [&](int b, std::vector<char>& allowed) {
        allowed.assign(n, 0);
        if (cfg.iid_mode) {
            std::fill(allowed.begin(), allowed.end(), 1);
            return;
        }
        i64 t0 = 0;
        bool any = false;
        for (int i = 0; i < n; ++i)
            if (buckets[i] == b) {
                if (!any) {
                    t0 = tsp[i];
                    any = true;
                } else
                    t0 = std::min(t0, tsp[i]);
            }
        if (!any) return;
        for (int i = 0; i < n; ++i)
            if (tsp[i] < t0 - embargo) allowed[i] = 1;
    };

    auto grow_head = [&](int head, const f64* gptr, const f64* hptr, std::mt19937& rng_ref) {
        auto sample = sample_blocks(buckets, yp, n, cfg.subsample, cfg.tail_buffer, rng_ref);
        if (cfg.causal_structure && !cfg.iid_mode && n_buckets > 2) {
            std::vector<i32> s2;
            s2.reserve(sample.size());
            for (i32 i : sample)
                if (buckets[i] < n_buckets - 1) s2.push_back(i);
            if (s2.size() >= static_cast<size_t>(2 * cfg.min_samples_leaf)) sample.swap(s2);
        }
        if (!cfg.iid_mode && cfg.structure_frac < 1.0 && n_buckets > 2) {
            const int cap = std::max(1, static_cast<int>(std::floor(cfg.structure_frac * n_buckets)));
            if (cap < n_buckets) {
                std::vector<i32> s2;
                for (i32 i : sample)
                    if (buckets[i] < cap) s2.push_back(i);
                if (s2.size() >= static_cast<size_t>(2 * cfg.min_samples_leaf)) sample.swap(s2);
            }
        }
        if (entp && cfg.entity_subsample > 0.0 && cfg.entity_subsample < 1.0) {
            std::vector<i64> ents;
            ents.reserve(sample.size());
            for (i32 i : sample) ents.push_back(entp[i]);
            std::sort(ents.begin(), ents.end());
            ents.erase(std::unique(ents.begin(), ents.end()), ents.end());
            std::shuffle(ents.begin(), ents.end(), rng_ref);
            const int keep_e =
                std::max(1, static_cast<int>(std::ceil(static_cast<f64>(ents.size()) * cfg.entity_subsample)));
            ents.resize(static_cast<size_t>(std::min(keep_e, static_cast<int>(ents.size()))));
            std::sort(ents.begin(), ents.end());
            std::vector<i32> s2;
            s2.reserve(sample.size());
            for (i32 i : sample)
                if (std::binary_search(ents.begin(), ents.end(), entp[i])) s2.push_back(i);
            if (s2.size() >= static_cast<size_t>(2 * cfg.min_samples_leaf)) sample.swap(s2);
        }
        std::vector<f64> g_goss, h_goss;
        const f64* g_use = gptr;
        const f64* h_use = hptr;
        const bool use_goss = (cfg.goss > 0) || (cfg.goss < 0 && n >= 250000);
        if (use_goss && cfg.goss_top < 1.0 && sample.size() > 64) {
            g_goss.assign(gptr, gptr + n);
            h_goss.assign(hptr, hptr + n);
            std::vector<i32> order = sample;
            std::sort(order.begin(), order.end(),
                      [&](i32 a, i32 b) { return std::abs(gptr[a]) > std::abs(gptr[b]); });
            const int n_top =
                std::min(static_cast<int>(order.size()),
                         std::max(1, static_cast<int>(std::ceil(cfg.goss_top * sample.size()))));
            std::vector<char> keep(n, 0);
            for (int i = 0; i < n_top; ++i) keep[order[i]] = 1;
            std::vector<i32> rest(order.begin() + n_top, order.end());
            int n_other = static_cast<int>(std::ceil(cfg.goss_other * sample.size()));
            n_other = std::min(n_other, static_cast<int>(rest.size()));
            std::shuffle(rest.begin(), rest.end(), rng_ref);
            const f64 scale = (1.0 - cfg.goss_top) / std::max(cfg.goss_other, 1e-12);
            for (int i = 0; i < n_other; ++i) {
                keep[rest[i]] = 1;
                g_goss[rest[i]] *= scale;
                h_goss[rest[i]] *= scale;
            }
            sample.clear();
            for (i32 i : order)
                if (keep[i]) sample.push_back(i);
            std::sort(sample.begin(), sample.end());
            g_use = g_goss.data();
            h_use = h_goss.data();
        }
        Tree tree = grow_tree(xb, xb_col, n, g_use, h_use, sample.data(), static_cast<int>(sample.size()),
                              n_f, n_bins, cfg, is_cat.data(), group_of.data(), monotone.data(),
                              rng_ref());
        std::vector<i32> nid(n);
        apply_all(tree, xb, n, n_f, n_bins, nid.data());
        const int nn = static_cast<int>(tree.nodes.size());
        tree.values.assign(nn, 0.0);
        tree.cover.assign(nn, 0);
        std::vector<f64> Gs(nn, 0), Hs(nn, 0);
        for (int i = 0; i < n; ++i) {
            Gs[nid[i]] += gptr[i];
            Hs[nid[i]] += hptr[i];
            tree.cover[nid[i]] += 1;
        }
        for (int k = 0; k < nn; ++k) tree.values[k] = leaf_value(Gs[k], Hs[k], cfg.l2, cfg.l1, cfg.max_delta_step);
        tree.n_buckets = n_buckets;
        tree.target = head;
        tree.n_targets = heads;
        tree.prefix_values.assign(static_cast<size_t>(n_buckets) * nn, 0.0);
        if (cfg.iid_mode) {
            for (int b = 0; b < n_buckets; ++b)
                for (int k = 0; k < nn; ++k) tree.prefix_values[b * nn + k] = tree.values[k];
        } else {
            const f64 min_cover = cfg.torb_min_cover > 0.0
                                     ? cfg.torb_min_cover
                                     : static_cast<f64>(2 * std::max(cfg.min_samples_leaf, 1));
            // Cuts are non-decreasing, so one sweep over the rows in time order fills every
            // bucket: carry running per-node sums and snapshot them at each cut. Accumulation
            // order matches the old per-bucket scan, so leaf values are bit-identical.
            std::vector<f64> pG(nn, 0), pH(nn, 0);
            int swept = 0;
            for (int b = 0; b < n_buckets; ++b) {
                f64 cover;
                if (torb_fast) {
                    const int cut = cut_of_bucket[b];
                    for (; swept < cut; ++swept) {
                        pG[nid[swept]] += gptr[swept];
                        pH[nid[swept]] += hptr[swept];
                    }
                    cover = static_cast<f64>(cut);
                } else {
                    std::vector<char> allowed;
                    prefix_mask(b, allowed);
                    std::fill(pG.begin(), pG.end(), 0.0);
                    std::fill(pH.begin(), pH.end(), 0.0);
                    cover = 0.0;
                    for (int i = 0; i < n; ++i)
                        if (allowed[i]) {
                            pG[nid[i]] += gptr[i];
                            pH[nid[i]] += hptr[i];
                            cover += 1.0;
                        }
                }
                // Sparse causal past → blend toward full-train leaf (avoids zeroed early buckets).
                const f64 alpha = cover <= 0.0 ? 0.0 : std::min(1.0, cover / std::max(min_cover, 1.0));
                for (int k = 0; k < nn; ++k) {
                    const f64 full = tree.values[k];
                    const f64 pref =
                        cover > 0.0 ? leaf_value(pG[k], pH[k], cfg.l2, cfg.l1, cfg.max_delta_step) : full;
                    tree.prefix_values[static_cast<size_t>(b) * nn + k] = alpha * pref + (1.0 - alpha) * full;
                }
            }
        }
        for (const auto& nd : tree.nodes) {
            if (nd.is_leaf || nd.feature < 0) continue;
            split_count[nd.feature] += 1;
            if (nd.feature2 >= 0) split_count[nd.feature2] += 0.5;
        }
        trees.push_back(std::move(tree));
        const Tree& T = trees.back();
        const f64 lr = cfg.learning_rate;
        if (cfg.iid_mode) {
            for (int i = 0; i < n; ++i) pred[static_cast<size_t>(i) * heads + head] += lr * T.values[nid[i]];
        } else {
            const int nn2 = static_cast<int>(T.nodes.size());
            for (int i = 0; i < n; ++i)
                pred[static_cast<size_t>(i) * heads + head] +=
                    lr * T.prefix_values[static_cast<size_t>(buckets[i]) * nn2 + nid[i]];
        }
    };

    for (int m = 0; m < cfg.n_estimators; ++m) {
        if (cfg.n_classes > 1) {
            obj.grad_hess_multiclass(yp, pred.data(), w.data(), n, heads, g_all.data(), h_all.data());
            for (int hd = 0; hd < heads; ++hd) {
                for (int i = 0; i < n; ++i) {
                    g[i] = g_all[static_cast<size_t>(i) * heads + hd];
                    h[i] = h_all[static_cast<size_t>(i) * heads + hd];
                }
                grow_head(hd, g.data(), h.data(), rng);
            }
        } else if (!cfg.quantiles.empty()) {
            for (int hd = 0; hd < heads; ++hd) {
                Objective o = obj;
                o.kind = ObjectiveKind::Quantile;
                o.quantile = cfg.quantiles[hd];
                std::vector<f64> ph(n);
                for (int i = 0; i < n; ++i) ph[i] = pred[static_cast<size_t>(i) * heads + hd];
                o.grad_hess(yp, ph.data(), w.data(), rgp, n, g.data(), h.data());
                grow_head(hd, g.data(), h.data(), rng);
            }
        } else if (y_wide) {
            for (int hd = 0; hd < heads; ++hd) {
                std::vector<f64> yh(n), ph(n);
                for (int i = 0; i < n; ++i) {
                    yh[i] = y_at(i, hd);
                    ph[i] = pred[static_cast<size_t>(i) * heads + hd];
                }
                obj.grad_hess(yh.data(), ph.data(), w.data(), rgp, n, g.data(), h.data());
                grow_head(hd, g.data(), h.data(), rng);
            }
        } else if (cfg.objective == ObjectiveKind::Custom && custom_grad) {
            custom_grad(yp, pred.data(), w.data(), n, g.data(), h.data());
            grow_head(0, g.data(), h.data(), rng);
        } else {
            obj.grad_hess(yp, pred.data(), w.data(), rgp, n, g.data(), h.data());
            grow_head(0, g.data(), h.data(), rng);
        }
        const f64 tr = (heads == 1) ? obj.loss(yp, pred.data(), w.data(), rgp, n)
                                    : obj.loss(yp, pred.data(), w.data(), rgp, n);
        evals.train.push_back(tr);
        if (n_eval > 0) {
            std::vector<f64> ev(static_cast<size_t>(n_eval) * heads);
            predict_multi_binned(xb_eval.data(), n_eval, ev.data());
            f64 vl;
            if (eval_ic) {
                std::vector<f64> ev0(n_eval);
                for (int i = 0; i < n_eval; ++i) ev0[i] = ev[static_cast<size_t>(i) * heads];
                vl = grouped_ic(yev.data(), ev0.data(), ts_eval, n_eval);
            } else {
                vl = obj.loss(yev.data(), ev.data(), wev.data(), nullptr, n_eval);
            }
            evals.valid.push_back(vl);
            const bool improved = eval_ic ? (vl > best_valid + 1e-12) : (vl + 1e-12 < best_valid);
            if (improved) {
                best_valid = vl;
                stall = 0;
                best_iteration = m + 1;
            } else {
                stall += 1;
                if (cfg.early_stopping_rounds > 0 && stall >= cfg.early_stopping_rounds) {
                    const int keep = best_iteration * heads;
                    if (keep > 0 && keep < static_cast<int>(trees.size())) trees.resize(keep);
                    break;
                }
            }
        }
        if (cfg.verbose > 0 && ((m + 1) % cfg.verbose == 0 || m == 0)) {
            const f64 vl = evals.valid.empty() ? 0.0 : evals.valid.back();
            std::fprintf(stderr, "[TideGBM] iter %d train %g valid %g\n", m + 1, tr, vl);
            if (log_fn) log_fn(m + 1, tr, vl);
        }
    }
    if (best_iteration <= 0) best_iteration = cfg.n_estimators;
    obj.row_cost = nullptr;
    fitted = true;
}

void Booster::predict_raw_binned(const u16* xb, int n, f64* out) const {
    const int heads = std::max(1, n_heads());
    const int n_trees = static_cast<int>(trees.size());
    const int nth = cfg.n_jobs > 0 ? cfg.n_jobs : 1;
    const int use_rounds = best_iteration > 0 ? best_iteration : cfg.n_estimators;
    const int use = std::min(n_trees, heads == 1 ? use_rounds : use_rounds * heads);
    std::vector<int> tids;
    std::vector<f64> lrs;
    tids.reserve(use);
    lrs.reserve(use);
    for (int t = 0; t < use; ++t) {
        if (trees[t].target != 0 && heads > 1) continue;
        const f64 decay = cfg.tree_decay < 1.0 ? std::pow(cfg.tree_decay, use - 1 - t) : 1.0;
        tids.push_back(t);
        lrs.push_back(cfg.learning_rate * decay);
    }
    const int nt = static_cast<int>(tids.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nth) if (n > 512 && nth > 1)
#endif
    for (int i = 0; i < n; ++i) {
        const u16* row = xb + static_cast<size_t>(i) * n_features;
        f64 s = init_score;
        for (int k = 0; k < nt; ++k) {
            const int nid = apply_one(trees[tids[k]], row, n_features, n_bins);
            s += lrs[k] * trees[tids[k]].values[nid];
        }
        out[i] = s;
    }
}

void Booster::predict_multi_binned(const u16* xb, int n, f64* out) const {
    const int heads = std::max(1, n_heads());
    const int nth = cfg.n_jobs > 0 ? cfg.n_jobs : 1;
    const int n_trees = static_cast<int>(trees.size());
    const int use_rounds = best_iteration > 0 ? best_iteration : cfg.n_estimators;
    const int use = std::min(n_trees, heads == 1 ? use_rounds : use_rounds * heads);
    std::vector<f64> lrs(use);
    for (int t = 0; t < use; ++t) {
        const f64 decay = cfg.tree_decay < 1.0 ? std::pow(cfg.tree_decay, use - 1 - t) : 1.0;
        lrs[t] = cfg.learning_rate * decay;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nth) if (n > 512 && nth > 1)
#endif
    for (int i = 0; i < n; ++i) {
        const u16* row = xb + static_cast<size_t>(i) * n_features;
        for (int h = 0; h < heads; ++h)
            out[static_cast<size_t>(i) * heads + h] =
                h < static_cast<int>(init_scores.size()) ? init_scores[h] : init_score;
        for (int t = 0; t < use; ++t) {
            const int nid = apply_one(trees[t], row, n_features, n_bins);
            out[static_cast<size_t>(i) * heads + trees[t].target] += lrs[t] * trees[t].values[nid];
        }
    }
}

void Booster::predict_binned(const u16* xb, int n, f64* out) const {
    predict_raw_binned(xb, n, out);
}

void Booster::predict_raw(const f64* X, int n, int n_f, f64* out) const {
    if (!fitted) throw TideError("not fitted");
    if (n_f != n_features) throw TideError("n_features mismatch");
    std::vector<u16> xb;
    bin_transform(X, n, n_f, xb);
    predict_raw_binned(xb.data(), n, out);
}

void Booster::predict_multi(const f64* X, int n, int n_f, f64* out) const {
    if (!fitted) throw TideError("not fitted");
    if (n_f != n_features) throw TideError("n_features mismatch");
    std::vector<u16> xb;
    bin_transform(X, n, n_f, xb);
    predict_multi_binned(xb.data(), n, out);
}

void Booster::predict(const f64* X, int n, int n_f, f64* out) const {
    const int heads = std::max(1, n_heads());
    if (heads > 1 && cfg.n_classes > 1) {
        std::vector<f64> raw(static_cast<size_t>(n) * heads);
        predict_multi(X, n, n_f, raw.data());
        obj.softmax_rows(raw.data(), n, heads, raw.data());
        for (int i = 0; i < n; ++i) {
            int arg = 0;
            f64 best = raw[static_cast<size_t>(i) * heads];
            for (int c = 1; c < heads; ++c)
                if (raw[static_cast<size_t>(i) * heads + c] > best) {
                    best = raw[static_cast<size_t>(i) * heads + c];
                    arg = c;
                }
            out[i] = static_cast<f64>(arg);
        }
        return;
    }
    std::vector<f64> raw(n);
    predict_raw(X, n, n_f, raw.data());
    obj.finalize(raw.data(), n, out);
}

void Booster::update(const f64* X, const f64* y, int n, int n_f, const f64* weight,
                     const i64* rank_groups, int extra_trees) {
    if (!fitted) throw TideError("not fitted");
    if (n_f != n_features) throw TideError("n_features mismatch");
    std::vector<u16> xb;
    bin_transform(X, n, n_f, xb);  // frozen binner
    std::vector<u16> xcol;
    const u16* xb_col = nullptr;
    if (cfg.col_major_hist) {
        xcol.resize(static_cast<size_t>(n) * n_f);
        transpose_bins(xb.data(), xcol.data(), n, n_f, cfg.n_jobs);
        xb_col = xcol.data();
    }
    std::vector<f64> pred(n), g(n), h(n), w(n, 1.0);
    if (weight) std::copy(weight, weight + n, w.begin());
    predict_raw_binned(xb.data(), n, pred.data());
    std::mt19937 rng(static_cast<uint32_t>(cfg.random_state) + static_cast<uint32_t>(trees.size()));
    std::vector<i32> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    for (int m = 0; m < extra_trees; ++m) {
        obj.grad_hess(y, pred.data(), w.data(), rank_groups, n, g.data(), h.data());
        Tree tree = grow_tree(xb.data(), xb_col, n, g.data(), h.data(), idx.data(), n, n_f, n_bins, cfg,
                              is_cat.data(), group_of.data(), monotone.data(), rng());
        std::vector<i32> nid(n);
        apply_all(tree, xb.data(), n, n_f, n_bins, nid.data());
        const int nn = static_cast<int>(tree.nodes.size());
        tree.values.assign(nn, 0);
        std::vector<f64> Gs(nn, 0), Hs(nn, 0);
        for (int i = 0; i < n; ++i) {
            Gs[nid[i]] += g[i];
            Hs[nid[i]] += h[i];
        }
        for (int k = 0; k < nn; ++k) tree.values[k] = leaf_value(Gs[k], Hs[k], cfg.l2, cfg.l1, cfg.max_delta_step);
        tree.n_buckets = 1;
        tree.prefix_values = tree.values;
        trees.push_back(std::move(tree));
        const Tree& T = trees.back();
        for (int i = 0; i < n; ++i) pred[i] += cfg.learning_rate * T.values[nid[i]];
    }
    best_iteration = static_cast<int>(trees.size());
}

void Booster::feature_importance(f64* out, const char* kind) const {
    std::fill(out, out + n_features, 0.0);
    const bool split = kind && std::string(kind) == "split";
    if (split && !split_count.empty()) {
        std::copy(split_count.begin(), split_count.end(), out);
    } else {
        for (const auto& t : trees) {
            for (const auto& n : t.nodes) {
                if (n.is_leaf || n.feature < 0) continue;
                out[n.feature] += std::max(n.gain, 0.0);
                if (n.feature2 >= 0) out[n.feature2] += 0.5 * std::max(n.gain, 0.0);
            }
        }
    }
    f64 s = 0;
    for (int i = 0; i < n_features; ++i) s += out[i];
    if (s > 0)
        for (int i = 0; i < n_features; ++i) out[i] /= s;
}

static f64 node_expect(const Tree& t, int nid);

static f64 node_expect(const Tree& t, int nid) {
    const Node& n = t.nodes[nid];
    if (n.is_leaf) return t.values.empty() ? n.value : t.values[nid];
    const f64 cl = std::max(t.cover[n.left], 1);
    const f64 cr = std::max(t.cover[n.right], 1);
    return (cl * node_expect(t, n.left) + cr * node_expect(t, n.right)) / (cl + cr);
}

void Booster::pred_contrib(const f64* X, int n, int n_f, f64* out) const {
    // Saabas path attributions + base. Shape (n, n_f+1), last col = base.
    if (n_f != n_features) throw TideError("n_features mismatch");
    std::vector<u16> xb;
    bin_transform(X, n, n_f, xb);
    const int cols = n_features + 1;
    std::fill(out, out + static_cast<size_t>(n) * cols, 0.0);
    f64 base = init_score;
    for (const auto& t : trees) base += cfg.learning_rate * node_expect(t, 0);
    for (int i = 0; i < n; ++i) {
        out[static_cast<size_t>(i) * cols + n_features] = base;
        const u16* row = xb.data() + static_cast<size_t>(i) * n_features;
        for (const auto& t : trees) {
            int nid = 0;
            f64 parent_e = node_expect(t, 0);
            for (;;) {
                const Node& nd = t.nodes[nid];
                if (nd.is_leaf) break;
                const int next = apply_one(t, row, n_features, n_bins);
                // walk one step
                bool go_left = false;
                if (nd.split_type == SplitType::Pair) {
                    auto b1 = row[nd.feature], b2 = row[nd.feature2];
                    if (b1 == kMissingBin || b2 == kMissingBin) go_left = nd.missing_go_left;
                    else go_left = pair_bin(b1, b2, n_bins) <= static_cast<u16>(nd.threshold_bin);
                } else if (nd.split_type == SplitType::Cat || nd.split_type == SplitType::Regime) {
                    go_left = cat_goes_left(nd, row[nd.feature]);
                } else {
                    auto b = row[nd.feature];
                    if (b == kMissingBin) go_left = nd.missing_go_left;
                    else go_left = b <= static_cast<u16>(nd.threshold_bin);
                }
                const int child = go_left ? nd.left : nd.right;
                const f64 child_e = node_expect(t, child);
                const f64 d = cfg.learning_rate * (child_e - parent_e);
                out[static_cast<size_t>(i) * cols + nd.feature] += d;
                if (nd.feature2 >= 0) out[static_cast<size_t>(i) * cols + nd.feature2] += 0.5 * d;
                parent_e = child_e;
                nid = child;
                (void)next;
            }
        }
    }
}

}  // namespace tidegbm
