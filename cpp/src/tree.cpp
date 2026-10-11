#include "tidegbm/tree.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tidegbm {

int apply_one(const Tree& t, const u16* row, int n_features, int n_bins) {
    int nid = 0;
    for (;;) {
        const Node& n = t.nodes[nid];
        if (n.is_leaf) return nid;
        bool go_left = false;
        if (n.split_type == SplitType::Pair) {
            const u16 b1 = row[n.feature];
            const u16 b2 = row[n.feature2];
            if (b1 == kMissingBin || b2 == kMissingBin) {
                go_left = n.missing_go_left != 0;
            } else {
                go_left = pair_bin(b1, b2, n_bins) <= static_cast<u16>(n.threshold_bin);
            }
        } else if (n.split_type == SplitType::Cat || n.split_type == SplitType::Regime) {
            go_left = cat_goes_left(n, row[n.feature]);
        } else {
            const u16 b = row[n.feature];
            if (b == kMissingBin) {
                go_left = n.missing_go_left != 0;
            } else {
                go_left = b <= static_cast<u16>(n.threshold_bin);
            }
        }
        nid = go_left ? n.left : n.right;
    }
}

void apply_all(const Tree& t, const u16* xb, int n, int n_features, int n_bins, i32* node_id) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        if (i + 16 < n) {
#if defined(__GNUC__)
            __builtin_prefetch(xb + static_cast<size_t>(i + 16) * n_features, 0, 3);
#endif
        }
        node_id[i] = apply_one(t, xb + static_cast<size_t>(i) * n_features, n_features, n_bins);
    }
}

void predict_tree(const Tree& t, const u16* xb, int n, int n_features, int n_bins, const f64* values,
                  f64* out) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        const int nid = apply_one(t, xb + static_cast<size_t>(i) * n_features, n_features, n_bins);
        out[i] = values[nid];
    }
}

static void split_indices(const u16* xb, int n_features, int n_bins, const i32* idx, int n_idx,
                          const Node& proto, std::vector<i32>& left, std::vector<i32>& right) {
    left.clear();
    right.clear();
    left.reserve(n_idx / 2);
    right.reserve(n_idx / 2);
    for (int k = 0; k < n_idx; ++k) {
        const i32 i = idx[k];
        const u16* row = xb + static_cast<size_t>(i) * n_features;
        bool go_left = false;
        if (proto.split_type == SplitType::Pair) {
            const u16 b1 = row[proto.feature], b2 = row[proto.feature2];
            if (b1 == kMissingBin || b2 == kMissingBin) go_left = proto.missing_go_left != 0;
            else go_left = pair_bin(b1, b2, n_bins) <= static_cast<u16>(proto.threshold_bin);
        } else if (proto.split_type == SplitType::Cat || proto.split_type == SplitType::Regime) {
            go_left = cat_goes_left(proto, row[proto.feature]);
        } else {
            const u16 b = row[proto.feature];
            if (b == kMissingBin) go_left = proto.missing_go_left != 0;
            else go_left = b <= static_cast<u16>(proto.threshold_bin);
        }
        (go_left ? left : right).push_back(i);
    }
}

static void fill_hist(const Config& cfg, const u16* xb_row, const u16* xb_col, int n_rows,
                      int n_features, int n_bins, const i32* idx, int n_idx, const f64* g,
                      const f64* h, Hist& hist) {
    hist.resize(n_features, n_bins);
    if (cfg.device == "gpu") {
        build_hist_gpu(xb_row, n_features, idx, n_idx, g, h, hist, n_rows);
    } else if (xb_col) {
        build_hist_col(xb_col, n_rows, n_features, idx, n_idx, g, h, hist, cfg.n_jobs);
    } else {
        build_hist(xb_row, n_features, idx, n_idx, g, h, hist, cfg.n_jobs);
    }
}

static SplitCand best_for_leaf(const u16* xb_row, const u16* xb_col, int n_rows, const f64* g,
                               const f64* h, const i32* idx, int n_idx, int n_features, int n_bins,
                               const Config& cfg, const int* is_cat, const i32* group_of,
                               const int8_t* monotone, const std::vector<int>& active, int pair_budget,
                               f64 parent_energy, Hist* cached) {
    Hist local;
    Hist* histp = cached;
    if (!histp || histp->n_features != n_features) {
        local.resize(n_features, n_bins);
        fill_hist(cfg, xb_row, xb_col, n_rows, n_features, n_bins, idx, n_idx, g, h, local);
        histp = &local;
    }
    Hist& hist = *histp;

    std::vector<SplitCand> per_f(n_features);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (static_cast<int>(active.size()) > 4 && cfg.n_jobs != 1)
#endif
    for (int ai = 0; ai < static_cast<int>(active.size()); ++ai) {
        const int f = active[ai];
        if (is_cat[f]) {
            per_f[f] = best_categorical(hist, f, cfg.l2, cfg.min_split_gain, cfg.min_samples_leaf,
                                        cfg.min_child_weight, cfg.cat_smooth, cfg.l1);
        } else {
            per_f[f] = best_numeric_missing(hist, f, cfg.l2, cfg.min_split_gain, cfg.min_samples_leaf,
                                            cfg.min_child_weight, monotone[f], cfg.l1);
        }
    }

    std::unordered_map<int, int> group_best;
    std::vector<int> allowed;
    allowed.reserve(active.size());
    for (int f : active) {
        const int grp = group_of ? group_of[f] : -1;
        if (grp >= 0) {
            auto it = group_best.find(grp);
            if (it == group_best.end()) {
                group_best[grp] = f;
            } else if (per_f[f].gain > per_f[it->second].gain) {
                it->second = f;
            }
        }
    }
    std::vector<char> seen_g(256, 0);
    for (int f : active) {
        const int grp = group_of ? group_of[f] : -1;
        if (grp >= 0) {
            if (grp < 256 && seen_g[grp]) continue;
            if (grp < 256) seen_g[grp] = 1;
            f = group_best[grp];
        }
        allowed.push_back(f);
    }

    SplitCand best;
    std::vector<SplitCand> cands;
    for (int f : allowed) {
        if (per_f[f].feature < 0 || per_f[f].gain < cfg.min_split_gain) continue;
        cands.push_back(per_f[f]);
        if (per_f[f].gain > best.gain) best = per_f[f];
    }

    if (cfg.regime_priority && cfg.regime_feature >= 0 && cfg.regime_feature < n_features) {
        const int rf = cfg.regime_feature;
        if (per_f[rf].feature >= 0 && per_f[rf].gain > 0) {
            if (best.feature < 0 || per_f[rf].gain >= 0.5 * std::max(best.gain, 0.0)) {
                best = per_f[rf];
                best.type = is_cat[rf] ? SplitType::Regime : SplitType::Axis;
            }
        }
    }

    if (cfg.halo && !cands.empty()) {
        std::sort(cands.begin(), cands.end(),
                  [](const SplitCand& a, const SplitCand& b) { return a.gain > b.gain; });
        const int topn = std::min(cfg.halo_candidates, static_cast<int>(cands.size()));
        for (int t = 0; t < topn; ++t) {
            Node proto;
            proto.split_type = cands[t].type;
            proto.feature = cands[t].feature;
            proto.threshold_bin = cands[t].threshold_bin;
            proto.missing_go_left = cands[t].missing_go_left;
            for (int k = 0; k < 4; ++k) proto.cat_left[k] = cands[t].cat_left[k];
            std::vector<i32> L, R;
            split_indices(xb_row, n_features, n_bins, idx, n_idx, proto, L, R);
            f64 extra = 0;
            for (auto* ch : {&L, &R}) {
                if (static_cast<int>(ch->size()) < 2 * cfg.min_samples_leaf) continue;
                Hist hc;
                hc.resize(n_features, n_bins);
                fill_hist(cfg, xb_row, xb_col, n_rows, n_features, n_bins, ch->data(),
                          static_cast<int>(ch->size()), g, h, hc);
                f64 cg = 0;
                for (int f : allowed) {
                    auto c = is_cat[f] ? best_categorical(hc, f, cfg.l2, 0, cfg.min_samples_leaf,
                                                          cfg.min_child_weight, cfg.cat_smooth, cfg.l1)
                                       : best_numeric_missing(hc, f, cfg.l2, 0, cfg.min_samples_leaf,
                                                              cfg.min_child_weight, monotone[f],
                                                              cfg.l1);
                    cg = std::max(cg, std::max(c.gain, 0.0));
                }
                extra += cg;
            }
            cands[t].halo = cands[t].gain + cfg.halo_gamma * extra;
        }
        best = *std::max_element(cands.begin(), cands.begin() + topn,
                                 [](const SplitCand& a, const SplitCand& b) { return a.halo < b.halo; });
    }

    const bool weak = (best.gain < cfg.pair_trigger * std::max(parent_energy * 0.01, 1e-12)) ||
                      best.gain <= 0;
    if (cfg.pair_splits && pair_budget > 0 && (allowed.size() >= 2 || !cfg.forced_pairs.empty()) &&
        weak) {
        std::vector<std::pair<int, int>> pairs;
        if (!cfg.forced_pairs.empty()) {
            pairs = cfg.forced_pairs;
        } else {
            std::vector<int> top = allowed;
            std::sort(top.begin(), top.end(),
                      [&](int a, int b) { return per_f[a].gain > per_f[b].gain; });
            const int k = std::min(cfg.pair_top_k, static_cast<int>(top.size()));
            for (int a = 0; a < k; ++a)
                for (int b = a + 1; b < k; ++b) pairs.emplace_back(top[a], top[b]);
        }
        std::vector<f64> gL(n_idx), hL(n_idx);
        for (int i = 0; i < n_idx; ++i) {
            gL[i] = g[idx[i]];
            hL[i] = h[idx[i]];
        }
        const int n_pairs = static_cast<int>(pairs.size());
        std::vector<SplitCand> pair_best(n_pairs);
#ifdef _OPENMP
        const int n_scratch = std::max(1, omp_get_max_threads());
#else
        const int n_scratch = 1;
#endif
        std::vector<Hist> pair_scratch(n_scratch);
        // Hashing the pair bin and accumulating it are fused into one pass over the leaf
        // (was: materialise z, then re-read it to build the histogram). Bins are visited in
        // the same order, so the resulting candidate is bit-identical to the two-pass form.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if (n_pairs > 1 && cfg.n_jobs != 1)
#endif
        for (int pi = 0; pi < n_pairs; ++pi) {
            const int f1 = pairs[pi].first, f2 = pairs[pi].second;
            if (f1 < 0 || f2 < 0 || f1 >= n_features || f2 >= n_features || f1 == f2) continue;
#ifdef _OPENMP
            const int slot = std::min(omp_get_thread_num(), n_scratch - 1);
#else
            const int slot = 0;
#endif
            Hist& hz = pair_scratch[slot];
            if (hz.n_bins != n_bins || hz.n_features != 1) hz.resize(1, n_bins);
            hz.zero();  // resize allocates only; this loop accumulates, so it must start clean
            HBin* HB = hz.B.data();
            for (int i = 0; i < n_idx; ++i) {
                const u16* row = xb_row + static_cast<size_t>(idx[i]) * n_features;
                const u16 b = pair_bin(row[f1], row[f2], n_bins);
                HB[b].g += gL[i];
                HB[b].h += hL[i];
                HB[b].c += 1;
            }
            auto cand = best_numeric_missing(hz, 0, cfg.pair_l2, cfg.min_split_gain,
                                             cfg.min_samples_leaf, cfg.min_child_weight, 0, cfg.l1);
            cand.type = SplitType::Pair;
            cand.feature = f1;
            cand.feature2 = f2;
            pair_best[pi] = cand;
        }
        for (int pi = 0; pi < n_pairs; ++pi) {
            if (pair_best[pi].feature2 < 0) continue;  // skipped or no valid split
            if (pair_best[pi].gain > best.gain) best = pair_best[pi];
        }
    }
    return best;
}

Tree grow_tree(const u16* xb_row, const u16* xb_col, int n_rows, const f64* g, const f64* h,
               const i32* sample_idx, int n_idx, int n_features, int n_bins, const Config& cfg,
               const int* is_cat, const i32* group_of, const int8_t* monotone, uint32_t seed) {
    Tree tree;
    tree.nodes.reserve(static_cast<size_t>(cfg.max_leaves) * 2);
    std::mt19937 rng(seed);
    std::vector<i32> sorted_idx(sample_idx, sample_idx + n_idx);
    std::sort(sorted_idx.begin(), sorted_idx.end());
    sample_idx = sorted_idx.data();
    std::vector<int> active(n_features);
    std::iota(active.begin(), active.end(), 0);
    if (cfg.colsample < 1.0 && n_features > 1) {
        const int k = std::max(1, static_cast<int>(std::ceil(n_features * cfg.colsample)));
        std::shuffle(active.begin(), active.end(), rng);
        active.resize(k);
    }
    if (cfg.regime_feature >= 0 && cfg.regime_feature < n_features) {
        if (std::find(active.begin(), active.end(), cfg.regime_feature) == active.end())
            active.push_back(cfg.regime_feature);
    }

    auto filter_active = [&](int ixg) {
        if (ixg < 0 || cfg.interaction_group.empty()) return active;
        std::vector<int> out;
        for (int f : active) {
            const int g = f < static_cast<int>(cfg.interaction_group.size()) ? cfg.interaction_group[f]
                                                                            : -1;
            if (g < 0 || g == ixg) out.push_back(f);
        }
        if (cfg.regime_feature >= 0) {
            if (std::find(out.begin(), out.end(), cfg.regime_feature) == out.end())
                out.push_back(cfg.regime_feature);
        }
        return out.empty() ? active : out;
    };

    Node root;
    root.is_leaf = true;
    root.n_samples = n_idx;
    root.depth = 0;
    f64 sg = 0, sh = 0;
    for (int k = 0; k < n_idx; ++k) {
        sg += g[sample_idx[k]];
        sh += h[sample_idx[k]];
    }
    root.value = leaf_value(sg, sh, cfg.l2, cfg.l1, cfg.max_delta_step);
    tree.nodes.push_back(root);
    std::vector<std::vector<i32>> leaf_idx(1);
    leaf_idx.reserve(static_cast<size_t>(cfg.max_leaves) * 2);
    leaf_idx[0].assign(sample_idx, sample_idx + n_idx);
    std::vector<Hist> leaf_hist(1);
    leaf_hist[0].resize(n_features, n_bins);
    fill_hist(cfg, xb_row, xb_col, n_rows, n_features, n_bins, sample_idx, n_idx, g, h, leaf_hist[0]);
    std::vector<int> leaf_ixg(1, -1);

    std::vector<SplitCand> cached_split(1);
    std::vector<char> split_ok(1, 0);
    const bool cache_splits = cfg.colsample_bynode >= 1.0;

    int pair_used = 0;
    for (int iter = 0; iter < cfg.max_leaves - 1; ++iter) {
        int best_leaf = -1;
        SplitCand best;
        f64 best_score = cfg.min_split_gain;
        for (int lid = 0; lid < static_cast<int>(tree.nodes.size()); ++lid) {
            if (!tree.nodes[lid].is_leaf) continue;
            if (tree.nodes[lid].depth >= cfg.max_depth) continue;
            auto& idxv = leaf_idx[lid];
            if (static_cast<int>(idxv.size()) < 2 * cfg.min_samples_leaf) continue;
            SplitCand rec;
            if (cache_splits && split_ok[lid]) {
                rec = cached_split[lid];
            } else {
                f64 energy = 0;
                for (i32 i : idxv) energy += g[i] * g[i];
                auto act = filter_active(leaf_ixg[lid]);
                if (cfg.colsample_bynode < 1.0 && static_cast<int>(act.size()) > 1) {
                    std::shuffle(act.begin(), act.end(), rng);
                    const int k =
                        std::max(1, static_cast<int>(std::ceil(act.size() * cfg.colsample_bynode)));
                    act.resize(k);
                }
                rec = best_for_leaf(xb_row, xb_col, n_rows, g, h, idxv.data(),
                                    static_cast<int>(idxv.size()), n_features, n_bins, cfg, is_cat,
                                    group_of, monotone, act, cfg.max_pair_splits_per_tree - pair_used,
                                    energy, &leaf_hist[lid]);
                if (cache_splits) {
                    if (static_cast<int>(cached_split.size()) <= lid) cached_split.resize(lid + 1);
                    if (static_cast<int>(split_ok.size()) <= lid) split_ok.resize(lid + 1, 0);
                    cached_split[lid] = rec;
                    split_ok[lid] = 1;
                }
            }
            const f64 score = cfg.halo ? rec.halo : rec.gain;
            if (rec.feature >= 0 && rec.gain >= cfg.min_split_gain && score > best_score) {
                best_score = score;
                best_leaf = lid;
                best = rec;
            }
        }
        if (best_leaf < 0) break;

        tree.nodes[best_leaf].is_leaf = false;
        tree.nodes[best_leaf].feature = best.feature;
        tree.nodes[best_leaf].feature2 = best.feature2;
        tree.nodes[best_leaf].threshold_bin = best.threshold_bin;
        tree.nodes[best_leaf].split_type = best.type;
        tree.nodes[best_leaf].missing_go_left = best.missing_go_left;
        tree.nodes[best_leaf].gain = best.gain;
        for (int k = 0; k < 4; ++k) tree.nodes[best_leaf].cat_left[k] = best.cat_left[k];
        if (best.type == SplitType::Pair) {
            pair_used += 1;
            std::fill(split_ok.begin(), split_ok.end(), 0);
        }
        if (best_leaf < static_cast<int>(split_ok.size())) split_ok[best_leaf] = 0;

        std::vector<i32> L, R;
        split_indices(xb_row, n_features, n_bins, leaf_idx[best_leaf].data(),
                      static_cast<int>(leaf_idx[best_leaf].size()), tree.nodes[best_leaf], L, R);
        Node left, right;
        left.is_leaf = right.is_leaf = true;
        left.depth = right.depth = tree.nodes[best_leaf].depth + 1;
        left.n_samples = static_cast<int>(L.size());
        right.n_samples = static_cast<int>(R.size());
        auto fill_val = [&](Node& nd, const std::vector<i32>& ix) {
            f64 G = 0, H = 0;
            for (i32 i : ix) {
                G += g[i];
                H += h[i];
            }
            nd.value = leaf_value(G, H, cfg.l2, cfg.l1, cfg.max_delta_step);
        };
        fill_val(left, L);
        fill_val(right, R);

        // Build only the smaller child, then turn the parent's own buffer into the larger one
        // by subtracting in place: one histogram allocation per split instead of three.
        Hist hL, hR;
        Hist parent_hist = std::move(leaf_hist[best_leaf]);
        Hist small;
        const bool left_is_small = L.size() <= R.size();
        const std::vector<i32>& small_ix = left_is_small ? L : R;
        small.resize(n_features, n_bins);
        fill_hist(cfg, xb_row, xb_col, n_rows, n_features, n_bins, small_ix.data(),
                  static_cast<int>(small_ix.size()), g, h, small);
        subtract_hist_inplace(parent_hist, small);
        if (left_is_small) {
            hL = std::move(small);
            hR = std::move(parent_hist);
        } else {
            hR = std::move(small);
            hL = std::move(parent_hist);
        }
        leaf_hist[best_leaf] = Hist();

        int child_ixg = leaf_ixg[best_leaf];
        if (child_ixg < 0 && best.feature >= 0 &&
            best.feature < static_cast<int>(cfg.interaction_group.size())) {
            const int g = cfg.interaction_group[best.feature];
            if (g >= 0) child_ixg = g;
        }

        const int left_id = static_cast<int>(tree.nodes.size());
        const int right_id = left_id + 1;
        tree.nodes[best_leaf].left = left_id;
        tree.nodes[best_leaf].right = right_id;
        tree.nodes.push_back(left);
        leaf_idx.push_back(std::move(L));
        leaf_hist.push_back(std::move(hL));
        leaf_ixg.push_back(child_ixg);
        tree.nodes.push_back(right);
        leaf_idx.push_back(std::move(R));
        leaf_hist.push_back(std::move(hR));
        leaf_ixg.push_back(child_ixg);
        cached_split.emplace_back();
        cached_split.emplace_back();
        split_ok.push_back(0);
        split_ok.push_back(0);
        leaf_idx[best_leaf].clear();
        leaf_idx[best_leaf].shrink_to_fit();
    }

    tree.values.resize(tree.nodes.size());
    tree.cover.assign(tree.nodes.size(), 0);
    for (size_t i = 0; i < tree.nodes.size(); ++i) {
        tree.values[i] = tree.nodes[i].value;
        tree.cover[i] = tree.nodes[i].n_samples;
    }
    return tree;
}

}  // namespace tidegbm
