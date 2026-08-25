#include "tidegbm/booster.hpp"

#include <cstring>
#include <sstream>
#include <vector>

namespace tidegbm {

static void put_u32(std::string& s, uint32_t v) {
    char b[4];
    b[0] = static_cast<char>(v & 255);
    b[1] = static_cast<char>((v >> 8) & 255);
    b[2] = static_cast<char>((v >> 16) & 255);
    b[3] = static_cast<char>((v >> 24) & 255);
    s.append(b, 4);
}
static void put_i32(std::string& s, int32_t v) { put_u32(s, static_cast<uint32_t>(v)); }
static void put_i64(std::string& s, int64_t v) {
    put_u32(s, static_cast<uint32_t>(v));
    put_u32(s, static_cast<uint32_t>(v >> 32));
}
static void put_f64(std::string& s, double v) {
    uint64_t u;
    std::memcpy(&u, &v, 8);
    put_u32(s, static_cast<uint32_t>(u));
    put_u32(s, static_cast<uint32_t>(u >> 32));
}
static void put_str(std::string& s, const std::string& x) {
    put_i32(s, static_cast<int32_t>(x.size()));
    s.append(x);
}

struct R {
    const char* p;
    const char* e;
    explicit R(const std::string& b) : p(b.data()), e(b.data() + b.size()) {}
    uint32_t u32() {
        if (e - p < 4) throw TideError("truncated model");
        uint32_t v = (uint8_t)p[0] | ((uint32_t)(uint8_t)p[1] << 8) | ((uint32_t)(uint8_t)p[2] << 16) |
                     ((uint32_t)(uint8_t)p[3] << 24);
        p += 4;
        return v;
    }
    int32_t i32() { return static_cast<int32_t>(u32()); }
    int64_t i64() {
        uint32_t lo = u32(), hi = u32();
        return static_cast<int64_t>(lo) | (static_cast<int64_t>(hi) << 32);
    }
    double f64() {
        uint32_t lo = u32(), hi = u32();
        uint64_t u = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
        double d;
        std::memcpy(&d, &u, 8);
        return d;
    }
    std::string str() {
        int n = i32();
        if (n < 0 || e - p < n) throw TideError("truncated string");
        std::string s(p, p + n);
        p += n;
        return s;
    }
};

std::string Booster::save_binary() const {
    std::string s;
    s.append(kMagic, 4);
    put_u32(s, kFormatVersion);
    put_i32(s, cfg.n_estimators);
    put_f64(s, cfg.learning_rate);
    put_i32(s, cfg.max_depth);
    put_i32(s, cfg.max_leaves);
    put_i32(s, cfg.min_samples_leaf);
    put_f64(s, cfg.min_child_weight);
    put_f64(s, cfg.l2);
    put_f64(s, cfg.min_split_gain);
    put_i32(s, cfg.max_bin);
    put_f64(s, cfg.subsample);
    put_f64(s, cfg.colsample);
    put_i32(s, cfg.n_time_buckets);
    put_i64(s, cfg.horizon);
    put_i64(s, cfg.embargo);
    put_i32(s, cfg.iid_mode ? 1 : 0);
    put_i32(s, cfg.halo ? 1 : 0);
    put_i32(s, static_cast<int>(cfg.objective));
    put_f64(s, cfg.quantile);
    put_f64(s, cfg.huber_delta);
    put_f64(s, cfg.rank_temperature);
    put_i32(s, cfg.random_state);
    put_i32(s, cfg.n_jobs);
    put_f64(s, cfg.tree_decay);
    put_i32(s, n_features);
    put_i32(s, n_bins);
    put_i32(s, n_time_buckets);
    put_i32(s, best_iteration);
    put_f64(s, init_score);
    put_i32(s, static_cast<int>(init_prefix.size()));
    for (f64 v : init_prefix) put_f64(s, v);
    put_i32(s, static_cast<int>(feature_names.size()));
    for (auto& n : feature_names) put_str(s, n);
    put_i32(s, static_cast<int>(is_cat.size()));
    for (int v : is_cat) put_i32(s, v);
    put_i32(s, static_cast<int>(group_of.size()));
    for (i32 v : group_of) put_i32(s, v);
    put_i32(s, static_cast<int>(monotone.size()));
    for (int8_t v : monotone) put_i32(s, v);
    // binner
    put_i32(s, binner.max_bin);
    put_i32(s, binner.n_features);
    for (int f = 0; f < binner.n_features; ++f) {
        put_i32(s, binner.is_cat[f]);
        put_i32(s, static_cast<int>(binner.edges[f].size()));
        for (f64 e : binner.edges[f]) put_f64(s, e);
        put_i32(s, static_cast<int>(binner.cat_values[f].size()));
        for (f64 e : binner.cat_values[f]) put_f64(s, e);
        put_i32(s, binner.n_bins_col[f]);
    }
    put_i32(s, static_cast<int>(trees.size()));
    for (const auto& t : trees) {
        put_i32(s, t.n_buckets);
        put_i32(s, static_cast<int>(t.nodes.size()));
        for (const auto& n : t.nodes) {
            put_i32(s, n.is_leaf ? 1 : 0);
            put_i32(s, static_cast<int>(n.split_type));
            put_i32(s, n.feature);
            put_i32(s, n.feature2);
            put_i32(s, n.threshold_bin);
            put_i32(s, n.left);
            put_i32(s, n.right);
            put_i32(s, n.missing_go_left);
            put_f64(s, n.value);
            put_f64(s, n.gain);
            put_i32(s, n.n_samples);
            put_i32(s, n.depth);
            for (int k = 0; k < 4; ++k) put_i64(s, static_cast<int64_t>(n.cat_left[k]));
        }
        put_i32(s, static_cast<int>(t.values.size()));
        for (f64 v : t.values) put_f64(s, v);
        put_i32(s, static_cast<int>(t.cover.size()));
        for (i32 v : t.cover) put_i32(s, v);
        put_i32(s, static_cast<int>(t.prefix_values.size()));
        for (f64 v : t.prefix_values) put_f64(s, v);
        put_i32(s, t.target);
        put_i32(s, t.n_targets);
    }
    put_i32(s, static_cast<int>(obj.kind));
    return s;
}

Booster Booster::load_binary(const std::string& blob) {
    if (blob.size() < 8 || std::memcmp(blob.data(), kMagic, 4) != 0) throw TideError("bad magic");
    R r(blob);
    r.p += 4;
    const uint32_t ver = r.u32();
    if (ver != kFormatVersion) throw TideError("unsupported model version");
    Booster b;
    b.cfg.n_estimators = r.i32();
    b.cfg.learning_rate = r.f64();
    b.cfg.max_depth = r.i32();
    b.cfg.max_leaves = r.i32();
    b.cfg.min_samples_leaf = r.i32();
    b.cfg.min_child_weight = r.f64();
    b.cfg.l2 = r.f64();
    b.cfg.min_split_gain = r.f64();
    b.cfg.max_bin = r.i32();
    b.cfg.subsample = r.f64();
    b.cfg.colsample = r.f64();
    b.cfg.n_time_buckets = r.i32();
    b.cfg.horizon = r.i64();
    b.cfg.embargo = r.i64();
    b.cfg.iid_mode = r.i32() != 0;
    b.cfg.halo = r.i32() != 0;
    b.cfg.objective = static_cast<ObjectiveKind>(r.i32());
    b.cfg.quantile = r.f64();
    b.cfg.huber_delta = r.f64();
    b.cfg.rank_temperature = r.f64();
    b.cfg.random_state = r.i32();
    b.cfg.n_jobs = r.i32();
    b.cfg.tree_decay = r.f64();
    b.n_features = r.i32();
    b.n_bins = r.i32();
    b.n_time_buckets = r.i32();
    b.best_iteration = r.i32();
    b.init_score = r.f64();
    int np = r.i32();
    b.init_prefix.resize(np);
    for (int i = 0; i < np; ++i) b.init_prefix[i] = r.f64();
    int nn = r.i32();
    b.feature_names.resize(nn);
    for (int i = 0; i < nn; ++i) b.feature_names[i] = r.str();
    int nc = r.i32();
    b.is_cat.resize(nc);
    for (int i = 0; i < nc; ++i) b.is_cat[i] = r.i32();
    int ng = r.i32();
    b.group_of.resize(ng);
    for (int i = 0; i < ng; ++i) b.group_of[i] = r.i32();
    int nm = r.i32();
    b.monotone.resize(nm);
    for (int i = 0; i < nm; ++i) b.monotone[i] = static_cast<int8_t>(r.i32());
    b.binner.max_bin = r.i32();
    b.binner.n_features = r.i32();
    b.binner.is_cat.resize(b.binner.n_features);
    b.binner.edges.resize(b.binner.n_features);
    b.binner.cat_values.resize(b.binner.n_features);
    b.binner.n_bins_col.resize(b.binner.n_features);
    for (int f = 0; f < b.binner.n_features; ++f) {
        b.binner.is_cat[f] = r.i32();
        int ne = r.i32();
        b.binner.edges[f].resize(ne);
        for (int i = 0; i < ne; ++i) b.binner.edges[f][i] = r.f64();
        int nv = r.i32();
        b.binner.cat_values[f].resize(nv);
        for (int i = 0; i < nv; ++i) b.binner.cat_values[f][i] = r.f64();
        b.binner.n_bins_col[f] = r.i32();
    }
    int nt = r.i32();
    b.trees.resize(nt);
    for (int ti = 0; ti < nt; ++ti) {
        auto& t = b.trees[ti];
        t.n_buckets = r.i32();
        int nnode = r.i32();
        t.nodes.resize(nnode);
        for (int k = 0; k < nnode; ++k) {
            auto& n = t.nodes[k];
            n.is_leaf = r.i32() != 0;
            n.split_type = static_cast<SplitType>(r.i32());
            n.feature = r.i32();
            n.feature2 = r.i32();
            n.threshold_bin = r.i32();
            n.left = r.i32();
            n.right = r.i32();
            n.missing_go_left = r.i32();
            n.value = r.f64();
            n.gain = r.f64();
            n.n_samples = r.i32();
            n.depth = r.i32();
            for (int j = 0; j < 4; ++j) n.cat_left[j] = static_cast<uint64_t>(r.i64());
        }
        int nv = r.i32();
        t.values.resize(nv);
        for (int i = 0; i < nv; ++i) t.values[i] = r.f64();
        int ncv = r.i32();
        t.cover.resize(ncv);
        for (int i = 0; i < ncv; ++i) t.cover[i] = r.i32();
        int npv = r.i32();
        t.prefix_values.resize(npv);
        for (int i = 0; i < npv; ++i) t.prefix_values[i] = r.f64();
        t.target = r.i32();
        t.n_targets = r.i32();
    }
    b.obj.kind = static_cast<ObjectiveKind>(r.i32());
    b.obj = make_objective(b.cfg);
    b.fitted = true;
    return b;
}

static void json_escape(std::ostringstream& o, const std::string& s) {
    o << '"';
    for (char c : s) {
        if (c == '"' || c == '\\') o << '\\' << c;
        else o << c;
    }
    o << '"';
}

std::string Booster::save_json() const {
    std::ostringstream o;
    o << "{\n  \"format\": \"tidegbm\",\n  \"version\": " << kFormatVersion << ",\n";
    o << "  \"n_features\": " << n_features << ",\n";
    o << "  \"n_bins\": " << n_bins << ",\n";
    o << "  \"best_iteration\": " << best_iteration << ",\n";
    o << "  \"init_score\": " << init_score << ",\n";
    o << "  \"learning_rate\": " << cfg.learning_rate << ",\n";
    o << "  \"horizon\": " << cfg.horizon << ",\n";
    o << "  \"iid_mode\": " << (cfg.iid_mode ? "true" : "false") << ",\n";
    o << "  \"feature_names\": [";
    for (size_t i = 0; i < feature_names.size(); ++i) {
        if (i) o << ", ";
        json_escape(o, feature_names[i]);
    }
    o << "],\n  \"binner\": {\n    \"max_bin\": " << binner.max_bin << ",\n    \"edges\": [";
    for (int f = 0; f < binner.n_features; ++f) {
        if (f) o << ", ";
        o << "[";
        for (size_t i = 0; i < binner.edges[f].size(); ++i) {
            if (i) o << ", ";
            o << binner.edges[f][i];
        }
        o << "]";
    }
    o << "],\n    \"is_cat\": [";
    for (int f = 0; f < binner.n_features; ++f) {
        if (f) o << ", ";
        o << binner.is_cat[f];
    }
    o << "],\n    \"cat_values\": [";
    for (int f = 0; f < binner.n_features; ++f) {
        if (f) o << ", ";
        o << "[";
        for (size_t i = 0; i < binner.cat_values[f].size(); ++i) {
            if (i) o << ", ";
            o << binner.cat_values[f][i];
        }
        o << "]";
    }
    o << "]\n  },\n  \"trees\": [\n";
    for (size_t ti = 0; ti < trees.size(); ++ti) {
        const auto& t = trees[ti];
        if (ti) o << ",\n";
        o << "    {\"target\": " << t.target << ", \"nodes\": [";
        for (size_t ni = 0; ni < t.nodes.size(); ++ni) {
            const auto& n = t.nodes[ni];
            if (ni) o << ", ";
            const f64 val = t.values.empty() ? n.value : t.values[ni];
            o << "{\"leaf\": " << (n.is_leaf ? "true" : "false") << ", \"f\": " << n.feature
              << ", \"thr\": " << n.threshold_bin << ", \"left\": " << n.left << ", \"right\": " << n.right
              << ", \"miss_left\": " << n.missing_go_left << ", \"v\": " << val << ", \"gain\": " << n.gain
              << "}";
        }
        o << "]}";
    }
    o << "\n  ]\n}\n";
    return o.str();
}

std::string Booster::to_onnx_json() const {
    std::ostringstream o;
    o << "{\n  \"ir_version\": 8,\n  \"producer_name\": \"tidegbm\",\n  \"graph\": {\n";
    o << "    \"node\": [{\"op_type\": \"TreeEnsembleRegressor\", \"attribute\": [\n";
    o << "      {\"name\": \"n_targets\", \"i\": 1},\n";
    o << "      {\"name\": \"aggregate_function\", \"s\": \"SUM\"},\n";
    o << "      {\"name\": \"base_values\", \"floats\": [" << init_score << "]},\n";
    o << "      {\"name\": \"nodes_treeids\": [";
    bool first = true;
    std::vector<int> treeids, nodeids, featids, lefts, rights;
    std::vector<f64> values, thresholds;
    std::vector<std::string> modes;
    for (int ti = 0; ti < static_cast<int>(trees.size()); ++ti) {
        const auto& t = trees[ti];
        for (int ni = 0; ni < static_cast<int>(t.nodes.size()); ++ni) {
            const auto& n = t.nodes[ni];
            treeids.push_back(ti);
            nodeids.push_back(ni);
            featids.push_back(n.is_leaf ? 0 : n.feature);
            lefts.push_back(n.left);
            rights.push_back(n.right);
            values.push_back(n.is_leaf ? (t.values.empty() ? n.value : t.values[ni]) * cfg.learning_rate : 0);
            f64 thr = static_cast<f64>(n.threshold_bin);
            if (!n.is_leaf && n.split_type == SplitType::Axis && n.feature >= 0 &&
                n.feature < binner.n_features && n.threshold_bin >= 1) {
                const auto& e = binner.edges[n.feature];
                const int ei = n.threshold_bin - 1;
                if (ei >= 0 && ei < static_cast<int>(e.size())) thr = e[ei];
            }
            thresholds.push_back(thr);
            modes.push_back(n.is_leaf ? "LEAF" : "BRANCH_LEQ");
        }
    }
    auto dump_i = [&](const char* name, const std::vector<int>& v) {
        o << "      {\"name\": \"" << name << "\", \"ints\": [";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) o << ", ";
            o << v[i];
        }
        o << "]},\n";
    };
    auto dump_f = [&](const char* name, const std::vector<f64>& v) {
        o << "      {\"name\": \"" << name << "\", \"floats\": [";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) o << ", ";
            o << v[i];
        }
        o << "]},\n";
    };
    dump_i("nodes_treeids", treeids);
    dump_i("nodes_nodeids", nodeids);
    dump_i("nodes_featureids", featids);
    dump_i("nodes_truenodeids", lefts);
    dump_i("nodes_falsenodeids", rights);
    dump_f("nodes_values", values);
    dump_f("nodes_thresholds", thresholds);
    o << "      {\"name\": \"nodes_modes\", \"strings\": [";
    for (size_t i = 0; i < modes.size(); ++i) {
        if (i) o << ", ";
        o << "\"" << modes[i] << "\"";
    }
    o << "]}\n    ]}],\n    \"comment\": \"Apply the saved TideGBM binner (JSON edges) before this graph if categories/missing are used. Numeric BRANCH_LEQ uses raw feature thresholds.\"\n  }\n}\n";
    (void)first;
    return o.str();
}

}  // namespace tidegbm
