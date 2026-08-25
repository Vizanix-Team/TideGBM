#include "tidegbm/booster.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <cstdint>
#include <memory>
#include <utility>

namespace py = pybind11;

using tidegbm::Booster;
using tidegbm::Config;
using tidegbm::ObjectiveKind;
using tidegbm::TideError;

static py::array_t<double> vec_to_array(const std::vector<double>& v) {
    py::array_t<double> a(v.size());
    std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(double));
    return a;
}

static Config config_from_kwargs(const py::kwargs& kw) {
    Config c;
    auto geti = [&](const char* k, int def) {
        if (kw.contains(k) && !kw[k].is_none()) return kw[k].cast<int>();
        return def;
    };
    auto getd = [&](const char* k, double def) {
        if (kw.contains(k) && !kw[k].is_none()) return kw[k].cast<double>();
        return def;
    };
    auto getb = [&](const char* k, bool def) {
        if (kw.contains(k) && !kw[k].is_none()) return kw[k].cast<bool>();
        return def;
    };
    auto gets = [&](const char* k, const std::string& def) {
        if (kw.contains(k) && !kw[k].is_none()) return kw[k].cast<std::string>();
        return def;
    };
    c.n_estimators = geti("n_estimators", c.n_estimators);
    c.learning_rate = getd("learning_rate", c.learning_rate);
    c.max_depth = geti("max_depth", c.max_depth);
    c.max_leaves = geti("max_leaves", c.max_leaves);
    c.min_samples_leaf = geti("min_samples_leaf", c.min_samples_leaf);
    c.min_child_weight = getd("min_child_weight", c.min_child_weight);
    c.l2 = getd("l2", c.l2);
    c.min_split_gain = getd("min_split_gain", c.min_split_gain);
    c.max_bin = geti("max_bin", c.max_bin);
    c.subsample = getd("subsample", c.subsample);
    c.colsample = getd("colsample", c.colsample);
    c.n_time_buckets = geti("n_time_buckets", c.n_time_buckets);
    c.horizon = kw.contains("horizon") && !kw["horizon"].is_none() ? kw["horizon"].cast<int64_t>() : -1;
    c.embargo = kw.contains("embargo") && !kw["embargo"].is_none() ? kw["embargo"].cast<int64_t>() : 0;
    c.iid_mode = getb("iid_mode", false);
    c.halo = getb("halo", false);
    c.halo_gamma = getd("halo_gamma", c.halo_gamma);
    c.halo_candidates = geti("halo_candidates", c.halo_candidates);
    c.pair_splits = getb("pair_splits", true);
    c.pair_trigger = getd("pair_trigger", c.pair_trigger);
    c.pair_top_k = geti("pair_top_k", c.pair_top_k);
    c.max_pair_splits_per_tree = geti("max_pair_splits_per_tree", c.max_pair_splits_per_tree);
    c.pair_l2 = getd("pair_l2", c.pair_l2);
    c.time_decay = getd("time_decay", 1.0);
    c.tail_buffer = getd("tail_buffer", 0.02);
    c.tree_decay = getd("tree_decay", 1.0);
    c.quantile = getd("quantile", 0.5);
    c.huber_delta = getd("huber_delta", 1.0);
    c.rank_temperature = getd("rank_temperature", 1.0);
    c.n_jobs = geti("n_jobs", 0);
    c.random_state = geti("random_state", 0);
    c.verbose = geti("verbose", 0);
    c.early_stopping_rounds = geti("early_stopping_rounds", 0);
    c.causal_structure = getb("causal_structure", c.causal_structure);
    c.finer_torb = getb("finer_torb", true);
    c.device = gets("device", "cpu");
    c.cost_threshold = getd("cost_threshold", 0.0);
    c.rank_pairwise = getb("rank_pairwise", false);
    c.l1 = getd("l1", 0.0);
    c.max_delta_step = getd("max_delta_step", 0.0);
    if (kw.contains("goss") && !kw["goss"].is_none()) {
        if (py::isinstance<py::bool_>(kw["goss"]))
            c.goss = kw["goss"].cast<bool>() ? 1 : 0;
        else
            c.goss = kw["goss"].cast<int>();
    }
    c.goss_top = getd("goss_top", 0.2);
    c.goss_other = getd("goss_other", 0.1);
    c.structure_frac = getd("structure_frac", c.structure_frac);
    c.causal_bins = getb("causal_bins", c.causal_bins);
    c.torb_min_cover = getd("torb_min_cover", c.torb_min_cover);
    c.cat_smooth = getd("cat_smooth", 1.0);
    c.col_major_hist = getb("col_major_hist", true);
    c.cat_min_count = geti("cat_min_count", 5);
    c.colsample_bynode = getd("colsample_bynode", 1.0);
    c.bin_sample = geti("bin_sample", 262144);
    c.entity_subsample = getd("entity_subsample", 1.0);
    c.eval_metric = gets("eval_metric", "auto");
    c.n_classes = geti("n_classes", 1);
    c.n_horizons = geti("n_horizons", 1);
    c.regime_feature = geti("regime_feature", -1);
    c.regime_priority = getb("regime_priority", true);
    if (kw.contains("init_score") && !kw["init_score"].is_none())
        c.init_score_given = kw["init_score"].cast<double>();
    if (kw.contains("quantiles") && !kw["quantiles"].is_none())
        c.quantiles = kw["quantiles"].cast<std::vector<double>>();
    if (kw.contains("interaction_group") && !kw["interaction_group"].is_none())
        c.interaction_group = kw["interaction_group"].cast<std::vector<int>>();
    if (kw.contains("forced_pairs") && !kw["forced_pairs"].is_none()) {
        auto pairs = kw["forced_pairs"].cast<std::vector<std::pair<int, int>>>();
        c.forced_pairs = std::move(pairs);
    }
    std::string obj = gets("objective", "mse");
    if (obj == "mse" || obj == "auto" || obj == "l2" || obj == "regression") c.objective = ObjectiveKind::MSE;
    else if (obj == "huber") c.objective = ObjectiveKind::Huber;
    else if (obj == "quantile" || obj == "pinball") c.objective = ObjectiveKind::Quantile;
    else if (obj == "logloss" || obj == "binary" || obj == "classification")
        c.objective = ObjectiveKind::Logloss;
    else if (obj == "rank" || obj == "listnet") c.objective = ObjectiveKind::RankListNet;
    else if (obj == "pairwise" || obj == "lambdarank") c.objective = ObjectiveKind::RankPairwise;
    else if (obj == "multiclass") c.objective = ObjectiveKind::Multiclass;
    else if (obj == "custom") c.objective = ObjectiveKind::Custom;
    else throw TideError("unknown objective: " + obj);
    return c;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "TideGBM C++ core";
    py::register_exception<TideError>(m, "TideError");

    py::class_<Booster>(m, "Booster")
        .def(py::init<>())
        .def_readwrite("fitted", &Booster::fitted)
        .def_readwrite("best_iteration", &Booster::best_iteration)
        .def_readwrite("n_features", &Booster::n_features)
        .def_readwrite("feature_names", &Booster::feature_names)
        .def_readwrite("init_score", &Booster::init_score)
        .def("set_config", [](Booster& b, py::kwargs kw) { b.cfg = config_from_kwargs(kw); })
        .def("fit",
             [](Booster& b, py::array_t<double, py::array::c_style | py::array::forcecast> X,
                py::array_t<double, py::array::c_style | py::array::forcecast> y,
                py::object timestamps, py::object weight, py::object is_cat, py::object group_of,
                py::object monotone, py::object rank_groups, py::object X_eval, py::object y_eval,
                py::object ts_eval, py::object entity, py::object cost, py::kwargs kw) {
                 if (kw.size()) b.cfg = config_from_kwargs(kw);
                 if (X.ndim() != 2) throw TideError("X must be 2d");
                 const int n = static_cast<int>(X.shape(0));
                 const int n_f = static_cast<int>(X.shape(1));
                 const double* Xp = X.data();
                 const double* yp = y.data();
                 std::vector<int64_t> ts;
                 const int64_t* tsp = nullptr;
                 if (!timestamps.is_none()) {
                     auto a = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(timestamps);
                     ts.assign(a.data(), a.data() + a.size());
                     tsp = ts.data();
                 }
                 std::vector<double> w;
                 const double* wp = nullptr;
                 if (!weight.is_none()) {
                     auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(weight);
                     w.assign(a.data(), a.data() + a.size());
                     wp = w.data();
                 }
                 std::vector<int> cat(n_f, 0);
                 if (!is_cat.is_none()) {
                     auto a = py::array_t<int, py::array::c_style | py::array::forcecast>::ensure(is_cat);
                     cat.assign(a.data(), a.data() + a.size());
                 }
                 std::vector<int32_t> grp(n_f, -1);
                 if (!group_of.is_none()) {
                     auto a = py::array_t<int32_t, py::array::c_style | py::array::forcecast>::ensure(group_of);
                     grp.assign(a.data(), a.data() + a.size());
                 }
                 std::vector<int8_t> mon(n_f, 0);
                 if (!monotone.is_none()) {
                     auto a = py::array_t<int8_t, py::array::c_style | py::array::forcecast>::ensure(monotone);
                     mon.assign(a.data(), a.data() + a.size());
                 }
                 std::vector<int64_t> rg;
                 const int64_t* rgp = nullptr;
                 if (!rank_groups.is_none()) {
                     auto a = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(rank_groups);
                     rg.assign(a.data(), a.data() + a.size());
                     rgp = rg.data();
                 }
                 const double* Xe = nullptr;
                 const double* ye = nullptr;
                 const int64_t* te = nullptr;
                 int n_eval = 0;
                 std::vector<int64_t> tse;
                 py::array_t<double> Xea, yea;
                 if (!X_eval.is_none()) {
                     Xea = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(X_eval);
                     yea = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(y_eval);
                     Xe = Xea.data();
                     ye = yea.data();
                     n_eval = static_cast<int>(Xea.shape(0));
                     if (!ts_eval.is_none()) {
                         auto a = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(ts_eval);
                         tse.assign(a.data(), a.data() + a.size());
                         te = tse.data();
                     }
                 }
                 std::vector<int64_t> ent;
                 const int64_t* entp = nullptr;
                 if (!entity.is_none()) {
                     auto a = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(entity);
                     ent.assign(a.data(), a.data() + a.size());
                     entp = ent.data();
                 }
                 std::vector<double> costv;
                 const double* costp = nullptr;
                 if (!cost.is_none()) {
                     auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(cost);
                     costv.assign(a.data(), a.data() + a.size());
                     costp = costv.data();
                 }
                 b.fit(Xp, yp, n, n_f, tsp, wp, cat.data(), grp.data(), mon.data(), rgp, Xe, ye, te,
                       n_eval, entp, costp);
             },
             py::arg("X"), py::arg("y"), py::arg("timestamps") = py::none(),
             py::arg("weight") = py::none(), py::arg("is_cat") = py::none(),
             py::arg("group_of") = py::none(), py::arg("monotone") = py::none(),
             py::arg("rank_groups") = py::none(), py::arg("X_eval") = py::none(),
             py::arg("y_eval") = py::none(), py::arg("ts_eval") = py::none(),
             py::arg("entity") = py::none(), py::arg("cost") = py::none())
        .def("predict",
             [](const Booster& b, py::array_t<double, py::array::c_style | py::array::forcecast> X) {
                 const int n = static_cast<int>(X.shape(0));
                 const int n_f = static_cast<int>(X.shape(1));
                 py::array_t<double> out(n);
                 b.predict(X.data(), n, n_f, out.mutable_data());
                 return out;
             })
        .def("raw_predict",
             [](const Booster& b, py::array_t<double, py::array::c_style | py::array::forcecast> X) {
                 const int n = static_cast<int>(X.shape(0));
                 const int n_f = static_cast<int>(X.shape(1));
                 py::array_t<double> out(n);
                 b.predict_raw(X.data(), n, n_f, out.mutable_data());
                 return out;
             })
        .def("update",
             [](Booster& b, py::array_t<double, py::array::c_style | py::array::forcecast> X,
                py::array_t<double, py::array::c_style | py::array::forcecast> y, int extra) {
                 b.update(X.data(), y.data(), static_cast<int>(X.shape(0)), static_cast<int>(X.shape(1)),
                          nullptr, nullptr, extra);
             },
             py::arg("X"), py::arg("y"), py::arg("extra") = 10)
        .def("feature_importance",
             [](const Booster& b, const std::string& kind) {
                 py::array_t<double> a(b.n_features);
                 b.feature_importance(a.mutable_data(), kind.c_str());
                 return a;
             },
             py::arg("kind") = "gain")
        .def("pred_contrib",
             [](const Booster& b, py::array_t<double, py::array::c_style | py::array::forcecast> X) {
                 const int n = static_cast<int>(X.shape(0));
                 const int n_f = static_cast<int>(X.shape(1));
                 py::array_t<double> a({n, n_f + 1});
                 b.pred_contrib(X.data(), n, n_f, a.mutable_data());
                 return a;
             })
        .def("predict_multi",
             [](const Booster& b, py::array_t<double, py::array::c_style | py::array::forcecast> X) {
                 const int n = static_cast<int>(X.shape(0));
                 const int n_f = static_cast<int>(X.shape(1));
                 const int h = std::max(1, b.n_heads());
                 py::array_t<double> out({n, h});
                 b.predict_multi(X.data(), n, n_f, out.mutable_data());
                 return out;
             })
        .def("predict_binned",
             [](const Booster& b, py::array_t<uint16_t, py::array::c_style | py::array::forcecast> xb) {
                 const int n = static_cast<int>(xb.shape(0));
                 py::array_t<double> out(n);
                 b.predict_binned(xb.data(), n, out.mutable_data());
                 return out;
             })
        .def("bin_transform",
             [](const Booster& b, py::array_t<double, py::array::c_style | py::array::forcecast> X) {
                 const int n = static_cast<int>(X.shape(0));
                 const int n_f = static_cast<int>(X.shape(1));
                 std::vector<uint16_t> xb;
                 b.bin_transform(X.data(), n, n_f, xb);
                 py::array_t<uint16_t> a({n, n_f});
                 std::memcpy(a.mutable_data(), xb.data(), xb.size() * sizeof(uint16_t));
                 return a;
             })
        .def("set_custom_objective",
             [](Booster& b, py::function fn) {
                 b.cfg.objective = tidegbm::ObjectiveKind::Custom;
                 b.custom_grad = [fn](const double* y, const double* pred, const double* w, int n, double* g,
                                      double* h) {
                     py::gil_scoped_acquire gil;
                     py::array_t<double> ya(n), pa(n), wa(n);
                     std::memcpy(ya.mutable_data(), y, n * sizeof(double));
                     std::memcpy(pa.mutable_data(), pred, n * sizeof(double));
                     std::memcpy(wa.mutable_data(), w, n * sizeof(double));
                     py::object res = fn(ya, pa, wa);
                     py::tuple t = py::reinterpret_borrow<py::tuple>(res);
                     auto ga = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(t[0]);
                     auto ha = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(t[1]);
                     std::memcpy(g, ga.data(), n * sizeof(double));
                     std::memcpy(h, ha.data(), n * sizeof(double));
                 };
             })
        .def("save", [](const Booster& b) { return py::bytes(b.save_binary()); })
        .def("save_json", &Booster::save_json)
        .def("to_json", &Booster::save_json)
        .def_static("load", [](py::bytes blob) { return Booster::load_binary(std::string(blob)); })
        .def("to_onnx_json", &Booster::to_onnx_json)
        .def("n_heads", &Booster::n_heads)
        .def("evals_result", [](const Booster& b) {
            py::dict d;
            d["train"] = b.evals.train;
            d["valid"] = b.evals.valid;
            return d;
        });
#ifdef TIDEGBM_CUDA
    m.attr("cuda_compiled") = true;
#else
    m.attr("cuda_compiled") = false;
#endif
}
