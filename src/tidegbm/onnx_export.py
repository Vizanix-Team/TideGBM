"""Optional ONNX protobuf export. Requires the ``onnx`` extra."""

from __future__ import annotations

import json


def booster_to_onnx_bytes(json_model: str) -> bytes:
    """Build an ONNX TreeEnsembleRegressor from TideGBM tree JSON.

    Axis-split thresholds are mapped through frozen bin edges when present.
    Categorical and pairwise nodes are exported as BRANCH_LEQ on bin ids and
    will not match native scoring on raw X. Leaf values include learning_rate.
    """
    try:
        from onnx import TensorProto, helper
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install onnx  (or tidegbm[onnx])") from e

    meta = json.loads(json_model)
    n_features = int(meta["n_features"])
    init_score = float(meta.get("init_score", 0.0))
    lr = float(meta.get("learning_rate", 1.0))
    edges = meta.get("binner", {}).get("edges", [])
    trees = meta.get("trees", [])

    treeids, nodeids, featids, lefts, rights = [], [], [], [], []
    thresholds, modes = [], []
    tgt_tree, tgt_node, tgt_id, tgt_w = [], [], [], []
    for ti, t in enumerate(trees):
        for ni, n in enumerate(t["nodes"]):
            leaf = bool(n["leaf"])
            treeids.append(ti)
            nodeids.append(ni)
            featids.append(0 if leaf else max(int(n["f"]), 0))
            lefts.append(max(int(n.get("left", 0)), 0))
            rights.append(max(int(n.get("right", 0)), 0))
            modes.append(b"LEAF" if leaf else b"BRANCH_LEQ")
            thr = float(n.get("thr", 0.0))
            f = int(n.get("f", -1))
            if not leaf and 0 <= f < len(edges):
                ei = int(thr) - 1
                col = edges[f]
                if 0 <= ei < len(col):
                    thr = float(col[ei])
            thresholds.append(0.0 if leaf else thr)
            if leaf:
                tgt_tree.append(ti)
                tgt_node.append(ni)
                tgt_id.append(0)
                tgt_w.append(lr * float(n["v"]))

    node = helper.make_node(
        "TreeEnsembleRegressor",
        inputs=["X"],
        outputs=["Y"],
        domain="ai.onnx.ml",
        n_targets=1,
        aggregate_function=b"SUM",
        base_values=[init_score],
        nodes_treeids=treeids,
        nodes_nodeids=nodeids,
        nodes_featureids=featids,
        nodes_truenodeids=lefts,
        nodes_falsenodeids=rights,
        nodes_values=thresholds,
        nodes_modes=modes,
        target_treeids=tgt_tree,
        target_nodeids=tgt_node,
        target_ids=tgt_id,
        target_weights=tgt_w,
        post_transform=b"NONE",
    )
    graph = helper.make_graph(
        [node],
        "tidegbm",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 8),
            helper.make_opsetid("ai.onnx.ml", 1),
        ],
        producer_name="tidegbm",
    )
    return model.SerializeToString()
