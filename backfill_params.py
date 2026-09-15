"""Add n_params to the tabular metrics_long.csv files.

run_method_matrix and run_fedheart_cv never recorded parameter counts, so our own rows show a dash
in the params column while every baseline row shows a number. That is the column the capacity
argument rests on, and it was missing on two of the three datasets.

No retraining needed: the parameter count is a deterministic function of the config, so the models
are rebuilt on CPU and counted.

Two counting decisions, both stated in the output rather than buried:

  inference params   encoders + head. This is what compares against a baseline's parameter count,
                     since that is the machinery that turns an input into a prediction.
  anchor params      the learnable class means and Cholesky factors. They shape the latent space
                     during training and are not used at inference, so folding them into the
                     headline number would overstate the model against baselines that have no
                     equivalent. Reported separately.

With a shared encoder the same module is registered under every group id, so parameters are
deduplicated by object identity rather than summed once per group.

  python backfill_params.py
"""
from __future__ import annotations
import copy, csv, os

import torch
import yaml

MATRICES = [
    ("runs/matrix_nhanes_nested", "nhanes", "experiments/nhanes_pergroup_gdro.yaml"),
    ("runs/matrix_nhanes_disjoint", "nhanes", "experiments/nhanes_disjoint_pergroup_gdro.yaml"),
    ("runs/fedheart_cv", "fedheart", "experiments/fedheart_exp_paper_hetagg_gdro.yaml"),
]
# (label, common_encoder, anchors_on) mirroring METHODS in the two runners
SHARED = {"ERM", "Shared_GDRO", "Shared_Anchors", "Shared_Anchors_GDRO"}
ANCHORED = {"Shared_Anchors", "Shared_Anchors_GDRO", "AnchorsOnly", "Ours_GDRO",
            "Ours_Regret", "Ours"}


def count(dataset, base_path, label):
    cfg = yaml.safe_load(open(base_path))
    cfg = copy.deepcopy(cfg)
    cfg["common_encoder"] = label in SHARED
    dev = torch.device("cpu")
    if dataset == "nhanes":
        from dro_hetero_anchors.src.datasets_nhanes import build_nhanes_loaders
        from dro_hetero_anchors.src.train_nhanes import build_models
        cfg.setdefault("data_split_seed", 100)
        _, _, info = build_nhanes_loaders(
            batch_size=cfg.get("batch_size", 128), seed=0,
            data_split_seed=cfg["data_split_seed"],
            feature_mode=cfg.get("feature_mode", "nested"))
        gfc = info["group_feature_counts"]
        if cfg["common_encoder"] and cfg.get("shared_common_features", True):
            n_common = len(sorted(set.intersection(
                *[set(v) for v in info["feature_indices"].values()])))
            for g in cfg["groups"]:
                g["input_dim"] = n_common
            gfc = {k: n_common for k in gfc}
        gc = info.get("train_group_counts") or info.get("group_counts")
        enc, head, anch, _ = build_models(cfg, gc, dev, gfc)
    else:
        from dro_hetero_anchors.src.datasets_fedheart import build_fedheart_loaders
        from dro_hetero_anchors.src.train_fedheart import build_models
        _, _, info = build_fedheart_loaders(
            batch_size=cfg.get("batch_size", 64), seed=1000, train_frac=0.8,
            feature_mask=cfg.get("feature_mask"),
            group_max_train_samples=cfg.get("group_max_train_samples"),
            subsample_seed=0, impute_missing=True)
        # true_hetero_input_dim only applies with per-group encoders, same gate as the trainer
        if (cfg.get("true_hetero_input_dim") and cfg.get("feature_mask")
                and not cfg["common_encoder"]):
            for gid, fm in enumerate(cfg["feature_mask"]):
                if fm is not None:
                    cfg["groups"][gid]["input_dim"] = len(fm)
        enc, head, anch, _ = build_models(cfg, info["train_group_counts"], dev)

    seen, infer = set(), 0
    for m in list(enc.values()) + [head]:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p)); infer += p.numel()
    anchor_n = sum(p.numel() for p in anch.parameters()) if label in ANCHORED else 0
    return infer, anchor_n


def main():
    for out, dataset, base in MATRICES:
        path = os.path.join(out, "metrics_long.csv")
        if not os.path.exists(path):
            print(f"{out}: no metrics_long.csv, skipped")
            continue
        rows = list(csv.DictReader(open(path)))
        if not rows:
            continue
        if "n_params" in rows[0]:
            print(f"{out}: already has n_params")
            continue
        cache = {}
        print(f"\n{out}")
        for r in rows:
            lab = r["method"]
            if lab not in cache:
                try:
                    cache[lab] = count(dataset, base, lab)
                except Exception as e:
                    print(f"  {lab}: FAILED {e}")
                    cache[lab] = (None, None)
            r["n_params"] = cache[lab][0] if cache[lab][0] is not None else ""
        for lab, (i, a) in sorted(cache.items()):
            if i is not None:
                print(f"  {lab:22} inference {i:>8,}   anchors {a:>7,}")
        fields = list(rows[0].keys())
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
