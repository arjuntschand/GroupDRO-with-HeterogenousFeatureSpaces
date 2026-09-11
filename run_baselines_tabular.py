"""Run the REMIND-paper baselines on our tabular datasets and emit metrics_long.csv.

Three methods, same metric set and same seeds as everything else so the rows drop straight
into the existing tables:

  Reweigh   fixed inverse-frequency group weights on a per-group-encoder model
  FlexMoE   shared block projections, learnable per-group stand-ins for absent blocks,
            Soft MoE fusion, shared head
  REMIND    FlexMoE plus a distributionally robust outer loop, lambda refreshed every N steps

Read the header of model/baselines.py before quoting these: they are reimplementations from
the paper, not the authors' released code, which we could not locate.

Block structure. A "modality" here is a block of features, which is exactly how the groups are
defined in both datasets. NHANES: survey (10, everyone), exam (3, G1 and G2), labs (7, G2).
Fed-Heart: one block per distinct site feature subset, built from the config's feature_mask.

  python run_baselines_tabular.py --dataset nhanes
"""
from __future__ import annotations
import argparse, copy, csv, json, os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import yaml

from dro_hetero_anchors.src.model.baselines import (
    FlexMoEModel, FlexMoESparse, ReweighLoss)


# NHANES feature layout, from datasets_nhanes.py _get_feature_config("nested")
NHANES_BLOCKS = [list(range(0, 10)), list(range(10, 13)), list(range(13, 20))]
NHANES_GROUP_BLOCKS = {0: [0], 1: [0, 1], 2: [0, 1, 2]}


def build_blocks(dataset: str, cfg: dict):
    """Return (block_feature_indices, group -> block ids)."""
    if dataset == "nhanes":
        return NHANES_BLOCKS, NHANES_GROUP_BLOCKS
    # Fed-Heart: derive blocks from the per-group feature masks. Block 0 is the intersection
    # (features every site records); after that one block per site for the features unique to
    # it. That reproduces the modality-combination structure the paper assumes.
    masks = [set(m) for m in cfg["feature_mask"]]
    shared = set.intersection(*masks)
    blocks = [sorted(shared)]
    gb: Dict[int, List[int]] = {g: [0] for g in range(len(masks))}
    for gi, m in enumerate(masks):
        extra = sorted(m - shared)
        if extra:
            blocks.append(extra)
            gb[gi].append(len(blocks) - 1)
    return blocks, gb


def evaluate(model, loader, blocks, device, num_groups, num_classes):
    from dro_hetero_anchors.src.model.baselines import FlexMoESparse
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="none")   # eval loss stays unweighted, as elsewhere
    cg = [0] * num_groups; tg = [0] * num_groups
    ls = [0.0] * num_groups; lc = [0] * num_groups
    tp = [[0] * num_classes for _ in range(num_groups)]
    fp = [[0] * num_classes for _ in range(num_groups)]
    fn = [[0] * num_classes for _ in range(num_groups)]
    with torch.no_grad():
        for x, y, g in loader:
            x, y, g = x.to(device), y.to(device), g.to(device)
            xb = [x[:, idx] for idx in blocks]
            out = model(xb, g, warmup=False) if isinstance(model, FlexMoESparse) else model(xb, g)
            logits = out[0]
            loss = ce(logits, y); pred = logits.argmax(1)
            for i in range(x.size(0)):
                gi, yi, pi = int(g[i]), int(y[i]), int(pred[i])
                tg[gi] += 1; ls[gi] += float(loss[i]); lc[gi] += 1
                if pi == yi: cg[gi] += 1; tp[gi][yi] += 1
                else: fp[gi][pi] += 1; fn[gi][yi] += 1
    acc = [cg[i] / max(1, tg[i]) for i in range(num_groups)]
    loss_g = [ls[i] / max(1, lc[i]) for i in range(num_groups)]
    f1 = []
    for gi in range(num_groups):
        per = []
        for c in range(num_classes):
            p = tp[gi][c] / max(1, tp[gi][c] + fp[gi][c])
            r = tp[gi][c] / max(1, tp[gi][c] + fn[gi][c])
            per.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
        f1.append(sum(per) / len(per))
    return acc, loss_g, f1, tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nhanes", "fedheart"], required=True)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[42, 1337, 7, 2024, 31337, 11, 22, 33, 44, 55])
    ap.add_argument("--methods", nargs="+", default=["Reweigh", "FlexMoE", "REMIND"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dataset == "nhanes":
        from dro_hetero_anchors.src.datasets_nhanes import build_nhanes_loaders as build
        base = yaml.safe_load(open("experiments/nhanes_pergroup_gdro.yaml"))
        rstar_path = "runs/rstar_nhanes_nested.json"
    else:
        from dro_hetero_anchors.src.datasets_fedheart import build_fedheart_loaders as build
        base = yaml.safe_load(open("experiments/fedheart_exp_paper_hetagg_gdro.yaml"))
        rstar_path = "runs/rstar_fedheart.json"

    out = args.out or f"runs/baselines_{args.dataset}"
    os.makedirs(out, exist_ok=True)
    device = torch.device("cpu")
    blocks, group_blocks = build_blocks(args.dataset, base)
    rs = json.load(open(rstar_path))["rstar"]
    rstar = [rs[str(i)] if str(i) in rs else rs.get(i, 0.0) for i in range(len(rs))]
    print(f"blocks: {[len(b) for b in blocks]}   group->blocks: {group_blocks}", flush=True)

    rows = []
    for method in args.methods:
        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            cfg = copy.deepcopy(base); cfg["seed"] = seed
            if args.dataset == "nhanes":
                tr, te, info = build(batch_size=cfg.get("batch_size", 128),
                                     seed=seed, stratified=cfg.get("stratified_batching", True),
                                     train_frac=cfg.get("train_frac", 0.8),
                                     use_post_pandemic=cfg.get("use_post_pandemic", True),
                                     data_split_seed=cfg.get("data_split_seed"),
                                     feature_mode=cfg.get("feature_mode", "nested"))
            else:
                tr, te, info = build(batch_size=cfg.get("batch_size", 64),
                                     seed=seed, stratified=cfg.get("stratified_batching", True),
                                     train_frac=cfg.get("train_frac", 0.8),
                                     feature_mask=cfg.get("feature_mask"),
                                     group_max_train_samples=cfg.get("group_max_train_samples"),
                                     impute_missing=True)
            ng = len(cfg["groups"]); nc = cfg["num_classes"]
            counts = info.get("group_counts") or info.get("train_group_counts") or [1] * ng
            # Inverse-frequency class weights, identical to train_nhanes.py:473. Without this
            # a 90/10 task is "solved" at ~89% by always predicting the majority class, and
            # the baseline numbers would not be comparable to any of our own arms.
            cls_w = None
            tcc = info.get("train_class_counts")
            if cfg.get("class_weight", "auto") == "auto" and tcc:
                tot = sum(tcc)
                cls_w = torch.tensor([tot / (len(tcc) * max(1, c)) for c in tcc],
                                     dtype=torch.float32, device=device)

            if method == "FlexMoE":
                # released defaults: 16 experts, top-k 4, hidden 128, 5 warm-up epochs
                model = FlexMoESparse(len(blocks), group_blocks, [len(b) for b in blocks],
                                      d_model=128, n_experts=16, top_k=4,
                                      num_classes=nc).to(device)
            else:
                model = FlexMoEModel(
                    [len(b) for b in blocks], group_blocks,
                    latent_dim=cfg.get("latent_dim", 64), num_classes=nc,
                    n_experts=4).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3),
                                   weight_decay=cfg.get("weight_decay", 1e-4))
            reweigh = (ReweighLoss(ng, counts, device, class_weight=cls_w)
                       if method == "Reweigh" else None)
            lam = torch.full((ng,), 1.0 / ng, device=device)
            ema = torch.zeros(ng, device=device)
            step = 0
            best = (-1.0, None)

            for ep in range(args.epochs):
                model.train()
                for x, y, g in tr:
                    x, y, g = x.to(device), y.to(device), g.to(device)
                    xb = [x[:, idx] for idx in blocks]
                    if method == "FlexMoE":
                        # warm up the experts through the generalised router first
                        logits, _ = model(xb, g, warmup=(ep < 5))
                    else:
                        logits, _ = model(xb, g)
                    if method == "Reweigh":
                        loss = reweigh(logits, y, g)
                    else:
                        per = torch.zeros(ng, device=device)
                        present = torch.zeros(ng, dtype=torch.bool, device=device)
                        for gid in range(ng):
                            m = (g == gid)
                            if bool(m.any()):
                                per[gid] = nn.functional.cross_entropy(logits[m], y[m],
                                                                       weight=cls_w)
                                present[gid] = True
                        if method == "REMIND":
                            loss = (lam * per * present).sum() / lam[present].sum().clamp_min(1e-8)
                            with torch.no_grad():
                                ema = 0.9 * ema + 0.1 * per.detach()
                                step += 1
                                if step % 50 == 0:          # refresh lambda every N steps
                                    lam = torch.softmax(ema, 0)
                        else:                                # FlexMoE: plain mean
                            loss = per[present].mean()
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                acc, lg, f1, tg = evaluate(model, te, blocks, device, ng, nc)
                if min(acc) > best[0]:
                    best = (min(acc), (acc, lg, f1, tg))
            acc, lg, f1, tg = best[1]
            print(f"  {method} s{seed}: worst={min(acc)*100:.2f} "
                  f"per-group={[round(a*100,1) for a in acc]}", flush=True)
            for gi in range(ng):
                rows.append(dict(method=method, seed=seed, group=f"g{gi}", n=tg[gi],
                                 accuracy=acc[gi], macro_f1=f1[gi], loss=lg[gi],
                                 R_star=rstar[gi] if gi < len(rstar) else "",
                                 excess_loss=(lg[gi] - rstar[gi]) if gi < len(rstar) else ""))

    p = os.path.join(out, "metrics_long.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "seed", "group", "n", "accuracy",
                                           "macro_f1", "loss", "R_star", "excess_loss"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {p}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
