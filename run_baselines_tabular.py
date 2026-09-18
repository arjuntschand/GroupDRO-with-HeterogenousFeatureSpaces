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
    # At released defaults Flex-MoE carries 536,354 parameters against our 30,818, a 17x
    # capacity advantage that has nothing to do with the method. --capacity-matched sizes it
    # to 4 experts at d=64 (35,722 params) so the comparison isolates the architecture. Both
    # settings get reported; neither is hidden.
    ap.add_argument("--capacity-matched", action="store_true")
    ap.add_argument("--dro-gamma", type=float, default=0.02,
                    help="REMIND sharpness gamma (paper: 0.02 on EMBED, swept over {0.5, 0.1, 0.02})")
    # Fed-Heart's own arms are evaluated with 5-fold CV over all 925 patients, so a single
    # split here would compare them against baselines measured on 185. Switzerland in
    # particular drops from 125 evaluated patients to 25, which is where most of the apparent
    # variance in the Fed-Heart baseline columns was coming from.
    ap.add_argument("--folds", type=int, default=1)
    # Our arms carve val_frac out of train to drive the DRO lambda signal. Baselines do not use
    # that signal, but they must lose the same rows, otherwise they train on 522 patients against
    # our 449 on Fed-Heart and the comparison measures sample size as much as method.
    # lets the baselines follow a variant config, e.g. the uncapped Fed-Heart robustness check.
    # Without this they read the capped config while our arms read the uncapped one, which would
    # give our side 83 and 136 training patients on the two groups that decide worst-group
    # accuracy against the baselines' 20 and 25.
    ap.add_argument("--base", default=None)
    ap.add_argument("--val-frac", type=float, default=None,
                    help="default: read val_frac from the same config our arms use")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dataset == "nhanes":
        from dro_hetero_anchors.src.datasets_nhanes import build_nhanes_loaders as build
        base = yaml.safe_load(open(args.base or "experiments/nhanes_pergroup_gdro.yaml"))
        rstar_path = "runs/rstar_nhanes_nested.json"
    else:
        from dro_hetero_anchors.src.datasets_fedheart import build_fedheart_loaders as build
        base = yaml.safe_load(open(args.base or "experiments/fedheart_exp_paper_hetagg_gdro.yaml"))
        rstar_path = "runs/rstar_fedheart.json"

    _VF = args.val_frac if args.val_frac is not None else float(base.get("val_frac", 0.0))
    print(f"val_frac = {_VF}  (matched to the config our arms use)", flush=True)

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
          fold_acc, fold_loss, fold_f1, fold_n = [], [], [], []
          for fold in range(args.folds):
            torch.manual_seed(seed); np.random.seed(seed)
            cfg = copy.deepcopy(base); cfg["seed"] = seed
            # fold identity goes through the loader's split seed, matching run_fedheart_cv,
            # which sets cfg["data_split_seed"] and train_fedheart passes it as the loader seed
            split_seed = (1000 + fold) if args.folds > 1 else cfg.get("data_split_seed", seed)
            frac = (1.0 - 1.0 / args.folds) if args.folds > 1 else cfg.get("train_frac", 0.8)
            if args.dataset == "nhanes":
                tr, te, info = build(batch_size=cfg.get("batch_size", 128),
                                     seed=seed, stratified=cfg.get("stratified_batching", True),
                                     train_frac=cfg.get("train_frac", 0.8),
                                     val_frac=_VF,
                                     use_post_pandemic=cfg.get("use_post_pandemic", True),
                                     data_split_seed=cfg.get("data_split_seed"),
                                     feature_mode=cfg.get("feature_mode", "nested"),
                                     # The Fed-Heart branch below passes the per-group training
                                     # cap; this branch did not, so a capped NHANES config gave
                                     # the baselines the full training set while our own arms
                                     # trained capped. Byte-identical CSVs across regimes were
                                     # the symptom. subsample_seed matches train_nhanes.py.
                                     group_max_train_samples=cfg.get("group_max_train_samples"),
                                     subsample_seed=seed if cfg.get("data_split_seed") is not None else None)
            else:
                tr, te, info = build(batch_size=cfg.get("batch_size", 64),
                                     seed=split_seed,
                                     stratified=cfg.get("stratified_batching", True),
                                     train_frac=frac, val_frac=_VF,
                                     feature_mask=cfg.get("feature_mask"),
                                     group_max_train_samples=cfg.get("group_max_train_samples"),
                                     impute_missing=True)
            # build_fedheart_loaders and build_nhanes_loaders both call torch.manual_seed with
            # the SPLIT seed internally. Under K-fold that split seed is 1000+fold, identical for
            # every experiment seed, so seeding before the loader leaves every seed with the same
            # model initialisation and all 10 "seeds" return byte-identical results. Reseed here,
            # after the loaders and before the model, so the seed actually reaches the weights.
            torch.manual_seed(seed); np.random.seed(seed)

            _val_loader = info.pop("val_loader", None)
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
                dm, ne = (64, 4) if args.capacity_matched else (128, 16)
                model = FlexMoESparse(len(blocks), group_blocks, [len(b) for b in blocks],
                                      d_model=dm, n_experts=ne, top_k=min(4, ne),
                                      num_classes=nc).to(device)
            else:
                model = FlexMoEModel(
                    [len(b) for b in blocks], group_blocks,
                    latent_dim=cfg.get("latent_dim", 64), num_classes=nc,
                    n_experts=2 if args.capacity_matched else 4,
                    group_routing=(method == "REMIND")).to(device)
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
                if method == "REMIND" and getattr(model, "moe", None) is not None \
                        and model.moe.phi_res is not None:
                    model.moe.residual_active = ep >= args.epochs // 2   # stage 2

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
                                if step % 50 == 0:          # paper eq. 4, every N steps
                                    lam = lam * torch.exp(args.dro_gamma * ema)
                                    lam = lam / lam.sum()
                        else:                                # FlexMoE: plain mean
                            loss = per[present].mean()
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                acc, lg, f1, tg = evaluate(model, te, blocks, device, ng, nc)
                # Select on VALIDATION, report TEST, matching what our own arms do. Selecting on
                # test gave the baselines a best-of-60 advantage on the very set the comparison
                # is scored on.
                if _val_loader is not None:
                    vacc, _, _, _ = evaluate(model, _val_loader, blocks, device, ng, nc)
                    sel = min(vacc)
                else:
                    sel = min(acc)
                if sel > best[0]:
                    best = (sel, (acc, lg, f1, tg))
            acc, lg, f1, tg = best[1]
            n_par = sum(p.numel() for p in model.parameters())
            fold_acc.append(acc); fold_loss.append(lg); fold_f1.append(f1); fold_n.append(tg)
          # pool folds the same way run_fedheart_cv does: weight each fold by its test count,
          # so every patient contributes exactly once across the K folds
          A = np.array(fold_acc); L = np.array(fold_loss); F = np.array(fold_f1)
          W = np.array(fold_n, dtype=float)
          den = np.maximum(W.sum(0), 1)
          acc = ((A * W).sum(0) / den).tolist()
          lg = ((L * W).sum(0) / den).tolist()
          f1 = ((F * W).sum(0) / den).tolist()
          tg = W.sum(0).astype(int).tolist()
          print(f"  {method} s{seed}: worst={min(acc)*100:.2f} params={n_par:,} "
                f"n/group={tg} per-group={[round(a*100,1) for a in acc]}", flush=True)
          for gi in range(ng):
                rows.append(dict(method=method, seed=seed, group=f"g{gi}", n=tg[gi],
                                 n_params=n_par,
                                 accuracy=acc[gi], macro_f1=f1[gi], loss=lg[gi],
                                 R_star=rstar[gi] if gi < len(rstar) else "",
                                 excess_loss=(lg[gi] - rstar[gi]) if gi < len(rstar) else ""))

    p = os.path.join(out, "metrics_long.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "seed", "group", "n", "n_params",
                                           "accuracy", "macro_f1", "loss", "R_star",
                                           "excess_loss"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {p}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
