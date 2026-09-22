"""Reweigh, FlexMoE and REMIND on EMBED, emitting the Step 6 metric schema.

EMBED is the paper's native setting, so the modality framing needs no adaptation here: a
modality is an image view and a group is the subset of views a breast actually has.

Reuses train_embed_xenia's data loading, patient-level split and R*_g estimation so these rows
are directly comparable to our own arms. Same seeds, same split seed, same cached ViT features.

Read the header of model/baselines.py first: these are reimplementations from the paper's
description, not the authors' released code, which we could not locate.

  python -m dro_hetero_anchors.src.run_baselines_embed --index ... --cache ...
"""
from __future__ import annotations
import argparse, csv, json, os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dro_hetero_anchors.src.train_embed_xenia import (
    load_group_tensors, patient_split, _subset, estimate_optimal_losses, NUM_CLASSES)
from dro_hetero_anchors.src.model.embed_xenia import VIEWS, GROUP_VIEWS, GROUPS
from dro_hetero_anchors.src.model.baselines import (
    FlexMoEEmbed, FlexMoESparseEmbed, inverse_frequency_weights)


def evaluate(model, data, masks, split, device, rstar):
    model.eval()
    rows = {}
    with torch.no_grad():
        for g in [g for g in GROUPS if g in data]:
            m = masks[g][split]
            if m.sum() == 0:
                continue
            sub = _subset(data[g], m, device)
            logits, _ = (model(g, sub["feats"], warmup=False)
                         if isinstance(model, FlexMoESparseEmbed) else model(g, sub["feats"]))
            y = sub["y"]
            loss = float(F.cross_entropy(logits, y))
            pred = logits.argmax(1)
            acc = float((pred == y).float().mean())
            f1s = []
            for c in range(NUM_CLASSES):
                tp = float(((pred == c) & (y == c)).sum())
                fp = float(((pred == c) & (y != c)).sum())
                fn = float(((pred != c) & (y == c)).sum())
                p = tp / max(1e-9, tp + fp); r = tp / max(1e-9, tp + fn)
                f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
            rows[g] = dict(n=int(m.sum()), acc=acc, macro_f1=sum(f1s) / len(f1s),
                           loss=loss, R_star=rstar.get(g, 0.0),
                           # signed, not clamped: below-zero means the shared model beats a
                           # model trained on that group alone
                           excess_loss=loss - rstar.get(g, 0.0))
    return rows


def train_one(method, data, masks, device, rstar, seed, epochs=20, lr=5e-5, wd=5e-5,
              batch=32, n_experts=128, capacity_matched=False, verbose=True, dro_gamma=0.02, v4=False):
    torch.manual_seed(seed); np.random.seed(seed)
    groups = [g for g in GROUPS if g in data]
    gv = {g: GROUP_VIEWS[g] for g in groups}
    if method == "FlexMoE":
        # released defaults 16 experts / d=128; capacity-matched 4 / d=64 sits beside our
        # 276,228 parameters so the comparison isolates the architecture
        dm, ne = (64, 4) if capacity_matched else (128, 16)
        model = FlexMoESparseEmbed(VIEWS, gv, d_model=dm, n_experts=ne,
                                   top_k=min(4, ne), num_classes=NUM_CLASSES).to(device)
    else:
        # Paper, Implementation Details: "a REMIND MoE transformer with 128 experts and one
        # slot per expert, embedding size 768, num of heads 8". Table 16 sweeps E in
        # {32, 64, 128} and reports 128 as best (80.7 average).
        ne_ = 8 if capacity_matched else n_experts
        # REMIND gets the group-specific residual routing (paper eq. 7-8); Reweigh is the
        # paper's Soft MoE backbone with fixed inverse-frequency group weights.
        model = FlexMoEEmbed(VIEWS, gv, num_classes=NUM_CLASSES, n_experts=ne_,
                             group_routing=(method == "REMIND")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)

    K = len(groups)
    counts = [int(masks[g]["train"].sum()) for g in groups]
    if method == "Reweigh" and not v4:
        lam = inverse_frequency_weights(counts, device)     # fixed, never updated
        # protocol v4: every step already draws the same number of samples from every group, so
        # the sampling frequencies are equal and inverse-frequency weights are uniform; 1/n_g on
        # top of balanced sampling squares the correction (falls through to the uniform branch)
    else:
        lam = torch.full((K,), 1.0 / K, device=device)
    ema = torch.zeros(K, device=device)

    train_sub = {g: (_subset(data[g], masks[g]["train"], device)
                     if masks[g]["train"].sum() > 0 else None) for g in groups}
    steps = max(1, int(np.ceil(max(counts) / batch)))
    gstep = 0
    best_sel, best_state = float("inf"), None
    curve = []
    warm = 5 if not v4 else max(1, int(epochs * 5 / 20))
    with torch.no_grad():
        curve.append({"epoch": -1, "lambda": (lam.detach().cpu().tolist() if method == "REMIND" else None),
                      "val_full": evaluate(model, data, masks, "val", device, rstar),
                      "test_full": evaluate(model, data, masks, "test", device, rstar)})

    for ep in range(epochs):
        model.train()
        # REMIND is trained in two stages: shared routing first, residual matrices in the
        # second half (paper Sec. 7.7: "modality-specific routing matrices in the second stage").
        if method == "REMIND" and hasattr(model, "moe") and model.moe.phi_res is not None:
            model.moe.residual_active = ep >= epochs // 2
        for _ in range(steps):
            opt.zero_grad()
            per = torch.zeros(K, device=device)
            present = torch.zeros(K, dtype=torch.bool, device=device)
            total = model.head.weight.new_zeros(())
            for gi, g in enumerate(groups):
                sub = train_sub[g]
                if sub is None:
                    continue
                n = sub["y"].shape[0]
                sel = torch.randint(0, n, (min(batch, n),), device=device)
                feats = {v: t[sel] for v, t in sub["feats"].items()}
                logits, _ = model(g, feats, warmup=(ep < warm)) \
                    if isinstance(model, FlexMoESparseEmbed) else model(g, feats)
                li = F.cross_entropy(logits, sub["y"][sel])
                per[gi] = li.detach(); present[gi] = True
                total = total + lam[gi] * li
            (total / lam[present].sum().clamp_min(1e-8)).backward()
            opt.step(); gstep += 1
            if method == "REMIND":
                with torch.no_grad():
                    ema = 0.9 * ema + 0.1 * per
                    if gstep % 50 == 0:                 # paper eq. 4, refreshed every N steps
                        lam = lam * torch.exp(dro_gamma * ema)
                        lam = lam / lam.sum()
        sched.step()
        val = evaluate(model, data, masks, "val", device, rstar)
        curve.append({"epoch": ep, "lambda": (lam.detach().cpu().tolist() if method == "REMIND" else None),
                      "val_full": val, "test_full": evaluate(model, data, masks, "test", device, rstar)})
        sel_metric = float(np.mean([v["loss"] for v in val.values()]))
        if sel_metric < best_sel:
            best_sel = sel_metric
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and ep % 5 == 0:
            print(f"    [{method} s{seed}] ep{ep:02d} val avg loss={sel_metric:.4f}", flush=True)
    if best_state:
        model.load_state_dict(best_state)
    model._curve = {"groups": groups, "curve": curve}
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="datasets/embed/index_production.parquet")
    ap.add_argument("--cache", default="datasets/embed/vit_cache_full")
    ap.add_argument("--out", default="runs/baselines_embed")
    ap.add_argument("--methods", nargs="+", default=["Reweigh", "FlexMoE", "REMIND"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--rstar-folds", type=int, default=5)
    ap.add_argument("--capacity-matched", action="store_true")
    ap.add_argument("--dro-gamma", type=float, default=0.02,
                    help="REMIND's sharpness gamma in lambda_k <- lambda_k exp(gamma R_k); the paper "
                         "sweeps {0.5, 0.1, 0.02} on EMBED and uses 0.02")
    ap.add_argument("--v4", action="store_true",
                    help="protocol v4: Reweigh's weights follow the (equal) sampling frequencies; Flex-MoE warm-up scales with the budget")
    ap.add_argument("--disjoint", action="store_true",
                    help="no-overlap variant: g1/g2/g3/g6 with one distinct view each (see model/embed_xenia.py)")
    args = ap.parse_args()
    if args.disjoint:
        from dro_hetero_anchors.src.model.embed_xenia import use_disjoint_views
        use_disjoint_views()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_group_tensors(args.index, args.cache)
    masks = patient_split(data, split_seed=args.split_seed)

    # Reuse the SAME R*_g as our own arms, which Step 6 requires or the excess columns are
    # not comparable across methods.
    rp = os.path.join(args.out, "rstar.json")
    src = "runs/embed_xenia_production/rstar.json"
    if os.path.exists(rp):
        rstar = json.load(open(rp))
    elif os.path.exists(src):
        rstar = json.load(open(src)); json.dump(rstar, open(rp, "w"), indent=2)
        print("reusing R*_g from the production run")
    else:
        rstar = estimate_optimal_losses(data, masks, device, folds=args.rstar_folds, seed=0)
        json.dump(rstar, open(rp, "w"), indent=2)
    print("R*_g = " + ", ".join(f"{g}:{rstar.get(g,0):.3f}" for g in GROUPS if g in data))

    rows = []
    for method in args.methods:
        for seed in args.seeds:
            model = train_one(method, data, masks, device, rstar, seed, epochs=args.epochs, dro_gamma=args.dro_gamma,
                              capacity_matched=args.capacity_matched, v4=args.v4)
            n_params = sum(p.numel() for p in model.parameters())
            json.dump({**model._curve, "n_params": n_params},
                      open(os.path.join(args.out, f"curve_{method}_s{seed}.json"), "w"))
            test = evaluate(model, data, masks, "test", device, rstar)
            worst = min(v["acc"] for v in test.values())
            print(f"[seed {seed}] {method}: worst={worst:.3f} "
                  f"overall={np.mean([v['acc'] for v in test.values()]):.3f} "
                  f"params={n_params:,}", flush=True)
            for g, v in test.items():
                rows.append(dict(method=method, seed=seed, group=g, n_params=n_params, **v))

    p = os.path.join(args.out, "metrics_long.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "seed", "group", "n_params", "n", "acc",
                                           "macro_f1", "loss", "R_star", "excess_loss"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {p} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
