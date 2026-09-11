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
from dro_hetero_anchors.src.model.baselines import FlexMoEEmbed, inverse_frequency_weights


def evaluate(model, data, masks, split, device, rstar):
    model.eval()
    rows = {}
    with torch.no_grad():
        for g in [g for g in GROUPS if g in data]:
            m = masks[g][split]
            if m.sum() == 0:
                continue
            sub = _subset(data[g], m, device)
            logits, _ = model(g, sub["feats"])
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
              batch=32, n_experts=4, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    groups = [g for g in GROUPS if g in data]
    gv = {g: GROUP_VIEWS[g] for g in groups}
    model = FlexMoEEmbed(VIEWS, gv, num_classes=NUM_CLASSES, n_experts=n_experts).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)

    K = len(groups)
    counts = [int(masks[g]["train"].sum()) for g in groups]
    if method == "Reweigh":
        lam = inverse_frequency_weights(counts, device)     # fixed, never updated
    else:
        lam = torch.full((K,), 1.0 / K, device=device)
    ema = torch.zeros(K, device=device)

    train_sub = {g: (_subset(data[g], masks[g]["train"], device)
                     if masks[g]["train"].sum() > 0 else None) for g in groups}
    steps = max(1, int(np.ceil(max(counts) / batch)))
    gstep = 0
    best_sel, best_state = float("inf"), None

    for ep in range(epochs):
        model.train()
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
                logits, _ = model(g, feats)
                li = F.cross_entropy(logits, sub["y"][sel])
                per[gi] = li.detach(); present[gi] = True
                total = total + lam[gi] * li
            (total / lam[present].sum().clamp_min(1e-8)).backward()
            opt.step(); gstep += 1
            if method == "REMIND":
                with torch.no_grad():
                    ema = 0.9 * ema + 0.1 * per
                    if gstep % 50 == 0:                 # refresh lambda every N steps
                        lam = torch.softmax(ema, 0)
        sched.step()
        val = evaluate(model, data, masks, "val", device, rstar)
        sel_metric = float(np.mean([v["loss"] for v in val.values()]))
        if sel_metric < best_sel:
            best_sel = sel_metric
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and ep % 5 == 0:
            print(f"    [{method} s{seed}] ep{ep:02d} val avg loss={sel_metric:.4f}", flush=True)
    if best_state:
        model.load_state_dict(best_state)
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
    args = ap.parse_args()

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
            model = train_one(method, data, masks, device, rstar, seed, epochs=args.epochs)
            n_params = sum(p.numel() for p in model.parameters())
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
