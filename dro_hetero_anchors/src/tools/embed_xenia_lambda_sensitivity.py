"""Sensitivity of the anchor weight (lambda_fit = lambda_sep = lam) for the 'ours'
method, holding everything else at Xenia's spec. Diagnoses whether the anchors can
help at a lower weight, and prints the loss-component magnitudes at init.

Reported as a clearly-labeled SENSITIVITY, separate from the faithful-spec result
(which fixes lam = 1.0). Not a replacement for the headline.

Usage: python -m dro_hetero_anchors.src.tools.embed_xenia_lambda_sensitivity
"""
from __future__ import annotations
import numpy as np, torch, torch.nn.functional as F
from ..train_embed_xenia import (
    load_group_tensors, patient_split, estimate_optimal_losses, train_one, evaluate, HEAD_GROUPS)
from ..model.embed_xenia import XeniaEmbedModel, anchor_fit_loss, anchor_sep_loss, GROUP_VIEWS

INDEX = "datasets/embed/index_xenia_offline.parquet"
CACHE = "datasets/embed/vit_cache"
SEEDS = [0, 1, 42]
LAMS = [0.0, 0.1, 0.3, 1.0]   # 0.0 == regret-only (no anchors)


def summarize(pg):
    accs = {g: v["acc"] for g, v in pg.items()}
    overall = np.average(list(accs.values()), weights=[pg[g]["n"] for g in accs])
    tail = np.mean([a for g, a in accs.items() if g not in HEAD_GROUPS])
    worst = min(accs.values())
    return overall, tail, worst


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_group_tensors(INDEX, CACHE)
    masks = patient_split(data, split_seed=0)

    # loss-component magnitudes at init (seed 0), to see if sep swamps task
    torch.manual_seed(0)
    m = XeniaEmbedModel().to(device)
    g = "g6"; sub = {v: data[g]["feats"][v][masks[g]["train"]][:64].to(device) for v in GROUP_VIEWS[g]}
    y = data[g]["y"][masks[g]["train"]][:64].to(device)
    logits, z = m(g, sub)
    print("== loss magnitudes at init (g6, 64 samples) ==")
    print(f"  L_task(CE) = {F.cross_entropy(logits,y).item():.3f}")
    print(f"  L_fit      = {anchor_fit_loss(z,y,m.anchors).item():.3f}")
    print(f"  L_sep      = {anchor_sep_loss(m).item():.3f}   (weighted by lam_sep)")

    print("\n== anchor-weight sensitivity for 'ours' (overall-wt / tail / worst), mean±std over seeds ==")
    rows = []
    for lam in LAMS:
        per = {"overall": [], "tail": [], "worst": []}
        for seed in SEEDS:
            rstar = estimate_optimal_losses(data, masks, device, folds=5, seed=seed)
            model, _ = train_one("ours", data, masks, device, rstar, seed,
                                 epochs=20, lam_fit=lam, lam_sep=lam, verbose=False)
            _, pg = evaluate(model, data, masks, "test", device, rstar)
            o, t, w = summarize(pg)
            per["overall"].append(o); per["tail"].append(t); per["worst"].append(w)
        a = {k: (np.mean(v), np.std(v)) for k, v in per.items()}
        tag = "regret-only" if lam == 0.0 else f"lam={lam}"
        print(f"  {tag:12s}: overall {a['overall'][0]:.3f}±{a['overall'][1]:.3f}  "
              f"tail {a['tail'][0]:.3f}±{a['tail'][1]:.3f}  worst {a['worst'][0]:.3f}±{a['worst'][1]:.3f}")
        rows.append((lam, a))
    print("\n(lam=1.0 is Xenia's faithful-spec 'ours'; lam=0.0 is regret-only.)")


if __name__ == "__main__":
    main()
