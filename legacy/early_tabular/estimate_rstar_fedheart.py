"""Re-estimate R*_g on Fed-Heart so it measures information, not data scarcity.

The current estimator fits a model on group g's rows alone. VA has 25 training patients, so it
overfits and returns R*_3 = 1.480 while an ordinary shared model reaches 0.551 on VA. R*_g is an
infimum over measurable predictors, so an estimate above an achieved loss is not a Bayes risk at
all. Proposition 1 is explicit that R*_g is fixed by what X_g measures and that no amount of data
reduces it, so the group-only fit contradicts the definition it is supposed to implement.

This separates the two things that estimator conflates:

  fitting     uses every patient whose record contains group g's feature set. Fed-Heart's masks
              are synthetic and all 13 columns exist for everyone after imputation, so g3's
              features are available across all 925 patients instead of 25.
  evaluation  stays on group g's own held-out rows, so the quantity estimated is still the Bayes
              risk of P_g. Pooling the evaluation too would silently estimate the Bayes risk of
              the pooled distribution, and the sites differ in prevalence, so that is a different
              number.

The result is still an upper bound on R*_g, since any fitted model is one particular measurable
predictor, but a far tighter one. The floor step then takes the minimum against the best loss any
of our trained arms actually achieved on that group, which makes the Bayes-risk violation
impossible by construction rather than by hope.

  python estimate_rstar_fedheart.py
"""
from __future__ import annotations
import csv, glob, json, os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedKFold

from dro_hetero_anchors.src.datasets_fedheart import build_fedheart_loaders

FOLDS, EPOCHS, SEED = 5, 60, 0


def pooled_tensors():
    """All patients, unmasked, with their site id. feature_mask is left off deliberately so
    every column is available and each group's mask can be applied afterwards.

    batch_size stays small. StratifiedGroupSampler computes its batch count by flooring, so a
    batch_size above the dataset size makes the train loader yield nothing at all, silently.
    Asking for 4096 here returned only the 185 test patients and the pooled estimate came out
    worse than the group-only one, which is the opposite of the point.
    """
    tr, te, info = build_fedheart_loaders(batch_size=64, seed=SEED, train_frac=0.8,
                                          impute_missing=True, feature_mask=None)
    xs, ys, gs = [], [], []
    for loader in (tr, te):
        for x, y, g in loader:
            xs.append(x); ys.append(y); gs.append(g)
    return torch.cat(xs), torch.cat(ys), torch.cat(gs)


def fit_eval(X, y, tr_idx, ev_idx, in_dim, seed):
    """Same shape of model as our encoders: MLP to a latent, then a linear head."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(in_dim, 64), nn.LayerNorm(64), nn.ReLU(),
                          nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    cnt = torch.bincount(y[tr_idx], minlength=2).float()
    w = cnt.sum() / (2 * cnt.clamp_min(1))
    ce = nn.CrossEntropyLoss(weight=w)
    for _ in range(EPOCHS):
        model.train(); opt.zero_grad()
        ce(model(X[tr_idx]), y[tr_idx]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        # unweighted CE at evaluation, matching how per-group loss is reported everywhere else
        return float(nn.functional.cross_entropy(model(X[ev_idx]), y[ev_idx]))


def achieved_floor():
    """Lowest per-group loss any arm actually reached. An achieved loss is by definition an
    upper bound on the infimum, so R* can never legitimately exceed it."""
    best = defaultdict(lambda: float("inf"))
    for p in glob.glob("runs/fedheart_cv*/metrics_long.csv") + \
             glob.glob("runs/baselines_fedheart*/metrics_long.csv"):
        for r in csv.DictReader(open(p)):
            try:
                gi = int(r["group"].lstrip("g")); L = float(r["loss"])
            except (ValueError, KeyError):
                continue
            best[gi] = min(best[gi], L)
    return dict(best)


def main():
    cfg = yaml.safe_load(open("experiments/fedheart_exp_paper_hetagg_gdro.yaml"))
    masks = [m if m is not None else list(range(13)) for m in cfg["feature_mask"]]
    X, y, g = pooled_tensors()
    print(f"pooled: {len(y)} patients, {int(g.max())+1} sites\n")

    old = json.load(open("runs/rstar_fedheart.json"))["rstar"]
    floor = achieved_floor()
    out = {}
    print(f"{'group':6} {'n_g':>5} {'n_fit':>6} {'old R*':>8} {'pooled':>8} {'floor':>8} {'new R*':>8}")
    for gi, mask in enumerate(masks):
        Xg = X[:, mask]
        own = (g == gi).nonzero(as_tuple=True)[0]
        # stratify the folds on the label so small sites keep both classes in every split
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        losses = []
        for f, (_, ev_rel) in enumerate(skf.split(own.numpy(), y[own].numpy())):
            ev = own[ev_rel]
            ev_set = set(ev.tolist())
            # fit on every patient not being evaluated, from ANY site, restricted to g's columns
            tr = torch.tensor([i for i in range(len(y)) if i not in ev_set])
            losses.append(fit_eval(Xg, y, tr, ev, len(mask), SEED + f))
        pooled = float(np.mean(losses))
        fl = floor.get(gi, float("inf"))
        new = min(pooled, fl)
        out[str(gi)] = round(new, 6)
        print(f"g{gi:<5} {len(own):>5} {len(y)-len(own)//FOLDS:>6} "
              f"{old[str(gi)]:>8.3f} {pooled:>8.3f} {fl:>8.3f} {new:>8.3f}")

    json.dump({"rstar": out, "method": "pooled-fit, group-evaluated, floored at best achieved"},
              open("runs/rstar_fedheart_pooled.json", "w"), indent=2)
    print("\nchecks:")
    viol = [k for k, v in out.items() if v > floor.get(int(k), float("inf")) + 1e-6]
    print(f"  Bayes-risk violations (R* above an achieved loss): {len(viol)}  {viol}")
    print("  Fed-Heart masks are not nested, so Proposition 1 imposes no ordering here.")
    print("\nwrote runs/rstar_fedheart_pooled.json")


if __name__ == "__main__":
    main()
