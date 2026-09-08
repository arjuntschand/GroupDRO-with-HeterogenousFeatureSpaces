"""Estimate the per-group reference losses R*_g for the tabular datasets, following
Xenia's Step 2 protocol (EMBED_Experiments_Description.docx):

  R*_g = the lowest loss achievable using only the features group g has.
  Estimated by training a DEDICATED model on group g alone (its own encoder + its own
  head, nothing shared) with 5-fold cross-validation, and averaging the per-sample
  out-of-fold cross-entropy. Never evaluate on data the model trained on.

These six/four numbers are constants for the rest of the project (no gradient), and are
what the regret-DRO objective subtracts: excess_g = max(0, L_g - R*_g).

Usage:
  python -m dro_hetero_anchors.src.tools.estimate_rstar_tabular --dataset nhanes \
      --base experiments/nhanes_disjoint_pergroup_gdro.yaml --out runs/rstar_nhanes_disjoint.json
  python -m dro_hetero_anchors.src.tools.estimate_rstar_tabular --dataset fedheart \
      --base experiments/fedheart_exp_paper_hetagg_gdro.yaml --out runs/rstar_fedheart.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml




def collect_group_tensors(loader, feature_indices=None):
    """Materialize (X, y) per group from a loader."""
    byg = {}
    for x, y, g in loader:
        for gid in torch.unique(g).tolist():
            m = (g == gid)
            xs = x[m]
            if feature_indices is not None and gid in feature_indices:
                xs = xs[:, feature_indices[gid]]
            byg.setdefault(gid, [[], []])
            byg[gid][0].append(xs)
            byg[gid][1].append(y[m])
    return {gid: (torch.cat(v[0]), torch.cat(v[1])) for gid, v in byg.items()}


class DedicatedModel(nn.Module):
    """One branch of the full method: this group's encoder + its OWN head."""

    def __init__(self, in_dim, latent, num_classes, hidden=64, dropout=0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, latent), nn.LayerNorm(latent), nn.ReLU(),
        )
        self.head = nn.Linear(latent, num_classes)

    def forward(self, x):
        return self.head(self.enc(x))


def rstar_for_group(X, y, num_classes, folds=5, epochs=200, lr=1e-3, wd=1e-4,
                    latent=64, seed=0, class_weight=None, device="cpu"):
    n = len(y)
    if n < folds * 2:
        folds = max(2, n // 2)
    rng = np.random.RandomState(seed)
    fold = rng.randint(0, folds, size=n)
    oof = np.full(n, np.nan)
    for f in range(folds):
        tr, te = fold != f, fold == f
        if tr.sum() < 2 or te.sum() == 0:
            continue
        torch.manual_seed(seed + f)
        model = DedicatedModel(X.shape[1], latent, num_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        Xtr, ytr = X[tr].to(device), y[tr].to(device)
        Xte, yte = X[te].to(device), y[te].to(device)
        cw = class_weight.to(device) if class_weight is not None else None
        # Nested CV: carve an INNER validation split out of the training folds and early-stop
        # on it. We must not checkpoint on lowest TRAIN loss (selects the most-overfit state
        # and inflates R*_g on small groups — the failure flagged in Xenia's Step 2), and we
        # must not peek at the outer held-out fold (that would leak).
        ntr = len(ytr)
        n_val = max(2, int(0.15 * ntr))
        perm = torch.randperm(ntr, generator=torch.Generator().manual_seed(seed + f))
        vi, ti = perm[:n_val], perm[n_val:]
        Xi, yi_, Xv, yv = Xtr[ti], ytr[ti], Xtr[vi], ytr[vi]
        best_state, best_val = None, float("inf")
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            F.cross_entropy(model(Xi), yi_, weight=cw).backward(); opt.step()
            if ep % 5 == 0 or ep == epochs - 1:
                model.eval()
                with torch.no_grad():
                    vl = F.cross_entropy(model(Xv), yv, weight=cw).item()
                if vl < best_val:
                    best_val = vl
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            ce = F.cross_entropy(model(Xte), yte, reduction="none").cpu().numpy()
        oof[te] = ce
    return float(np.nanmean(oof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nhanes", "fedheart"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.base))
    device = "cpu"
    if args.dataset == "nhanes":
        from ..datasets_nhanes import build_nhanes_loaders
        tr, te, info = build_nhanes_loaders(
            data_root=cfg.get("root", "datasets"), batch_size=cfg.get("batch_size", 64),
            num_workers=0, train_frac=cfg.get("train_frac", 0.8),
            seed=cfg.get("data_split_seed", cfg.get("seed", 42)),
            data_split_seed=cfg.get("data_split_seed"),
            subsample_seed=cfg.get("seed"),
            feature_mode=cfg.get("feature_mode", "nested"),
            use_post_pandemic=cfg.get("use_post_pandemic", True),
        )
        fi = None
        fidx = info.get("feature_indices")
        if fidx is not None:
            fi = {int(k): (v if torch.is_tensor(v) else torch.tensor(v)) for k, v in fidx.items()}
    else:
        from ..datasets_fedheart import build_fedheart_loaders
        tr, te, info = build_fedheart_loaders(
            data_root=cfg.get("root", "datasets"), batch_size=cfg.get("batch_size", 64),
            num_workers=0, train_frac=cfg.get("train_frac", 0.8),
            seed=cfg.get("data_split_seed", cfg.get("seed", 42)),
            group_max_train_samples=cfg.get("group_max_train_samples"),
            subsample_seed=cfg.get("seed"),
        )
        fi = None
        fm = cfg.get("feature_mask")
        if fm and cfg.get("true_hetero_input_dim", False):
            fi = {i: torch.tensor(m) for i, m in enumerate(fm) if m is not None}

    num_classes = cfg.get("num_classes", 2)
    data = collect_group_tensors(tr, fi)
    rstar = {}
    print(f"[{args.dataset}] estimating R*_g with {args.folds}-fold OOF dedicated models "
          f"({len(data)} groups)")
    for gid in sorted(data):
        X, y = data[gid]
        r = rstar_for_group(X, y, num_classes, folds=args.folds, epochs=args.epochs,
                            seed=args.seed, device=device)
        rstar[int(gid)] = r
        print(f"  group {gid}: n={len(y):5d}  features={X.shape[1]:3d}  R*_g = {r:.4f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"rstar": rstar, "config": args.base, "folds": args.folds,
               "epochs": args.epochs, "seed": args.seed}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
