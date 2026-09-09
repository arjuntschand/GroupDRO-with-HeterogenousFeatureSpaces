"""Synthetic scaling study: sweep the axes of heterogeneity one at a time.

Real datasets give us two or three points on the heterogeneity spectrum. To claim that the
method's benefit *scales with how different the feature spaces are*, we need a controlled
sweep where every other factor is held fixed. This generates data where we set exactly:

  --sweep overlap        fraction of features shared between groups (1.0 = homogeneous,
                         0.0 = fully disjoint). Directly tests the paper's core thesis.
  --sweep private        number of private features per group.
  --sweep groups         number of groups (clients), 2 to 16.
  --sweep samples        samples per group (data scarcity).
  --sweep severity       group imbalance severity: how much smaller the tail groups are.

Data model. There is one shared latent cause y. Each group g observes a feature vector made
of (i) shared features driven by y, and (ii) private features also driven by y but through a
group-specific random projection, so different groups genuinely require different decoders.
Noise and class balance are identical across groups, so the ONLY thing changing along a
sweep is the axis being swept.

Every point reports mean and std over seeds plus a paired t-test of the method against the
shared-ERM baseline.

Usage:
  python run_synthetic_scaling.py --sweep overlap --seeds 5
  python run_synthetic_scaling.py --sweep groups --seeds 5
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from dro_hetero_anchors.src.model.anchors import AnchorModule
from dro_hetero_anchors.src.model.losses import (
    per_class_batch_moments, anchor_fit_loss, anchor_sep_loss)
from dro_hetero_anchors.src.model.groupdro import GroupDRO


def make_data(n_groups=3, n_shared=10, n_private=5, overlap=None, n_per_group=800,
              severity=1.0, noise=1.6, seed=0, n_classes=2):
    """Generate heterogeneous-feature-space groups.

    overlap, if given, overrides n_shared/n_private: total feature budget is fixed at 15 and
    overlap is the fraction of it that is shared (so overlap=1.0 means fully homogeneous).
    severity shrinks the later ("tail") groups: group g gets n_per_group * severity**g samples.
    """
    rng = np.random.RandomState(seed)
    total = n_shared + n_private
    if overlap is not None:
        total = 15
        n_shared = int(round(overlap * total))
        n_private = total - n_shared

    # one shared latent cause
    w_shared = rng.randn(max(n_shared, 1))
    groups = []
    for g in range(n_groups):
        n = max(60, int(n_per_group * (severity ** g)))
        y = rng.randint(0, n_classes, size=n)
        signal = (y - 0.5) * 2.0
        X = np.zeros((n, total), dtype=np.float32)
        if n_shared > 0:
            # shared block: same generative weights for every group
            X[:, :n_shared] = (signal[:, None] * w_shared[None, :] * 0.55
                               + rng.randn(n, n_shared) * noise)
        if n_private > 0:
            # private block: group-specific projection, so each group needs its own decoder
            w_priv = rng.randn(n_private) * 1.2
            X[:, n_shared:] = (signal[:, None] * w_priv[None, :] * 0.55
                               + rng.randn(n, n_private) * noise)
        groups.append((torch.tensor(X), torch.tensor(y, dtype=torch.long)))
    return groups, n_shared, n_private, total


class Enc(nn.Module):
    def __init__(self, in_dim, latent=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                 nn.Linear(hidden, latent))

    def forward(self, x):
        return self.net(x)


def run_one(groups, per_group: bool, gdro: bool, anchors_on: bool, seed=0,
            latent=32, epochs=200, lr=1e-3, n_classes=2, hidden=64, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    K = len(groups)
    # 80/20 split per group
    tr, te = [], []
    for X, y in groups:
        n = len(y); idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        c = int(0.8 * n)
        tr.append((X[idx[:c]].to(device), y[idx[:c]].to(device)))
        te.append((X[idx[c:]].to(device), y[idx[c:]].to(device)))

    in_dim = groups[0][0].shape[1]
    if per_group:
        encs = nn.ModuleList([Enc(in_dim, latent, hidden) for _ in range(K)]).to(device)
    else:
        shared = Enc(in_dim, latent, hidden).to(device)
        encs = nn.ModuleList([shared] * K)
    head = nn.Linear(latent, n_classes).to(device)
    anc = AnchorModule(n_classes, latent, eps=1e-4).to(device)
    params = list({id(p): p for p in list(encs.parameters()) + list(head.parameters())
                   + list(anc.parameters())}.values())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    dro = GroupDRO(num_groups=K, eta=1.0, gamma=0.9, update_mode="softmax",
                   device=torch.device(device), uniform_init=True) if gdro else None

    for ep in range(epochs):
        opt.zero_grad()
        losses, zs, ys = [], [], []
        for g, (X, y) in enumerate(tr):
            z = encs[g](X)
            losses.append(F.cross_entropy(head(z), y))
            zs.append(z); ys.append(y)
        if dro is not None:
            dro.update_weights({i: l.detach() for i, l in enumerate(losses)},
                               {i: len(tr[i][1]) for i in range(K)})
            q = dro.q
            loss = sum(q[i] * losses[i] for i in range(K))
        else:
            loss = sum(losses) / K
        if anchors_on:
            z_all = torch.cat(zs); y_all = torch.cat(ys)
            mom = per_class_batch_moments(z_all, y_all, n_classes, 1e-4)
            m_a, S_a, L_n = anc.forward()
            loss = loss + 0.1 * anchor_fit_loss(m_a, S_a, mom, 1e-4)
            loss = loss + 0.1 * anchor_sep_loss(m_a, S_a, L_n, head, n_classes, 8,
                                                torch.device(device), sep_method="classifier",
                                                margin=2.0, eps=1e-4)
        loss.backward(); opt.step()

    accs = []
    with torch.no_grad():
        for g, (X, y) in enumerate(te):
            accs.append((head(encs[g](X)).argmax(1) == y).float().mean().item())
    return {"worst": min(accs), "mean": float(np.mean(accs)), "per_group": accs}


SWEEPS = {
    "overlap":  ("feature overlap between groups", [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
    "private":  ("private features per group",     [0, 2, 5, 8, 12]),
    "groups":   ("number of groups",               [2, 3, 4, 6, 8, 12, 16]),
    "samples":  ("samples per group",              [80, 150, 300, 600, 1200]),
    "severity": ("group imbalance (size ratio)",   [1.0, 0.7, 0.5, 0.3, 0.15]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, choices=list(SWEEPS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/synthetic")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    label, values = SWEEPS[args.sweep]
    seeds = list(range(args.seeds))
    ARMS = [("shared_ERM", False, False, False), ("pergroup_ERM", True, False, False),
            ("pergroup_GDRO", True, True, False), ("ours", True, True, True)]

    print(f"\n########## SYNTHETIC SWEEP: {label} ##########")
    print(f"{args.sweep:>10} | " + " | ".join(f"{a[0]:>14}" for a in ARMS) + " | ours vs shared")
    out = {}
    for v in values:
        kw = dict(n_groups=3, n_shared=10, n_private=5, n_per_group=300, severity=0.45)
        if args.sweep == "overlap":   kw["overlap"] = v
        if args.sweep == "private":   kw["n_private"] = v
        if args.sweep == "groups":    kw["n_groups"] = v
        if args.sweep == "samples":   kw["n_per_group"] = v
        if args.sweep == "severity":  kw["severity"] = v
        per_arm = {a[0]: [] for a in ARMS}
        for s in seeds:
            groups, ns, npv, tot = make_data(seed=s, **kw)
            for name, pg, gd, an in ARMS:
                per_arm[name].append(run_one(groups, pg, gd, an, seed=s)["worst"] * 100)
        base = np.array(per_arm["shared_ERM"]); ours = np.array(per_arm["ours"])
        t, p = stats.ttest_rel(ours, base) if len(base) > 2 else (np.nan, np.nan)
        cells = " | ".join(f"{np.mean(per_arm[a[0]]):6.2f}±{np.std(per_arm[a[0]]):4.2f}" for a in ARMS)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print(f"{v:>10} | {cells} | {ours.mean()-base.mean():+6.2f} (p={p:.3f}) {sig}")
        out[str(v)] = {a[0]: per_arm[a[0]] for a in ARMS}
    json.dump({"sweep": args.sweep, "label": label, "values": values, "results": out},
              open(f"{args.out}/{args.sweep}.json", "w"), indent=2)
    print(f"\nwrote {args.out}/{args.sweep}.json")


if __name__ == "__main__":
    main()
