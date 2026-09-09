"""EMBED training per Xenia's spec — runs entirely on cached ViT embeddings.

Objective (min over params theta, max over group weights lambda):
    min_theta max_{lambda in simplex}
        sum_g lambda_g [ (L_g^task - R*_g) + lam_fit * L_g^fit ]  +  lam_sep * L^sep

  - L_g^task : cross-entropy for group g
  - L_g^fit  : diagonal-W2 anchor fit for group g
  - L^sep    : global anchor separation
  - R*_g     : group g's best achievable CE (5-fold OOF dedicated model). R*=0 -> plain GroupDRO.
  - lambda update: EMA of per-group excess, lambda_g <- lambda_g * exp(gamma * excess_g), renormalize.

Seven methods (Step 5 of the spec), selected by --method:
  erm            dro=off  anchors=off  regret=off        (uniform lambda, CE only)
  groupdro       dro=on   anchors=off  regret=off        (R*=0)      <- headline baseline
  align_only     dro=on   anchors=ON   regret=off        (ablation: anchors)
  regret_only    dro=on   anchors=off  regret=ON         (ablation: regret)
  ours           dro=on   anchors=ON   regret=ON         (both)      <- headline method
  group_only     one dedicated model per group           (also yields R*_g)
  # remind: cite published numbers, not run here

Rows {groupdro, align_only, regret_only, ours} form the 2x2 (anchors x regret);
groupdro vs ours is the headline comparison.
"""
from __future__ import annotations
import argparse, copy, json, os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .model.embed_xenia import (
    XeniaEmbedModel, GROUP_VIEWS, GROUPS, VIEWS, anchor_fit_loss, anchor_sep_loss)

HEAD_GROUPS = {"g4", "g6"}      # >15% frequency; the rest are tail
NUM_CLASSES = 4


# ------------------------------------------------------------ data layer ----

def load_group_tensors(index_path: str, cache_dir: str):
    """Return {group -> dict(feats={view:(N,768)}, y=(N,), empi=(N,))} on cached ViT vecs."""
    idx = pd.read_parquet(index_path)
    emb = np.load(os.path.join(cache_dir, "embeddings.f16.npy")).astype(np.float32)
    path2row = json.load(open(os.path.join(cache_dir, "paths.json")))
    out = {}
    missing = 0
    for g in GROUPS:
        rows = idx[idx.group == g]
        views = GROUP_VIEWS[g]
        feats = {v: [] for v in views}
        ys, empis = [], []
        for _, r in rows.iterrows():
            d = r["paths"] if isinstance(r["paths"], dict) else dict(r["paths"])
            if not all(v in d and d[v] in path2row for v in views):
                missing += 1
                continue
            for v in views:
                feats[v].append(emb[path2row[d[v]]])
            ys.append(int(r["label"])); empis.append(int(r["empi_anon"]))
        if not ys:
            continue
        out[g] = {
            "feats": {v: torch.tensor(np.stack(feats[v])) for v in views},
            "y": torch.tensor(ys, dtype=torch.long),
            "empi": np.array(empis),
        }
    if missing:
        print(f"[data] skipped {missing} breast-rows lacking a cached view embedding")
    return out


def patient_split(data: Dict, split_seed: int = 0, fracs=(0.7, 0.1, 0.2)):
    """Split by patient (empi) so no patient leaks across train/val/test. Returns
    {group -> {'train':idx,'val':idx,'test':idx}} as boolean masks. Fixed by split_seed."""
    all_empi = np.unique(np.concatenate([d["empi"] for d in data.values()]))
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(all_empi)
    n = len(perm); a = int(fracs[0] * n); b = int((fracs[0] + fracs[1]) * n)
    train_e, val_e, test_e = set(perm[:a]), set(perm[a:b]), set(perm[b:])
    masks = {}
    for g, d in data.items():
        e = d["empi"]
        masks[g] = {
            "train": np.array([x in train_e for x in e]),
            "val": np.array([x in val_e for x in e]),
            "test": np.array([x in test_e for x in e]),
        }
    return masks


def _subset(d, m, device):
    return {
        "feats": {v: t[m].to(device) for v, t in d["feats"].items()},
        "y": d["y"][m].to(device),
    }


# ------------------------------------------------------------ R*_g (Step 2) --

def estimate_optimal_losses(data, masks, device, folds=5, epochs=40, lr=5e-4, seed=0):
    """R*_g = 5-fold out-of-fold CE of a dedicated per-group model (its own view
    projections + MLP_g + head, CE only). Uses train+val rows of each group."""
    rstar = {}
    for g, d in data.items():
        m = masks[g]["train"] | masks[g]["val"]
        y = d["y"][m]
        feats = {v: t[m] for v, t in d["feats"].items()}
        n = len(y)
        if n < folds:
            rstar[g] = 0.0
            continue
        rng = np.random.RandomState(seed)
        fold = rng.randint(0, folds, size=n)
        oof = np.zeros(n)
        for f in range(folds):
            tr, te = fold != f, fold == f
            if te.sum() == 0 or tr.sum() == 0:
                continue
            torch.manual_seed(seed + f)
            model = XeniaEmbedModel().to(device)   # only group g's params get trained
            params = list(model.mlp[g].parameters()) + list(model.head.parameters())
            for v in GROUP_VIEWS[g]:
                params += list(model.proj[v].parameters())
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=5e-5)
            ftr = {v: feats[v][tr].to(device) for v in GROUP_VIEWS[g]}
            ytr = y[tr].to(device)
            fte = {v: feats[v][te].to(device) for v in GROUP_VIEWS[g]}
            yte = y[te].to(device)
            for _ in range(epochs):
                model.train(); opt.zero_grad()
                logits, _ = model(g, ftr)
                loss = F.cross_entropy(logits, ytr)
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                logits, _ = model(g, fte)
                ce = F.cross_entropy(logits, yte, reduction="none").cpu().numpy()
            oof[te] = ce
        rstar[g] = float(oof.mean())
    return rstar


# ------------------------------------------------------------ evaluation ----

@torch.no_grad()
def evaluate(model, data, masks, split, device, rstar=None):
    model.eval()
    per_group = {}
    for g, d in data.items():
        m = masks[g][split]
        if m.sum() == 0:
            continue
        sub = _subset(d, m, device)
        logits, _ = model(g, sub["feats"])
        y = sub["y"]
        ce = F.cross_entropy(logits, y).item()
        pred = logits.argmax(1)
        acc = (pred == y).float().mean().item()
        # macro-F1
        f1s = []
        for c in range(NUM_CLASSES):
            tp = ((pred == c) & (y == c)).sum().item()
            fp = ((pred == c) & (y != c)).sum().item()
            fn = ((pred != c) & (y == c)).sum().item()
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        rs = (rstar or {}).get(g, 0.0)
        per_group[g] = {
            "n": int(m.sum()), "acc": acc, "macro_f1": float(np.mean(f1s)),
            "loss": ce, "R_star": rs, "excess_loss": max(0.0, ce - rs),
        }
    accs = [v["acc"] for v in per_group.values()]
    overall = {
        "overall_acc": float(np.mean(accs)),
        "worst_group_acc": float(np.min(accs)),
        "worst_group": min(per_group, key=lambda k: per_group[k]["acc"]),
        "tail_acc": float(np.mean([v["acc"] for k, v in per_group.items() if k not in HEAD_GROUPS])),
        "max_excess": float(np.max([v["excess_loss"] for v in per_group.values()])),
        "avg_loss": float(np.mean([v["loss"] for v in per_group.values()])),
        "worst_group_loss": float(np.max([v["loss"] for v in per_group.values()])),
    }
    return overall, per_group


# ------------------------------------------------------------ training ------

METHOD_FLAGS = {
    "erm":         dict(dro=False, anchors=False, regret=False, select="avg_loss"),
    # anchors WITHOUT GroupDRO. Xenia's spec does not define this arm, but without it the
    # 2x2 (anchors x DRO) has an empty cell and we cannot tell whether the anchors help on
    # their own or only in combination with DRO, which is exactly the synergy question.
    "anchors_only": dict(dro=False, anchors=True, regret=False, select="avg_loss"),
    "groupdro":    dict(dro=True,  anchors=False, regret=False, select="worst_group_loss"),
    "align_only":  dict(dro=True,  anchors=True,  regret=False, select="avg_loss"),
    "regret_only": dict(dro=True,  anchors=False, regret=True,  select="max_excess"),
    "ours":        dict(dro=True,  anchors=True,  regret=True,  select="max_excess"),
}


@torch.no_grad()
def _group_val_excess(model, data, masks, rstar_t, groups, anchors_on, lam_fit, device):
    """Per-group excess on the VALIDATION split: (val_task - R*) + lam_fit*val_fit.
    Used by dro_signal='val' so the max player is not fooled by tail memorization."""
    model.eval()
    ex = []
    for gi, g in enumerate(groups):
        m = masks[g]["val"]
        if m.sum() == 0:
            ex.append(torch.zeros((), device=device)); continue
        sub = _subset(data[g], m, device)
        logits, z = model(g, sub["feats"])
        lt = F.cross_entropy(logits, sub["y"])
        lf = anchor_fit_loss(z, sub["y"], model.anchors) if anchors_on else torch.zeros((), device=device)
        ex.append((lt - rstar_t[gi]) + lam_fit * lf)
    return torch.stack(ex)


def train_one(method, data, masks, device, rstar, seed,
              epochs=20, lr=5e-5, wd=5e-5, batch=32,
              gamma=0.02, decay=0.9, lam_fit=1.0, lam_sep=1.0, dro_signal="train",
              uniform_lambda_init=False, verbose=True):
    flags = METHOD_FLAGS[method]
    groups = [g for g in GROUPS if g in data]
    torch.manual_seed(seed); np.random.seed(seed)
    model = XeniaEmbedModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)

    K = len(groups)
    rstar_t = torch.tensor([rstar.get(g, 0.0) if flags["regret"] else 0.0 for g in groups],
                           device=device)
    # Xenia spec: lambda_g initialised to empirical group proportion p_g; ema_loss init = R*_g
    ptrain = np.array([int(masks[g]["train"].sum()) for g in groups], dtype=float)
    if uniform_lambda_init:
        lam = torch.full((len(groups),), 1.0 / len(groups), device=device, dtype=torch.float32)
    else:
        lam = torch.tensor(ptrain / ptrain.sum(), device=device, dtype=torch.float32)
    ema = rstar_t.clone()

    N = 50           # lambda update stride (Xenia)
    gstep = 0
    best_sel, best_state, curve = float("inf"), None, []
    # steps per epoch = ceil(max group train size / batch); group-stratified sampling
    max_n = max(int(masks[g]["train"].sum()) for g in groups)
    steps = max(1, int(np.ceil(max_n / batch)))

    # Materialise each group's TRAIN split on the device ONCE. Doing this inside the step
    # loop (as before) re-sliced and re-copied the full group tensor every single step —
    # for the 51k-row head groups that dominated runtime and left the GPU at ~17%.
    train_sub = {}
    for g in groups:
        m = masks[g]["train"]
        train_sub[g] = _subset(data[g], m, device) if m.sum() > 0 else None

    for ep in range(epochs):
        model.train()
        for _ in range(steps):
            opt.zero_grad()
            raw = []   # per-group L_task + lam_fit*L_fit (no R*), for the EMA
            task_total = model.head.weight.new_zeros(())
            for gi, g in enumerate(groups):
                sub = train_sub[g]
                if sub is None:
                    raw.append(None); continue
                # group-stratified: up to `batch` samples from this group
                nn_ = sub["y"].shape[0]
                sel = torch.randint(0, nn_, (min(batch, nn_),), device=device)
                feats = {v: t[sel] for v, t in sub["feats"].items()}
                y = sub["y"][sel]
                logits, z = model(g, feats)
                lt = F.cross_entropy(logits, y)
                lf = anchor_fit_loss(z, y, model.anchors) if flags["anchors"] else torch.zeros((), device=device)
                raw.append((lt + lam_fit * lf).detach())
                comp = (lt - rstar_t[gi]) + lam_fit * lf     # objective term (R* is a constant)
                task_total = task_total + lam[gi] * comp
            lsep = anchor_sep_loss(model) if flags["anchors"] else torch.zeros((), device=device)
            loss = task_total + lam_sep * lsep
            loss.backward(); opt.step()
            gstep += 1
            # max player (Xenia): EMA of raw per-group loss (present groups), update lambda every N steps
            if flags["dro"] and dro_signal == "train":
                for gi in range(K):
                    if raw[gi] is not None:
                        ema[gi] = decay * ema[gi] + (1 - decay) * raw[gi]
                if gstep % N == 0:
                    excess = torch.clamp(ema - rstar_t, min=0.0)   # clamp: R* is an estimate
                    lam = lam * torch.exp(gamma * excess)
                    lam = torch.clamp(lam, min=1e-8); lam = lam / lam.sum()
        # lambda update (max player) — validation-loss signal, once per epoch
        if flags["dro"] and dro_signal == "val":
            ex = _group_val_excess(model, data, masks, rstar_t, groups,
                                   flags["anchors"], lam_fit, device)
            ema = decay * ema + (1 - decay) * ex
            lam = lam * torch.exp(gamma * steps * ema)  # scale per-epoch step to match per-step cumulative
            lam = torch.clamp(lam, min=1e-8); lam = lam / lam.sum()
            model.train()
        sched.step()
        ov, _ = evaluate(model, data, masks, "val", device, rstar)
        sel = ov[flags["select"]]
        curve.append({"epoch": ep, **{k: ov[k] for k in
                    ["overall_acc", "worst_group_acc", "tail_acc", "avg_loss", "max_excess"]}})
        if sel < best_sel:
            best_sel, best_state = sel, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  [{method} s{seed}] ep{ep:02d} val acc={ov['overall_acc']:.3f} "
                  f"worst={ov['worst_group_acc']:.3f}({ov['worst_group']}) tail={ov['tail_acc']:.3f} "
                  f"sel({flags['select']})={sel:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"lambda": lam.detach().cpu().numpy().tolist(), "groups": groups, "curve": curve}


def group_only_models(data, masks, device, seed, epochs=40, lr=5e-4):
    """Row 3: a dedicated model per group (test-set metrics). Also mirrors R*_g."""
    rows = []
    for g in [g for g in GROUPS if g in data]:
        tr = masks[g]["train"] | masks[g]["val"]
        torch.manual_seed(seed)
        model = XeniaEmbedModel().to(device)
        params = list(model.mlp[g].parameters()) + list(model.head.parameters())
        for v in GROUP_VIEWS[g]:
            params += list(model.proj[v].parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=5e-5)
        ftr = {v: data[g]["feats"][v][tr].to(device) for v in GROUP_VIEWS[g]}
        ytr = data[g]["y"][tr].to(device)
        for _ in range(epochs):
            model.train(); opt.zero_grad()
            logits, _ = model(g, ftr)
            F.cross_entropy(logits, ytr).backward(); opt.step()
        _, pg = evaluate(model, {g: data[g]}, {g: masks[g]}, "test", device)
        rows.append((g, pg[g]))
    return rows


# ------------------------------------------------------------ driver --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="datasets/embed/index_xenia_6group.parquet")
    ap.add_argument("--cache", default="datasets/embed/vit_cache")
    ap.add_argument("--out", default="runs/embed_xenia")
    ap.add_argument("--methods", nargs="+",
                    default=["erm", "anchors_only", "groupdro", "align_only", "regret_only",
                             "ours", "group_only"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42])
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--rstar-folds", type=int, default=5)
    ap.add_argument("--dro-gamma", type=float, default=0.02,
                    help="lambda step size. Xenia's spec is 0.02, which on EMBED's 1:1000 "
                         "group imbalance is too gentle to move lambda off the initial "
                         "proportions, making GroupDRO behave identically to ERM.")
    ap.add_argument("--uniform-lambda-init", action="store_true",
                    help="initialise lambda uniformly instead of at group proportions. With "
                         "proportional init the four tail groups share only 1.5%% of the "
                         "gradient weight on EMBED.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    print(f"device={device}  loading cached ViT vectors ...")
    data = load_group_tensors(args.index, args.cache)
    masks = patient_split(data, split_seed=args.split_seed)
    for g in GROUPS:
        if g in data:
            tr = int(masks[g]["train"].sum()); te = int(masks[g]["test"].sum())
            print(f"  {g}: N={len(data[g]['y'])} train={tr} test={te}")

    long_rows = []
    n_params = sum(p.numel() for p in XeniaEmbedModel().parameters())

    # R*_g is a property of the DATA, not of a training seed. Xenia's spec says to freeze the
    # six numbers and treat them as constants, so estimate them once and reuse across seeds
    # (previously this 5-fold x 6-group estimation re-ran for every seed).
    rstar_path = os.path.join(args.out, "rstar.json")
    if os.path.exists(rstar_path):
        rstar = {k: float(v) for k, v in json.load(open(rstar_path)).items()}
        print("R*_g (cached) = " + ", ".join(f"{g}:{rstar.get(g,0):.3f}" for g in GROUPS if g in data))
    else:
        rstar = estimate_optimal_losses(data, masks, device, folds=args.rstar_folds, seed=0)
        json.dump(rstar, open(rstar_path, "w"), indent=2)
        print("R*_g = " + ", ".join(f"{g}:{rstar.get(g,0):.3f}" for g in GROUPS if g in data))

    for seed in args.seeds:
        for method in args.methods:
            if method == "group_only":
                for g, pg in group_only_models(data, masks, device, seed):
                    long_rows.append(dict(method="group_only", seed=seed, group=g, n_params=n_params, **pg))
                continue
            if method not in METHOD_FLAGS:
                print(f"  (skip unknown method {method})"); continue
            model, info = train_one(method, data, masks, device, rstar, seed, epochs=args.epochs,
                                    gamma=args.dro_gamma,
                                    uniform_lambda_init=args.uniform_lambda_init)
            ov, pg = evaluate(model, data, masks, "test", device, rstar)
            print(f"[seed {seed}] {method}: test overall={ov['overall_acc']:.3f} "
                  f"worst={ov['worst_group_acc']:.3f}({ov['worst_group']}) tail={ov['tail_acc']:.3f}")
            for g, row in pg.items():
                long_rows.append(dict(method=method, seed=seed, group=g, n_params=n_params, **row))
            with open(os.path.join(args.out, f"curve_{method}_s{seed}.json"), "w") as f:
                json.dump(info, f)

    df = pd.DataFrame(long_rows)
    df.to_csv(os.path.join(args.out, "metrics_long.csv"), index=False)
    print(f"\nwrote {os.path.join(args.out, 'metrics_long.csv')}  ({len(df)} rows)")
    # headline: groupdro vs ours, worst-group acc (mean over seeds)
    if {"groupdro", "ours"}.issubset(set(df.method)):
        for meth in ["groupdro", "ours"]:
            sub = df[df.method == meth]
            wg = sub.groupby("seed").apply(lambda s: s.set_index("group")["acc"].min(), include_groups=False)
            print(f"  {meth}: worst-group acc = {wg.mean():.3f} +/- {wg.std():.3f}")


if __name__ == "__main__":
    main()
