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

def _joint_oof(data, masks, device, folds=5, epochs=40, lr=5e-4, seed=0, return_samples=False):
    """Out-of-fold CE per group from ONE model trained on every group at once.

    The per-group estimator below fits group g in isolation. That is the same flaw the tabular
    R* had: a shared head trained across all groups transfers information an isolated fit cannot
    reach, so the isolated number sits above what a real model achieves. On EMBED it left 5 of 6
    groups with an R* ABOVE a loss some arm actually reached, g5 worst at 1.280 against 0.836.
    Since any fitted model is one measurable predictor, its honest out-of-fold risk upper-bounds
    R*_g, and taking the min of several estimators tightens the bound without invalidating it.
    """
    groups = [g for g in GROUPS if g in data]
    rng = np.random.RandomState(seed)
    fold = {g: rng.randint(0, folds, size=int((masks[g]["train"] | masks[g]["val"]).sum()))
            for g in groups}
    oof = {g: [] for g in groups}
    for f in range(folds):
        torch.manual_seed(seed + f)
        model = XeniaEmbedModel().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
        sub = {}
        for g in groups:
            m = masks[g]["train"] | masks[g]["val"]
            y = data[g]["y"][m]; feats = {v: t[m] for v, t in data[g]["feats"].items()}
            tr, te = fold[g] != f, fold[g] == f
            sub[g] = (feats, y, tr, te)
        for _ in range(epochs):
            model.train(); opt.zero_grad()
            tot = None
            for g in groups:
                feats, y, tr, te = sub[g]
                if tr.sum() == 0:
                    continue
                ftr = {v: feats[v][tr].to(device) for v in GROUP_VIEWS[g]}
                lg, _ = model(g, ftr)
                l = F.cross_entropy(lg, y[tr].to(device))
                tot = l if tot is None else tot + l
            if tot is not None:
                tot.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            for g in groups:
                feats, y, tr, te = sub[g]
                if te.sum() == 0:
                    continue
                fte = {v: feats[v][te].to(device) for v in GROUP_VIEWS[g]}
                lg, _ = model(g, fte)
                oof[g] += F.cross_entropy(lg, y[te].to(device),
                                          reduction="none").cpu().numpy().tolist()
    if return_samples:
        return {g: np.asarray(v) for g, v in oof.items() if v}
    return {g: float(np.mean(v)) for g, v in oof.items() if v}


def estimate_optimal_losses(data, masks, device, folds=5, epochs=40, lr=5e-4, seed=0,
                            eq13=False, detail=None):
    """R*_g = out-of-fold CE, minimised over a per-group fit and a joint fit.

    The per-group fit is Xenia's Step 2 model. The joint fit shares the head across groups,
    which is what the deployed method does and what a small group benefits from. Each is a valid
    upper bound on the Bayes risk, so the minimum is the tighter honest estimate. Scored strictly
    out of fold on train+val rows, never on test."""
    joint_s = _joint_oof(data, masks, device, folds=folds, epochs=epochs, lr=lr, seed=seed,
                         return_samples=True)
    joint = {g: float(v.mean()) for g, v in joint_s.items()}
    n_groups = len(data)
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
        per_group = float(oof.mean())
        rstar[g] = min(per_group, joint.get(g, float("inf")))
        if eq13:
            # eq. 13: R~_g = min{fitted, constant predictor} - c_g. The constant predictor is the
            # train-fold class frequency, scored out of fold; c_g is the half-width of a bootstrap
            # percentile interval for the mean per-sample loss, union-bounded over the G groups.
            const = np.zeros(n)
            yn = y.numpy()
            for f in range(folds):
                tr, te = fold != f, fold == f
                if te.sum() == 0 or tr.sum() == 0:
                    continue
                freq = np.bincount(yn[tr], minlength=NUM_CLASSES).astype(float)
                freq = np.clip(freq / freq.sum(), 1e-6, None); freq = freq / freq.sum()
                const[te] = -np.log(freq[yn[te]])
            cands = {"group_fit": oof, "constant": const}
            if g in joint_s:
                cands["joint_fit"] = joint_s[g]
            best = min(cands, key=lambda k: cands[k].mean())
            samp = cands[best]
            brng = np.random.RandomState(1234)
            means = samp[brng.randint(0, len(samp), size=(4000, len(samp)))].mean(1)
            a = 0.05 / n_groups
            lo, hi = np.quantile(means, [a / 2, 1 - a / 2])
            c_g = float((hi - lo) / 2)
            rstar[g] = max(0.0, float(samp.mean()) - c_g)
            if detail is not None:
                detail[g] = {"n": int(n), "chosen": best, "margin_c_g": c_g, "rstar_eq13": rstar[g],
                             **{k: float(v.mean()) for k, v in cands.items()}}
            print(f"  R~[{g}] = {rstar[g]:.3f}  ({best} {samp.mean():.3f} - c_g {c_g:.3f}; "
                  + ", ".join(f"{k} {v.mean():.3f}" for k, v in cands.items()) + ")", flush=True)
        if joint.get(g, float("inf")) < per_group:
            print(f"  R*[{g}]: joint fit {joint[g]:.3f} beats group-only {per_group:.3f}",
                  flush=True)
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
            # signed, per Step 6; run_baselines_embed already reports signed, so clamping
            # here made our EMBED arms unable to show a negative excess while baselines could
            "loss": ce, "R_star": rs, "excess_loss": ce - rs,
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
    # Mechanism control: identical to "align_only" except each sample is pulled toward a
    # RANDOM class anchor instead of its own. Same loss, same parameters, same magnitude,
    # but the class-conditional structure is destroyed. On both NHANES modes this control
    # matched the real anchors, which would mean the class structure is not what helps.
    # EMBED has 4 classes rather than 2, so random assignment is a far harsher scramble
    # and is the sharper test of that story.
    "rand_anchor":  dict(dro=True,  anchors=True, regret=False, select="avg_loss",
                         random_anchor_targets=True),
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
              uniform_lambda_init=False, verbose=True, anchor_lr=None, signed_excess=True):
    flags = METHOD_FLAGS[method]
    groups = [g for g in GROUPS if g in data]
    torch.manual_seed(seed); np.random.seed(seed)
    model = XeniaEmbedModel().to(device)
    # One optimizer, one loss. anchor_lr gives the anchor parameters (means and log-variances)
    # their own step size as a second parameter group; at the base 5e-5 for 20 epochs the four
    # class anchors barely separate (they sit almost on top of each other in the latent scatter).
    if anchor_lr is not None and flags["anchors"]:
        anc_ids = {id(p) for p in model.anchors.parameters()}
        rest = [p for p in model.parameters() if id(p) not in anc_ids]
        opt = torch.optim.AdamW([{"params": rest, "lr": lr},
                                 {"params": list(model.anchors.parameters()), "lr": anchor_lr}],
                                lr=lr, weight_decay=wd)
    else:
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
                if flags["anchors"]:
                    y_anchor = y
                    if flags.get("random_anchor_targets"):
                        y_anchor = torch.randint(0, NUM_CLASSES, y.shape, device=y.device)
                    lf = anchor_fit_loss(z, y_anchor, model.anchors)
                else:
                    lf = torch.zeros((), device=device)
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
                    # signed excess (review of 2026-09-20): the exponentiated update is invariant
                    # to a constant shift of the payoffs, the [.]_+ clamp is not, and it froze
                    # the weights whenever every group sat below its reference. Subtracting the
                    # max is a constant shift (it cancels on renormalisation) and guards overflow.
                    excess = ema - rstar_t
                    if not signed_excess:
                        excess = torch.clamp(excess, min=0.0)
                    scaled = gamma * excess
                    lam = lam * torch.exp(scaled - scaled.max())
                    lam = torch.clamp(lam, min=1e-12); lam = lam / lam.sum()
        # lambda update (max player) — validation-loss signal, once per epoch
        if flags["dro"] and dro_signal == "val":
            ex = _group_val_excess(model, data, masks, rstar_t, groups,
                                   flags["anchors"], lam_fit, device)
            ema = decay * ema + (1 - decay) * ex
            # One update per epoch at the SAME step size the train path uses per update.
            #
            # This previously read exp(gamma * steps * ema). On EMBED steps is about 1594
            # (largest group ~51k rows at batch 32), so with gamma 0.5 the exponent reached
            # ~797 and overflowed to inf; after normalisation lambda became exactly one-hot and
            # the model trained on a single group, giving 9.7% overall accuracy. The intent was
            # to match the train path's cumulative movement, but that path applies
            # exp(gamma * excess) once every N=50 steps, so the per-epoch exponent is
            # gamma * excess * steps/N, not gamma * excess * steps. The old line was 50x too
            # large and applied it in one jump against a stale EMA.
            #
            # Using plain gamma moves lambda more slowly than the train variant by design,
            # which is the point: the validation signal is the thing we want to react to
            # carefully, since it is the one that is not memorised.
            step_exp = torch.clamp(gamma * ema, max=20.0)   # exp(20) is already ~5e8
            lam = lam * torch.exp(step_exp)
            lam = torch.clamp(lam, min=1e-8); lam = lam / lam.sum()
            model.train()
        sched.step()
        ov, pg_val = evaluate(model, data, masks, "val", device, rstar)
        sel = ov[flags["select"]]
        # Per-group train and test loss are logged for the dynamics plots only; epoch selection
        # reads the validation numbers above.
        _, pg_tr = evaluate(model, data, masks, "train", device, rstar)
        _, pg_te = evaluate(model, data, masks, "test", device, rstar)
        curve.append({"epoch": ep, "lambda": lam.detach().cpu().numpy().tolist(),
                      "per_group_loss": {g_: float(r_["loss"]) for g_, r_ in pg_val.items()},
                      "per_group_loss_train": {g_: float(r_["loss"]) for g_, r_ in pg_tr.items()},
                      "per_group_loss_test": {g_: float(r_["loss"]) for g_, r_ in pg_te.items()},
                      "per_group_acc": {g_: float(r_["acc"]) for g_, r_ in pg_val.items()},
                      **{k: ov[k] for k in
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
    # The spec's "Sizes you can change" table says to always sweep lam_fit over 0.1, 1, 10.
    # It was only ever run at the 1.0 default because there was no flag for it.
    ap.add_argument("--lam-fit", type=float, default=1.0,
                    help="weight on the anchor alignment term; spec says sweep 0.1, 1, 10")
    ap.add_argument("--lam-sep", type=float, default=1.0,
                    help="weight on the anchor separation term")
    # Driving the max player from TRAIN loss lets it be fooled by tail memorisation: g5 has
    # roughly 40 training rows, so its train loss collapses, its excess goes to zero, and its
    # lambda never grows. Validation excess is not memorised.
    ap.add_argument("--dro-signal", choices=["train", "val"], default="train",
                    help="whether the lambda update reads train or validation excess")
    ap.add_argument("--uniform-lambda-init", action="store_true",
                    help="initialise lambda uniformly instead of at group proportions. With "
                         "proportional init the four tail groups share only 1.5%% of the "
                         "gradient weight on EMBED.")
    ap.add_argument("--clamp-excess", action="store_true",
                    help="pre-2026-09-20 weight update, [L_g - R_g]_+ (default is the signed excess)")
    ap.add_argument("--anchor-lr", type=float, default=None,
                    help="separate learning rate for the anchor parameters (default: same as --lr)")
    ap.add_argument("--rstar-eq13", action="store_true",
                    help="R~_g = min{fitted, constant predictor} - bootstrap margin c_g (draft eq. 13)")
    ap.add_argument("--save-latents", default=None,
                    help="directory to write the TEST latents (z, y, group) and anchor means per method and seed")
    ap.add_argument("--disjoint", action="store_true",
                    help="no-overlap variant: g1/g2/g3/g6 with one distinct view each (see model/embed_xenia.py)")
    args = ap.parse_args()
    if args.disjoint:
        from dro_hetero_anchors.src.model.embed_xenia import use_disjoint_views
        use_disjoint_views()

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
        _detail = {}
        rstar = estimate_optimal_losses(data, masks, device, folds=args.rstar_folds, seed=0,
                                        eq13=args.rstar_eq13, detail=_detail)
        json.dump(rstar, open(rstar_path, "w"), indent=2)
        if _detail:
            json.dump(_detail, open(os.path.join(args.out, "rstar_eq13_detail.json"), "w"), indent=2)
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
                                    lam_fit=args.lam_fit, lam_sep=args.lam_sep,
                                    dro_signal=args.dro_signal,
                                    uniform_lambda_init=args.uniform_lambda_init,
                                    anchor_lr=args.anchor_lr,
                                    signed_excess=not args.clamp_excess)
            ov, pg = evaluate(model, data, masks, "test", device, rstar)
            if args.save_latents:
                os.makedirs(args.save_latents, exist_ok=True)
                model.eval(); _zs, _ys, _gs = [], [], []
                _names = [g_ for g_ in GROUPS if g_ in data]
                with torch.no_grad():
                    for _gi, _g in enumerate(_names):
                        _m = masks[_g]["test"]
                        if _m.sum() == 0:
                            continue
                        _sub = _subset(data[_g], _m, device)
                        _, _z = model(_g, _sub["feats"])
                        _zs.append(_z.cpu().numpy()); _ys.append(_sub["y"].cpu().numpy())
                        _gs.append(np.full(len(_sub["y"]), _gi))
                np.savez_compressed(os.path.join(args.save_latents, f"{method}_s{seed}.npz"),
                                    z=np.concatenate(_zs), y=np.concatenate(_ys), g=np.concatenate(_gs),
                                    anchor_m=model.anchors.m.detach().cpu().numpy(),
                                    anchor_S=np.stack([np.diag(v) for v in model.anchors.var().detach().cpu().numpy()]),
                                    groups=np.array(_names))
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
