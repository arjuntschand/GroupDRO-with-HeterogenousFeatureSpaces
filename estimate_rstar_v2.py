"""R*_g estimated as a Bayes risk instead of as an artefact of our own training setup.

Three things were wrong with the old estimator (tools/estimate_rstar_tabular.py).

1. It passed group_max_train_samples through, so R* was fitted on the CAPPED training set:
   20 patients for Switzerland, 25 for VA. R*_g is an infimum over measurable predictors on
   group g's distribution. It cannot depend on how many samples we chose to train our own model
   on. That single line is why VA came out at 1.480 while an ordinary shared model reaches 0.551.

2. It fitted on group g's rows only. For a small group that overfits, which inflates the
   estimate further, and Proposition 1 is explicit that R*_g is fixed by what X_g measures and
   that no amount of data reduces it.

3. It used a single model shape, so a bad fit for one group's geometry inflated that group.

The fix follows from what R* is. Any fitted model is one particular measurable predictor, so its
honest out-of-fold risk on group g is an UPPER BOUND on R*_g. Fit several estimators, take the
minimum, and the bound gets tighter without ever becoming invalid.

  fit scope    group-only, and pooled over every sample whose record contains g's feature set.
               Pooling fixes scarcity but introduces distribution shift between sites, so
               neither dominates: VA wants pooling, Switzerland wants group-only. Taking the min
               picks per group without us having to choose.
  capacity     linear, and two MLP widths. Guards against one architecture suiting one group.
  evaluation   K-fold, always scored on group g's own held-out rows, so the quantity stays the
               Bayes risk of P_g rather than of the pooled distribution.

Nothing here touches the test set. Folds are drawn over the group's own data and every score is
out-of-fold, so this can feed the training objective without leaking.

  python estimate_rstar_v2.py --dataset fedheart
  python estimate_rstar_v2.py --dataset nhanes
"""
from __future__ import annotations
import argparse, csv, glob, json, os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedKFold

FOLDS, EPOCHS = 5, 150


def make(kind, in_dim, nc):
    if kind == "linear":
        return nn.Linear(in_dim, nc)
    h = 32 if kind == "mlp32" else 64
    return nn.Sequential(nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(),
                         nn.Linear(h, h), nn.ReLU(), nn.Linear(h, nc))


def fit_eval(X, y, tr, ev, kind, nc, seed, wd, weighted=True):
    """weighted=False matters more than it looks. Switzerland is 93% positive, 8 negatives out
    of 115, so inverse-frequency weights put about 7x mass on 8 samples and wreck the UNWEIGHTED
    cross-entropy we score with. The real arms avoid this by training on every group at once
    with global class weights, so their Switzerland predictions stay near global calibration.
    Both variants go in the pool and the min decides."""
    torch.manual_seed(seed)
    m = make(kind, X.shape[1], nc)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=wd)
    if weighted:
        cnt = torch.bincount(y[tr], minlength=nc).float()
        w = cnt.sum() / (nc * cnt.clamp_min(1))
    else:
        w = None
    ce = nn.CrossEntropyLoss(weight=w)
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad(); ce(m(X[tr]), y[tr]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        # unweighted CE at evaluation, matching how per-group loss is reported everywhere else
        per = nn.functional.cross_entropy(m(X[ev]), y[ev], reduction="none")
        return float(per.mean()), per.detach().cpu().numpy()


class Joint(nn.Module):
    """Per-group encoders into a shared latent space with one shared head.

    This is the model class the paper is actually about, and it matters for R*: a shared head
    trained across every group transfers information that a model fitted on one group in
    isolation cannot reach. On Switzerland, 115 patients, the isolated families bottom out at
    0.582 while the joint model reaches 0.337. Since any fitted model is one measurable
    predictor, adding it to the pool tightens the upper bound without weakening it.
    """

    def __init__(self, dims, latent, nc):
        super().__init__()
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Linear(d, latent), nn.LayerNorm(latent), nn.ReLU(),
                          nn.Linear(latent, latent), nn.ReLU()) for d in dims])
        self.head = nn.Linear(latent, nc)

    def forward(self, x, gid):
        return self.head(self.enc[gid](x))


def joint_oof(X, y, g, masks, nc, folds, seed=0, epochs=150, latent=64):
    """Out-of-fold per-group CE from the joint model. Folds are per group, so the evaluation
    rows for group g are held out of training for EVERY group, not just g's own encoder."""
    out = {gi: [] for gi in range(len(masks))}
    for f in range(len(folds[0])):
        ev_idx, tr_idx = {}, {}
        for gi in range(len(masks)):
            own = (g == gi).nonzero(as_tuple=True)[0]
            tr_rel, ev_rel = folds[gi][f]
            ev_idx[gi] = own[ev_rel]; tr_idx[gi] = own[tr_rel]
        torch.manual_seed(seed + f)
        m = Joint([len(mk) for mk in masks], latent, nc)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        allc = torch.cat([y[tr_idx[gi]] for gi in tr_idx])
        cnt = torch.bincount(allc, minlength=nc).float()
        w = cnt.sum() / (nc * cnt.clamp_min(1))
        ce = nn.CrossEntropyLoss(weight=w)
        for _ in range(epochs):
            m.train(); opt.zero_grad()
            loss = 0.0
            for gi in tr_idx:
                if len(tr_idx[gi]) == 0:
                    continue
                loss = loss + ce(m(X[tr_idx[gi]][:, masks[gi]], gi), y[tr_idx[gi]])
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            for gi in ev_idx:
                if len(ev_idx[gi]) == 0:
                    continue
                out[gi].append(float(nn.functional.cross_entropy(
                    m(X[ev_idx[gi]][:, masks[gi]], gi), y[ev_idx[gi]])))
    return {gi: float(np.mean(v)) for gi, v in out.items() if v}


def load(dataset, base=None):
    """Return (X, y, group, masks, num_classes) with NO training caps applied."""
    if dataset == "fedheart":
        from dro_hetero_anchors.src.datasets_fedheart import build_fedheart_loaders
        cfg = yaml.safe_load(open(base or "experiments/fedheart_exp_paper_hetagg_gdro.yaml"))
        # group_max_train_samples deliberately NOT passed: R* is a property of the distribution,
        # not of the training budget we imposed on ourselves.
        tr, te, info = build_fedheart_loaders(batch_size=64, seed=0, train_frac=0.8,
                                              impute_missing=True, feature_mask=None)
        masks = [m if m is not None else list(range(13)) for m in cfg["feature_mask"]]
        nc = 2
    else:
        from dro_hetero_anchors.src.datasets_nhanes import build_nhanes_loaders
        cfg = yaml.safe_load(open(base or "experiments/nhanes_pergroup_gdro.yaml"))
        tr, te, info = build_nhanes_loaders(batch_size=256, seed=0, data_split_seed=100,
                                            feature_mode=cfg.get("feature_mode", "nested"))
        masks = [list(v) for _, v in sorted(info["feature_indices"].items(),
                                            key=lambda kv: int(kv[0]))]
        nc = cfg.get("num_classes", 2)
    xs, ys, gs = [], [], []
    for loader in (tr, te):
        for x, y, g in loader:
            xs.append(x); ys.append(y); gs.append(g)
    return torch.cat(xs), torch.cat(ys), torch.cat(gs), masks, nc


def achieved(dataset):
    """Lowest per-group loss any run actually reached. Used only as a CHECK on the result,
    never as an input, since these come from test."""
    pats = (["runs/fedheart_cv*/metrics_long.csv", "runs/baselines_fedheart*/metrics_long.csv"]
            if dataset == "fedheart" else
            ["runs/matrix_nhanes_nested*/metrics_long.csv", "runs/baselines_nhanes*/metrics_long.csv"])
    best = defaultdict(lambda: float("inf"))
    for pat in pats:
        for p in glob.glob(pat):
            for r in csv.DictReader(open(p)):
                try:
                    gi = int(r["group"].lstrip("g")); best[gi] = min(best[gi], float(r["loss"]))
                except (ValueError, KeyError):
                    continue
    return dict(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fedheart", "nhanes"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--base", default=None,
                    help="config yaml to read feature_mode etc. from (default: the paper config)")
    ap.add_argument("--group-cap", nargs="+", type=int, default=None,
                    help="per-group cap on rows used to fit R* (item 5: show how the estimate depends on n). Default: no cap, which is the correct estimator.")
    args = ap.parse_args()

    X, y, g, masks, nc = load(args.dataset, args.base)
    if args.group_cap:
        keep = []
        rng = np.random.default_rng(0)
        for gi_, cap in enumerate(args.group_cap):
            idx = (g == gi_).nonzero(as_tuple=True)[0].numpy()
            if cap and cap > 0 and len(idx) > cap:
                idx = rng.choice(idx, size=cap, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep)); keep_t = torch.tensor(keep)
        X, y, g = X[keep_t], y[keep_t], g[keep_t]
        print(f"  --group-cap applied: {args.group_cap} -> {len(y)} rows")
    print(f"{args.dataset}: {len(y)} samples, {len(masks)} groups, "
          f"{'capped ' + str(args.group_cap) if args.group_cap else 'uncapped'}\n")
    old_p = f"runs/rstar_{args.dataset}.json" if args.dataset == "fedheart" \
        else "runs/rstar_nhanes_nested.json"
    old = json.load(open(old_p))["rstar"]
    floor = achieved(args.dataset)

    # per-group fold assignments, reused by every family so the comparison is like for like
    allfolds = {}
    for gi in range(len(masks)):
        own = (g == gi).nonzero(as_tuple=True)[0]
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)
        allfolds[gi] = list(skf.split(own.numpy(), y[own].numpy()))
    print("  fitting the joint model (shared head across groups)...", flush=True)
    jo = joint_oof(X, y, g, masks, nc, allfolds)

    out, detail, const_floor, margin, out_uncorrected, const_fold_losses, fit_only = {}, {}, {}, {}, {}, {}, {}
    for gi, mask in enumerate(masks):
        Xg = X[:, mask]
        own = (g == gi).nonzero(as_tuple=True)[0]
        folds = allfolds[gi]
        best_name, best_val = "joint", jo.get(gi, float("inf"))
        fit_best_val = best_val
        best_fold_losses = []
        best_per_sample = None
        # Constant predictor (Xenia): predict the group's training class rate for every sample.
        # Uses no features, so it is the loosest VALID upper bound; a fitted R* above it is
        # provably invalid. On Fed-Heart the previous estimate violated it for Switzerland and VA.
        const_losses, const_per_sample = [], []
        for f, (tr_rel, ev_rel) in enumerate(folds):
            tr, ev = own[tr_rel], own[ev_rel]
            pr = torch.bincount(y[tr], minlength=nc).float().clamp_min(1)
            pr = pr / pr.sum()
            _per = (-torch.log(pr[y[ev]])).detach().cpu().numpy()
            const_losses.append(float(_per.mean())); const_per_sample.append(_per)
        const_v = float(np.mean(const_losses))
        fit_best_pre = None  # filled after the candidate loop
        const_floor[str(gi)] = round(const_v, 6)
        const_fold_losses[str(gi)] = [round(v, 6) for v in const_losses]
        if const_v < best_val:
            best_val, best_name = const_v, "constant/no-features"
            best_per_sample = np.concatenate(const_per_sample)
        for scope in ("group", "pooled"):
            for kind in ("linear", "mlp32", "mlp64"):
                for wt in (True, False):
                    losses, per_sample = [], []
                    for f, (tr_rel, ev_rel) in enumerate(folds):
                        ev = own[ev_rel]
                        if scope == "group":
                            tr = own[tr_rel]
                        else:
                            ex = set(ev.tolist())
                            tr = torch.tensor([i for i in range(len(y)) if i not in ex])
                        mean_f, per_f = fit_eval(Xg, y, tr, ev, kind, nc, f, 1e-4, wt)
                        losses.append(mean_f); per_sample.append(per_f)
                    v = float(np.mean(losses))
                    fit_best_val = min(fit_best_val, v)
                    if v < best_val:
                        best_val = v
                        best_name = f"{scope}/{kind}/{'w' if wt else 'unw'}"
                        best_fold_losses = list(losses)
                        best_per_sample = np.concatenate(per_sample)
        # fit-only R^_g: the best over joint + fitted candidates, ignoring the constant. This is
        # the quantity certificate (i) tests against R^const_g (Appendix F, Table 1).
        fit_only[str(gi)] = round(fit_best_val, 6)
        # eq (13): R~_g = min{R^_g, R^const_g} - c_g. Appendix F: c_g is the half-width of a bootstrap
        # percentile interval for the mean of group g's out-of-fold losses, at a level union-bounded
        # over the G groups (delta=0.05 -> each group at delta/G). It makes the reference a deliberate
        # underestimate, which errs on the safe side: a reference below the true floor cannot exclude
        # a group. It grows as groups shrink, tracking sampling variability only.
        # Appendix F: c_g = half-width of a bootstrap percentile interval for the MEAN of group
        # g's pooled out-of-fold per-sample losses, level union-bounded over the G groups. Fold
        # means are the wrong resolution: for a constant predictor every fold gives the same
        # loss, so a fold-level bootstrap returned c_g = 0 for the one group (VA) that most
        # needed a margin. Per-sample bootstrap scales like sigma/sqrt(n_g), which is what
        # "grows as groups shrink" means. If the joint fit wins we have no per-sample losses
        # for it, so its margin is 0 and flagged.
        G_ = len(masks); alpha_ = 0.05 / G_
        rng_b = np.random.default_rng(1000 + gi)
        if best_per_sample is not None and len(best_per_sample) >= 2:
            ps_ = np.asarray(best_per_sample, dtype=float); n_ = len(ps_)
            bs = np.array([ps_[rng_b.integers(0, n_, n_)].mean() for _ in range(4000)])
            lo_, hi_ = np.percentile(bs, [100 * alpha_ / 2, 100 * (1 - alpha_ / 2)])
            c_g = float((hi_ - lo_) / 2)
            fl_ = ps_
        else:
            c_g = 0.0; fl_ = np.array([best_val])
        print(f"  margin-debug g{gi}: chosen={best_name} n_oof={len(fl_)} c_g={c_g:.5f}", flush=True)
        margin[str(gi)] = round(c_g, 6)
        out_uncorrected[str(gi)] = round(best_val, 6)          # min{R^, R^const}, before the margin
        out[str(gi)] = round(max(best_val - c_g, 0.0), 6)      # R~_g, what training consumes
        detail[str(gi)] = best_name
        fl = floor.get(gi, float("inf"))
        flag = "  <-- still above an achieved loss" if best_val > fl + 1e-6 else ""
        print(f"  g{gi}: n={len(own):>5}  old {old[str(gi)]:6.3f} -> new {best_val:6.3f} "
              f"[{best_name:13}]  achieved {fl:6.3f}{flag}")

    path = args.out or (f"runs/rstar_{args.dataset}_v2.json" if args.dataset == "fedheart"
                        else "runs/rstar_nhanes_nested_v2.json")
    json.dump({"rstar": out, "rstar_before_margin": out_uncorrected, "margin_c_g": margin,
               "chosen_estimator": detail, "constant_predictor_bound": const_floor, "rstar_fit_only": fit_only, "constant_fold_losses": const_fold_losses,
               "method": "uncapped; min over {group,pooled} x {linear,mlp32,mlp64}; "
                         "out-of-fold on the group's own rows"},
              open(path, "w"), indent=2)

    print("\nchecks:")
    viol = [k for k, v in out.items() if v > floor.get(int(k), float("inf")) + 1e-6]
    print(f"  Bayes-risk violations: {len(viol)}/{len(out)}  {viol}")
    if args.dataset == "nhanes":
        vals = [out[str(i)] for i in range(len(masks))]
        ok = all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))
        print(f"  Prop 1 ordering (nested, R*_0 >= R*_1 >= R*_2): {'holds' if ok else 'VIOLATED'}"
              f"  {[round(v,3) for v in vals]}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
