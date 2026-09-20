"""R*_g estimated without leaking the test set and without the winner's curse.

This is a correction of estimate_rstar_v2.py. The three structural ideas in v2 were right and
are kept: do not cap the fitting sample, widen the candidate pool, and admit pooled and jointly
trained predictors, since any fitted model is one measurable predictor on X_g and its honest
out-of-fold risk upper-bounds R*_g. What was wrong was the machinery around them.

WHAT CHANGED, and why. Each item is marked [FIX n] at the point of change in the code.

[FIX 1] TEST-SET LEAKAGE.
  v2's load() concatenated the train AND test loaders, so folds were drawn over test rows and
  R~_g depended on test data. R~_g feeds the training objective through lambda, so this is a
  path from test to training. We now load the training split only. --include-test reproduces
  the old behaviour for comparison, and prints a warning.

[FIX 2] THE MINIMUM OVER CANDIDATES IS NOT AN UPPER BOUND IN FINITE SAMPLES.
  E[min_k Rhat_k] <= min_k E[Rhat_k]: selecting the winner on the same out-of-fold rows that
  are then reported absorbs the most favourable noise draw. With K candidates and out-of-fold
  standard error s the downward bias is about s*sqrt(2 log K); with K=14 that is ~1.8s, and
  since s ~ 1/sqrt(n_g) it is largest exactly for the smallest groups. That makes the reference
  error UNEVEN across groups, which is the quantity that does the damage (Prop. spread).
  We now use NESTED cross-fitting: within each outer training portion, an inner CV picks the
  candidate family; that family is refitted on the full outer training portion and scored once
  on the held-out outer fold. Selection and evaluation never see the same rows.
  --naive restores the v2 estimator so the size of the curse can be measured; both numbers are
  written to the output json.

[FIX 3] THE MARGIN WAS COMPUTED DIFFERENTLY FOR DIFFERENT GROUPS.
  v2's joint_oof returned fold MEANS, so when the joint model won there were no per-sample
  losses, c_g fell back to 0, and that group alone got no margin while its neighbours got ~0.19.
  That manufactures a reference-error spread out of an implementation detail, and it hits the
  small groups, which are the ones where the joint model wins. Every candidate now returns
  per-sample losses, so c_g is constructed identically for every group.

[FIX 4] THE CANDIDATES WERE UNDERTRAINED.
  `for _ in range(EPOCHS): opt.step()` is 150 full-batch Adam steps at lr=1e-3, not 150 epochs.
  A linear model is roughly converged; mlp64 is not. Undertraining inflates a candidate's score,
  which the min then partly undoes, by an amount that differs per group. We now train to a
  plateau (higher lr, cosine decay, more steps, patience on the training loss) and report the
  fraction of fits that hit the step cap so undertraining is visible rather than silent.

[FIX 5] THE ORDERING CHECK TESTED THE WRONG QUANTITY.
  Certificate (ii) is a statement about the ESTIMATES Rhat_g. v2 ran it on the post-margin R~_g,
  so uneven margins could make it pass or fail for reasons unrelated to the estimates. It now
  runs on the pre-margin estimates, and the post-margin values are reported separately.

[FIX 6] IMPUTATION BEFORE FOLDING.
  impute_missing=True is applied inside the loader, i.e. before folds are drawn. If it uses
  global column statistics that is a second (smaller) leak. We cannot fix it from here, so we
  warn; the fix belongs in the loader, fitting the imputer on training folds only.

[FIX 7] SMALLER POINTS.
  - The pooled scope now excludes every group's evaluation rows, not just group g's.
  - The bootstrap treats pooled out-of-fold losses as i.i.d. although samples sharing a training
    fold are correlated, so the interval is slightly too narrow. Documented, and a fold-cluster
    variant is available with --cluster-bootstrap.
  - Per-fold selection counts are written out, so a group whose winner is unstable is visible.

  python estimate_rstar_v3.py --dataset fedheart
  python estimate_rstar_v3.py --dataset nhanes
  python estimate_rstar_v3.py --dataset fedheart --naive     # v2 behaviour, for comparison
"""
from __future__ import annotations
import argparse, csv, glob, json, os, warnings
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedKFold

OUTER_FOLDS, INNER_FOLDS = 5, 3
JOINT_HEAD_HIDDEN = 0   # [ADD b] 0 = linear head (as written); set by --joint-head-hidden to match
                        # the deployed model (head_hidden: 32 on the tabular datasets)
MAX_STEPS, PATIENCE, LR, WD = 1500, 60, 1e-2, 1e-4     # [FIX 4]
LATENT = 64
_CONVERGENCE = {"hit_cap": 0, "total": 0}               # [FIX 4] undertraining telemetry


# ---------------------------------------------------------------- models
def make(kind, in_dim, nc):
    if kind == "linear":
        return nn.Linear(in_dim, nc)
    h = 32 if kind == "mlp32" else 64
    return nn.Sequential(nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(),
                         nn.Linear(h, h), nn.ReLU(), nn.Linear(h, nc))


def _train_to_plateau(m, closure, max_steps=MAX_STEPS, patience=PATIENCE):
    """[FIX 4] Train until the objective stops improving, instead of a fixed step count.

    v2 ran exactly 150 full-batch steps at lr=1e-3 regardless of model size, which leaves the
    wider MLPs far from converged and inflates their out-of-fold scores.
    """
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)
    best, bad = float("inf"), 0
    for step in range(max_steps):
        m.train(); opt.zero_grad()
        loss = closure()
        loss.backward(); opt.step(); sched.step()
        v = float(loss)
        if v < best - 1e-5:
            best, bad = v, 0
        else:
            bad += 1
            if bad >= patience:
                _CONVERGENCE["total"] += 1
                return
    _CONVERGENCE["hit_cap"] += 1
    _CONVERGENCE["total"] += 1


def fit_eval(X, y, tr, ev, kind, nc, seed, weighted):
    """Fit one flat model and return PER-SAMPLE out-of-fold losses.

    weighted=False matters: Switzerland is 93% positive, so inverse-frequency weights put ~7x
    mass on 8 samples and wreck the UNWEIGHTED cross-entropy we score with. Both variants stay
    in the pool; [FIX 2] means the choice between them is now made on inner folds.
    """
    torch.manual_seed(seed)
    m = make(kind, X.shape[1], nc)
    if weighted:
        cnt = torch.bincount(y[tr], minlength=nc).float()
        w = cnt.sum() / (nc * cnt.clamp_min(1))
    else:
        w = None
    ce = nn.CrossEntropyLoss(weight=w)
    _train_to_plateau(m, lambda: ce(m(X[tr]), y[tr]))
    m.eval()
    with torch.no_grad():
        # unweighted CE at evaluation, matching how per-group loss is reported everywhere else
        return nn.functional.cross_entropy(m(X[ev]), y[ev], reduction="none").cpu().numpy()


class Joint(nn.Module):
    """Per-group encoders into a shared latent space with one shared head.

    Keeping this in the pool is correct and was one of v2's good ideas: its group-g component is
    a function of X_g alone, so it is a legitimate element of F_g, and a shared head reaches
    information a model fitted on 115 Switzerland patients cannot. Note what this makes the
    estimand: regret is then measured against the best of {isolated, pooled, joint-ERM}, so it
    answers "how much do DRO and the rest add over joint ERM?" rather than "how far is this
    group from its Bayes risk?". That is defensible, but the paper should say it.
    """

    def __init__(self, dims, latent, nc):
        super().__init__()
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Linear(d, latent), nn.LayerNorm(latent), nn.ReLU(),
                          nn.Linear(latent, latent), nn.ReLU()) for d in dims])
        self.head = (nn.Linear(latent, nc) if not JOINT_HEAD_HIDDEN else
                     nn.Sequential(nn.Linear(latent, JOINT_HEAD_HIDDEN), nn.ReLU(),
                                   nn.Linear(JOINT_HEAD_HIDDEN, nc)))

    def forward(self, x, gid):
        return self.head(self.enc[gid](x))


def joint_per_sample(X, y, masks, nc, tr_idx, ev_idx, seed=0):
    """[FIX 3] Returns PER-SAMPLE losses per group, not fold means.

    v2 returned float(mean) here, so when the joint model won the selection there were no
    per-sample losses to bootstrap and c_g silently became 0 for that group alone.
    """
    torch.manual_seed(seed)
    m = Joint([len(mk) for mk in masks], LATENT, nc)
    allc = torch.cat([y[tr_idx[gi]] for gi in tr_idx if len(tr_idx[gi])])
    cnt = torch.bincount(allc, minlength=nc).float()
    ce = nn.CrossEntropyLoss(weight=cnt.sum() / (nc * cnt.clamp_min(1)))

    def closure():
        loss = 0.0
        for gi in tr_idx:
            if len(tr_idx[gi]) == 0:
                continue
            loss = loss + ce(m(X[tr_idx[gi]][:, masks[gi]], gi), y[tr_idx[gi]])
        return loss

    _train_to_plateau(m, closure)
    m.eval()
    out = {}
    with torch.no_grad():
        for gi in ev_idx:
            if len(ev_idx[gi]) == 0:
                continue
            out[gi] = nn.functional.cross_entropy(
                m(X[ev_idx[gi]][:, masks[gi]], gi), y[ev_idx[gi]], reduction="none").cpu().numpy()
    return out


# ---------------------------------------------------------------- candidate pool
def candidate_names():
    names = ["constant", "joint"]
    for scope in ("group", "pooled"):
        for kind in ("linear", "mlp32", "mlp64"):
            for wt in (True, False):
                names.append(f"{scope}/{kind}/{'w' if wt else 'unw'}")
    return names


def all_candidate_losses(X, y, masks, nc, tr_idx, ev_idx, seed=0):
    """Per-sample losses of every candidate on every group, for one train/eval split.

    Used for BOTH the inner selection split and the outer scoring split, so selection and
    evaluation are computed the same way on disjoint rows ([FIX 2]).
    """
    res = defaultdict(dict)
    G = len(masks)

    # constant / no-features predictor: the loosest VALID upper bound on R*_g, since the
    # constants lie in F_g. A fitted estimate above it is provably invalid (certificate (i)).
    for gi in range(G):
        if len(ev_idx[gi]) == 0 or len(tr_idx[gi]) == 0:
            continue
        pr = torch.bincount(y[tr_idx[gi]], minlength=nc).float().clamp_min(1)
        pr = pr / pr.sum()
        res["constant"][gi] = (-torch.log(pr[y[ev_idx[gi]]])).cpu().numpy()

    for gi, ps in joint_per_sample(X, y, masks, nc, tr_idx, ev_idx, seed).items():
        res["joint"][gi] = ps

    # [FIX 7] pooled training rows exclude EVERY group's evaluation rows, not only group g's
    pooled_tr = torch.cat([tr_idx[gi] for gi in range(G) if len(tr_idx[gi])])
    for scope in ("group", "pooled"):
        for kind in ("linear", "mlp32", "mlp64"):
            for wt in (True, False):
                name = f"{scope}/{kind}/{'w' if wt else 'unw'}"
                for gi in range(G):
                    if len(ev_idx[gi]) == 0 or len(tr_idx[gi]) == 0:
                        continue
                    tr = tr_idx[gi] if scope == "group" else pooled_tr
                    res[name][gi] = fit_eval(X[:, masks[gi]], y, tr, ev_idx[gi], kind, nc, seed, wt)
    return res


# ---------------------------------------------------------------- estimators
def nested_estimate(X, y, g, masks, nc, outer, seed=0):
    """[FIX 2] Nested cross-fitting.

    For each outer fold: run an inner CV over the outer TRAINING rows to choose the candidate
    family per group, then refit that family on all outer training rows and score it once on the
    held-out outer fold. The reported per-sample losses therefore come from a family that never
    saw those rows during selection, so min-over-candidates cannot bias them downward.
    """
    G = len(masks)
    per_sample = {gi: [] for gi in range(G)}
    fold_of = {gi: [] for gi in range(G)}          # [FIX 7] for the cluster bootstrap
    chosen = {gi: [] for gi in range(G)}

    for k in range(len(outer[0])):
        tr_idx = {gi: outer[gi][k][0] for gi in range(G)}
        ev_idx = {gi: outer[gi][k][1] for gi in range(G)}

        # ---- inner CV on the outer training rows only: pick a family per group
        inner_scores = defaultdict(lambda: defaultdict(list))
        for gi in range(G):
            own = tr_idx[gi]
            if len(own) < INNER_FOLDS * 2:
                continue
            skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
            for j, (itr, iev) in enumerate(skf.split(own.numpy(), y[own].numpy())):
                sub_tr = {h: (own[itr] if h == gi else tr_idx[h]) for h in range(G)}
                sub_ev = {h: (own[iev] if h == gi else own[iev][:0]) for h in range(G)}
                got = all_candidate_losses(X, y, masks, nc, sub_tr, sub_ev, seed + 100 * j)
                for name, d in got.items():
                    if gi in d:
                        inner_scores[gi][name].append(float(np.mean(d[gi])))

        # ---- refit the winners on the full outer training rows, score on the outer fold
        outer_got = all_candidate_losses(X, y, masks, nc, tr_idx, ev_idx, seed + 7)
        for gi in range(G):
            if gi not in outer_got.get("constant", {}) and gi not in outer_got.get("joint", {}):
                continue
            scores = {n: float(np.mean(v)) for n, v in inner_scores[gi].items() if v}
            pick = min(scores, key=scores.get) if scores else "constant"
            if gi not in outer_got.get(pick, {}):
                pick = "constant"
            per_sample[gi].append(outer_got[pick][gi])
            fold_of[gi].append(np.full(len(outer_got[pick][gi]), k))
            chosen[gi].append(pick)

    est, ps_out, fold_out = {}, {}, {}
    for gi in range(G):
        if not per_sample[gi]:
            continue
        ps_out[gi] = np.concatenate(per_sample[gi])
        fold_out[gi] = np.concatenate(fold_of[gi])
        est[gi] = float(ps_out[gi].mean())
    return est, ps_out, fold_out, {gi: Counter(v) for gi, v in chosen.items()}


def naive_estimate(X, y, g, masks, nc, outer, seed=0):
    """The v2 estimator: min over candidates scored on the SAME out-of-fold rows.

    Kept only so the winner's curse can be measured: the gap between this and the nested
    estimate is the selection bias, and it should be largest for the smallest groups.
    """
    G = len(masks)
    acc = defaultdict(lambda: defaultdict(list))
    for k in range(len(outer[0])):
        tr_idx = {gi: outer[gi][k][0] for gi in range(G)}
        ev_idx = {gi: outer[gi][k][1] for gi in range(G)}
        got = all_candidate_losses(X, y, masks, nc, tr_idx, ev_idx, seed + 7)
        for name, d in got.items():
            for gi, v in d.items():
                acc[gi][name].append(v)
    out = {}
    for gi in acc:
        means = {n: float(np.concatenate(v).mean()) for n, v in acc[gi].items()}
        out[gi] = min(means.values())
    return out


def bootstrap_margin(ps, folds=None, n_groups=1, n_boot=4000, alpha=0.05, seed=0, cluster=False):
    """c_g: half-width of a bootstrap percentile interval for the MEAN out-of-fold loss.

    Level is union-bounded over the G groups. [FIX 3] every group reaches this with real
    per-sample losses, so c_g is built the same way everywhere.
    [FIX 7] samples sharing a training fold are correlated, so the i.i.d. bootstrap is slightly
    too narrow; --cluster-bootstrap resamples folds first, which is wider but noisier with 5
    folds. Either way c_g measures SAMPLING NOISE only -- it does not see the learning-curve
    bias E_g(m), which is the separate error discussed in the appendix.
    """
    ps = np.asarray(ps, dtype=float)
    if len(ps) < 2:
        return 0.0
    rng = np.random.default_rng(1000 + seed)
    a = alpha / n_groups
    if cluster and folds is not None:
        uf = np.unique(folds)
        groups = [ps[folds == f] for f in uf]
        bs = np.array([np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
                       for _ in range(n_boot)])
    else:
        n = len(ps)
        bs = np.array([ps[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [100 * a / 2, 100 * (1 - a / 2)])
    return float((hi - lo) / 2)


# ---------------------------------------------------------------- data
def load(dataset, base=None, include_test=False, n_folds=0, fold_index=0):
    """[FIX 1] Training split only by default.

    v2 did `for loader in (tr, te)`, so R~_g depended on test rows and then entered training
    through lambda. --include-test restores that, for comparison only.
    """
    if dataset == "fedheart":
        from dro_hetero_anchors.src.datasets_fedheart import build_fedheart_loaders
        cfg = yaml.safe_load(open(base or "experiments/fedheart_exp_paper_hetagg_gdro.yaml"))
        # group_max_train_samples deliberately NOT passed: R*_g is a property of the
        # distribution, not of the training budget we imposed on ourselves.
        # [ADD a] (2026-09-20, item 7 of the review) with --n-folds the references are built
        # from the NON-TEST rows of outer fold k only, using the same fixed per-site stratified
        # K-fold assignment the trainer uses, so no row of a test fold ever reaches a reference.
        # The loader now fits the imputation medians on those training rows too, which is the
        # fix [FIX 6] asked for.
        tr, te, info = build_fedheart_loaders(batch_size=64, seed=0, train_frac=0.8,
                                              impute_missing=True, feature_mask=None,
                                              n_folds=n_folds, fold_index=fold_index)
        masks = [m if m is not None else list(range(13)) for m in cfg["feature_mask"]]
        nc = 2
        # [FIX 6]
        if not n_folds:
            warnings.warn("no --n-folds: references come from one 80/20 training split, not "
                          "from the outer fold they will be used in.")
    else:
        from dro_hetero_anchors.src.datasets_nhanes import build_nhanes_loaders
        cfg = yaml.safe_load(open(base or "experiments/nhanes_pergroup_gdro.yaml"))
        tr, te, info = build_nhanes_loaders(batch_size=256, seed=0, data_split_seed=100,
                                            feature_mode=cfg.get("feature_mode", "nested"))
        masks = [list(v) for _, v in sorted(info["feature_indices"].items(),
                                            key=lambda kv: int(kv[0]))]
        nc = cfg.get("num_classes", 2)

    loaders = (tr, te) if include_test else (tr,)
    if include_test:
        warnings.warn("--include-test: the estimate will depend on test rows and must not be "
                      "used for any reported result.")
    xs, ys, gs = [], [], []
    for loader in loaders:
        for x, yy, gg in loader:
            xs.append(x); ys.append(yy); gs.append(gg)
    return torch.cat(xs), torch.cat(ys), torch.cat(gs), masks, nc


def achieved(dataset):
    """Lowest per-group loss any run actually reached. A CHECK on the result, never an input."""
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


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fedheart", "nhanes"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--include-test", action="store_true",
                    help="[FIX 1] reproduce v2's leak; for comparison only")
    ap.add_argument("--naive", action="store_true",
                    help="[FIX 2] also report the v2 min-over-candidates estimate")
    ap.add_argument("--cluster-bootstrap", action="store_true", help="[FIX 7]")
    ap.add_argument("--n-folds", type=int, default=0, help="[ADD a] outer K of the experiment")
    ap.add_argument("--fold", type=int, default=0, help="[ADD a] which outer fold's training rows to use")
    ap.add_argument("--joint-head-hidden", type=int, default=0,
                    help="[ADD b] hidden width of the joint candidate's head; 0 = linear")
    ap.add_argument("--group-cap", nargs="+", type=int, default=None,
                    help="per-group cap on rows used to fit R*, to show how the estimate "
                         "depends on n. Default: no cap, which is the correct estimator.")
    args = ap.parse_args()

    global JOINT_HEAD_HIDDEN
    JOINT_HEAD_HIDDEN = args.joint_head_hidden
    X, y, g, masks, nc = load(args.dataset, args.base, args.include_test, args.n_folds, args.fold)
    if args.group_cap:
        keep, rng = [], np.random.default_rng(0)
        for gi_, cap in enumerate(args.group_cap):
            idx = (g == gi_).nonzero(as_tuple=True)[0].numpy()
            if cap and cap > 0 and len(idx) > cap:
                idx = rng.choice(idx, size=cap, replace=False)
            keep.append(idx)
        keep_t = torch.tensor(np.sort(np.concatenate(keep)))
        X, y, g = X[keep_t], y[keep_t], g[keep_t]
        print(f"  --group-cap applied: {args.group_cap} -> {len(y)} rows")

    G = len(masks)
    print(f"{args.dataset}: {len(y)} samples, {G} groups, "
          f"{'train+TEST (leaky)' if args.include_test else 'train split only'}\n")
    floor = achieved(args.dataset)

    # outer fold assignments, shared by every candidate so the comparison is like for like
    outer = {}
    for gi in range(G):
        own = (g == gi).nonzero(as_tuple=True)[0]
        skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=0)
        outer[gi] = [(own[a], own[b]) for a, b in skf.split(own.numpy(), y[own].numpy())]

    print("  nested cross-fitting (inner CV selects, outer fold scores)...", flush=True)
    est, ps, folds_of, chosen = nested_estimate(X, y, g, masks, nc, outer)

    naive = naive_estimate(X, y, g, masks, nc, outer) if args.naive else {}

    # constant-predictor bound, computed on the same outer folds (certificate (i))
    const_bound = {}
    for gi in range(G):
        vals = []
        for tr, ev in outer[gi]:
            pr = torch.bincount(y[tr], minlength=nc).float().clamp_min(1)
            pr = pr / pr.sum()
            vals.append(float((-torch.log(pr[y[ev]])).mean()))
        const_bound[gi] = float(np.mean(vals))

    out, out_pre, margin, detail = {}, {}, {}, {}
    for gi in sorted(est):
        c_g = bootstrap_margin(ps[gi], folds_of[gi], n_groups=G, seed=gi,
                               cluster=args.cluster_bootstrap)
        pre = min(est[gi], const_bound[gi])                # truncation enforces certificate (i)
        out_pre[str(gi)] = round(pre, 6)
        margin[str(gi)] = round(c_g, 6)
        out[str(gi)] = round(max(pre - c_g, 0.0), 6)       # R~_g, what training consumes
        detail[str(gi)] = dict(chosen[gi])
        n_g = int((g == gi).sum())
        extra = f"  naive {naive[gi]:6.3f} (curse {est[gi] - naive[gi]:+.3f})" if naive else ""
        fl = floor.get(gi, float("inf"))
        flag = "  <-- above an achieved loss" if pre > fl + 1e-6 else ""
        print(f"  g{gi}: n={n_g:>5}  nested {est[gi]:6.3f}  const {const_bound[gi]:6.3f}  "
              f"c_g {c_g:.3f}  R~ {out[str(gi)]:6.3f}{extra}  achieved {fl:6.3f}{flag}")
        print(f"        selected per fold: {dict(chosen[gi])}")

    path = args.out or (f"runs/rstar_{args.dataset}_v3.json" if args.dataset == "fedheart"
                        else "runs/rstar_nhanes_nested_v3.json")
    json.dump({"rstar": out,
               "rstar_before_margin": out_pre,
               "rstar_nested_raw": {str(k): round(v, 6) for k, v in est.items()},
               "rstar_naive_min": {str(k): round(v, 6) for k, v in naive.items()},
               "margin_c_g": margin,
               "constant_predictor_bound": {str(k): round(v, 6) for k, v in const_bound.items()},
               "selected_per_fold": detail,
               "outer_fold": ({"n_folds": args.n_folds, "fold": args.fold} if args.n_folds else None),
               "train_rows_sha1": __import__("hashlib").sha1(X.numpy().tobytes() + y.numpy().tobytes()).hexdigest(),
               "n_rows": int(len(y)),
               "joint_head_hidden": JOINT_HEAD_HIDDEN,
               "method": "nested CV (inner selects, outer scores); train split only; "
                         "per-sample bootstrap margin; truncated at the constant predictor"},
              open(path, "w"), indent=2)

    print("\nchecks:")
    viol = [k for k, v in out.items() if v > floor.get(int(k), float("inf")) + 1e-6]
    print(f"  above an achieved loss: {len(viol)}/{len(out)}  {viol}")
    cert_i = [k for k, v in est.items() if v > const_bound[k] + 1e-6]
    print(f"  certificate (i) failures (estimate worse than base rate): {cert_i}")
    if args.dataset == "nhanes":
        # [FIX 5] certificate (ii) is about the ESTIMATES, so test the pre-margin values
        vals = [out_pre[str(i)] for i in range(G) if str(i) in out_pre]
        ok = all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))
        print(f"  certificate (ii) ordering on ESTIMATES: {'holds' if ok else 'VIOLATED'}"
              f"  {[round(v, 3) for v in vals]}")
        if not ok:
            inv = max(vals[i + 1] - vals[i] for i in range(len(vals) - 1))
            print(f"    certified reference-error spread >= {inv:.3f}")
    hc, tot = _CONVERGENCE["hit_cap"], max(_CONVERGENCE["total"], 1)
    print(f"  fits that hit the step cap (possible undertraining): {hc}/{tot} = {hc / tot:.1%}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()