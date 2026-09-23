"""Protocol v4 trainer for the tabular datasets (documentation/PROTOCOL_V4_2026-09-22.md).

One code path for every method, ours and the published baselines, so the sampler, the budget and
the logging are identical by construction:

  * equal-group batches: every step draws --per-group samples from every group
  * uniform initial group weights, logged as epoch 0
  * constant learning rate for every method; fixed --epochs (10), no early stopping; every epoch stores per-group validation and test metrics
    and the group weights, so any validation-based selection rule can be applied afterwards
    (report_v4.py)
  * our arms follow Algorithm 1 of the draft literally:
        J = sum_g lambda_g [ (L_g - R_g) + a_align L_align_g ] + a_sep L_sep
        Lbar_g <- rho Lbar_g + (1 - rho) (L_g + a_align L_align_g)      (training batches)
        every N steps: lambda_g <- lambda_g exp(gamma (Lbar_g - R_g)), renormalised   (signed excess)
    with R_g = 0 for GroupDRO. N = min(50, steps per epoch).

Class weights: 'auto' on NHANES (10% positive) for every method, none on Fed-Heart for every method.
The training objective uses the class-weighted loss; the weight signal and every reported loss use
the unweighted loss, which is what the references R_g are estimated in.

  python train_v4_tabular.py --dataset nhanes --base experiments/nhanes_v3.yaml \
      --rstar runs/rstar_v3/nhanes/nested.json --out runs/v4/nhanes
  python train_v4_tabular.py --dataset fedheart --base experiments/fedheart_v3.yaml \
      --rstar runs/rstar_v3/fedheart --folds 5 --out runs/v4/fedheart
"""
import argparse, csv, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from dro_hetero_anchors.src.encoders import ENCODER_REGISTRY
from dro_hetero_anchors.src.model.head import MLPHead, LinearHead
from dro_hetero_anchors.src.model.anchors import AnchorModule
from dro_hetero_anchors.src.model.losses import anchor_sep_loss
from dro_hetero_anchors.src.model.wasserstein import diagonal_gaussian_w2_squared
from dro_hetero_anchors.src.model.baselines import FlexMoEModel, FlexMoESparse

# label: (shared encoder, DRO, regret, anchors, independent heads)
OURS = {
    "ERM":                 (True,  False, False, False, False),
    "Shared_GDRO":         (True,  True,  False, False, False),
    "Shared_Anchors_GDRO": (True,  True,  False, True,  False),
    "PerGroupOnly":        (False, False, False, False, False),
    "AnchorsOnly":         (False, False, False, True,  False),
    "GroupDRO":            (False, True,  False, False, False),
    "RegretDRO":           (False, True,  True,  False, False),
    "Ours_GDRO":           (False, True,  False, True,  False),
    "Ours_Regret":         (False, True,  True,  True,  False),
    "Independent":         (False, False, False, False, True),
}
BASELINES = ["Reweigh", "FlexMoE", "REMIND"]
HAS_STEP = {m for m, f in OURS.items() if f[1]} | {"REMIND"}


# ------------------------------------------------------------------ data ----
def _stack(loader):
    xs, ys, gs = [], [], []
    for x, y, g in loader:
        xs.append(x); ys.append(y); gs.append(g)
    return torch.cat(xs).float(), torch.cat(ys).long(), torch.cat(gs).long()


def load_split(dataset, cfg, fold, n_folds):
    """Materialise train / val / test tensors once; the loaders' own batching is not used."""
    if dataset == "nhanes":
        from dro_hetero_anchors.src.datasets_nhanes import build_nhanes_loaders
        tr, te, info = build_nhanes_loaders(
            batch_size=4096, seed=cfg.get("seed", 42), stratified=False,
            train_frac=cfg.get("train_frac", 0.8), val_frac=cfg.get("val_frac", 0.15),
            use_post_pandemic=cfg.get("use_post_pandemic", True),
            data_split_seed=cfg.get("data_split_seed"), feature_mode=cfg.get("feature_mode", "nested"))
        feat = {int(g): list(v) for g, v in info["feature_indices"].items()}
    else:
        from dro_hetero_anchors.src.datasets_fedheart import build_fedheart_loaders
        tr, te, info = build_fedheart_loaders(
            batch_size=4096, seed=0, stratified=False, feature_mask=cfg.get("feature_mask"),
            impute_missing=True, val_frac=cfg.get("val_frac", 0.15), n_folds=n_folds, fold_index=fold)
        masks = cfg.get("feature_mask") or [None] * 4
        feat = {g: (list(m) if m is not None else list(range(13))) for g, m in enumerate(masks)}
    va = info["val_loader"]
    return _stack(tr), _stack(va), _stack(te), feat, info


class EqualGroupBatches:
    """Every step: `per_group` indices from every group. Each group walks its own shuffled
    permutation without replacement and reshuffles when it runs out."""
    def __init__(self, g, per_group, rng):
        self.idx = [np.where(g.numpy() == k)[0] for k in range(int(g.max()) + 1)]
        self.m, self.rng = per_group, rng
        self.perm = [rng.permutation(ix) for ix in self.idx]
        self.pos = [0] * len(self.idx)
        self.steps_per_epoch = int(math.ceil(max(len(ix) for ix in self.idx) / per_group))

    def next(self):
        out = []
        for k, ix in enumerate(self.idx):
            take = []
            while len(take) < self.m:
                if self.pos[k] >= len(ix):
                    self.perm[k] = self.rng.permutation(ix); self.pos[k] = 0
                need = self.m - len(take)
                take.extend(self.perm[k][self.pos[k]:self.pos[k] + need]); self.pos[k] += need
            out.append(np.asarray(take[:self.m]))
        return np.concatenate(out)


class ProportionalBatches:
    """Protocol v4b: every batch keeps the training group proportions (the stratified sampler every
    earlier run used), at least one sample per group. Each group walks its own shuffled permutation."""
    def __init__(self, g, batch, rng):
        self.idx = [np.where(g.numpy() == k)[0] for k in range(int(g.max()) + 1)]
        n = np.array([len(ix) for ix in self.idx], dtype=float)
        per = np.maximum(1, np.round(n / n.sum() * batch)).astype(int)
        self.m = per.tolist()
        self.rng = rng; self.perm = [rng.permutation(ix) for ix in self.idx]; self.pos = [0] * len(self.idx)
        self.steps_per_epoch = int(math.ceil(n.sum() / sum(self.m)))

    def next(self):
        out = []
        for k, ix in enumerate(self.idx):
            take = []
            while len(take) < self.m[k]:
                if self.pos[k] >= len(ix):
                    self.perm[k] = self.rng.permutation(ix); self.pos[k] = 0
                need = self.m[k] - len(take)
                take.extend(self.perm[k][self.pos[k]:self.pos[k] + need]); self.pos[k] += need
            out.append(np.asarray(take[:self.m[k]]))
        return np.concatenate(out)


# --------------------------------------------------------------- metrics ----
def group_metrics(logits, y, g, G):
    from sklearn.metrics import roc_auc_score
    out = []
    p1 = torch.softmax(logits, 1)[:, 1]
    pred = logits.argmax(1)
    for k in range(G):
        m = g == k
        n = int(m.sum())
        if n == 0:
            out.append(dict(n=0, loss=float("nan"), acc=float("nan"), f1=float("nan"), auroc=float("nan"))); continue
        yk, pk = y[m], pred[m]
        f1s = []
        for c in (0, 1):
            tp = int(((pk == c) & (yk == c)).sum()); fp = int(((pk == c) & (yk != c)).sum()); fn = int(((pk != c) & (yk == c)).sum())
            pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
            f1s.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
        au = float(roc_auc_score(yk.numpy(), p1[m].numpy())) if len(set(yk.tolist())) == 2 else float("nan")
        out.append(dict(n=n, loss=float(F.cross_entropy(logits[m], yk)), acc=float((pk == yk).float().mean()),
                        f1=sum(f1s) / 2, auroc=au))
    return out


# ---------------------------------------------------------------- models ----
class OursModel(nn.Module):
    def __init__(self, cfg, feat, G, shared, indep, common_idx):
        super().__init__()
        k = cfg["latent_dim"]; gc = cfg["groups"]
        enc_cls = ENCODER_REGISTRY[gc[0]["encoder"]]
        mk = lambda d, gcfg: enc_cls(k, input_dim=d, hidden_dim=gcfg.get("hidden_dim", 64), dropout=gcfg.get("dropout", 0.1))
        self.shared, self.G, self.k = shared, G, k
        self.idx = [torch.tensor(common_idx if shared else feat[g]) for g in range(G)]
        if shared:
            self.enc = nn.ModuleList([mk(len(common_idx), gc[0])])
        else:
            self.enc = nn.ModuleList([mk(len(feat[g]), gc[g]) for g in range(G)])
        mkh = lambda: (MLPHead(k, cfg["head_hidden"], cfg["num_classes"]) if cfg.get("head_hidden", 0) > 0
                       else LinearHead(k, cfg["num_classes"]))
        self.heads = nn.ModuleList([mkh() for _ in range(G if indep else 1)])
        self.indep = indep

    def encode(self, x, g):
        z = x.new_zeros(x.size(0), self.k)
        for k in range(self.G):
            m = g == k
            if m.any():
                z[m] = self.enc[0 if self.shared else k](x[m][:, self.idx[k]])
        return z

    def classify(self, z, g):
        if not self.indep:
            return self.heads[0](z)
        out = z.new_zeros(z.size(0), 2)
        for k in range(self.G):
            m = g == k
            if m.any():
                out[m] = self.heads[k](z[m])
        return out

    def forward(self, x, g):
        z = self.encode(x, g)
        return self.classify(z, g), z


def align_loss(z, y, anchors, n_min=2):
    """Equation (6) for one group: mean over the classes with >= n_min samples of the closed-form
    W2^2 between the class cloud's diagonal Gaussian and the class anchor."""
    m_anc, var_anc = anchors.m, anchors.variance()
    terms = []
    for c in range(m_anc.size(0)):
        zc = z[y == c]
        if zc.size(0) < n_min:
            continue
        terms.append(diagonal_gaussian_w2_squared(zc.mean(0), zc.var(0, unbiased=True) + anchors.eps, m_anc[c], var_anc[c]))
    return torch.stack(terms).mean() if terms else None


# ------------------------------------------------------------------ train ----
def run_one(method, step, seed, fold, data, feat, cfg, rstar, args, blocks=None, group_blocks=None):
    (Xtr, ytr, gtr), (Xva, yva, gva), (Xte, yte, gte) = data
    G = int(gtr.max()) + 1
    torch.manual_seed(seed * 1000 + fold); rng = np.random.default_rng(seed * 1000 + fold)
    sampler = (ProportionalBatches(gtr, args.batch, rng) if args.sampler == "proportional"
               else EqualGroupBatches(gtr, args.per_group, rng))
    spe = sampler.steps_per_epoch
    N = min(50, spe)
    cls_w = None
    if args.dataset == "nhanes":
        cnt = torch.bincount(ytr, minlength=2).float()
        cls_w = cnt.sum() / (2 * cnt.clamp_min(1))
    R = torch.tensor([float(rstar[str(k)]) for k in range(G)])

    ours = method in OURS
    anchors = None
    if ours:
        shared, dro, regret, use_anchors, indep = OURS[method]
        common = sorted(set.intersection(*[set(feat[k]) for k in range(G)])) if args.dataset == "nhanes" else list(range(13))
        model = OursModel(cfg, feat, G, shared, indep, common)
        params = list(model.parameters())
        n_params = sum(p.numel() for p in params)
        if use_anchors:
            anchors = AnchorModule(cfg["num_classes"], cfg["latent_dim"], eps=cfg["anchor_eps"], diagonal=True)
            params += list(anchors.parameters())
        fwd = lambda X, g, **kw: model(X, g)[0]
    else:
        dro, regret, use_anchors = (method == "REMIND"), False, False
        if method == "FlexMoE":
            model = FlexMoESparse(len(blocks), group_blocks, [len(b) for b in blocks], d_model=128, n_experts=16,
                                  top_k=4, num_classes=2)
        else:
            model = FlexMoEModel([len(b) for b in blocks], group_blocks, latent_dim=cfg.get("latent_dim", 64),
                                 num_classes=2, n_experts=4, group_routing=(method == "REMIND"))
        params = list(model.parameters()); n_params = sum(p.numel() for p in params)
        def fwd(X, g, warmup=False):
            xb = [X[:, idx] for idx in blocks]
            return (model(xb, g, warmup=warmup) if method == "FlexMoE" else model(xb, g))[0]

    opt = torch.optim.Adam(params, lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=cfg.get("lr_min", 1e-5))
             if args.schedule == "cosine" else None)          # v4a: constant for everyone; v4b: cosine for everyone
    a_fit, a_sep = args.alpha_align, args.alpha_sep
    lam = torch.full((G,), 1.0 / G)
    Lbar = R.clone() if regret else None                          # Algorithm 1 line 6; GroupDRO: first batch
    Rref = R if regret else torch.zeros(G)
    rows, gstep = [], 0

    def log_epoch(ep):
        model.eval()
        with torch.no_grad():
            mv = group_metrics(fwd(Xva, gva), yva, gva, G); mt = group_metrics(fwd(Xte, gte), yte, gte, G)
        model.train()
        for k in range(G):
            rows.append(dict(method=method, step=step, seed=seed, fold=fold, epoch=ep, group=f"g{k}",
                             n_val=mv[k]["n"], val_loss=mv[k]["loss"], val_acc=mv[k]["acc"], val_auroc=mv[k]["auroc"],
                             n_test=mt[k]["n"], test_loss=mt[k]["loss"], test_acc=mt[k]["acc"], test_f1=mt[k]["f1"],
                             test_auroc=mt[k]["auroc"], weight=(float(lam[k]) if dro else float("nan")),
                             R_star=float(R[k]), n_params=n_params))
    log_epoch(0)
    warm_eps = max(1, round(args.epochs * 5 / 60)) if args.sampler == "equal" else 5   # Flex-MoE: 5 warm-up epochs at release
    for ep in range(1, args.epochs + 1):
        if method == "REMIND" and getattr(model, "moe", None) is not None and model.moe.phi_res is not None:
            model.moe.residual_active = ep > args.epochs // 2      # REMIND stage 2
        for _ in range(spe):
            ix = torch.from_numpy(sampler.next())
            x, y, g = Xtr[ix], ytr[ix], gtr[ix]
            if ours:
                logits, z = model(x, g)
            else:
                logits = fwd(x, g, warmup=(method == "FlexMoE" and ep <= warm_eps)); z = None
            ce_w = torch.stack([F.cross_entropy(logits[g == k], y[g == k], weight=cls_w) for k in range(G)])
            ce_u = torch.stack([F.cross_entropy(logits[g == k], y[g == k]) for k in range(G)]).detach()
            al = torch.zeros(G)
            if anchors is not None:
                y_anc = y[torch.randperm(y.shape[0])] if args.random_anchors else y      # control: labels permuted within the batch
                al = torch.stack([(lambda a: a if a is not None else z.new_zeros(()))(align_loss(z[g == k], y_anc[g == k], anchors))
                                  for k in range(G)])
            if dro:
                loss = (lam * (ce_w + a_fit * al)).sum()
            elif args.sampler == "proportional":
                loss = F.cross_entropy(logits, y, weight=cls_w) + a_fit * al.mean()   # plain ERM over the batch
            else:
                loss = ce_w.mean() + a_fit * al.mean()             # equal-group batch: group-balanced mean
            if anchors is not None:
                m_anc, S_anc, L_norm = anchors.forward()
                loss = loss + a_sep * anchor_sep_loss(m_anc, S_anc, L_norm, model.heads[0], cfg["num_classes"],
                                                      cfg.get("sep_samples_per_class", 4), torch.device("cpu"),
                                                      sep_method=cfg.get("sep_method", "classifier"),
                                                      margin=cfg.get("sep_margin", 1.0), eps=cfg["anchor_eps"])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.get("grad_clip", 1.0)); opt.step()
            gstep += 1
            if dro:
                raw = ce_u + a_fit * al.detach()
                Lbar = raw.clone() if Lbar is None else args.rho * Lbar + (1 - args.rho) * raw
                if gstep % N == 0:
                    s = step * (Lbar - Rref)                       # signed excess, no clamp
                    lam = lam * torch.exp(s - s.max()); lam = lam / lam.sum()
        if sched is not None:
            sched.step()
        log_epoch(ep)
    if args.save_latents and ours and fold == args.latents_fold:
        model.eval()
        with torch.no_grad():
            z = model.encode(Xte, gte)
        os.makedirs(args.save_latents, exist_ok=True)
        tag = "random_anchors" if args.random_anchors else ("real_anchors" if anchors is not None else "no_anchors")
        np.savez_compressed(os.path.join(args.save_latents, f"{tag}_s{seed}.npz"), z=z.numpy(), y=yte.numpy(), g=gte.numpy(),
                            anchor_m=(anchors.m.detach().numpy() if anchors is not None else np.zeros((2, z.size(1)))),
                            anchor_S=(np.stack([np.diag(v) for v in anchors.variance().detach().numpy()]) if anchors is not None else np.zeros((2, z.size(1), z.size(1)))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nhanes", "fedheart"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--rstar", required=True, help="json (nhanes) or directory of fold<k>.json (fedheart)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=list(OURS) + BASELINES)
    ap.add_argument("--steps", nargs="+", type=float, default=[0.1, 0.5, 2.0, 10.0], help="eta grid for our DRO arms")
    ap.add_argument("--remind-steps", nargs="+", type=float, default=[0.02, 0.5, 2.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 7, 2024, 31337, 11, 22, 33, 44, 55])
    ap.add_argument("--folds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--per-group", type=int, default=None, help="samples per group per step; default 50 (NHANES), 16 (Fed-Heart)")
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--sampler", choices=["equal", "proportional"], default="equal")
    ap.add_argument("--batch", type=int, default=None, help="proportional sampler: batch size (default from the config)")
    ap.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    ap.add_argument("--alpha-align", type=float, default=0.1)
    ap.add_argument("--alpha-sep", type=float, default=0.1)
    ap.add_argument("--shard", default="0/1", help="i/n: this process handles jobs with index % n == i")
    ap.add_argument("--save-latents", default=None, help="write test latents (z, y, g, anchors) for plot_latent_scatter.py")
    ap.add_argument("--latents-fold", type=int, default=0)
    ap.add_argument("--random-anchors", action="store_true", help="control: alignment targets use permuted labels")
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.per_group is None:
        args.per_group = 50 if args.dataset == "nhanes" else 16
    cfg = yaml.safe_load(open(args.base))
    if args.batch is None:
        args.batch = int(cfg.get("batch_size", 64))
    si, sn = map(int, args.shard.split("/"))
    os.makedirs(os.path.join(args.out, "epochs"), exist_ok=True)

    blocks = group_blocks = None
    if any(m in BASELINES for m in args.methods):
        from run_baselines_tabular import build_blocks
        blocks, group_blocks = build_blocks(args.dataset, cfg)

    jobs = []
    for m in args.methods:
        grid = args.steps if m in OURS and m in HAS_STEP else args.remind_steps if m == "REMIND" else [0.0]
        for st in grid:
            for sd in args.seeds:
                jobs.append((m, st, sd))
    jobs = [j for i, j in enumerate(jobs) if i % sn == si]

    cache = {}
    t0 = time.time()
    for ji, (m, st, sd) in enumerate(jobs):
        path = os.path.join(args.out, "epochs", f"{m}__step{st:g}__s{sd}.csv")
        if os.path.exists(path):
            continue
        rows = []
        for fold in range(args.folds):
            if fold not in cache:
                tr, va, te, feat, _ = load_split(args.dataset, cfg, fold, args.folds if args.folds > 1 else 0)
                cache[fold] = (tr, va, te, feat)
            tr, va, te, feat = cache[fold]
            if m in OURS and OURS[m][0] and args.dataset == "nhanes" and not set.intersection(*[set(v) for v in feat.values()]):
                rows = None; break                                  # no common feature: no shared-encoder arm
            rs = (json.load(open(args.rstar))["rstar"] if args.dataset == "nhanes"
                  else json.load(open(os.path.join(args.rstar, f"fold{fold}.json")))["rstar"])
            rows += run_one(m, st, sd, fold, (tr, va, te), feat, cfg, rs, args, blocks, group_blocks)
        if rows is None:
            print(f"skip {m}: groups share no feature", flush=True); continue
        with open(path + ".tmp", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        os.replace(path + ".tmp", path)
        print(f"[{ji+1}/{len(jobs)}] {m} step={st:g} s{sd}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
