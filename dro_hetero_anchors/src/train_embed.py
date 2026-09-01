"""Training script for EMBED mammography (density classification).

Setup mirrors the REMIND paper (arXiv 2603.00046): groups = modality
combinations, metric = accuracy reported per head / tail / overall. Our method =
shared multi-view encoder (mask-driven over missing modalities) → shared latent +
class-conditional Gaussian anchors + GroupDRO reweighting of tail groups.

Unlike the tabular trainers, EMBED uses ONE multi-view encoder that natively
handles missing modalities via the presence mask, so there is no per-group
encoder dict — group ids feed GroupDRO for reweighting only.

Run:
    python -m dro_hetero_anchors.src.train_embed --config experiments/embed_baseline.yaml
    python -m dro_hetero_anchors.src.train_embed --config experiments/embed_groupdro.yaml
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import set_seed, ensure_dir, Meter, console
from .results_logger import ResultsLogger
from .datasets_embed import build_embed_loaders, print_embed_summary
from .encoders import ENCODER_REGISTRY
from .model.head import LinearHead, MLPHead
from .model.anchors import AnchorModule
from .model.losses import per_class_batch_moments, anchor_fit_loss, anchor_sep_loss
from .model.groupdro import GroupDRO


def build_models(cfg, info: Dict, device):
    latent_dim = cfg["latent_dim"]
    num_groups = info["num_groups"]

    enc_cls = ENCODER_REGISTRY[cfg.get("encoder", "mammo_multiview")]
    encoder = enc_cls(
        latent_dim,
        num_view_types=cfg.get("num_view_types", 4),
        pretrained=cfg.get("pretrained", True),
        dropout=cfg.get("dropout", 0.2),
        backbone=cfg.get("backbone", "resnet18"),
        per_view_encoders=cfg.get("per_view_encoders", False),
    )

    head = (MLPHead(latent_dim, cfg["head_hidden"], info["num_classes"])
            if cfg.get("head_hidden", 0) > 0
            else LinearHead(latent_dim, info["num_classes"]))
    anchors = AnchorModule(info["num_classes"], latent_dim, eps=cfg["anchor_eps"])

    groupdro = None
    if cfg.get("groupdro_enabled", False):
        groupdro = GroupDRO(
            num_groups=num_groups, eta=cfg.get("groupdro_eta", 0.1), device=device,
            update_mode=cfg.get("groupdro_update_mode", "exp_smooth"),
            robust_objective=cfg.get("groupdro_objective", "weighted"),
            gamma=cfg.get("groupdro_gamma", 0.9),
            group_counts=info["train_group_counts"],
            kl_lambda=cfg.get("groupdro_kl_lambda", 0.0),
            uniform_init=cfg.get("groupdro_uniform_init", True),
        )
    return encoder, head, anchors, groupdro


def _apply_view_dropout(mask: torch.Tensor, p: float, gen: torch.Generator) -> torch.Tensor:
    """Randomly drop present views with prob p, keeping >=1 present per sample.
    Simulates missing modalities at inference. Returns a new mask (views are zeroed
    downstream via the mask in the encoder's masked-mean pooling)."""
    if p <= 0:
        return mask
    m = mask.clone()
    B, V = m.shape
    drop = (torch.rand(B, V, generator=gen, device=m.device) < p) & (m > 0)
    new = m * (~drop).float()
    # ensure at least one present view: rows that lost everything keep one original view
    empty = new.sum(dim=1) == 0
    if empty.any():
        for i in torch.nonzero(empty, as_tuple=False).flatten().tolist():
            present = torch.nonzero(m[i] > 0, as_tuple=False).flatten()
            if len(present):
                keep = present[torch.randint(len(present), (1,), generator=gen, device=m.device)]
                new[i, keep] = 1.0
    return new


@torch.no_grad()
def evaluate(encoder, head, loader, device, info, view_dropout: float = 0.0,
             dropout_seed: int = 0) -> Dict[str, Any]:
    encoder.eval(); head.eval()
    num_groups, num_classes = info["num_groups"], info["num_classes"]
    gen = torch.Generator(device=device); gen.manual_seed(dropout_seed)
    correct_g = [0] * num_groups
    total_g = [0] * num_groups
    correct = total = 0
    for views, mask, y, g in loader:
        views, mask, y = views.to(device), mask.to(device), y.to(device)
        if view_dropout > 0:
            mask = _apply_view_dropout(mask, view_dropout, gen)
            views = views * mask.view(mask.size(0), mask.size(1), 1, 1, 1)
        z = encoder(views, mask)
        pred = head(z).argmax(dim=1).cpu()
        y = y.cpu()
        for i in range(len(y)):
            gi = int(g[i]); ok = int(pred[i] == y[i])
            total_g[gi] += 1; correct_g[gi] += ok
            total += 1; correct += ok

    per_group_acc = [correct_g[i] / max(1, total_g[i]) for i in range(num_groups)]
    head_gids, tail_gids = info["head_gids"], info["tail_gids"]

    def _pooled(gids):
        c = sum(correct_g[i] for i in gids); t = sum(total_g[i] for i in gids)
        return c / max(1, t)

    return {
        "overall_acc": correct / max(1, total),
        "head_acc": _pooled(head_gids) if head_gids else 0.0,
        "tail_acc": _pooled(tail_gids) if tail_gids else 0.0,
        "worst_group_acc": min(per_group_acc) if per_group_acc else 0.0,
        "per_group_acc": per_group_acc,
        "per_group_counts": total_g,
    }


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.log(f"Using device: {device}")
    ensure_dir(cfg["run_dir"])

    train_loader, test_loader, info = build_embed_loaders(
        data_root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
        image_size=cfg.get("image_size", 224),
        train_frac=cfg.get("train_frac", 0.8),
        data_split_seed=cfg.get("data_split_seed", 42),
        tail_threshold=cfg.get("tail_threshold", 0.15),
        stratified=cfg.get("stratified_batching", True),
        group_max_train_samples=cfg.get("group_max_train_samples"),
        require_local_images=cfg.get("require_local_images", False),
        min_group_size=cfg.get("min_group_size", 0),
        tail_train_cap=cfg.get("tail_train_cap"),
        head_group_names=cfg.get("head_group_names"),
    )
    # Principled head/tail: head = named complete-modality combos, everything else = tail.
    # (Stable across subset proportions, unlike the frequency-based default.)
    if cfg.get("head_group_names"):
        heads = set(cfg["head_group_names"])
        name_by_gid = {g["gid"]: g["name"] for g in info["groups"]}
        info["head_gids"] = [gid for gid, nm in name_by_gid.items() if nm in heads]
        info["tail_gids"] = [gid for gid, nm in name_by_gid.items() if nm not in heads]
        for g in info["groups"]:
            g["is_tail"] = g["name"] not in heads
    print_embed_summary(info)
    set_seed(cfg["seed"])

    encoder, head, anchors, groupdro = build_models(cfg, info, device)
    encoder, head, anchors = encoder.to(device), head.to(device), anchors.to(device)

    # Optional inverse-frequency class weighting for density imbalance (A~10%, D~5%)
    class_weight = None
    if cfg.get("class_weight") == "auto":
        counts = torch.tensor(info["train_label_counts"], dtype=torch.float, device=device)
        class_weight = (counts.sum() / (len(counts) * counts.clamp_min(1)))
        console.log(f"class_weight (auto): {[round(w,3) for w in class_weight.tolist()]}")

    params = list(encoder.parameters()) + list(head.parameters()) + list(anchors.parameters())
    # Optimizer (config: optimizer = adamw|adam|sgd)
    optname = cfg.get("optimizer", "adamw").lower()
    if optname == "sgd":
        opt = optim.SGD(params, lr=cfg["lr"], momentum=cfg.get("momentum", 0.9),
                        nesterov=True, weight_decay=cfg["weight_decay"])
    elif optname == "adam":
        opt = optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        opt = optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # LR scheduler (config: scheduler = none|cosine|step|cosine_warmup)
    import math as _math
    sched, schname = None, cfg.get("scheduler")
    if schname == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    elif schname == "step":
        sched = optim.lr_scheduler.StepLR(opt, step_size=max(1, cfg["epochs"] // 3), gamma=0.3)
    elif schname == "cosine_warmup":
        _warm = max(1, cfg["epochs"] // 10)
        def _lrl(ep):
            if ep < _warm:
                return (ep + 1) / _warm
            return 0.5 * (1 + _math.cos(_math.pi * (ep - _warm) / max(1, cfg["epochs"] - _warm)))
        sched = optim.lr_scheduler.LambdaLR(opt, _lrl)
    console.log(f"optimizer={optname} lr={cfg['lr']} scheduler={schname}")

    writer = SummaryWriter(log_dir=str(cfg["run_dir"]))
    logger = ResultsLogger(cfg["run_dir"])
    lambda_fit, lambda_sep = cfg["lambda_fit"], cfg["lambda_sep"]
    J, eps = cfg["sep_samples_per_class"], cfg["anchor_eps"]
    num_classes = info["num_classes"]
    best_tail = best_overall = 0.0
    best_overall_metrics = best_tail_metrics = None
    step = 0

    for epoch in range(1, cfg["epochs"] + 1):
        encoder.train(); head.train(); anchors.train()
        lm, am = Meter(), Meter()
        for views, mask, y, g in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}"):
            views, mask, y, g = views.to(device), mask.to(device), y.to(device), g.to(device)
            z = encoder(views, mask)
            logits = head(z)

            if groupdro is not None:
                ce = groupdro.forward(logits, y, g, num_classes=num_classes,
                                      class_weight=class_weight)
            else:
                ce = nn.functional.cross_entropy(logits, y, weight=class_weight)

            moments = per_class_batch_moments(z, y, num_classes, eps)
            m_anc, S_anc, L_norm = anchors.forward()
            l_fit = anchor_fit_loss(m_anc, S_anc, moments, eps)
            l_sep = anchor_sep_loss(m_anc, S_anc, L_norm, head, num_classes, J, device,
                                    sep_method=cfg.get("sep_method", "classifier"),
                                    margin=cfg.get("sep_margin", 1.0), eps=eps)
            loss = ce + lambda_fit * l_fit + lambda_sep * l_sep

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("grad_clip", 0) > 0:
                nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            opt.step()
            if groupdro is not None:
                groupdro.update_weights(groupdro._last_group_losses, groupdro._last_group_counts)

            acc = (logits.argmax(1) == y).float().mean()
            lm.update(loss.item(), y.size(0)); am.update(acc.item(), y.size(0))
            if step % cfg.get("log_interval", 50) == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/l_fit", l_fit.item(), step)
                writer.add_scalar("train/l_sep", l_sep.item(), step)
            step += 1

        m = evaluate(encoder, head, test_loader, device, info)
        console.log(f"[bold]Epoch {epoch}[/bold] train_loss={lm.avg:.4f} "
                    f"overall={m['overall_acc']:.4f} head={m['head_acc']:.4f} "
                    f"tail={m['tail_acc']:.4f} worst_group={m['worst_group_acc']:.4f}")
        for k in ("overall_acc", "head_acc", "tail_acc", "worst_group_acc"):
            writer.add_scalar(f"test/{k}", m[k], epoch)
        logger.log_epoch({"config": cfg, "epoch": epoch, "train_loss": float(lm.avg),
                          **{f"test_{k}": v for k, v in m.items()}})
        logger.save()

        save = {"cfg": cfg, "epoch": epoch, "encoder": encoder.state_dict(),
                "head": head.state_dict(), "anchors": anchors.state_dict(), "metrics": m}
        if m["tail_acc"] > best_tail:
            best_tail = m["tail_acc"]; best_tail_metrics = m
            torch.save(save, Path(cfg["run_dir"]) / "best_tail.ckpt")
            console.log(f"[green]New best tail acc: {best_tail:.4f}[/green]")
        if m["overall_acc"] > best_overall:
            best_overall = m["overall_acc"]; best_overall_metrics = m
            torch.save(save, Path(cfg["run_dir"]) / "best_overall.ckpt")
        torch.save(save, Path(cfg["run_dir"]) / "last.ckpt")
        if sched is not None:
            sched.step()

    # Missingness-robustness curve: reload best-overall model, eval under view dropout.
    dropout_curve = None
    if cfg.get("dropout_eval_ps"):
        ck = torch.load(Path(cfg["run_dir"]) / "best_overall.ckpt", map_location=device)
        encoder.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
        dropout_curve = {}
        for p in cfg["dropout_eval_ps"]:
            md = evaluate(encoder, head, test_loader, device, info,
                          view_dropout=float(p), dropout_seed=1234)
            dropout_curve[str(p)] = {k: md[k] for k in
                                     ("overall_acc", "head_acc", "tail_acc", "worst_group_acc")}
            console.log(f"[dropout p={p}] overall={md['overall_acc']:.4f} "
                        f"tail={md['tail_acc']:.4f} worst={md['worst_group_acc']:.4f}")

    console.rule("Training Complete")
    console.log(f"Best tail acc: {best_tail:.4f} | Best overall acc: {best_overall:.4f}")
    writer.close()
    return {"best_tail_acc": best_tail, "best_overall_acc": best_overall,
            "best_overall_metrics": best_overall_metrics,
            "best_tail_metrics": best_tail_metrics,
            "dropout_curve": dropout_curve,
            "num_groups": info["num_groups"],
            "head_gids": info["head_gids"], "tail_gids": info["tail_gids"],
            "group_names": [g["name"] for g in sorted(info["groups"], key=lambda x: x["gid"])],
            "train_group_counts": info.get("train_group_counts"),
            "test_group_counts": info.get("test_group_counts")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
