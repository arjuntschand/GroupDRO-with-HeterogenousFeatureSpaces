"""Training script for NHANES CVD dataset.

NHANES CVD: 3 groups by feature availability, tabular features, binary classification.
- G0 (survey_only): 10 features
- G1 (exam): 13 features
- G2 (vitals_labs): 20 features

Features:
- Stratified batch sampling to maintain group proportions π
- GroupDRO with KL divergence penalty
- Per-group, per-class, per-group-per-class metrics tracking
- Per-group encoders with true heterogeneous input dimensions
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import set_seed, ensure_dir, Meter, console
from .results_logger import ResultsLogger
from .datasets_nhanes import (
    build_nhanes_loaders,
    print_nhanes_summary,
)
from .encoders import ENCODER_REGISTRY
from .model.head import LinearHead, MLPHead
from .model.anchors import AnchorModule
from .model.losses import per_class_batch_moments, anchor_fit_loss, anchor_sep_loss, FocalLoss, LabelSmoothingLoss
from .model.groupdro import GroupDRO


def build_models(cfg, group_counts: List[int], device: torch.device,
                 group_feature_counts: Optional[Dict[int, int]] = None
                 ) -> Tuple[Dict[int, nn.Module], nn.Module, AnchorModule, Optional[GroupDRO]]:
    """Build all model components.

    Args:
        cfg: Configuration dictionary
        group_counts: Per-group sample counts (for π initialization)
        device: PyTorch device
        group_feature_counts: Per-group feature dimensions (for true heterogeneous input dims)

    Returns:
        encoders, head, anchors, groupdro
    """
    latent_dim = cfg["latent_dim"]
    num_groups = len(cfg["groups"])

    # Build encoders
    encoders: Dict[int, nn.Module] = {}
    if cfg.get("common_encoder", False):
        g0 = cfg["groups"][0]
        enc_name = g0["encoder"]
        enc_cls = ENCODER_REGISTRY[enc_name]
        # Shared encoder takes max_features as input
        input_dim = g0.get("input_dim", 20)
        hidden_dim = g0.get("hidden_dim", 64)
        dropout = g0.get("dropout", 0.1)
        shared_enc = enc_cls(latent_dim, input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        for gid in range(num_groups):
            encoders[gid] = shared_enc
    else:
        for gid, g in enumerate(cfg["groups"]):
            enc_name = g["encoder"]
            enc_cls = ENCODER_REGISTRY[enc_name]
            # Per-group encoder: use true feature count if available
            if group_feature_counts is not None and gid in group_feature_counts:
                input_dim = group_feature_counts[gid]
            else:
                input_dim = g.get("input_dim", 25)   # fallback: max features across modes
            hidden_dim = g.get("hidden_dim", 64)
            dropout = g.get("dropout", 0.1)
            encoders[gid] = enc_cls(latent_dim, input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)

    # Classification head
    head = (MLPHead(latent_dim, cfg["head_hidden"], cfg["num_classes"])
            if cfg.get("head_hidden", 0) > 0
            else LinearHead(latent_dim, cfg["num_classes"]))

    # Anchor module
    anchors = AnchorModule(cfg["num_classes"], latent_dim, eps=cfg["anchor_eps"])

    # GroupDRO
    groupdro = None
    if cfg.get("groupdro_enabled", False):
        groupdro = GroupDRO(
            num_groups=num_groups,
            eta=cfg.get("groupdro_eta", 0.1),
            device=device,
            update_mode=cfg.get("groupdro_update_mode", "exp"),
            robust_objective=cfg.get("groupdro_objective", "weighted"),
            gamma=cfg.get("groupdro_gamma", 1.0),
            group_counts=group_counts,
            kl_lambda=cfg.get("groupdro_kl_lambda", 0.1),
            uniform_init=cfg.get("groupdro_uniform_init", False),
            use_regret=cfg.get("use_regret", False),
            optimal_losses=cfg.get("optimal_losses"),
        )

    return encoders, head, anchors, groupdro


def evaluate(encoders: Dict[int, nn.Module], head: nn.Module, loader,
             device: torch.device, num_groups: int, num_classes: int,
             feature_indices: Optional[Dict[int, torch.Tensor]] = None) -> Dict[str, Any]:
    """Comprehensive evaluation with per-group, per-class, per-group-per-class metrics."""
    head.eval()
    for e in encoders.values():
        e.eval()

    correct = 0
    total = 0
    correct_g = [0] * num_groups
    total_g = [0] * num_groups
    correct_c = [0] * num_classes
    total_c = [0] * num_classes
    correct_gc = [[0] * num_classes for _ in range(num_groups)]
    total_gc = [[0] * num_classes for _ in range(num_groups)]

    ce = nn.CrossEntropyLoss(reduction="none")
    loss_sums_g = [0.0] * num_groups
    loss_counts_g = [0] * num_groups

    # For AUROC computation
    all_probs_g = [[] for _ in range(num_groups)]
    all_labels_g = [[] for _ in range(num_groups)]

    latent_dim = list(head.parameters())[0].shape[1]

    with torch.no_grad():
        for x, y, g in loader:
            x, y, g = x.to(device), y.to(device), g.to(device)

            z = torch.zeros((x.size(0), latent_dim), device=device)
            for gid, enc in encoders.items():
                mask = (g == gid)
                if mask.sum() == 0:
                    continue
                x_g = x[mask]
                if feature_indices is not None and gid in feature_indices:
                    x_g = x_g[:, feature_indices[gid]]
                z[mask] = enc(x_g)

            logits = head(z)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            losses = ce(logits, y)

            correct += (pred == y).sum().item()
            total += y.numel()

            for i in range(x.size(0)):
                yi = int(y[i].item())
                gi = int(g[i].item())
                is_correct = int(pred[i].item()) == yi

                total_c[yi] += 1
                total_g[gi] += 1
                total_gc[gi][yi] += 1
                loss_sums_g[gi] += float(losses[i].item())
                loss_counts_g[gi] += 1

                all_probs_g[gi].append(float(probs[i, 1].item()))
                all_labels_g[gi].append(yi)

                if is_correct:
                    correct_c[yi] += 1
                    correct_g[gi] += 1
                    correct_gc[gi][yi] += 1

    overall_acc = correct / max(1, total)
    per_group_acc = [
        correct_g[g] / max(1, total_g[g]) if total_g[g] > 0 else 0.0
        for g in range(num_groups)
    ]
    balanced_acc = sum(per_group_acc) / max(1, len(per_group_acc))
    worst_group_acc = min(per_group_acc) if per_group_acc else 0.0
    best_group_acc = max(per_group_acc) if per_group_acc else 0.0

    per_class_acc = [
        correct_c[c] / max(1, total_c[c]) if total_c[c] > 0 else 0.0
        for c in range(num_classes)
    ]
    per_group_per_class_acc = [
        [
            correct_gc[g][c] / total_gc[g][c] if total_gc[g][c] > 0 else None
            for c in range(num_classes)
        ]
        for g in range(num_groups)
    ]
    per_group_loss = [
        loss_sums_g[g] / max(1, loss_counts_g[g]) if loss_counts_g[g] > 0 else 0.0
        for g in range(num_groups)
    ]

    # Clinical metrics: sensitivity (recall for CVD+), specificity, F1, AUROC per group
    from sklearn.metrics import roc_auc_score
    per_group_sensitivity = []
    per_group_specificity = []
    per_group_f1 = []
    per_group_auroc = []
    for g_idx in range(num_groups):
        # Sensitivity = TP / (TP + FN) = correct_gc[g][1] / total_gc[g][1]
        tp = correct_gc[g_idx][1] if num_classes > 1 else 0
        fn = total_gc[g_idx][1] - tp if total_gc[g_idx][1] > 0 else 0
        tn = correct_gc[g_idx][0]
        fp = total_gc[g_idx][0] - tn if total_gc[g_idx][0] > 0 else 0

        sensitivity = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / max(1, tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * sensitivity / max(1e-9, precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        per_group_sensitivity.append(sensitivity)
        per_group_specificity.append(specificity)
        per_group_f1.append(f1)

        # AUROC
        try:
            if len(set(all_labels_g[g_idx])) > 1 and len(all_probs_g[g_idx]) > 0:
                auroc = roc_auc_score(all_labels_g[g_idx], all_probs_g[g_idx])
            else:
                auroc = 0.0
        except Exception:
            auroc = 0.0
        per_group_auroc.append(auroc)

    # Overall clinical metrics
    all_probs = sum(all_probs_g, [])
    all_labels = sum(all_labels_g, [])
    try:
        overall_auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    except Exception:
        overall_auroc = 0.0

    # Worst-group balanced accuracy: (sensitivity + specificity) / 2 per group, then min
    per_group_bal_acc = [(s + sp) / 2 for s, sp in zip(per_group_sensitivity, per_group_specificity)]
    worst_group_bal_acc = min(per_group_bal_acc) if per_group_bal_acc else 0.0

    return {
        "overall_acc": overall_acc,
        "balanced_acc": balanced_acc,
        "worst_group_acc": worst_group_acc,
        "best_group_acc": best_group_acc,
        "worst_group_bal_acc": worst_group_bal_acc,
        "per_group_bal_acc": per_group_bal_acc,
        "per_group_acc": per_group_acc,
        "per_group_loss": per_group_loss,
        "per_class_acc": per_class_acc,
        "per_group_per_class_acc": per_group_per_class_acc,
        "per_group_counts": total_g,
        # Clinical metrics
        "overall_auroc": overall_auroc,
        "per_group_sensitivity": per_group_sensitivity,
        "per_group_specificity": per_group_specificity,
        "per_group_f1": per_group_f1,
        "per_group_auroc": per_group_auroc,
        # group-robustness F1 summaries (mean / worst over groups)
        "mean_group_f1": (sum(per_group_f1) / len(per_group_f1)) if per_group_f1 else 0.0,
        "worst_group_f1": min(per_group_f1) if per_group_f1 else 0.0,
    }


def print_hyperparameters(cfg: Dict, dataset_info: Dict):
    """Print all hyperparameters and dataset info."""
    console.rule("Experiment Configuration")
    console.log(f"Run name: {cfg.get('run_name', 'unnamed')}")
    console.log(f"Run directory: {cfg.get('run_dir', 'runs/unnamed')}")
    console.log(f"Timestamp: {datetime.now().isoformat()}")
    console.log("")

    console.log("[bold]Dataset: NHANES CVD[/bold]")
    console.log(f"  Total training samples: {dataset_info['train_total']}")
    console.log(f"  Total test samples: {dataset_info['test_total']}")
    console.log(f"  Number of groups: {dataset_info['num_groups']}")
    console.log(f"  Number of classes: {dataset_info['num_classes']}")
    console.log(f"  Group proportions π: {[f'{p:.4f}' for p in dataset_info['group_proportions']]}")
    console.log("")

    gfc = dataset_info.get("group_feature_counts", {})
    console.log("[bold]Per-group sample counts (training):[/bold]")
    for g, count in enumerate(dataset_info["train_group_counts"]):
        pct = 100 * count / dataset_info["train_total"]
        name = dataset_info.get("group_names", ["G0","G1","G2","G3"])[g]
        n_feat = gfc.get(g, "?")
        console.log(f"  G{g} ({name}): {count:5d} samples ({pct:5.1f}%), {n_feat} features")
    console.log("")

    console.log("[bold]Per-class sample counts (training):[/bold]")
    for c, count in enumerate(dataset_info["train_class_counts"]):
        pct = 100 * count / dataset_info["train_total"]
        console.log(f"  Class {c}: {count:5d} samples ({pct:5.1f}%)")
    console.log("")

    console.log("[bold]Model:[/bold]")
    console.log(f"  Latent dimension: {cfg['latent_dim']}")
    console.log(f"  Head hidden: {cfg.get('head_hidden', 0)}")
    if cfg.get("common_encoder", False):
        console.log(f"  Encoder: COMMON (shared) — {cfg['groups'][0]['encoder']}")
    else:
        console.log(f"  Encoders: per-group")
        for g, gcfg in enumerate(cfg["groups"]):
            console.log(f"    G{g}: {gcfg['encoder']} (input_dim={gfc.get(g, '?')})")
    console.log("")

    console.log("[bold]Training:[/bold]")
    console.log(f"  Epochs: {cfg['epochs']}")
    console.log(f"  Batch size: {cfg['batch_size']}")
    console.log(f"  Learning rate: {cfg['lr']}")
    console.log(f"  Weight decay: {cfg['weight_decay']}")
    console.log(f"  Optimizer: {cfg.get('optimizer', 'adam')}")
    console.log(f"  Gradient clipping: {cfg.get('grad_clip', 0)}")
    console.log("")

    console.log("[bold]Anchor losses:[/bold]")
    console.log(f"  λ_fit: {cfg['lambda_fit']}")
    console.log(f"  λ_sep: {cfg['lambda_sep']}")
    console.log(f"  Sep samples per class: {cfg['sep_samples_per_class']}")
    console.log("")

    if cfg.get("groupdro_enabled", False):
        console.log("[bold]GroupDRO:[/bold]")
        console.log(f"  Enabled: True")
        console.log(f"  η (eta): {cfg.get('groupdro_eta', 0.1)}")
        console.log(f"  γ (gamma): {cfg.get('groupdro_gamma', 1.0)}")
        console.log(f"  Update mode: {cfg.get('groupdro_update_mode', 'exp')}")
        console.log(f"  Objective: {cfg.get('groupdro_objective', 'weighted')}")
        console.log(f"  KL λ: {cfg.get('groupdro_kl_lambda', 0.0)}")
    else:
        console.log("[bold]GroupDRO:[/bold] Disabled (baseline ERM)")
    console.rule("")


def print_epoch_results(epoch: int, train_loss: float, train_acc: float,
                        test_metrics: Dict, groupdro: Optional[GroupDRO],
                        num_groups: int, num_classes: int,
                        g_names: Optional[List[str]] = None):
    """Print epoch results."""
    if g_names is None:
        g_names = [f"G{i}" for i in range(num_groups)]
    console.log(f"\n[bold]Epoch {epoch} Results:[/bold]")
    console.log(f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
    console.log(f"  Test overall acc: {test_metrics['overall_acc']:.4f}")
    console.log(f"  Test balanced acc: {test_metrics['balanced_acc']:.4f}")
    console.log(f"  Test worst group acc: {test_metrics['worst_group_acc']:.4f}")
    console.log(f"  Test best group acc: {test_metrics['best_group_acc']:.4f}")

    console.log("\n  [bold]Per-group test accuracy:[/bold]")
    for g in range(num_groups):
        acc = test_metrics["per_group_acc"][g]
        loss = test_metrics["per_group_loss"][g]
        console.log(f"    G{g} ({g_names[g]}): acc={acc:.4f}, loss={loss:.4f}")

    console.log("\n  [bold]Per-class test accuracy:[/bold]")
    for c in range(num_classes):
        acc = test_metrics["per_class_acc"][c]
        label = "No CVD" if c == 0 else "CVD"
        console.log(f"    Class {c} ({label}): {acc:.4f}")

    if groupdro is not None:
        weights = groupdro.q.detach().cpu().tolist()
        console.log(f"\n  [bold]GroupDRO weights q:[/bold] {[f'{w:.4f}' for w in weights]}")


def train(cfg):
    """Main training loop for NHANES CVD."""
    if torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.log(f"Using device: {device}")

    ensure_dir(cfg["run_dir"])

    # Build data loaders FIRST (uses data_split_seed, not model seed)
    console.log("Loading NHANES CVD dataset...")
    data_seed = cfg.get("data_split_seed", cfg["seed"])
    subsample_seed = cfg["seed"] if cfg.get("data_split_seed") is not None else None

    train_loader, test_loader, dataset_info = build_nhanes_loaders(
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 0),
        seed=data_seed,
        stratified=cfg.get("stratified_batching", True),
        data_root=cfg.get("data_root"),
        train_frac=cfg.get("train_frac", 0.8),
        use_post_pandemic=cfg.get("use_post_pandemic", True),
        group_max_train_samples=cfg.get("group_max_train_samples"),
        data_split_seed=cfg.get("data_split_seed"),
        subsample_seed=subsample_seed,
        feature_mode=cfg.get("feature_mode", "nested"),
        class_balanced=cfg.get("class_balanced", False),
    )

    # Set model seed AFTER data loading so model init varies across runs
    set_seed(cfg["seed"])

    # Extract group info
    num_groups = dataset_info["num_groups"]
    num_classes = dataset_info["num_classes"]
    g_names = dataset_info.get("group_names", [f"G{i}" for i in range(num_groups)])

    print_nhanes_summary(dataset_info)
    print_hyperparameters(cfg, dataset_info)

    # Build feature_indices tensors for per-group feature selection
    feature_indices: Dict[int, torch.Tensor] = {}
    fi_raw = dataset_info["feature_indices"]
    use_true_hetero = not cfg.get("common_encoder", False)

    if use_true_hetero:
        # Per-group encoders: each group gets its own feature set
        for gid, idx_list in fi_raw.items():
            feature_indices[int(gid)] = torch.tensor(idx_list, dtype=torch.long, device=device)
        console.log("[bold]True heterogeneous feature spaces:[/bold]")
        gfc = dataset_info["group_feature_counts"]
        for gid in sorted(gfc.keys()):
            console.log(f"  G{gid} ({g_names[gid]}): {gfc[gid]} features")
    elif cfg.get("shared_common_features", True):
        # Shared encoder sees ONLY features common to ALL groups (intersection)
        # This is the principled comparison: without per-group models, you can only use common features
        common_idx = sorted(set.intersection(*[set(v) for v in fi_raw.values()]))
        for gid in fi_raw:
            feature_indices[int(gid)] = torch.tensor(common_idx, dtype=torch.long, device=device)
        n_common = len(common_idx)
        console.log(f"[bold]Shared encoder: common features only ({n_common} features)[/bold]")
        # Override input_dim for the shared encoder
        for g in cfg["groups"]:
            g["input_dim"] = n_common
    else:
        # Legacy: shared encoder sees ALL features (zero-padded)
        feature_indices = None

    # Build models
    group_counts = dataset_info["train_group_counts"]
    gfc = dataset_info["group_feature_counts"] if use_true_hetero else None
    encoders, head, anchors, groupdro = build_models(cfg, group_counts, device, gfc)

    # Move to device
    for gid in encoders:
        encoders[gid] = encoders[gid].to(device)
    head = head.to(device)
    anchors = anchors.to(device)

    # Optimizer (deduplicate shared encoder params)
    params = list(head.parameters()) + list(anchors.parameters())
    seen_enc_ids = set()
    for enc in encoders.values():
        if id(enc) not in seen_enc_ids:
            params += list(enc.parameters())
            seen_enc_ids.add(id(enc))

    opt_name = cfg.get("optimizer", "adam").lower()
    if opt_name == "sgd":
        opt = optim.SGD(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                        momentum=cfg.get("momentum", 0.9))
    elif opt_name == "adamw":
        opt = optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        opt = optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # Class weights for imbalanced data
    class_weight = None
    if cfg.get("class_weight", "auto") == "auto":
        train_class_counts = dataset_info["train_class_counts"]
        total = sum(train_class_counts)
        n_classes = len(train_class_counts)
        # Inverse frequency weighting: weight_c = total / (n_classes * count_c)
        class_weight = torch.tensor(
            [total / (n_classes * max(1, c)) for c in train_class_counts],
            dtype=torch.float32, device=device,
        )
        console.log(f"Class weights (auto): {[f'{w:.3f}' for w in class_weight.tolist()]}")
    elif cfg.get("class_weight") is not None and cfg.get("class_weight") != "none":
        class_weight = torch.tensor(cfg["class_weight"], dtype=torch.float32, device=device)
        console.log(f"Class weights (manual): {class_weight.tolist()}")

    # Loss function
    loss_type = cfg.get("loss_type", "cross_entropy").lower()
    if loss_type == "focal":
        loss_fn = FocalLoss(gamma=cfg.get("focal_gamma", 2.0), reduction="mean")
        console.log(f"Using Focal Loss (gamma={cfg.get('focal_gamma', 2.0)})")
    elif loss_type == "label_smoothing":
        loss_fn = LabelSmoothingLoss(smoothing=cfg.get("label_smoothing", 0.1), reduction="mean")
    else:
        loss_fn = None
        console.log("Using standard Cross-Entropy Loss")

    # Logging
    writer = SummaryWriter(log_dir=str(cfg["run_dir"]))
    results_logger = ResultsLogger(cfg["run_dir"])

    # Training constants
    lambda_fit = cfg["lambda_fit"]
    lambda_sep = cfg["lambda_sep"]
    J = cfg["sep_samples_per_class"]
    eps = cfg["anchor_eps"]
    sep_method = cfg.get("sep_method", "classifier")
    sep_margin = cfg.get("sep_margin", 1.0)

    run_meta = {
        "config": cfg,
        "dataset_info": {k: v for k, v in dataset_info.items() if k != "feature_indices"},
        "groupdro_config": groupdro.get_config() if groupdro else None,
        "start_time": datetime.now().isoformat(),
    }

    # Best tracking
    best_worst_group_acc = 0.0
    best_balanced_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_patience = cfg.get("early_stopping_patience", 0)
    global_step = 0

    # LR scheduler
    lr_scheduler = None
    if cfg.get("lr_scheduler") == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        lr_scheduler = CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg.get("lr_min", 1e-5))
    elif cfg.get("lr_scheduler") == "reduce_on_plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        lr_scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5, min_lr=1e-5)

    for epoch in range(1, cfg["epochs"] + 1):
        # Linear LR schedule if configured
        lr_start = cfg.get("lr_start")
        lr_end = cfg.get("lr_end")
        if lr_start is not None and lr_end is not None:
            t = (epoch - 1) / max(1, cfg["epochs"] - 1)
            current_lr = (1 - t) * lr_start + t * lr_end
            for pg in opt.param_groups:
                pg["lr"] = current_lr
        else:
            current_lr = cfg["lr"]

        head.train()
        for enc in encoders.values():
            enc.train()
        anchors.train()

        loss_meter = Meter()
        acc_meter = Meter()
        _q_epoch: List[List[float]] = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} (lr={current_lr:.4g})")
        for x, y, g in pbar:
            x, y, g = x.to(device), y.to(device), g.to(device)

            latent_dim = cfg["latent_dim"]
            z = torch.zeros((x.size(0), latent_dim), device=device)
            for gid, enc in encoders.items():
                mask = (g == gid)
                if mask.sum() == 0:
                    continue
                x_g = x[mask]
                if feature_indices is not None and gid in feature_indices:
                    x_g = x_g[:, feature_indices[gid]]
                z[mask] = enc(x_g)

            logits = head(z)

            # Classification loss
            if groupdro is not None:
                ce_loss = groupdro.forward(logits, y, g, num_classes=num_classes,
                                           class_weight=class_weight)
            else:
                if loss_fn is not None:
                    ce_loss = loss_fn(logits, y)
                else:
                    ce_loss = nn.functional.cross_entropy(logits, y, weight=class_weight)

            # Anchor losses
            # MECHANISM CONTROL (c): break the correspondence between a sample and its own
            # class anchor, and see whether the gain survives.
            #
            # Two ways to do that, and they are not equivalent:
            #   "permute" (default) shuffles the batch's real labels. Each pseudo-class keeps
            #       the true class proportions and the true count, so the per-class moments are
            #       estimated from the same number of samples as in the real run. The ONLY thing
            #       that changes is which sample belongs to which class.
            #   "randint" draws labels uniformly. On an imbalanced task that also changes the
            #       class proportions (NHANES is roughly 90/10, uniform draws give 50/50), so it
            #       confounds "class structure destroyed" with "moments estimated from different
            #       subset sizes". Kept for comparison with the earlier runs.
            y_anchor = y
            _mode = cfg.get("random_anchor_targets", False)
            if _mode:
                if _mode == "randint":
                    y_anchor = torch.randint(0, num_classes, y.shape, device=y.device)
                else:
                    y_anchor = y[torch.randperm(y.shape[0], device=y.device)]
            m_anc, S_anc, L_norm = anchors.forward()
            # Two versions of the fit loss appear in the write-up and they are NOT equivalent.
            #   pooled  (eq. 11-13, the centralized section): class moments pooled over every
            #           group, so the loss only pulls the *global* class-c cloud to anchor c.
            #           No group is constrained on its own, so cross-group alignment is at best
            #           a side effect of shrinking the space.
            #   pergroup (eq. 18, the GroupDRO section, which is the setting we actually run):
            #           moments computed per (group, class), so each group's class-c cloud is
            #           pulled to anchor c separately. This is what actually forces g1's class c
            #           to land on top of g2's class c, i.e. the alignment the paper claims.
            # Only the pergroup form makes the class-conditional structure load-bearing, so it
            # is also the only form under which the random-target control is a real test.
            if cfg.get("per_group_fit", False):
                l_fit = z.new_zeros(())
                n_gr = 0
                for gid in encoders.keys():
                    gm = (g == gid)
                    if gm.sum() < 2:
                        continue
                    mom_g = per_class_batch_moments(z[gm], y_anchor[gm], num_classes, eps)
                    if not mom_g:
                        continue
                    l_fit = l_fit + anchor_fit_loss(m_anc, S_anc, mom_g, eps)
                    n_gr += 1
                if n_gr:
                    l_fit = l_fit / n_gr
            else:
                moments = per_class_batch_moments(z, y_anchor, num_classes, eps)
                l_fit = anchor_fit_loss(m_anc, S_anc, moments, eps)
            l_sep = anchor_sep_loss(m_anc, S_anc, L_norm, head, num_classes, J, device,
                                    sep_method=sep_method, margin=sep_margin, eps=eps)

            # MECHANISM CONTROL (b): a generic L2 penalty on the latent, standing in
            # for 'any regularizer would have helped'.
            lat_l2 = cfg.get("latent_l2", 0.0)
            loss = ce_loss + lambda_fit * l_fit + lambda_sep * l_sep
            if lat_l2:
                loss = loss + lat_l2 * z.pow(2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.get("grad_clip", 0) > 0:
                nn.utils.clip_grad_norm_(params, cfg["grad_clip"])

            opt.step()

            if groupdro is not None:
                groupdro.update_weights(groupdro._last_group_losses, groupdro._last_group_counts)
                # Accumulate q across the epoch. In softmax mode update_weights OVERWRITES q
                # from the current batch alone, so q at the end of an epoch reflects only the
                # final batch. That batch is usually a partial remainder and often holds a
                # single group, which makes the mask-absent-with-minus-inf path produce an
                # exact one-hot. Logging that value made GroupDRO look permanently collapsed
                # when the weights actually hover near uniform throughout training. The mean
                # over the epoch's updates is what genuinely drove the gradients.
                _q_epoch.append(groupdro.q.detach().cpu().tolist())

            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean()
            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(acc.item(), x.size(0))

            if global_step % cfg.get("log_interval", 50) == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/acc", acc.item(), global_step)
                writer.add_scalar("train/ce", ce_loss.item(), global_step)
                writer.add_scalar("train/l_fit", l_fit.item(), global_step)
                writer.add_scalar("train/l_sep", l_sep.item(), global_step)

                if groupdro is not None:
                    for gid in range(num_groups):
                        writer.add_scalar(f"train/group_{gid}_weight", groupdro.q[gid].item(), global_step)

            global_step += 1
            pbar.set_postfix({"loss": f"{loss_meter.avg:.3f}", "acc": f"{acc_meter.avg:.3f}"})

        # Evaluation
        test_metrics = evaluate(encoders, head, test_loader, device, num_groups, num_classes,
                                feature_indices=feature_indices)

        print_epoch_results(epoch, loss_meter.avg, acc_meter.avg, test_metrics, groupdro, num_groups, num_classes, g_names)

        # TensorBoard
        writer.add_scalar("test/overall_acc", test_metrics["overall_acc"], epoch)
        writer.add_scalar("test/balanced_acc", test_metrics["balanced_acc"], epoch)
        writer.add_scalar("test/worst_group_acc", test_metrics["worst_group_acc"], epoch)
        writer.add_scalar("test/best_group_acc", test_metrics["best_group_acc"], epoch)
        for gid in range(num_groups):
            writer.add_scalar(f"test/group_{gid}_acc", test_metrics["per_group_acc"][gid], epoch)
            writer.add_scalar(f"test/group_{gid}_loss", test_metrics["per_group_loss"][gid], epoch)
        for cid in range(num_classes):
            writer.add_scalar(f"test/class_{cid}_acc", test_metrics["per_class_acc"][cid], epoch)

        # Results logger
        epoch_data = {
            **run_meta,
            "epoch": epoch,
            "train_loss": float(loss_meter.avg),
            "train_acc": float(acc_meter.avg),
            "learning_rate": current_lr,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "groupdro_weights": groupdro.q.detach().cpu().tolist() if groupdro else None,
            # mean over the epoch's updates; use THIS for training-dynamics plots
            "groupdro_weights_mean": (
                [sum(c)/len(c) for c in zip(*_q_epoch)] if _q_epoch else None),
            "groupdro_pi": groupdro.pi.detach().cpu().tolist() if groupdro else None,
        }
        results_logger.log_epoch(epoch_data)
        results_logger.save()

        # Checkpointing
        save_dict = {
            "cfg": cfg,
            "epoch": epoch,
            "dataset_info": {k: v for k, v in dataset_info.items() if k != "feature_indices"},
            "encoders": {gid: enc.state_dict() for gid, enc in encoders.items()},
            "head": head.state_dict(),
            "anchors": anchors.state_dict(),
            "test_metrics": test_metrics,
        }
        if groupdro is not None:
            save_dict["groupdro"] = {
                "weights": groupdro.q.detach(),
                "pi": groupdro.pi.detach(),
                "config": groupdro.get_config(),
            }

        wg_metric = test_metrics["worst_group_acc"]
        if wg_metric > best_worst_group_acc:
            best_worst_group_acc = wg_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(save_dict, Path(cfg["run_dir"]) / "best_worst_group.ckpt")
            console.log(f"[green]New best worst-group acc: {best_worst_group_acc:.4f} (epoch {epoch})[/green]")
        else:
            epochs_without_improvement += 1

        if test_metrics["balanced_acc"] > best_balanced_acc:
            best_balanced_acc = test_metrics["balanced_acc"]
            torch.save(save_dict, Path(cfg["run_dir"]) / "best_balanced.ckpt")
            console.log(f"[green]New best balanced acc: {best_balanced_acc:.4f}[/green]")

        if epoch % cfg.get("save_every", 5) == 0 or epoch == cfg["epochs"]:
            ckpt_path = Path(cfg["run_dir"]) / "last.ckpt"
            torch.save(save_dict, ckpt_path)

        # LR scheduler
        if lr_scheduler is not None:
            if cfg.get("lr_scheduler") == "cosine":
                lr_scheduler.step()
            elif cfg.get("lr_scheduler") == "reduce_on_plateau":
                lr_scheduler.step(wg_metric)

        # Early stopping
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            console.log(f"[yellow]Early stopping at epoch {epoch} (best worst-group: {best_worst_group_acc:.4f} at epoch {best_epoch})[/yellow]")
            break

    console.rule("Training Complete")
    console.log(f"Best worst-group accuracy: {best_worst_group_acc:.4f}")
    console.log(f"Best balanced accuracy: {best_balanced_acc:.4f}")
    console.log(f"Results saved to: {cfg['run_dir']}")
    writer.close()

    out = {
        "best_worst_group_acc": best_worst_group_acc,
        "best_balanced_acc": best_balanced_acc,
        "final_test_metrics": test_metrics,
    }

    # Optional: hand back the final test latents and anchor means so the mechanism can be
    # inspected directly (are the anchors separated? do classes align to them? are the
    # per-group latent clouds actually pulled together?) rather than inferred from accuracy.
    if cfg.get("return_latents", False):
        head.eval()
        for e in encoders.values():
            e.eval()
        zs, ys, gs = [], [], []
        with torch.no_grad():
            for x, y, g in test_loader:
                x, y, g = x.to(device), y.to(device), g.to(device)
                z = torch.zeros((x.size(0), cfg["latent_dim"]), device=device)
                for gid, enc in encoders.items():
                    m = (g == gid)
                    if m.sum() == 0:
                        continue
                    xg = x[m]
                    if feature_indices is not None and gid in feature_indices:
                        xg = xg[:, feature_indices[gid]]
                    z[m] = enc(xg)
                zs.append(z.cpu()); ys.append(y.cpu()); gs.append(g.cpu())
        m_anc, _, _ = anchors.forward()
        out["final_latents"] = {"z": torch.cat(zs), "y": torch.cat(ys),
                                "g": torch.cat(gs), "anchor_m": m_anc.detach().cpu()}
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Train on NHANES CVD dataset")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
