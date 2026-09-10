"""Training script for Fed-Heart Disease dataset.

Fed-Heart Disease: 4 hospitals (groups), 13 tabular features, binary classification.

Features:
- Stratified batch sampling to maintain group proportions π
- GroupDRO with KL divergence penalty
- Per-group, per-class, per-group-per-class metrics tracking
- Comprehensive hyperparameter and statistics logging
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
from .datasets_fedheart import (
    build_fedheart_loaders,
    print_fedheart_summary,
)
from .encoders import ENCODER_REGISTRY
from .model.head import LinearHead, MLPHead
from .model.anchors import AnchorModule
from .model.losses import per_class_batch_moments, anchor_fit_loss, anchor_sep_loss, FocalLoss, LabelSmoothingLoss
from .model.groupdro import GroupDRO


def build_models(cfg, group_counts: List[int], device: torch.device) -> Tuple[Dict[int, nn.Module], nn.Module, AnchorModule, Optional[GroupDRO]]:
    """Build all model components and optionally initialize GroupDRO.
    
    Args:
        cfg: Configuration dictionary
        group_counts: List of sample counts per group (for π initialization)
        device: PyTorch device
        
    Returns:
        encoders: Dict mapping group_id -> encoder module
        head: Classification head
        anchors: Anchor module
        groupdro: GroupDRO module (or None if disabled)
    """
    latent_dim = cfg["latent_dim"]
    num_groups = len(cfg["groups"])
    
    # Build encoders: either one shared encoder or one per group
    encoders: Dict[int, nn.Module] = {}
    if cfg.get("common_encoder", False):
        g0 = cfg["groups"][0]
        enc_name = g0["encoder"]
        enc_cls = ENCODER_REGISTRY[enc_name]
        if "mlp_tabular" in enc_name:
            input_dim = g0.get("input_dim", 13)
            hidden_dim = g0.get("hidden_dim", 64)
            dropout = g0.get("dropout", 0.1)
            shared_enc = enc_cls(latent_dim, input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        else:
            shared_enc = enc_cls(latent_dim)
        for gid in range(num_groups):
            encoders[gid] = shared_enc
    else:
        for gid, g in enumerate(cfg["groups"]):
            enc_name = g["encoder"]
            enc_cls = ENCODER_REGISTRY[enc_name]
            if "mlp_tabular" in enc_name:
                input_dim = g.get("input_dim", 13)
                hidden_dim = g.get("hidden_dim", 64)
                dropout = g.get("dropout", 0.1)
                encoders[gid] = enc_cls(latent_dim, input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
            else:
                encoders[gid] = enc_cls(latent_dim)
    
    # Build classification head
    head = (MLPHead(latent_dim, cfg["head_hidden"], cfg["num_classes"])
            if cfg.get("head_hidden", 0) > 0 
            else LinearHead(latent_dim, cfg["num_classes"]))
    
    # Build anchor module
    anchors = AnchorModule(cfg["num_classes"], latent_dim, eps=cfg["anchor_eps"])
    
    # Initialize GroupDRO if enabled
    groupdro = None
    if cfg.get("groupdro_enabled", False):
        groupdro = GroupDRO(
            num_groups=num_groups,
            eta=cfg.get("groupdro_eta", 0.1),
            device=device,
            update_mode=cfg.get("groupdro_update_mode", "exp"),
            robust_objective=cfg.get("groupdro_objective", "weighted"),
            gamma=cfg.get("groupdro_gamma", 1.0),
            group_counts=group_counts,  # Initialize π proportional to group sizes
            kl_lambda=cfg.get("groupdro_kl_lambda", 0.1),  # KL divergence penalty
            uniform_init=cfg.get("groupdro_uniform_init", False),  # Start with equal weights
            use_regret=cfg.get("use_regret", False),          # Regret-DRO: weight by L_g - R*_g
            optimal_losses=cfg.get("optimal_losses"),          # per-group reference losses R*_g
        )
    
    return encoders, head, anchors, groupdro


def evaluate(encoders: Dict[int, nn.Module], head: nn.Module, loader, 
             device: torch.device, num_groups: int, num_classes: int,
             feature_indices: Optional[Dict[int, torch.Tensor]] = None) -> Dict[str, Any]:
    """Comprehensive evaluation with per-group, per-class, and per-group-per-class metrics.
    
    Returns:
        Dictionary with all metrics:
            - overall_acc: float
            - balanced_acc: float (unweighted mean of group accuracies)
            - worst_group_acc: float
            - best_group_acc: float
            - per_group_acc: List[float]
            - per_group_loss: List[float]
            - per_class_acc: List[float]
            - per_group_per_class_acc: List[List[float or None]]
            - per_group_counts: List[int]
    """
    head.eval()
    for e in encoders.values():
        e.eval()
    
    # Counters
    correct = 0
    total = 0
    correct_g = [0] * num_groups
    total_g = [0] * num_groups
    correct_c = [0] * num_classes
    total_c = [0] * num_classes
    correct_gc = [[0] * num_classes for _ in range(num_groups)]
    total_gc = [[0] * num_classes for _ in range(num_groups)]
    # confusion counts for macro-F1 (overall and per group)
    tp = [0] * num_classes; fp = [0] * num_classes; fn = [0] * num_classes
    tp_g = [[0] * num_classes for _ in range(num_groups)]
    fp_g = [[0] * num_classes for _ in range(num_groups)]
    fn_g = [[0] * num_classes for _ in range(num_groups)]
    
    # Loss tracking
    ce = nn.CrossEntropyLoss(reduction="none")
    loss_sums_g = [0.0] * num_groups
    loss_counts_g = [0] * num_groups
    
    latent_dim = list(head.parameters())[0].shape[1]
    
    with torch.no_grad():
        for x, y, g in loader:
            x, y, g = x.to(device), y.to(device), g.to(device)
            
            # Encode each group (with optional per-group feature selection)
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
            pred = logits.argmax(dim=1)
            losses = ce(logits, y)
            
            correct += (pred == y).sum().item()
            total += y.numel()
            
            # Per-sample statistics
            for i in range(x.size(0)):
                yi = int(y[i].item())
                gi = int(g[i].item())
                pi = int(pred[i].item())
                is_correct = pi == yi

                total_c[yi] += 1
                total_g[gi] += 1
                total_gc[gi][yi] += 1
                loss_sums_g[gi] += float(losses[i].item())
                loss_counts_g[gi] += 1

                # confusion counts for macro-F1
                if pi == yi:
                    tp[yi] += 1; tp_g[gi][yi] += 1
                else:
                    fp[pi] += 1; fp_g[gi][pi] += 1
                    fn[yi] += 1; fn_g[gi][yi] += 1

                if is_correct:
                    correct_c[yi] += 1
                    correct_g[gi] += 1
                    correct_gc[gi][yi] += 1
    
    # Compute metrics
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
    
    def _macro_f1(TP, FP, FN):
        f1s = []
        for c in range(num_classes):
            prec = TP[c] / (TP[c] + FP[c]) if (TP[c] + FP[c]) else 0.0
            rec = TP[c] / (TP[c] + FN[c]) if (TP[c] + FN[c]) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        return sum(f1s) / max(1, num_classes)

    overall_macro_f1 = _macro_f1(tp, fp, fn)
    per_group_f1 = [_macro_f1(tp_g[g], fp_g[g], fn_g[g]) for g in range(num_groups)]

    return {
        "overall_acc": overall_acc,
        "balanced_acc": balanced_acc,
        "worst_group_acc": worst_group_acc,
        "best_group_acc": best_group_acc,
        "per_group_acc": per_group_acc,
        "per_group_loss": per_group_loss,
        "per_class_acc": per_class_acc,
        "per_group_per_class_acc": per_group_per_class_acc,
        "per_group_counts": total_g,
        "overall_macro_f1": overall_macro_f1,
        "per_group_f1": per_group_f1,
        "worst_group_f1": min(per_group_f1) if per_group_f1 else 0.0,
        "worst_group_loss": max(per_group_loss) if per_group_loss else 0.0,
    }


def print_hyperparameters(cfg: Dict, dataset_info: Dict):
    """Print all hyperparameters and dataset info at the start of training."""
    console.rule("Experiment Configuration")
    console.log(f"Run name: {cfg.get('run_name', 'unnamed')}")
    console.log(f"Run directory: {cfg.get('run_dir', 'runs/unnamed')}")
    console.log(f"Timestamp: {datetime.now().isoformat()}")
    console.log("")
    
    console.log("[bold]Dataset:[/bold]")
    console.log(f"  Total training samples: {dataset_info['train_total']}")
    console.log(f"  Total test samples: {dataset_info['test_total']}")
    console.log(f"  Number of groups: {dataset_info['num_groups']}")
    console.log(f"  Number of classes: {dataset_info['num_classes']}")
    console.log(f"  Group proportions π: {[f'{p:.4f}' for p in dataset_info['group_proportions']]}")
    console.log("")
    
    console.log("[bold]Per-group sample counts (training):[/bold]")
    for g, count in enumerate(dataset_info['train_group_counts']):
        pct = 100 * count / dataset_info['train_total']
        console.log(f"  Group {g}: {count:4d} samples ({pct:5.1f}%)")
    console.log("")
    
    console.log("[bold]Per-class sample counts (training):[/bold]")
    for c, count in enumerate(dataset_info['train_class_counts']):
        pct = 100 * count / dataset_info['train_total']
        console.log(f"  Class {c}: {count:4d} samples ({pct:5.1f}%)")
    console.log("")
    
    console.log("[bold]Model:[/bold]")
    console.log(f"  Latent dimension: {cfg['latent_dim']}")
    console.log(f"  Head hidden: {cfg.get('head_hidden', 0)}")
    if cfg.get("common_encoder", False):
        console.log(f"  Encoder: COMMON (shared) — {cfg['groups'][0]['encoder']}")
    else:
        console.log(f"  Encoders: {[g['encoder'] for g in cfg['groups']]} (per-group)")
    console.log("")
    
    console.log("[bold]Training:[/bold]")
    console.log(f"  Epochs: {cfg['epochs']}")
    console.log(f"  Batch size: {cfg['batch_size']}")
    console.log(f"  Learning rate: {cfg['lr']}")
    console.log(f"  Weight decay: {cfg['weight_decay']}")
    console.log(f"  Optimizer: {cfg.get('optimizer', 'adam')}")
    console.log(f"  Gradient clipping: {cfg.get('grad_clip', 0)}")
    console.log(f"  Stratified batching: {cfg.get('stratified_batching', True)}")
    console.log("")
    
    console.log("[bold]Anchor losses:[/bold]")
    console.log(f"  λ_fit: {cfg['lambda_fit']}")
    console.log(f"  λ_sep: {cfg['lambda_sep']}")
    console.log(f"  Sep samples per class: {cfg['sep_samples_per_class']}")
    console.log(f"  Sep method: {cfg.get('sep_method', 'classifier')}")
    console.log("")
    
    if cfg.get("groupdro_enabled", False):
        console.log("[bold]GroupDRO:[/bold]")
        console.log(f"  Enabled: True")
        console.log(f"  η (eta): {cfg.get('groupdro_eta', 0.1)}")
        console.log(f"  γ (gamma): {cfg.get('groupdro_gamma', 1.0)}")
        console.log(f"  Update mode: {cfg.get('groupdro_update_mode', 'exp')}")
        console.log(f"  Objective: {cfg.get('groupdro_objective', 'weighted')}")
        console.log(f"  KL λ (kl_lambda): {cfg.get('groupdro_kl_lambda', 0.0)}")
        console.log(f"  Initial weights π: {[f'{p:.4f}' for p in dataset_info['group_proportions']]}")
    else:
        console.log("[bold]GroupDRO:[/bold] Disabled (baseline ERM)")
    
    console.rule("")


def print_epoch_results(epoch: int, train_loss: float, train_acc: float, 
                        test_metrics: Dict, groupdro: Optional[GroupDRO], 
                        num_groups: int, num_classes: int):
    """Print comprehensive epoch results."""
    console.log(f"\n[bold]Epoch {epoch} Results:[/bold]")
    console.log(f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
    console.log(f"  Test overall acc: {test_metrics['overall_acc']:.4f}")
    console.log(f"  Test balanced acc: {test_metrics['balanced_acc']:.4f}")
    console.log(f"  Test worst group acc: {test_metrics['worst_group_acc']:.4f}")
    console.log(f"  Test best group acc: {test_metrics['best_group_acc']:.4f}")
    
    console.log("\n  [bold]Per-group test accuracy:[/bold]")
    for g in range(num_groups):
        acc = test_metrics['per_group_acc'][g]
        loss = test_metrics['per_group_loss'][g]
        console.log(f"    Group {g}: acc={acc:.4f}, loss={loss:.4f}")
    
    console.log("\n  [bold]Per-class test accuracy:[/bold]")
    for c in range(num_classes):
        acc = test_metrics['per_class_acc'][c]
        label = "No Disease" if c == 0 else "Disease"
        console.log(f"    Class {c} ({label}): {acc:.4f}")
    
    console.log("\n  [bold]Per-group-per-class test accuracy:[/bold]")
    for g in range(num_groups):
        accs = test_metrics['per_group_per_class_acc'][g]
        accs_str = [f"{a:.4f}" if a is not None else "N/A" for a in accs]
        console.log(f"    Group {g}: {accs_str}")
    
    if groupdro is not None:
        weights = groupdro.q.detach().cpu().tolist()
        console.log(f"\n  [bold]GroupDRO weights q:[/bold] {[f'{w:.4f}' for w in weights]}")
        console.log(f"  [bold]Reference π:[/bold] {[f'{p:.4f}' for p in groupdro.pi.detach().cpu().tolist()]}")
        if groupdro.kl_lambda > 0:
            console.log(f"  [bold]Last KL penalty:[/bold] {groupdro._last_kl_penalty:.6f}")


def train(cfg):
    """Main training loop for Fed-Heart Disease."""
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("cpu")  # MPS has issues with some ops
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.log(f"Using device: {device}")
    
    set_seed(cfg["seed"])
    ensure_dir(cfg["run_dir"])
    
    # Build data loaders
    console.log("Loading Fed-Heart Disease dataset...")
    # data_split_seed: fixed seed for train/test split (like FLamby).
    # When set, all experiment seeds share the same data split; only model init + subsampling varies.
    data_seed = cfg.get("data_split_seed", cfg["seed"])
    # subsample_seed: controls which samples are selected when capping groups.
    # Uses experiment seed so each run gets different capped samples while keeping the split fixed.
    subsample_seed = cfg["seed"] if cfg.get("data_split_seed") is not None else None
    train_loader, test_loader, dataset_info = build_fedheart_loaders(
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 0),
        seed=data_seed,
        stratified=cfg.get("stratified_batching", True),
        data_root=cfg.get("data_root"),
        group_max_train_samples=cfg.get("group_max_train_samples"),
        train_frac=cfg.get("train_frac", 0.66),  # 0.8 for standard 80/20 train/test split
        label_noise_rate=cfg.get("label_noise_rate"),
        feature_mask=cfg.get("feature_mask"),
        input_noise_std=cfg.get("input_noise_std"),
        subsample_seed=subsample_seed,
        impute_missing=cfg.get("impute_missing", False),
    )

    # Print dataset summary and hyperparameters
    print_fedheart_summary(dataset_info)
    print_hyperparameters(cfg, dataset_info)
    
    # Build per-group feature index tensors.
    # IMPORTANT:
    # - Default behavior (matching your earlier runs): we only *mask features* (zero them)
    #   in the dataset, while encoders still take `input_dim=13` for all groups.
    # - Optional strict behavior: if `true_hetero_input_dim=true`, then we also slice inputs
    #   so each encoder only sees the kept feature columns (and we override per-group
    #   `input_dim` accordingly). This is a different experiment regime.
    feature_indices: Optional[Dict[int, torch.Tensor]] = None
    raw_mask = cfg.get("feature_mask")
    use_true_hetero = (
        bool(cfg.get("true_hetero_input_dim", False))
        and raw_mask is not None
        and not cfg.get("common_encoder", False)
    )
    if use_true_hetero:
        feature_indices = {}
        for gid, fm in enumerate(raw_mask):
            if fm is not None:
                feature_indices[gid] = torch.tensor(fm, dtype=torch.long, device=device)
        if not feature_indices:
            feature_indices = None
        else:
            # Override input_dim per group to match actual feature count
            for gid in range(len(cfg["groups"])):
                if raw_mask[gid] is not None:
                    cfg["groups"][gid]["input_dim"] = len(raw_mask[gid])
            console.log("[bold]True heterogeneous feature spaces:[/bold]")
            for gid, g in enumerate(cfg["groups"]):
                n_feat = g.get("input_dim", 13)
                console.log(f"  Group {gid} ({g['name']}): {n_feat} features")

    # Build models with group counts for proper π initialization
    group_counts = dataset_info["train_group_counts"]
    encoders, head, anchors, groupdro = build_models(cfg, group_counts, device)
    
    # Move to device
    for gid in encoders:
        encoders[gid] = encoders[gid].to(device)
    head = head.to(device)
    anchors = anchors.to(device)
    
    # Setup optimizer (deduplicate if common_encoder shares the same module)
    params = list(head.parameters()) + list(anchors.parameters())
    seen_enc_ids = set()
    for enc in encoders.values():
        if id(enc) not in seen_enc_ids:
            params += list(enc.parameters())
            seen_enc_ids.add(id(enc))
    
    opt_name = cfg.get("optimizer", "adam").lower()
    if opt_name == "sgd":
        opt = optim.SGD(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                        momentum=cfg.get("momentum", 0.9), nesterov=cfg.get("nesterov", False))
    elif cfg.get("optimizer", "adam").lower() == "adamw":
        opt = optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        opt = optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    
    # Setup loss function
    loss_type = cfg.get("loss_type", "cross_entropy").lower()
    if loss_type == "focal":
        focal_gamma = cfg.get("focal_gamma", 2.0)
        loss_fn = FocalLoss(gamma=focal_gamma, reduction='mean')
        console.log(f"Using Focal Loss (gamma={focal_gamma})")
    elif loss_type == "label_smoothing":
        smoothing = cfg.get("label_smoothing", 0.1)
        loss_fn = LabelSmoothingLoss(smoothing=smoothing, reduction='mean')
        console.log(f"Using Label Smoothing Loss (smoothing={smoothing})")
    else:
        loss_fn = None  # Will use F.cross_entropy directly
        console.log("Using standard Cross-Entropy Loss")
    
    # Setup logging
    writer = SummaryWriter(log_dir=str(cfg["run_dir"]))
    results_logger = ResultsLogger(cfg["run_dir"])

    # Training constants
    num_groups = dataset_info["num_groups"]
    num_classes = dataset_info["num_classes"]
    lambda_fit = cfg["lambda_fit"]
    lambda_sep = cfg["lambda_sep"]
    J = cfg["sep_samples_per_class"]
    eps = cfg["anchor_eps"]
    sep_method = cfg.get("sep_method", "classifier")
    sep_margin = cfg.get("sep_margin", 1.0)
    
    # Metadata for logging
    run_meta = {
        "config": cfg,
        "dataset_info": dataset_info,
        "groupdro_config": groupdro.get_config() if groupdro else None,
        "start_time": datetime.now().isoformat(),
    }
    
    # Best tracking and early stopping
    best_worst_group_acc = 0.0
    best_balanced_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_patience = cfg.get("early_stopping_patience", 0)  # 0 = disabled
    global_step = 0
    
    # Optional LR scheduler (cosine or reduce_on_plateau)
    lr_scheduler = None
    if cfg.get("lr_scheduler") == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        lr_scheduler = CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg.get("lr_min", 1e-5))
    elif cfg.get("lr_scheduler") == "reduce_on_plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        lr_scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5, min_lr=1e-5)
    
    for epoch in range(1, cfg["epochs"] + 1):
        # Learning rate schedule (linear decay from lr_start to lr_end takes precedence if set)
        lr_start = cfg.get("lr_start")
        lr_end = cfg.get("lr_end")
        if lr_start is not None and lr_end is not None:
            t = (epoch - 1) / max(1, cfg["epochs"] - 1)
            current_lr = (1 - t) * lr_start + t * lr_end
            for pg in opt.param_groups:
                pg["lr"] = current_lr
        else:
            current_lr = cfg["lr"]
        
        # Training mode
        head.train()
        for enc in encoders.values():
            enc.train()
        anchors.train()
        
        loss_meter = Meter()
        acc_meter = Meter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} (lr={current_lr:.4g})")
        for x, y, g in pbar:
            x, y, g = x.to(device), y.to(device), g.to(device)
            
            # Encode (with optional per-group feature selection)
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
                ce = groupdro.forward(logits, y, g, num_classes=num_classes)
            else:
                # Use configured loss function (focal, label smoothing, or cross-entropy)
                if loss_fn is not None:
                    ce = loss_fn(logits, y)
                else:
                    ce = nn.functional.cross_entropy(logits, y)
            
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
            # See train_nhanes.py for the full note. pooled = eq. 11-13 (class moments pooled
            # over all groups); pergroup = eq. 18, the form the GroupDRO section specifies,
            # where each group's class-c cloud is pulled to anchor c on its own.
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
            
            # Total loss
            # MECHANISM CONTROL (b): a generic L2 penalty on the latent, standing in
            # for 'any regularizer would have helped'.
            lat_l2 = cfg.get("latent_l2", 0.0)
            loss = ce + lambda_fit * l_fit + lambda_sep * l_sep
            if lat_l2:
                loss = loss + lat_l2 * z.pow(2).mean()
            
            # Backward
            opt.zero_grad(set_to_none=True)
            loss.backward()
            
            if cfg.get("grad_clip", 0) > 0:
                nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            
            opt.step()
            
            # Update GroupDRO weights (after backward)
            if groupdro is not None:
                groupdro.update_weights(groupdro._last_group_losses, groupdro._last_group_counts)
            
            # Metrics
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean()
            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(acc.item(), x.size(0))
            
            # TensorBoard logging
            if global_step % cfg.get("log_interval", 50) == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/acc", acc.item(), global_step)
                writer.add_scalar("train/ce", ce.item(), global_step)
                writer.add_scalar("train/l_fit", l_fit.item(), global_step)
                writer.add_scalar("train/l_sep", l_sep.item(), global_step)
                
                if groupdro is not None:
                    for gid in range(num_groups):
                        writer.add_scalar(f"train/group_{gid}_weight", groupdro.q[gid].item(), global_step)
                    if groupdro.kl_lambda > 0:
                        writer.add_scalar("train/kl_penalty", groupdro._last_kl_penalty, global_step)
            
            global_step += 1
            pbar.set_postfix({"loss": f"{loss_meter.avg:.3f}", "acc": f"{acc_meter.avg:.3f}"})
        
        # Evaluation
        test_metrics = evaluate(encoders, head, test_loader, device, num_groups, num_classes, feature_indices=feature_indices)
        
        # Print epoch results
        print_epoch_results(epoch, loss_meter.avg, acc_meter.avg, test_metrics, groupdro, num_groups, num_classes)
        
        # TensorBoard test metrics
        writer.add_scalar("test/overall_acc", test_metrics["overall_acc"], epoch)
        writer.add_scalar("test/balanced_acc", test_metrics["balanced_acc"], epoch)
        writer.add_scalar("test/worst_group_acc", test_metrics["worst_group_acc"], epoch)
        writer.add_scalar("test/best_group_acc", test_metrics["best_group_acc"], epoch)
        for gid in range(num_groups):
            writer.add_scalar(f"test/group_{gid}_acc", test_metrics["per_group_acc"][gid], epoch)
            writer.add_scalar(f"test/group_{gid}_loss", test_metrics["per_group_loss"][gid], epoch)
        for cid in range(num_classes):
            writer.add_scalar(f"test/class_{cid}_acc", test_metrics["per_class_acc"][cid], epoch)
        
        # Log to results file
        epoch_data = {
            **run_meta,
            "epoch": epoch,
            "train_loss": float(loss_meter.avg),
            "train_acc": float(acc_meter.avg),
            "learning_rate": current_lr,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "groupdro_weights": groupdro.q.detach().cpu().tolist() if groupdro else None,
            "groupdro_pi": groupdro.pi.detach().cpu().tolist() if groupdro else None,
            "groupdro_kl_penalty": groupdro._last_kl_penalty if groupdro else None,
        }
        results_logger.log_epoch(epoch_data)
        results_logger.save()
        
        # Build save_dict once for checkpoints
        save_dict = {
            "cfg": cfg,
            "epoch": epoch,
            "dataset_info": dataset_info,
            "encoders": {gid: enc.state_dict() for gid, enc in encoders.items()},
            "head": head.state_dict(),
            "anchors": anchors.state_dict(),
            "test_metrics": test_metrics,
        }
        if groupdro is not None:
            save_dict["groupdro"] = {
                "weights": groupdro.q.detach(),
                "pi": groupdro.pi.detach(),
                "stats": groupdro.group_stats,
                "config": groupdro.get_config(),
            }
        
        # Save best by worst-group every epoch (so we don't miss best when early stopping)
        if test_metrics["worst_group_acc"] > best_worst_group_acc:
            best_worst_group_acc = test_metrics["worst_group_acc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_ckpt_path = Path(cfg["run_dir"]) / "best_worst_group.ckpt"
            torch.save(save_dict, best_ckpt_path)
            console.log(f"[green]New best worst-group acc: {best_worst_group_acc:.4f} (epoch {epoch})[/green]")
        else:
            epochs_without_improvement += 1
        
        # Save best by balanced accuracy
        if test_metrics["balanced_acc"] > best_balanced_acc:
            best_balanced_acc = test_metrics["balanced_acc"]
            best_bal_path = Path(cfg["run_dir"]) / "best_balanced.ckpt"
            torch.save(save_dict, best_bal_path)
            console.log(f"[green]New best balanced acc: {best_balanced_acc:.4f}[/green]")
        
        # Periodic full checkpoint
        if epoch % cfg.get("save_every", 5) == 0 or epoch == cfg["epochs"]:
            ckpt_path = Path(cfg["run_dir"]) / "last.ckpt"
            torch.save(save_dict, ckpt_path)
            console.log(f"Saved checkpoint to {ckpt_path}")
        
        # LR scheduler step
        if lr_scheduler is not None:
            if cfg.get("lr_scheduler") == "cosine":
                lr_scheduler.step()
            elif cfg.get("lr_scheduler") == "reduce_on_plateau":
                lr_scheduler.step(test_metrics["worst_group_acc"])
        
        # Early stopping
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            console.log(f"[yellow]Early stopping: no improvement in worst-group acc for {early_stopping_patience} epochs (best: {best_worst_group_acc:.4f} at epoch {best_epoch}).[/yellow]")
            break
    
    # Final summary
    console.rule("Training Complete")
    console.log(f"Best worst-group accuracy: {best_worst_group_acc:.4f}")
    console.log(f"Best balanced accuracy: {best_balanced_acc:.4f}")
    console.log(f"Results saved to: {cfg['run_dir']}")
    
    writer.close()
    
    return {
        "best_worst_group_acc": best_worst_group_acc,
        "best_balanced_acc": best_balanced_acc,
        "final_test_metrics": test_metrics,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Train on Fed-Heart Disease dataset")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
