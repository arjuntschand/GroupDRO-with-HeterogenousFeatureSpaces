"""
Training script for TextCaps Multi-Modal GroupDRO experiments.

Handles the multi-modal batch structure where each batch contains:
- Visual samples (group 0): images
- Text samples (group 1): tokenized captions

This demonstrates GroupDRO with truly heterogeneous feature spaces
(image vs text modalities).
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import set_seed, ensure_dir, Meter, console
from .results_logger import ResultsLogger
from .datasets_textcaps import (
    build_textcaps_loaders, 
    build_textcaps_loaders_hf,
    SimpleTextEncoder,
    HF_AVAILABLE,
)
from collections import defaultdict
from .encoders import ENCODER_REGISTRY
from .model.head import LinearHead, MLPHead
from .model.anchors import AnchorModule
from .model.losses import per_class_batch_moments, anchor_fit_loss, anchor_sep_loss
from .model.groupdro import GroupDRO


class FusionLayer(nn.Module):
    """Fusion layer for combining visual and text latent representations.
    
    If fusion_hidden is set, uses two layers: (2*latent_dim) -> fusion_hidden -> latent_dim
    for more capacity. Otherwise single layer: (2*latent_dim) -> latent_dim.
    """
    
    def __init__(self, latent_dim: int, dropout: float = 0.1, fusion_hidden: Optional[int] = None):
        super().__init__()
        if fusion_hidden is not None and fusion_hidden > 0:
            self.fusion = nn.Sequential(
                nn.Linear(latent_dim * 2, fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, latent_dim),
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
    
    def forward(self, z_visual: torch.Tensor, z_text: torch.Tensor) -> torch.Tensor:
        """Fuse visual and text latents by concatenation then projection."""
        z_concat = torch.cat([z_visual, z_text], dim=1)
        return self.fusion(z_concat)


class FusionLayerCrossAttn(nn.Module):
    """Fuse visual and text via self-attention over the 2 tokens (visual, text) per sample.
    Each sample gets a combined representation that mixes both modalities. More capacity than concat-only.
    """
    def __init__(self, latent_dim: int, dropout: float = 0.1, num_heads: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        assert latent_dim % num_heads == 0
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),  # concat mean-pooled + last token or similar
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, z_visual: torch.Tensor, z_text: torch.Tensor) -> torch.Tensor:
        # (B, 2, d): per-sample sequence of [visual, text]
        x = torch.stack([z_visual, z_text], dim=1)
        x = self.self_attn(x)   # (B, 2, d)
        # Mean over the 2 tokens, then concat with original visual for residual
        x_mean = x.mean(dim=1)
        z_concat = torch.cat([z_visual, x_mean], dim=1)
        return self.out_proj(z_concat)


def build_textcaps_models(cfg, text_encoder: SimpleTextEncoder, device: torch.device, 
                          group_counts: Optional[List[int]] = None):
    """Build models for TextCaps multi-modal setup.
    
    Args:
        cfg: Configuration dict
        text_encoder: Text tokenizer
        device: Torch device
        group_counts: List of sample counts per group for π-proportional init
    """
    latent_dim = cfg["latent_dim"]
    include_combined = cfg.get("include_combined_group", False)
    num_groups = 3 if include_combined else 2
    
    encoders: Dict[int, nn.Module] = {}
    
    # Group 0: Visual encoder
    visual_enc_name = cfg["groups"][0]["encoder"]
    visual_enc_cls = ENCODER_REGISTRY[visual_enc_name]
    encoders[0] = visual_enc_cls(latent_dim)
    
    # Group 1: Text encoder
    text_enc_name = cfg["groups"][1]["encoder"]
    text_enc_cls = ENCODER_REGISTRY[text_enc_name]
    text_kwargs = dict(
        latent_dim=latent_dim,
        vocab_size=text_encoder.vocab_size,
        max_len=text_encoder.max_len,
    )
    # Stronger transformer: optional embed_dim, num_heads, num_layers (raise text-group ceiling)
    if text_enc_name == "transformer_text":
        if cfg.get("text_encoder_embed_dim") is not None:
            text_kwargs["embed_dim"] = cfg["text_encoder_embed_dim"]
        if cfg.get("text_encoder_num_heads") is not None:
            text_kwargs["num_heads"] = cfg["text_encoder_num_heads"]
        if cfg.get("text_encoder_num_layers") is not None:
            text_kwargs["num_layers"] = cfg["text_encoder_num_layers"]
    encoders[1] = text_enc_cls(**text_kwargs)
    
    # Fusion layer for combined group (group 2) if enabled
    fusion_layer = None
    if include_combined:
        fusion_type = cfg.get("fusion_type", "concat").lower()
        if fusion_type == "cross_attn":
            fusion_layer = FusionLayerCrossAttn(
                latent_dim,
                dropout=cfg.get("fusion_dropout", 0.1),
                num_heads=cfg.get("fusion_num_heads", 4),
            )
        else:
            fusion_layer = FusionLayer(
                latent_dim,
                dropout=cfg.get("fusion_dropout", 0.1),
                fusion_hidden=cfg.get("fusion_hidden"),
            )
    
    # Build head and anchors (head_dropout reduces overfitting)
    head = (MLPHead(latent_dim, cfg["head_hidden"], cfg["num_classes"], dropout=cfg.get("head_dropout", 0.3))
            if cfg.get("head_hidden", 0) > 0 
            else LinearHead(latent_dim, cfg["num_classes"]))
    
    anchors = AnchorModule(cfg["num_classes"], latent_dim, eps=cfg["anchor_eps"])
    
    # Initialize GroupDRO if enabled, with π-proportional weights
    groupdro = None
    if cfg.get("groupdro_enabled", False):
        groupdro = GroupDRO(
            num_groups=num_groups,
            eta=cfg.get("groupdro_eta", 0.1),
            device=device,
            update_mode=cfg.get("groupdro_update_mode", "exp"),
            robust_objective=cfg.get("groupdro_objective", "weighted"),
            gamma=cfg.get("groupdro_gamma", 1.0),
            group_counts=group_counts,  # π-proportional initialization
            kl_lambda=cfg.get("groupdro_kl_lambda", 0.1),  # KL penalty coefficient
        )
    
    return encoders, head, anchors, groupdro, fusion_layer


def evaluate_textcaps(
    encoders: Dict[int, nn.Module],
    head: nn.Module,
    loader,
    device,
    num_classes: int = 10,
    fusion_layer: Optional[nn.Module] = None,
    include_combined: bool = False,
):
    """Evaluate on TextCaps multi-modal test set.
    
    Returns:
        acc: Overall accuracy
        acc_by_group: [visual_acc, text_acc, (combined_acc if enabled)]
        worst_group_acc: Minimum group accuracy
        balanced_acc: Average of group accuracies
        metrics: Dict with F1, precision, recall (per-class and macro), and per-class per-group acc
    """
    head.eval()
    for e in encoders.values():
        e.eval()
    if fusion_layer is not None:
        fusion_layer.eval()
    
    num_groups = 3 if include_combined else 2
    correct_g = [0] * num_groups
    total_g = [0] * num_groups
    loss_sum_g = [0.0] * num_groups  # sum of CE loss per group for test loss

    # For classification metrics: track TP, FP, FN per class (overall)
    tp_per_class = [0] * num_classes
    fp_per_class = [0] * num_classes
    fn_per_class = [0] * num_classes
    
    # Per-class per-group accuracy tracking
    correct_per_class_per_group = defaultdict(lambda: defaultdict(int))  # [group][class] -> correct
    total_per_class_per_group = defaultdict(lambda: defaultdict(int))    # [group][class] -> total
    
    with torch.no_grad():
        for batch in loader:
            # Process visual samples (group 0)
            if batch['visual_x'].size(0) > 0:
                x_v = batch['visual_x'].to(device)
                y_v = batch['visual_y'].to(device)
                z_v = encoders[0](x_v)
                logits_v = head(z_v)
                loss_sum_g[0] += nn.functional.cross_entropy(logits_v, y_v, reduction='sum').item()
                pred_v = logits_v.argmax(dim=1)
                correct_g[0] += (pred_v == y_v).sum().item()
                total_g[0] += y_v.size(0)
                
                # Update per-class metrics and per-class-per-group
                for i in range(y_v.size(0)):
                    true_class = int(y_v[i].item())
                    pred_class = int(pred_v[i].item())
                    total_per_class_per_group[0][true_class] += 1
                    if pred_class == true_class:
                        tp_per_class[true_class] += 1
                        correct_per_class_per_group[0][true_class] += 1
                    else:
                        fp_per_class[pred_class] += 1
                        fn_per_class[true_class] += 1
            
            # Process text samples (group 1)
            if batch['text_x'].size(0) > 0:
                x_t = batch['text_x'].to(device)
                y_t = batch['text_y'].to(device)
                z_t = encoders[1](x_t)
                logits_t = head(z_t)
                loss_sum_g[1] += nn.functional.cross_entropy(logits_t, y_t, reduction='sum').item()
                pred_t = logits_t.argmax(dim=1)
                correct_g[1] += (pred_t == y_t).sum().item()
                total_g[1] += y_t.size(0)
                
                # Update per-class metrics and per-class-per-group
                for i in range(y_t.size(0)):
                    true_class = int(y_t[i].item())
                    pred_class = int(pred_t[i].item())
                    total_per_class_per_group[1][true_class] += 1
                    if pred_class == true_class:
                        tp_per_class[true_class] += 1
                        correct_per_class_per_group[1][true_class] += 1
                    else:
                        fp_per_class[pred_class] += 1
                        fn_per_class[true_class] += 1
            
            # Process combined samples (group 2) if enabled
            if include_combined and batch['combined_visual_x'].size(0) > 0:
                x_cv = batch['combined_visual_x'].to(device)
                x_ct = batch['combined_text_x'].to(device)
                y_c = batch['combined_y'].to(device)
                
                # Encode both modalities
                z_cv = encoders[0](x_cv)
                z_ct = encoders[1](x_ct)
                
                # Fuse
                z_c = fusion_layer(z_cv, z_ct)
                logits_c = head(z_c)
                loss_sum_g[2] += nn.functional.cross_entropy(logits_c, y_c, reduction='sum').item()
                pred_c = logits_c.argmax(dim=1)
                correct_g[2] += (pred_c == y_c).sum().item()
                total_g[2] += y_c.size(0)
                
                # Update per-class-per-group
                for i in range(y_c.size(0)):
                    true_class = int(y_c[i].item())
                    pred_class = int(pred_c[i].item())
                    total_per_class_per_group[2][true_class] += 1
                    if pred_class == true_class:
                        correct_per_class_per_group[2][true_class] += 1
    
    total = sum(total_g)
    correct = sum(correct_g)
    
    acc = correct / max(1, total)
    acc_by_group = [correct_g[g] / max(1, total_g[g]) for g in range(num_groups)]
    worst_group_acc = min(acc_by_group) if acc_by_group else 0.0
    balanced_acc = sum(acc_by_group) / max(1, len(acc_by_group))
    
    # Compute classification metrics
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for c in range(num_classes):
        tp = tp_per_class[c]
        fp = fp_per_class[c]
        fn = fn_per_class[c]
        
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    
    # Macro averages (average across classes)
    macro_precision = sum(precision_per_class) / max(1, len(precision_per_class))
    macro_recall = sum(recall_per_class) / max(1, len(recall_per_class))
    macro_f1 = sum(f1_per_class) / max(1, len(f1_per_class))
    
    # Compute per-class per-group accuracy
    per_class_per_group_acc = {}
    group_names = ['visual', 'text', 'combined'] if include_combined else ['visual', 'text']
    for g in range(num_groups):
        per_class_per_group_acc[group_names[g]] = {}
        for c in range(num_classes):
            total_c = total_per_class_per_group[g][c]
            correct_c = correct_per_class_per_group[g][c]
            per_class_per_group_acc[group_names[g]][c] = correct_c / max(1, total_c)

    # Per-group test (CE) loss
    test_loss_per_group = [loss_sum_g[g] / max(1, total_g[g]) for g in range(num_groups)]
    
    metrics = {
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_per_group_acc": per_class_per_group_acc,
        "test_loss_per_group": test_loss_per_group,
    }
    
    return acc, acc_by_group, worst_group_acc, balanced_acc, metrics


def train_textcaps(cfg, resume: bool = False, extra_epochs: Optional[int] = None, no_ema: bool = False):
    """Main training loop for TextCaps multi-modal.
    
    If resume=True and run_dir/last.ckpt exists, loads it and continues from
    the next epoch. Use num_workers=0 when resuming to avoid multiprocessing
    temp-dir issues that can crash long runs.
    
    When no_ema=True: do not load EMA weights from checkpoint and do not save
    EMA weights in checkpoints (use only regular model weights).
    
    To extend training (e.g. 50 more epochs after a finished 50-epoch run):
    - Option A: set epochs in the config to the new total (e.g. 100) and run with --resume.
    - Option B: run with --resume --extra-epochs 50; max epoch becomes (last_ckpt_epoch + 50).
    """
    use_ema = not no_ema
    
    # Device setup - try MPS (Apple Silicon GPU) first
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        console.log("Using Apple MPS GPU!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    console.log(f"Using device: {device}")
    
    set_seed(cfg["seed"])
    ensure_dir(cfg["run_dir"])
    writer = SummaryWriter(log_dir=str(cfg["run_dir"]))

    # Avoid DataLoader worker spawn / temp-dir issues when resuming long runs
    if resume:
        cfg["num_workers"] = 0
        console.log("Resume mode: using num_workers=0 to avoid multiprocessing temp-dir issues.")

    include_combined = cfg.get("include_combined_group", False)
    use_stratified = cfg.get("use_stratified_sampling", True)
    
    # Build data loaders
    use_huggingface = cfg.get("textcaps_use_huggingface", True)  # Default to HF
    
    if use_huggingface and HF_AVAILABLE:
        console.log("Building TextCaps data loaders from Hugging Face...")
        
        # Spurious correlation settings
        spurious_correlation = cfg.get("spurious_correlation", False)
        spurious_majority_frac = cfg.get("spurious_majority_frac", 0.8)
        spurious_group_classes = cfg.get("spurious_group_classes")  # None = use default
        
        train_loader, test_loader, class_to_idx, text_encoder, dataset_info = build_textcaps_loaders_hf(
            batch_size=cfg["batch_size"],
            num_workers=cfg.get("num_workers", 0),
            num_classes=cfg.get("textcaps_num_classes", 10),
            class_names=cfg.get("textcaps_class_names"),
            train_frac=cfg.get("textcaps_train_frac"),
            max_train_samples=cfg.get("textcaps_max_train_samples"),
            max_test_samples=cfg.get("textcaps_max_test_samples"),
            seed=cfg["seed"],
            include_combined=include_combined,
            use_stratified_sampling=use_stratified,
            spurious_correlation=spurious_correlation,
            spurious_majority_frac=spurious_majority_frac,
            spurious_group_classes=spurious_group_classes,
        )
    else:
        console.log("Building TextCaps data loaders from local files...")
        train_loader, test_loader, class_to_idx, text_encoder = build_textcaps_loaders(
            root=cfg["root"],
            batch_size=cfg["batch_size"],
            num_workers=cfg.get("num_workers", 0),
            num_classes=cfg.get("textcaps_num_classes", 10),
            max_train_samples=cfg.get("textcaps_max_train_samples"),
            max_test_samples=cfg.get("textcaps_max_test_samples"),
            seed=cfg["seed"],
        )
        # Create a basic dataset_info for local loader
        dataset_info = {
            'train_group_counts': {0: len(train_loader.dataset) // 2, 1: len(train_loader.dataset) // 2},
            'test_group_counts': {0: len(test_loader.dataset) // 2, 1: len(test_loader.dataset) // 2},
            'train_total': len(train_loader.dataset),
            'test_total': len(test_loader.dataset),
            'num_groups': 2,
            'group_names': ['visual', 'text'],
        }
    
    # Get actual number of classes (may differ from config if using all classes)
    actual_num_classes = len(class_to_idx)
    if cfg.get("num_classes") != actual_num_classes:
        console.log(f"Config has num_classes={cfg.get('num_classes')}, but dataset has {actual_num_classes} classes. Using {actual_num_classes}.")
        cfg["num_classes"] = actual_num_classes  # Update config to match actual
    
    console.log(f"Classes ({actual_num_classes} total): {list(class_to_idx.keys())[:10]}..." if actual_num_classes > 10 else f"Classes: {list(class_to_idx.keys())}")
    
    # Extract group counts for π-proportional initialization
    train_group_counts = dataset_info['train_group_counts']
    group_counts_list = [train_group_counts.get(i, 0) for i in range(dataset_info['num_groups'])]
    
    # Build models (pass group counts for π-proportional init)
    encoders, head, anchors, groupdro, fusion_layer = build_textcaps_models(
        cfg, text_encoder, device, group_counts=group_counts_list
    )
    
    # Compute π (group distribution) for baseline weighting
    total_train = sum(group_counts_list)
    pi = [c / total_train for c in group_counts_list] if total_train > 0 else [1.0 / len(group_counts_list)] * len(group_counts_list)
    pi_tensor = torch.tensor(pi, device=device, dtype=torch.float32)
    
    # =========================================================================
    # PRINT COMPREHENSIVE EXPERIMENT INFO
    # =========================================================================
    console.log("=" * 60)
    console.log("EXPERIMENT CONFIGURATION")
    console.log("=" * 60)
    console.log(f"Run name: {cfg.get('run_name', 'unnamed')}")
    console.log(f"Run directory: {cfg['run_dir']}")
    console.log(f"Seed: {cfg['seed']}")
    console.log("")
    
    console.log("--- Dataset Statistics ---")
    console.log(f"Total training samples: {dataset_info['train_total']}")
    console.log(f"Total test samples: {dataset_info['test_total']}")
    console.log(f"Number of groups: {dataset_info['num_groups']}")
    console.log(f"Group names: {dataset_info.get('group_names', ['visual', 'text'])}")
    console.log(f"Training samples per group: {train_group_counts}")
    console.log(f"Group distribution (π): {[f'{p:.4f}' for p in pi]}")
    if 'train_class_counts' in dataset_info:
        console.log(f"Training samples per class: {dataset_info['train_class_counts']}")
    console.log("")
    
    console.log("--- Model Configuration ---")
    console.log(f"Latent dimension: {cfg['latent_dim']}")
    console.log(f"Number of classes: {cfg['num_classes']}")
    console.log(f"Head hidden: {cfg.get('head_hidden', 0)}")
    console.log(f"Head dropout: {cfg.get('head_dropout', 0.0)}")
    console.log(f"Include combined group: {include_combined}")
    console.log("")
    
    console.log("--- Training Configuration ---")
    console.log(f"Epochs: {cfg['epochs']}")
    console.log(f"Batch size: {cfg['batch_size']}")
    console.log(f"Learning rate: {cfg['lr']}")
    console.log(f"Weight decay: {cfg.get('weight_decay', 0)}")
    console.log(f"Grad clip: {cfg.get('grad_clip', 0)}")
    console.log(f"Visual encoder LR scale: {cfg.get('visual_encoder_lr_scale', 1.0)}")
    console.log(f"Stratified sampling: {use_stratified}")
    console.log("")
    
    console.log("--- GroupDRO Configuration ---")
    console.log(f"GroupDRO enabled: {cfg.get('groupdro_enabled', False)}")
    if cfg.get('groupdro_enabled', False):
        console.log(f"  eta: {cfg.get('groupdro_eta', 0.1)}")
        console.log(f"  gamma: {cfg.get('groupdro_gamma', 1.0)}")
        console.log(f"  update_mode: {cfg.get('groupdro_update_mode', 'exp')}")
        console.log(f"  objective: {cfg.get('groupdro_objective', 'weighted')}")
        console.log(f"  KL lambda: {cfg.get('groupdro_kl_lambda', 0.1)}")
        console.log(f"  warmup_epochs: {cfg.get('groupdro_warmup_epochs', 3)}")
        console.log(f"  Initial weights (π): {[f'{p:.4f}' for p in pi]}")
    else:
        console.log(f"  Baseline: using π-weighted loss with π={[f'{p:.4f}' for p in pi]}")
    console.log("=" * 60)
    
    # Move to device
    for k in encoders:
        encoders[k] = encoders[k].to(device)
    head = head.to(device)
    anchors = anchors.to(device)
    if fusion_layer is not None:
        fusion_layer = fusion_layer.to(device)
    
    # Optimizer: visual encoder gets a smaller LR so that when we unfreeze it,
    # it fine-tunes gently and doesn't wreck the head (avoids the post-unfreeze crash).
    visual_lr_scale = cfg.get("visual_encoder_lr_scale", 0.01)  # visual_lr = main_lr * scale
    
    # Build parameter groups
    main_params = list(head.parameters()) + list(anchors.parameters()) + list(encoders[1].parameters())
    if fusion_layer is not None:
        main_params += list(fusion_layer.parameters())
    
    param_groups = [
        {"params": main_params, "lr": cfg["lr"]},
        {"params": list(encoders[0].parameters()), "lr": cfg["lr"] * visual_lr_scale},
    ]
    opt = optim.Adam(param_groups, weight_decay=cfg.get("weight_decay", 0))
    trainable_params = [p for pg in param_groups for p in pg["params"]]

    # Results logger
    results_logger = ResultsLogger(cfg["run_dir"])

    # Training constants
    lambda_fit = cfg["lambda_fit"]
    lambda_sep = cfg["lambda_sep"]
    J = cfg["sep_samples_per_class"]
    num_classes = cfg["num_classes"]
    eps = cfg["anchor_eps"]

    best_worst_group_acc = 0.0
    global_step = 0
    start_epoch = 1
    ema_state_dict = None  # When EMA is implemented, set to {"ema_encoders": ..., "ema_head": ..., ...}; when no_ema, never set.

    # Resume from checkpoint if requested
    if resume:
        ckpt_path = Path(cfg["run_dir"]) / "last.ckpt"
        if ckpt_path.exists():
            console.log(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            # Always load regular model weights
            for gid, enc in encoders.items():
                enc.load_state_dict(ckpt["encoders"][gid])
            head.load_state_dict(ckpt["head"])
            anchors.load_state_dict(ckpt["anchors"])
            if fusion_layer is not None and "fusion_layer" in ckpt:
                fusion_layer.load_state_dict(ckpt["fusion_layer"])
            if groupdro is not None and "groupdro" in ckpt:
                groupdro.q.copy_(ckpt["groupdro"]["weights"].to(device))
            # Load EMA weights into model only when use_ema and checkpoint has them
            if use_ema and "ema_encoders" in ckpt:
                for gid, enc in encoders.items():
                    enc.load_state_dict(ckpt["ema_encoders"][gid])
                head.load_state_dict(ckpt["ema_head"])
                anchors.load_state_dict(ckpt["ema_anchors"])
                if fusion_layer is not None and "ema_fusion_layer" in ckpt:
                    fusion_layer.load_state_dict(ckpt["ema_fusion_layer"])
                console.log("Loaded EMA weights into model for resumed training.")
            start_epoch = ckpt["epoch"] + 1
            best_worst_group_acc = ckpt.get("best_worst_group_acc", ckpt.get("worst_group_acc", 0.0))
            if extra_epochs is not None:
                cfg["epochs"] = ckpt["epoch"] + extra_epochs
                console.log(f"Extended training: --extra-epochs {extra_epochs} -> will run through epoch {cfg['epochs']}")
            console.log(f"Resumed from epoch {ckpt['epoch']}, starting at epoch {start_epoch}; best_worst_group_acc={best_worst_group_acc:.4f}")
        else:
            console.log(f"Resume requested but no checkpoint at {ckpt_path}; starting from scratch.")

    console.log("Starting training...")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        # Freeze visual encoder for first K epochs to create visual vs text disparity
        freeze_visual_epochs = cfg.get("freeze_visual_encoder_epochs", 0)
        if freeze_visual_epochs > 0:
            for p in encoders[0].parameters():
                p.requires_grad = (epoch > freeze_visual_epochs)
        
        # Optional linear LR decay across epochs: lr_start -> lr_end
        lr_start = cfg.get("lr_start")
        lr_end = cfg.get("lr_end")
        if lr_start is not None and lr_end is not None:
            t = (epoch - 1) / max(1, (cfg["epochs"] - 1))
            current_lr = (1 - t) * lr_start + t * lr_end
            opt.param_groups[0]["lr"] = current_lr
            opt.param_groups[1]["lr"] = current_lr * visual_lr_scale
        else:
            current_lr = cfg["lr"]
            opt.param_groups[0]["lr"] = current_lr
            opt.param_groups[1]["lr"] = current_lr * visual_lr_scale
        
        head.train()
        for e in encoders.values():
            e.train()
        anchors.train()
        
        loss_meter = Meter()
        acc_meter = Meter()
        num_groups_epoch = len(pi)
        loss_meters_per_group = [Meter() for _ in range(num_groups_epoch)]

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} (lr={current_lr:.4g})")
        
        for batch in pbar:
            # Get visual and text data
            visual_x = batch['visual_x'].to(device)
            visual_y = batch['visual_y'].to(device)
            visual_g = batch['visual_g'].to(device)
            
            text_x = batch['text_x'].to(device)
            text_y = batch['text_y'].to(device)
            text_g = batch['text_g'].to(device)
            
            # Get combined data if enabled
            if include_combined:
                combined_visual_x = batch['combined_visual_x'].to(device)
                combined_text_x = batch['combined_text_x'].to(device)
                combined_y = batch['combined_y'].to(device)
                combined_g = batch['combined_g'].to(device)
            
            # Encode all modalities
            z_list = []
            y_list = []
            g_list = []
            
            if visual_x.size(0) > 0:
                z_v = encoders[0](visual_x)
                z_list.append(z_v)
                y_list.append(visual_y)
                g_list.append(visual_g)
            
            if text_x.size(0) > 0:
                z_t = encoders[1](text_x)
                z_list.append(z_t)
                y_list.append(text_y)
                g_list.append(text_g)
            
            if include_combined and combined_visual_x.size(0) > 0:
                # Encode both modalities and fuse
                z_cv = encoders[0](combined_visual_x)
                z_ct = encoders[1](combined_text_x)
                z_c = fusion_layer(z_cv, z_ct)
                z_list.append(z_c)
                y_list.append(combined_y)
                g_list.append(combined_g)
            
            if not z_list:
                continue
            
            # Concatenate
            z = torch.cat(z_list, dim=0)
            y = torch.cat(y_list, dim=0)
            g = torch.cat(g_list, dim=0)
            
            # Classification
            logits = head(z)
            
            # Loss computation
            if groupdro is not None:
                ce = groupdro.forward(logits, y, g)
            else:
                # Baseline: π-weighted per-group loss
                # L = sum_g π_g * L_g where L_g is CE on group g samples
                ce = torch.tensor(0.0, device=device)
                num_groups = len(pi)
                for gid in range(num_groups):
                    mask = (g == gid)
                    if mask.sum() > 0:
                        ce = ce + pi_tensor[gid] * nn.functional.cross_entropy(logits[mask], y[mask])
            
            # Accuracy
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean()
            
            # Anchor losses
            moments = per_class_batch_moments(z, y, num_classes, eps)
            m_anc, S_anc, L_norm = anchors.forward()
            l_fit = anchor_fit_loss(m_anc, S_anc, moments, eps)
            l_sep = anchor_sep_loss(
                m_anc, S_anc, L_norm, head, num_classes, J, device,
                sep_method=cfg.get("sep_method", "classifier"),
                margin=cfg.get("sep_margin", 1.0),
                eps=eps
            )
            
            # Total loss
            loss = ce + lambda_fit * l_fit + lambda_sep * l_sep
            
            # Backward
            opt.zero_grad(set_to_none=True)
            loss.backward()
            
            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(trainable_params, cfg["grad_clip"])
            
            opt.step()
            
            # GroupDRO weight update
            if groupdro is not None and hasattr(groupdro, "_last_group_losses"):
                warmup_epochs = cfg.get("groupdro_warmup_epochs", 3)  # Default 3-epoch warmup
                if epoch <= warmup_epochs:
                    # During warmup, keep weights at π (initial distribution)
                    with torch.no_grad():
                        groupdro.q.copy_(groupdro.pi)
                else:
                    groupdro.update_weights(
                        groupdro._last_group_losses,
                        groupdro._last_group_counts
                    )
            
            # Update meters
            loss_meter.update(loss.item(), z.size(0))
            acc_meter.update(acc.item(), z.size(0))
            # Per-group training loss (CE only, for logging)
            with torch.no_grad():
                for gid in range(num_groups_epoch):
                    mask = (g == gid)
                    if mask.sum() > 0:
                        ce_g = nn.functional.cross_entropy(logits[mask], y[mask])
                        loss_meters_per_group[gid].update(ce_g.item(), mask.sum().item())

            # Logging
            if global_step % cfg["log_interval"] == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/acc", acc.item(), global_step)
                writer.add_scalar("train/ce", ce.item(), global_step)
                writer.add_scalar("train/l_fit", l_fit.item(), global_step)
                writer.add_scalar("train/l_sep", l_sep.item(), global_step)
                
                if groupdro is not None:
                    writer.add_scalar("train/q_visual", groupdro.q[0].item(), global_step)
                    writer.add_scalar("train/q_text", groupdro.q[1].item(), global_step)
                    if include_combined and len(groupdro.q) > 2:
                        writer.add_scalar("train/q_combined", groupdro.q[2].item(), global_step)
                    if hasattr(groupdro, '_last_kl_penalty'):
                        writer.add_scalar("train/kl_penalty", groupdro._last_kl_penalty, global_step)
            
            global_step += 1
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.3f}",
                "acc": f"{acc_meter.avg:.3f}"
            })
        
        # Evaluation
        test_acc, test_acc_by_group, worst_group_acc, balanced_acc, test_metrics = evaluate_textcaps(
            encoders, head, test_loader, device, num_classes=num_classes,
            fusion_layer=fusion_layer, include_combined=include_combined
        )
        
        # Log epoch results
        q_snapshot = None
        if groupdro is not None:
            with torch.no_grad():
                q_snapshot = groupdro.q.detach().cpu().tolist()
        
        # Build group accuracy string
        group_names = dataset_info.get('group_names', ['visual', 'text'])
        group_acc_str = " | ".join([f"{group_names[i]}={test_acc_by_group[i]:.4f}" for i in range(len(test_acc_by_group))])
        
        console.log(
            f"Epoch {epoch}: train_loss={loss_meter.avg:.4f} | train_acc={acc_meter.avg:.4f}"
        )
        console.log(
            f"Epoch {epoch}: test_acc={test_acc:.4f} | balanced={balanced_acc:.4f} | "
            f"worst_group={worst_group_acc:.4f} | {group_acc_str}"
        )
        train_loss_pg = [float(m.avg) for m in loss_meters_per_group]
        test_loss_pg = test_metrics.get("test_loss_per_group", [])
        train_loss_pg_str = " | ".join([f"{group_names[i]}={train_loss_pg[i]:.4f}" for i in range(len(train_loss_pg))])
        test_loss_pg_str = " | ".join([f"{group_names[i]}={test_loss_pg[i]:.4f}" for i in range(len(test_loss_pg))]) if test_loss_pg else "n/a"
        console.log(f"Epoch {epoch}: train_loss_per_group: {train_loss_pg_str}")
        console.log(f"Epoch {epoch}: test_loss_per_group:  {test_loss_pg_str}")
        console.log(
            f"Epoch {epoch}: F1={test_metrics['macro_f1']:.4f} | "
            f"Precision={test_metrics['macro_precision']:.4f} | "
            f"Recall={test_metrics['macro_recall']:.4f}"
        )
        if q_snapshot:
            console.log(f"Epoch {epoch}: GroupDRO weights q={[f'{w:.4f}' for w in q_snapshot]}")
            if groupdro is not None and hasattr(groupdro, '_last_kl_penalty'):
                console.log(f"Epoch {epoch}: KL(q||π) = {groupdro._last_kl_penalty:.6f}")
        
        # Print per-class per-group accuracy
        if 'per_class_per_group_acc' in test_metrics:
            console.log(f"Epoch {epoch}: Per-class accuracy by group:")
            for gname, class_accs in test_metrics['per_class_per_group_acc'].items():
                class_names_list = dataset_info.get('class_names', [str(i) for i in range(num_classes)])
                acc_strs = [f"{class_names_list[c] if c < len(class_names_list) else c}: {acc:.2%}" 
                           for c, acc in sorted(class_accs.items())]
                console.log(f"  {gname}: {', '.join(acc_strs)}")
        
        # TensorBoard
        writer.add_scalar("test/acc", test_acc, epoch)
        writer.add_scalar("test/worst_group_acc", worst_group_acc, epoch)
        writer.add_scalar("test/balanced_acc", balanced_acc, epoch)
        for i, gname in enumerate(group_names):
            if i < len(test_acc_by_group):
                writer.add_scalar(f"test/acc_{gname}", test_acc_by_group[i], epoch)
        for i, gname in enumerate(group_names):
            if i < len(test_metrics.get("test_loss_per_group", [])):
                writer.add_scalar(f"test/loss_{gname}", test_metrics["test_loss_per_group"][i], epoch)
        writer.add_scalar("test/macro_f1", test_metrics['macro_f1'], epoch)
        writer.add_scalar("test/macro_precision", test_metrics['macro_precision'], epoch)
        writer.add_scalar("test/macro_recall", test_metrics['macro_recall'], epoch)
        
        # Results logger - include comprehensive info
        epoch_record = {
            "metrics_version": "2.0",
            "epoch": epoch,
            "train_loss": float(loss_meter.avg),
            "train_loss_per_group": [float(m.avg) for m in loss_meters_per_group],
            "test_loss_per_group": [float(x) for x in test_metrics.get("test_loss_per_group", [])],
            "train_acc": float(acc_meter.avg),
            "test_acc": float(test_acc),
            "worst_group_acc": float(worst_group_acc),
            "balanced_acc": float(balanced_acc),
            "per_group_acc": test_acc_by_group,
            "group_names": group_names,
            "train_group_weights": q_snapshot,
            "macro_f1": float(test_metrics['macro_f1']),
            "macro_precision": float(test_metrics['macro_precision']),
            "macro_recall": float(test_metrics['macro_recall']),
            "f1_per_class": test_metrics['f1_per_class'],
            "precision_per_class": test_metrics['precision_per_class'],
            "recall_per_class": test_metrics['recall_per_class'],
            "per_class_per_group_acc": test_metrics.get('per_class_per_group_acc', {}),
        }
        
        # Add KL penalty info if available
        if groupdro is not None and hasattr(groupdro, '_last_kl_penalty'):
            epoch_record["kl_penalty"] = groupdro._last_kl_penalty
            epoch_record["pi"] = pi  # Reference distribution
        
        # Add experiment config on first epoch
        if epoch == 1:
            epoch_record["config"] = {
                "run_name": cfg.get("run_name"),
                "seed": cfg["seed"],
                "epochs": cfg["epochs"],
                "batch_size": cfg["batch_size"],
                "lr": cfg["lr"],
                "weight_decay": cfg.get("weight_decay", 0),
                "latent_dim": cfg["latent_dim"],
                "num_classes": cfg["num_classes"],
                "groupdro_enabled": cfg.get("groupdro_enabled", False),
                "groupdro_eta": cfg.get("groupdro_eta"),
                "groupdro_gamma": cfg.get("groupdro_gamma"),
                "groupdro_kl_lambda": cfg.get("groupdro_kl_lambda"),
                "groupdro_warmup_epochs": cfg.get("groupdro_warmup_epochs", 3),
                "include_combined_group": include_combined,
                "use_stratified_sampling": use_stratified,
            }
            epoch_record["dataset_info"] = dataset_info
        
        results_logger.log_epoch(epoch_record)
        results_logger.save()
        
        # Checkpointing
        if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            ckpt_path = Path(cfg["run_dir"]) / "last.ckpt"
            save_dict = {
                "cfg": cfg,
                "epoch": epoch,
                "encoders": {gid: enc.state_dict() for gid, enc in encoders.items()},
                "head": head.state_dict(),
                "anchors": anchors.state_dict(),
                "worst_group_acc": worst_group_acc,
                "best_worst_group_acc": best_worst_group_acc,
                "class_to_idx": class_to_idx,
                "dataset_info": dataset_info,
                "pi": pi,
            }
            if fusion_layer is not None:
                save_dict["fusion_layer"] = fusion_layer.state_dict()
            if groupdro is not None:
                save_dict["groupdro"] = {
                    "weights": groupdro.q,
                    "pi": groupdro.pi,
                    "stats": groupdro.group_stats,
                }
            # Save EMA weights only when use_ema (not --no_ema). If EMA state is maintained
            # during training (e.g. ema_state_dict populated elsewhere), add it here.
            if use_ema and ema_state_dict is not None:
                save_dict.update(ema_state_dict)
            torch.save(save_dict, ckpt_path)
            console.log(f"Saved checkpoint to {ckpt_path}")
            
            # Best model
            if worst_group_acc > best_worst_group_acc:
                best_worst_group_acc = worst_group_acc
                best_ckpt_path = Path(cfg["run_dir"]) / "best.ckpt"
                torch.save(save_dict, best_ckpt_path)
                console.log(f"New best worst-group acc: {worst_group_acc:.4f}")
    
    console.log("Training complete!")
    console.log(f"Best worst-group accuracy: {best_worst_group_acc:.4f}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume", action="store_true", help="Resume from run_dir/last.ckpt if it exists")
    ap.add_argument("--extra-epochs", type=int, default=None, metavar="N",
                    help="When resuming: train N more epochs (max_epoch = last_ckpt_epoch + N). Ignored if not resuming.")
    ap.add_argument("--no_ema", action="store_true",
                    help="Do not load EMA weights from checkpoint and do not save EMA weights.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_textcaps(cfg, resume=args.resume, extra_epochs=args.extra_epochs, no_ema=args.no_ema)
