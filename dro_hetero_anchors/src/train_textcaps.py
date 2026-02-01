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
from typing import Dict, Optional, Tuple

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
from .encoders import ENCODER_REGISTRY
from .model.head import LinearHead, MLPHead
from .model.anchors import AnchorModule
from .model.losses import per_class_batch_moments, anchor_fit_loss, anchor_sep_loss
from .model.groupdro import GroupDRO


def build_textcaps_models(cfg, text_encoder: SimpleTextEncoder, device: torch.device):
    """Build models for TextCaps multi-modal setup."""
    latent_dim = cfg["latent_dim"]
    
    encoders: Dict[int, nn.Module] = {}
    
    # Group 0: Visual encoder
    visual_enc_name = cfg["groups"][0]["encoder"]
    visual_enc_cls = ENCODER_REGISTRY[visual_enc_name]
    encoders[0] = visual_enc_cls(latent_dim)
    
    # Group 1: Text encoder
    text_enc_name = cfg["groups"][1]["encoder"]
    text_enc_cls = ENCODER_REGISTRY[text_enc_name]
    # Text encoder needs vocab_size
    encoders[1] = text_enc_cls(
        latent_dim=latent_dim,
        vocab_size=text_encoder.vocab_size,
        max_len=text_encoder.max_len,
    )
    
    # Build head and anchors
    head = (MLPHead(latent_dim, cfg["head_hidden"], cfg["num_classes"])
            if cfg.get("head_hidden", 0) > 0 
            else LinearHead(latent_dim, cfg["num_classes"]))
    
    anchors = AnchorModule(cfg["num_classes"], latent_dim, eps=cfg["anchor_eps"])
    
    # Initialize GroupDRO if enabled
    groupdro = None
    if cfg.get("groupdro_enabled", False):
        groupdro = GroupDRO(
            num_groups=2,
            eta=cfg.get("groupdro_eta", 0.1),
            device=device,
            update_mode=cfg.get("groupdro_update_mode", "exp"),
            robust_objective=cfg.get("groupdro_objective", "weighted"),
            gamma=cfg.get("groupdro_gamma", 1.0),
            min_group_weight=cfg.get("groupdro_min_group_weight", 0.0),
        )
    
    return encoders, head, anchors, groupdro


def evaluate_textcaps(
    encoders: Dict[int, nn.Module],
    head: nn.Module,
    loader,
    device,
    num_classes: int = 10,
):
    """Evaluate on TextCaps multi-modal test set.
    
    Returns:
        acc: Overall accuracy
        acc_by_group: [visual_acc, text_acc]
        worst_group_acc: Minimum group accuracy
        balanced_acc: Average of group accuracies
        metrics: Dict with F1, precision, recall (per-class and macro)
    """
    head.eval()
    for e in encoders.values():
        e.eval()
    
    correct = 0
    total = 0
    correct_g = [0, 0]
    total_g = [0, 0]
    
    # For classification metrics: track TP, FP, FN per class
    tp_per_class = [0] * num_classes
    fp_per_class = [0] * num_classes
    fn_per_class = [0] * num_classes
    
    with torch.no_grad():
        for batch in loader:
            # Process visual samples
            if batch['visual_x'].size(0) > 0:
                x_v = batch['visual_x'].to(device)
                y_v = batch['visual_y'].to(device)
                z_v = encoders[0](x_v)
                logits_v = head(z_v)
                pred_v = logits_v.argmax(dim=1)
                correct_g[0] += (pred_v == y_v).sum().item()
                total_g[0] += y_v.size(0)
                
                # Update per-class metrics
                for i in range(y_v.size(0)):
                    true_class = int(y_v[i].item())
                    pred_class = int(pred_v[i].item())
                    if pred_class == true_class:
                        tp_per_class[true_class] += 1
                    else:
                        fp_per_class[pred_class] += 1
                        fn_per_class[true_class] += 1
            
            # Process text samples
            if batch['text_x'].size(0) > 0:
                x_t = batch['text_x'].to(device)
                y_t = batch['text_y'].to(device)
                z_t = encoders[1](x_t)
                logits_t = head(z_t)
                pred_t = logits_t.argmax(dim=1)
                correct_g[1] += (pred_t == y_t).sum().item()
                total_g[1] += y_t.size(0)
                
                # Update per-class metrics
                for i in range(y_t.size(0)):
                    true_class = int(y_t[i].item())
                    pred_class = int(pred_t[i].item())
                    if pred_class == true_class:
                        tp_per_class[true_class] += 1
                    else:
                        fp_per_class[pred_class] += 1
                        fn_per_class[true_class] += 1
    
    total = total_g[0] + total_g[1]
    correct = correct_g[0] + correct_g[1]
    
    acc = correct / max(1, total)
    acc_by_group = [
        correct_g[0] / max(1, total_g[0]),
        correct_g[1] / max(1, total_g[1]),
    ]
    worst_group_acc = min(acc_by_group)
    balanced_acc = sum(acc_by_group) / 2
    
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
    
    metrics = {
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }
    
    return acc, acc_by_group, worst_group_acc, balanced_acc, metrics


def train_textcaps(cfg):
    """Main training loop for TextCaps multi-modal."""
    
    # Device setup - try MPS (Apple Silicon GPU) first
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        console.log("🚀 Using Apple MPS GPU!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    console.log(f"Using device: {device}")
    
    set_seed(cfg["seed"])
    ensure_dir(cfg["run_dir"])
    writer = SummaryWriter(log_dir=str(cfg["run_dir"]))
    
    # Build data loaders
    use_huggingface = cfg.get("textcaps_use_huggingface", True)  # Default to HF
    
    if use_huggingface and HF_AVAILABLE:
        console.log("Building TextCaps data loaders from Hugging Face...")
        train_loader, test_loader, class_to_idx, text_encoder = build_textcaps_loaders_hf(
            batch_size=cfg["batch_size"],
            num_workers=cfg.get("num_workers", 0),
            num_classes=cfg.get("textcaps_num_classes", 10),
            max_train_samples=cfg.get("textcaps_max_train_samples"),
            max_test_samples=cfg.get("textcaps_max_test_samples"),
            seed=cfg["seed"],
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
    
    # Get actual number of classes (may differ from config if using all classes)
    actual_num_classes = len(class_to_idx)
    if cfg.get("num_classes") != actual_num_classes:
        console.log(f"⚠️  Config has num_classes={cfg.get('num_classes')}, but dataset has {actual_num_classes} classes. Using {actual_num_classes}.")
        cfg["num_classes"] = actual_num_classes  # Update config to match actual
    
    console.log(f"Classes ({actual_num_classes} total): {list(class_to_idx.keys())[:10]}..." if actual_num_classes > 10 else f"Classes: {list(class_to_idx.keys())}")
    
    # Build models (pass device for GroupDRO)
    encoders, head, anchors, groupdro = build_textcaps_models(cfg, text_encoder, device)
    
    # Move to device
    for k in encoders:
        encoders[k] = encoders[k].to(device)
    head = head.to(device)
    anchors = anchors.to(device)
    
    # Optimizer
    params = list(head.parameters()) + list(anchors.parameters())
    for k in encoders:
        params += list(encoders[k].parameters())
    
    opt = optim.Adam(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0))
    
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
    
    console.log("Starting training...")
    
    for epoch in range(1, cfg["epochs"] + 1):
        # Optional linear LR decay across epochs: lr_start -> lr_end
        lr_start = cfg.get("lr_start")
        lr_end = cfg.get("lr_end")
        if lr_start is not None and lr_end is not None:
            t = (epoch - 1) / max(1, (cfg["epochs"] - 1))
            current_lr = (1 - t) * lr_start + t * lr_end
            for pg in opt.param_groups:
                pg["lr"] = current_lr
        else:
            current_lr = cfg["lr"]
        
        head.train()
        for e in encoders.values():
            e.train()
        anchors.train()
        
        loss_meter = Meter()
        acc_meter = Meter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} (lr={current_lr:.4g})")
        
        for batch in pbar:
            # Get visual and text data
            visual_x = batch['visual_x'].to(device)
            visual_y = batch['visual_y'].to(device)
            visual_g = batch['visual_g'].to(device)
            
            text_x = batch['text_x'].to(device)
            text_y = batch['text_y'].to(device)
            text_g = batch['text_g'].to(device)
            
            # Encode both modalities
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
                ce = nn.functional.cross_entropy(logits, y)
            
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
                nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            
            opt.step()
            
            # GroupDRO weight update
            if groupdro is not None and hasattr(groupdro, "_last_group_losses"):
                warmup_epochs = cfg.get("groupdro_warmup_epochs", 0)
                if epoch <= warmup_epochs:
                    with torch.no_grad():
                        groupdro.q.copy_(torch.ones_like(groupdro.q) / len(groupdro.q))
                else:
                    groupdro.update_weights(
                        groupdro._last_group_losses,
                        groupdro._last_group_counts
                    )
            
            # Update meters
            loss_meter.update(loss.item(), z.size(0))
            acc_meter.update(acc.item(), z.size(0))
            
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
            
            global_step += 1
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.3f}",
                "acc": f"{acc_meter.avg:.3f}"
            })
        
        # Evaluation
        test_acc, test_acc_by_group, worst_group_acc, balanced_acc, test_metrics = evaluate_textcaps(
            encoders, head, test_loader, device, num_classes=num_classes
        )
        
        # Log epoch results
        q_snapshot = None
        if groupdro is not None:
            with torch.no_grad():
                q_snapshot = groupdro.q.detach().cpu().tolist()
        
        console.log(
            f"Epoch {epoch}: train_loss={loss_meter.avg:.4f} | train_acc={acc_meter.avg:.4f}"
        )
        console.log(
            f"Epoch {epoch}: test_acc={test_acc:.4f} | balanced={balanced_acc:.4f} | "
            f"worst_group={worst_group_acc:.4f} | visual={test_acc_by_group[0]:.4f} | "
            f"text={test_acc_by_group[1]:.4f}"
        )
        console.log(
            f"Epoch {epoch}: F1={test_metrics['macro_f1']:.4f} | "
            f"Precision={test_metrics['macro_precision']:.4f} | "
            f"Recall={test_metrics['macro_recall']:.4f}"
        )
        if q_snapshot:
            console.log(f"Epoch {epoch}: GroupDRO weights q={q_snapshot}")
        
        # TensorBoard
        writer.add_scalar("test/acc", test_acc, epoch)
        writer.add_scalar("test/worst_group_acc", worst_group_acc, epoch)
        writer.add_scalar("test/balanced_acc", balanced_acc, epoch)
        writer.add_scalar("test/acc_visual", test_acc_by_group[0], epoch)
        writer.add_scalar("test/acc_text", test_acc_by_group[1], epoch)
        writer.add_scalar("test/macro_f1", test_metrics['macro_f1'], epoch)
        writer.add_scalar("test/macro_precision", test_metrics['macro_precision'], epoch)
        writer.add_scalar("test/macro_recall", test_metrics['macro_recall'], epoch)
        
        # Results logger
        results_logger.log_epoch({
            "metrics_version": "1.0",
            "epoch": epoch,
            "train_loss": float(loss_meter.avg),
            "train_acc": float(acc_meter.avg),
            "test_acc": float(test_acc),
            "worst_group_acc": float(worst_group_acc),
            "balanced_acc": float(balanced_acc),
            "per_group_acc": test_acc_by_group,
            "group_names": ["visual", "text"],
            "train_group_weights": q_snapshot,
            "macro_f1": float(test_metrics['macro_f1']),
            "macro_precision": float(test_metrics['macro_precision']),
            "macro_recall": float(test_metrics['macro_recall']),
            "f1_per_class": test_metrics['f1_per_class'],
            "precision_per_class": test_metrics['precision_per_class'],
            "recall_per_class": test_metrics['recall_per_class'],
        })
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
                "class_to_idx": class_to_idx,
            }
            if groupdro is not None:
                save_dict["groupdro"] = {
                    "weights": groupdro.q,
                    "stats": groupdro.group_stats,
                }
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
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_textcaps(cfg)
