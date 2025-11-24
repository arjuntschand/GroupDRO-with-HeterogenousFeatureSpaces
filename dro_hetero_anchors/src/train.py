import argparse, yaml
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .utils import set_seed, ensure_dir, Meter, console
from .results_logger import ResultsLogger
from .datasets import (
    build_loaders,
    build_skewed_mnist_usps_loaders,
    build_skewed_mnist_usps_mnist32_loaders,
    build_usps_only_balanced_loaders,
    build_mnist_usps_balanced_subset_loaders,
    build_skewed_mnist_usps_train_subset_loaders,
)
from .encoders import ENCODER_REGISTRY
from .model.head import LinearHead, MLPHead
from .model.anchors import AnchorModule
from .model.losses import per_class_batch_moments, anchor_fit_loss, anchor_sep_loss
from .model.groupdro import GroupDRO

def build_models(cfg) -> Tuple[Dict[int, nn.Module], nn.Module, AnchorModule, Optional[GroupDRO]]:
    """Build all model components and optionally initialize GroupDRO."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = cfg["latent_dim"]
    
    # Build per-group encoders
    encoders: Dict[int, nn.Module] = {}
    for gid, g in enumerate(cfg["groups"]):
        enc_cls = ENCODER_REGISTRY[g["encoder"]]
        encoders[gid] = enc_cls(latent_dim)
        
    # Build head and anchors
    head = (MLPHead(latent_dim, cfg["head_hidden"], cfg["num_classes"])
            if cfg.get("head_hidden", 0) > 0 else LinearHead(latent_dim, cfg["num_classes"]))
    anchors = AnchorModule(cfg["num_classes"], latent_dim, eps=cfg["anchor_eps"])
    
    # Initialize GroupDRO if enabled
    groupdro = None
    if cfg.get("groupdro_enabled", False):
        groupdro = GroupDRO(
            num_groups=len(cfg["groups"]),
            eta=cfg.get("groupdro_eta", 0.1),
            device=device
        )
    
    return encoders, head, anchors, groupdro

def evaluate(encoders: Dict[int, nn.Module], head: nn.Module, loader, device, num_groups: int):
    head.eval()
    for e in encoders.values(): e.eval()
    correct = 0; total = 0
    # per-group totals
    correct_g = [0 for _ in range(num_groups)]
    total_g = [0 for _ in range(num_groups)]
    # per-class totals (assume classes are 0..C-1, infer C from head)
    C = list(head.parameters())[-1].shape[0]
    correct_c = [0 for _ in range(C)]
    total_c = [0 for _ in range(C)]
    # per-group per-class matrix
    correct_gc = [[0 for _ in range(C)] for _ in range(num_groups)]
    total_gc = [[0 for _ in range(C)] for _ in range(num_groups)]

    with torch.no_grad():
        for x, y, g in loader:
            x, y, g = x.to(device), y.to(device), g.to(device)
            latent_dim = list(head.parameters())[0].shape[1]
            z = torch.zeros((x.size(0), latent_dim), device=device)
            for gid, enc in encoders.items():
                mask = (g == gid)
                if mask.sum() == 0: continue
                z_gid = enc(x[mask])  # compute encodings
                z[mask] = z_gid  # safe assignment to pre-allocated tensor
            logits = head(z)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item(); total += y.numel()

            for i in range(x.size(0)):
                yi = int(y[i].item())
                gi = int(g[i].item())
                total_c[yi] += 1
                total_g[gi] += 1
                total_gc[gi][yi] += 1
                if int(pred[i].item()) == yi:
                    correct_c[yi] += 1
                    correct_g[gi] += 1
                    correct_gc[gi][yi] += 1

    acc = correct / max(1, total)
    acc_by_group = [ (correct_g[i] / max(1, total_g[i])) if total_g[i] > 0 else 0.0 for i in range(num_groups) ]
    acc_by_class = [ (correct_c[i] / max(1, total_c[i])) if total_c[i] > 0 else 0.0 for i in range(C) ]
    acc_by_group_by_class = [[ (correct_gc[g][c] / max(1, total_gc[g][c])) if total_gc[g][c] > 0 else 0.0
                                for c in range(C)] for g in range(num_groups)]
    return acc, acc_by_group, acc_by_class, acc_by_group_by_class

def compute_dataset_summary(loader, num_classes: int, num_groups: int):
    """Return dict with per-class counts, per-group counts, and group-by-class counts."""
    class_counts = [0] * num_classes
    group_counts = [0] * num_groups
    group_class_counts = [[0] * num_classes for _ in range(num_groups)]
    total = 0
    with torch.no_grad():
        for _, y, g in loader:
            total += y.size(0)
            for i in range(y.size(0)):
                yi = int(y[i].item())
                gi = int(g[i].item())
                class_counts[yi] += 1
                group_counts[gi] += 1
                group_class_counts[gi][yi] += 1
    return {
        "total": total,
        "class_counts": class_counts,
        "group_counts": group_counts,
        "group_class_counts": group_class_counts,
    }

def print_dataset_summary(summary, num_classes, num_groups):
    """Pretty-print the dataset summary returned by compute_dataset_summary()."""
    total = summary["total"]
    class_counts = summary["class_counts"]
    group_counts = summary["group_counts"]
    group_class_counts = summary["group_class_counts"]
    console.log(f"Dataset Summary ({total} samples):")
    console.log("Per-class distribution:")
    for cid in range(num_classes):
        pct = 100.0 * class_counts[cid] / max(1, total)
        console.log(f"  Class {cid:2d}: {class_counts[cid]:5d} ({pct:5.1f}%)")
    console.log("Per-group distribution:")
    for gid in range(num_groups):
        pct = 100.0 * group_counts[gid] / max(1, total)
        console.log(f"  Group {gid}: {group_counts[gid]:5d} ({pct:5.1f}%)")
    console.log("Per-group-per-class breakdown:")
    for gid in range(num_groups):
        console.log(f"  Group {gid}: {group_class_counts[gid]}")

def train(cfg):
    # Enable anomaly detection globally
    torch.autograd.set_detect_anomaly(True)
    
    # Initialize best accuracy tracker
    best_worst_group_acc = 0.0
    
    # Force CPU if on MPS (M1/M2 Mac) to avoid eigh autograd issues
    if torch.backends.mps.is_available():
        device = torch.device("cpu")
        torch.set_default_device(device)  # Ensure all new tensors are on CPU
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(cfg["seed"])
    ensure_dir(cfg["run_dir"])
    writer = SummaryWriter(log_dir=str(cfg["run_dir"]))
    
    # Setup data and models
    # Check if using skewed MNIST+USPS (2-group) or MNIST+USPS+MNIST32 (3-group)
    if cfg.get("use_skewed_mnist_usps_mnist32", False):
        console.log("Loading skewed MNIST+USPS+MNIST32 (3 groups) datasets...")
        train_loader, test_loader = build_skewed_mnist_usps_mnist32_loaders(
            cfg["root"], cfg["batch_size"], cfg["num_workers"],
            mnist28_size=cfg.get("mnist28_size", 15000),
            usps_size=cfg.get("usps_size", 2000),
            mnist32_size=cfg.get("mnist32_size", 15000),
            mnist28_majority=cfg.get("mnist28_majority", [0, 1, 2, 3]),
            usps_majority=cfg.get("usps_majority", [4, 5, 6]),
            mnist32_majority=cfg.get("mnist32_majority", [7, 8, 9]),
            seed=cfg["seed"],
            majority_frac=cfg.get("majority_frac", 0.8),
        )
    elif cfg.get("use_skewed_mnist_usps", False):
        console.log("Loading skewed MNIST+USPS datasets...")
        train_loader, test_loader = build_skewed_mnist_usps_loaders(
            cfg["root"], cfg["batch_size"], cfg["num_workers"],
            mnist_size=cfg.get("mnist_size", 30000),
            usps_size=cfg.get("usps_size", 1000),
            mnist_majority=cfg.get("mnist_majority", [0, 1, 2, 3, 4]),
            usps_majority=cfg.get("usps_majority", [5, 6, 7, 8, 9]),
            seed=cfg["seed"],
            majority_frac=cfg.get("majority_frac", 0.8),
        )
    elif cfg.get("use_usps_only_balanced", False):
        console.log("Loading balanced USPS-only dataset (single group)...")
        train_loader, test_loader = build_usps_only_balanced_loaders(
            cfg["root"], cfg["batch_size"], cfg["num_workers"],
            usps_size=cfg.get("usps_size", 5000),
            seed=cfg["seed"],
            max_train_frac=cfg.get("max_train_frac", 0.9),
        )
    elif cfg.get("use_skewed_mnist_usps_train_subset", False):
        console.log("Loading skewed MNIST+USPS with fixed train subset sizes and flipped 30/70 per-class ratios...")
        train_loader, test_loader = build_skewed_mnist_usps_train_subset_loaders(
            root=cfg["root"],
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            mnist_train_size=cfg.get("mnist_train_size", 200),
            usps_train_size=cfg.get("usps_train_size", 4000),
            mnist_high_ratio=cfg.get("mnist_high_ratio", 0.7),
            usps_high_ratio=cfg.get("usps_high_ratio", 0.3),
            use_full_pool=cfg.get("use_full_pool", True),
            seed=cfg.get("seed", 1337),
        )
    elif cfg.get("use_mnist_usps_balanced", False):
        console.log("Loading balanced MNIST+USPS subset with per-class 80/20 splits (enhanced pools)...")
        train_loader, test_loader = build_mnist_usps_balanced_subset_loaders(
            root=cfg["root"],
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            mnist_total_size=cfg.get("mnist_total_size", 200),
            usps_total_size=cfg.get("usps_total_size", 6000),
            train_frac=cfg.get("train_frac", 0.8),
            seed=cfg.get("seed", 1337),
            mnist_use_full_pool=cfg.get("mnist_use_full_pool", False),
            usps_use_full_pool=cfg.get("usps_use_full_pool", False),
        )
    else:
        train_loader, test_loader = build_loaders(
            cfg["root"], cfg["groups"], cfg["batch_size"], cfg["num_workers"])
    
    # Compute and print training & test dataset summaries for verification
    console.rule("Training Dataset Summary")
    train_summary = compute_dataset_summary(train_loader, cfg["num_classes"], len(cfg["groups"]))
    print_dataset_summary(train_summary, cfg["num_classes"], len(cfg["groups"]))
    console.rule("Test Dataset Summary")
    test_summary = compute_dataset_summary(test_loader, cfg["num_classes"], len(cfg["groups"]))
    print_dataset_summary(test_summary, cfg["num_classes"], len(cfg["groups"]))
    console.rule("")

    # Initialize results logger in run dir
    results_logger = ResultsLogger(cfg["run_dir"]) 
    # Static run metadata
    # Normalize group descriptors: allow simple int or str entries instead of dicts
    normalized_groups = []
    for i, g in enumerate(cfg["groups"]):
        if isinstance(g, dict):
            normalized_groups.append(g.get("name", str(i)))
        else:
            normalized_groups.append(str(g))

    run_meta = {
        "mnist_size": cfg.get("mnist_size"),
        "usps_size": cfg.get("usps_size"),
        "mnist28_size": cfg.get("mnist28_size"),
        "mnist32_size": cfg.get("mnist32_size"),
        "num_epochs": cfg.get("epochs"),
        "batch_size": cfg.get("batch_size"),
        "seed": cfg.get("seed"),
        "groups": normalized_groups,
        "train_dataset_summary": train_summary,
        "test_dataset_summary": test_summary,
    }
    
    encoders, head, anchors, groupdro = build_models(cfg)
    
    # Move models to device
    for k in encoders: encoders[k] = encoders[k].to(device)
    head = head.to(device)
    anchors = anchors.to(device)
    
    # Prepare optimizer
    params = list(head.parameters()) + list(anchors.parameters())
    for k in encoders: params += list(encoders[k].parameters())
    opt = optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    
    # Constants from config
    lambda_fit = cfg["lambda_fit"]
    lambda_sep = cfg["lambda_sep"]
    J = cfg["sep_samples_per_class"]
    num_classes = cfg["num_classes"]
    eps = cfg["anchor_eps"]
    sep_method = cfg.get("sep_method", "classifier")
    sep_margin = cfg.get("sep_margin", 1.0)
    num_groups = len(cfg["groups"])
    global_step = 0
    debug_one_batch = cfg.get("debug_one_batch", False)
    for epoch in range(1, cfg["epochs"] + 1):
        head.train(); [e.train() for e in encoders.values()]; anchors.train()
        loss_meter = Meter(); acc_meter = Meter()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}")
        for x, y, g in pbar:
            x, y, g = x.to(device), y.to(device), g.to(device)
            z = torch.zeros((x.size(0), cfg["latent_dim"]), device=device)
            for gid, enc in encoders.items():
                mask = (g == gid)
                if mask.sum() == 0: continue
                z_gid = enc(x[mask])  # compute encodings for group
                z.index_copy_(0, torch.where(mask)[0], z_gid)  # safe index copy
            logits = head(z)
            
            # Compute loss based on GroupDRO or standard CE
            if groupdro is not None:
                ce = groupdro.forward(logits, y, g)  # handles weight updates internally
            else:
                ce = nn.functional.cross_entropy(logits, y)
            
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean()

            # Compute anchor losses with normalized parameters and optionally run
            # the whole forward+backward inside anomaly-detection so we get a
            # precise stack trace for in-place/NaN errors.
            if debug_one_batch:
                with torch.autograd.set_detect_anomaly(True):
                    moments = per_class_batch_moments(z, y, num_classes, eps)
                    m_anc, S_anc, L_norm = anchors.forward()
                    l_fit = anchor_fit_loss(m_anc, S_anc, moments, eps)
                    # Use normalized L and S for separation loss; method selectable via config
                    sep_method = cfg.get("sep_method", "classifier")
                    sep_margin = cfg.get("sep_margin", 1.0)
                    l_sep = anchor_sep_loss(m_anc, S_anc, L_norm, head, num_classes, J, device,
                                            sep_method=sep_method, margin=sep_margin, eps=eps)

                    # Total loss and optimization
                    loss = ce + lambda_fit * l_fit + lambda_sep * l_sep
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
            else:
                moments = per_class_batch_moments(z, y, num_classes, eps)
                m_anc, S_anc, L_norm = anchors.forward()
                l_fit = anchor_fit_loss(m_anc, S_anc, moments, eps)
                # Use normalized L and S for separation loss; method selectable via config
                sep_method = cfg.get("sep_method", "classifier")
                sep_margin = cfg.get("sep_margin", 1.0)
                l_sep = anchor_sep_loss(m_anc, S_anc, L_norm, head, num_classes, J, device,
                                        sep_method=sep_method, margin=sep_margin, eps=eps)

                # Total loss and optimization
                loss = ce + lambda_fit * l_fit + lambda_sep * l_sep
                opt.zero_grad(set_to_none=True)
                loss.backward()
            
            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            
            opt.step()
            # Deferred GroupDRO update: update group weights after backward/step
            if groupdro is not None and hasattr(groupdro, "_last_group_losses"):
                try:
                    groupdro.update_weights(groupdro._last_group_losses, groupdro._last_group_counts)
                except Exception:
                    # Don't let weight update crash training; log later if needed
                    pass
            if debug_one_batch:
                # break after first batch to surface anomaly stacktrace quickly
                return
            # Update meters
            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(acc.item(), x.size(0))
            
            # Logging
            if global_step % cfg["log_interval"] == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/acc", acc.item(), global_step)
                writer.add_scalar("train/ce", ce.item(), global_step)
                writer.add_scalar("train/l_fit", l_fit.item(), global_step)
                writer.add_scalar("train/l_sep", l_sep.item(), global_step)
                
                if groupdro is not None:
                    writer.add_scalar("train/worst_group_acc", 
                                    groupdro.worst_group_acc, global_step)
                    # Log group weights
                    for gid in range(num_groups):
                        writer.add_scalar(f"train/group_{gid}_weight", 
                                        groupdro.q[gid].item(), global_step)
            
            global_step += 1
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.3f}",
                "acc": f"{acc_meter.avg:.3f}"
            })
        
        # Evaluation (returns overall, per-group, per-class, and per-group-by-class)
        test_acc, test_acc_by_group, test_acc_by_class, test_acc_by_group_by_class = \
            evaluate(encoders, head, test_loader, device, num_groups)
        worst_group_acc = min(test_acc_by_group) if len(test_acc_by_group) > 0 else 0.0
        
        # Logging
        writer.add_scalar("test/acc", test_acc, epoch)
        writer.add_scalar("test/worst_group_acc", worst_group_acc, epoch)
        for gid, acc in enumerate(test_acc_by_group):
            writer.add_scalar(f"test/acc_group_{gid}", acc, epoch)
        # per-class logging
        for cid, acc in enumerate(test_acc_by_class):
            writer.add_scalar(f"test/acc_class_{cid}", acc, epoch)
        
        console.log(
            f"Epoch {epoch}: test acc={test_acc:.4f} | "
            f"worst group={worst_group_acc:.4f} | "
            f"per-group={test_acc_by_group}"
        )
        # Print per-class and per-group-by-class matrices for debugging/diagnostics
        console.log(f"Per-class accuracies: {test_acc_by_class}")
        console.log("Per-group-by-class accuracies:")
        for gid in range(len(test_acc_by_group_by_class)):
            console.log(f"  Group {gid}: {test_acc_by_group_by_class[gid]}")

        # Persist results for this epoch
        results_logger.log_epoch({
            "metrics_version": "1.0",
            **run_meta,
            "epoch": epoch,
            "train_loss": float(loss_meter.avg),
            "train_acc": float(acc_meter.avg),
            "test_acc": float(test_acc),
            "worst_group_acc": float(worst_group_acc),
            "per_group_acc": test_acc_by_group,
            "per_class_acc": test_acc_by_class,
            "per_group_by_class_acc": test_acc_by_group_by_class,
        })
        # Flush to disk each epoch to avoid data loss
        results_logger.save()
        
        # Save checkpoint
        if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            ckpt_path = Path(cfg["run_dir"]) / "last.ckpt"
            save_dict = {
                "cfg": cfg,
                "epoch": epoch,
                "encoders": {gid: enc.state_dict() for gid, enc in encoders.items()},
                "head": head.state_dict(),
                "anchors": anchors.state_dict(),
                "worst_group_acc": worst_group_acc,
            }
            # record experiment metadata
            save_dict["experiment"] = {"sep_method": sep_method, "sep_margin": sep_margin}
            if groupdro is not None:
                save_dict["groupdro"] = {
                    "weights": groupdro.q,
                    "stats": groupdro.group_stats
                }
            torch.save(save_dict, ckpt_path)
            console.log(f"Saved checkpoint to {ckpt_path}")
            
            # Save best checkpoint by worst-group accuracy
            if worst_group_acc > best_worst_group_acc:
                best_worst_group_acc = worst_group_acc
                best_ckpt_path = Path(cfg["run_dir"]) / "best.ckpt"
                torch.save(save_dict, best_ckpt_path)
                console.log(f"New best worst-group acc: {worst_group_acc:.4f}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
