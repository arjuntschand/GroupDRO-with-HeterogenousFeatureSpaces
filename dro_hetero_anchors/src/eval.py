import argparse, yaml, torch
from .utils import console
from .datasets import (
    build_loaders,
    build_skewed_mnist_usps_loaders,
    build_skewed_mnist_usps_mnist32_loaders,
)
from .encoders import ENCODER_REGISTRY
from .model.head import LinearHead, MLPHead
from .model.anchors import AnchorModule
from .train import evaluate, compute_dataset_summary, print_dataset_summary

def load_models(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]; latent_dim = cfg["latent_dim"]
    encoders = {}
    for gid, g in enumerate(cfg["groups"]):
        enc_cls = ENCODER_REGISTRY[g["encoder"]]
        enc = enc_cls(latent_dim); enc.load_state_dict(ckpt["encoders"][gid])
        encoders[gid] = enc
    head_hidden = cfg.get("head_hidden", 0)
    head = (MLPHead(latent_dim, head_hidden, cfg["num_classes"]) if head_hidden > 0
            else LinearHead(latent_dim, cfg["num_classes"]))
    head.load_state_dict(ckpt["head"])
    anchors = AnchorModule(cfg["num_classes"], latent_dim, eps=cfg["anchor_eps"])
    anchors.load_state_dict(ckpt["anchors"])
    return cfg, encoders, head, anchors

def main(cfg_path: str, ckpt_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build the appropriate loaders depending on experiment type
    if cfg.get("use_skewed_mnist_usps_mnist32", False):
        _, test_loader = build_skewed_mnist_usps_mnist32_loaders(
            cfg["root"], cfg["batch_size"], cfg["num_workers"],
            mnist28_size=cfg.get("mnist28_size", 15000),
            usps_size=cfg.get("usps_size", 2000),
            mnist32_size=cfg.get("mnist32_size", 15000),
            mnist28_majority=cfg.get("mnist28_majority", [0, 1, 2, 3]),
            usps_majority=cfg.get("usps_majority", [4, 5, 6]),
            mnist32_majority=cfg.get("mnist32_majority", [7, 8, 9]),
            seed=cfg.get("seed", 1337),
        )
    elif cfg.get("use_skewed_mnist_usps", False):
        _, test_loader = build_skewed_mnist_usps_loaders(
            cfg["root"], cfg["batch_size"], cfg["num_workers"],
            mnist_size=cfg.get("mnist_size", 30000),
            usps_size=cfg.get("usps_size", 1000),
            mnist_majority=cfg.get("mnist_majority", [0, 1, 2, 3, 4]),
            usps_majority=cfg.get("usps_majority", [5, 6, 7, 8, 9]),
            seed=cfg.get("seed", 1337),
        )
    else:
        _, test_loader = build_loaders(cfg["root"], cfg["groups"], cfg["batch_size"], cfg["num_workers"])
    cfg_ckpt, encoders, head, anchors = load_models(ckpt_path)
    assert cfg_ckpt["groups"] == cfg["groups"], "Config groups must match checkpoint"
    for k in encoders: encoders[k] = encoders[k].to(device)
    head = head.to(device)
    # Print test dataset summary to validate distribution
    console.rule("Test Dataset Summary")
    test_summary = compute_dataset_summary(test_loader, cfg["num_classes"], len(cfg["groups"]))
    print_dataset_summary(test_summary, cfg["num_classes"], len(cfg["groups"]))

    acc, acc_by_group, acc_by_class, acc_by_group_by_class = evaluate(
        encoders, head, test_loader, device, len(cfg["groups"]))
    console.rule("Evaluation")
    console.print(f"Overall accuracy: {acc:.4f}")
    for gid, a in enumerate(acc_by_group):
        console.print(f"Group {gid} acc: {a:.4f}")
    console.print("Per-class accuracies:")
    for cid, a in enumerate(acc_by_class):
        console.print(f"  Class {cid} acc: {a:.4f}")
    console.print("Per-group-by-class accuracies:")
    for gid in range(len(acc_by_group_by_class)):
        console.print(f"  Group {gid}: {acc_by_group_by_class[gid]}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.ckpt)
