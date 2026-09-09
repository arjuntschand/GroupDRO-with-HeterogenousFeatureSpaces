"""Fine-tune the ViT-Base backbone on EMBED, then re-cache embeddings from it.

Why. Xenia's spec freezes the ViT and trains small MLPs on cached features. That is cheap and
it is what the frozen-backbone results use. But REMIND fine-tunes ViT-Base end to end, and
their reported EMBED accuracy (78.5-80.7) is several points above what frozen features can
reach. To compare like for like we need a fine-tuned backbone too.

Fine-tuning all 6 methods x 10 seeds end to end is not affordable: one method-run would be
about 12 GPU-hours, so 60 runs is weeks. Instead we use the standard transfer-learning
protocol:

  1. Fine-tune ViT-Base ONCE on the density task, using ONLY the training-split patients.
  2. Re-extract embeddings for every image from that fine-tuned backbone.
  3. Run all 6 methods x 10 seeds on the improved embeddings, exactly as before (3.5 h).

The backbone is therefore identical across methods, so method comparisons stay valid, and it
never sees a validation or test patient, so there is no leakage. This is a shared-encoder
adaptation step, and the paper should describe it as such rather than implying each method
fine-tuned its own backbone.

Usage:
  python -m dro_hetero_anchors.src.finetune_embed_vit --epochs 5
  python -m dro_hetero_anchors.src.finetune_embed_vit --extract-only   # after training
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .train_embed_xenia import patient_split, load_group_tensors  # split reuse
from .model.embed_xenia import GROUPS

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
NUM_CLASSES = 4


def build_vit(device, pretrained=True):
    import timm
    m = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
    return m.to(device)


def image_level_split(index_path, img_map, split_seed=0):
    """Map each cached image to (row_index, label, split). A row's label applies to all of its
    images. Split is by PATIENT, reusing the same partition as the downstream experiment."""
    idx = pd.read_parquet(index_path)
    all_empi = np.unique(idx.empi_anon.values)
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(all_empi)
    n = len(perm); a = int(0.7 * n); b = int(0.8 * n)
    train_e, val_e = set(perm[:a]), set(perm[a:b])

    rows, labels, splits = [], [], []
    for _, r in idx.iterrows():
        d = r["paths"] if isinstance(r["paths"], dict) else dict(r["paths"])
        sp = "train" if r["empi_anon"] in train_e else ("val" if r["empi_anon"] in val_e else "test")
        for p in d.values():
            if isinstance(p, str) and p in img_map:
                rows.append(img_map[p]); labels.append(int(r["label"])); splits.append(sp)
    return np.array(rows), np.array(labels), np.array(splits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="datasets/embed/index_production.parquet")
    ap.add_argument("--img-cache", default="datasets/embed/img_cache")
    ap.add_argument("--out-cache", default="datasets/embed/vit_cache_finetuned")
    ap.add_argument("--ckpt", default="runs/embed_vit_finetuned.pt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)      # small: fine-tuning a pretrained ViT
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    imgs = np.load(os.path.join(args.img_cache, "images.u8.npy"), mmap_mode="r")
    img_map = json.load(open(os.path.join(args.img_cache, "paths.json")))
    print(f"image cache: {imgs.shape}, {len(img_map):,} paths", flush=True)

    vit = build_vit(device)
    head = nn.Linear(768, NUM_CLASSES).to(device)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    def batch_to_gpu(rows):
        x = torch.from_numpy(np.ascontiguousarray(imgs[rows])).to(device).float() / 255.0
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        return (x - mean) / std

    if not args.extract_only:
        rows, labels, splits = image_level_split(args.index, img_map)
        tr = np.where(splits == "train")[0]
        va = np.where(splits == "val")[0]
        print(f"fine-tune on {len(tr):,} train images; {len(va):,} val images "
              f"(patient-level split, test never touched)", flush=True)
        y_all = torch.tensor(labels, dtype=torch.long)

        opt = torch.optim.AdamW(list(vit.parameters()) + list(head.parameters()),
                                lr=args.lr, weight_decay=5e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
        best_val, best_state = 0.0, None

        for ep in range(args.epochs):
            vit.train(); head.train()
            perm = np.random.RandomState(ep).permutation(tr)
            t0, seen, correct = time.time(), 0, 0
            for i in range(0, len(perm), args.batch):
                sel = perm[i:i + args.batch]
                x = batch_to_gpu(rows[sel]); y = y_all[sel].to(device)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits = head(vit(x))
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                seen += len(sel); correct += (logits.argmax(1) == y).sum().item()
                if (i // args.batch) % 200 == 0:
                    print(f"  ep{ep} {seen:,}/{len(perm):,} train_acc={correct/max(1,seen):.3f} "
                          f"({seen/max(1e-9,time.time()-t0):.0f} img/s)", flush=True)
            # validation
            vit.eval(); head.eval(); vc = vn = 0
            with torch.no_grad():
                for i in range(0, len(va), 256):
                    sel = va[i:i + 256]
                    x = batch_to_gpu(rows[sel]); y = y_all[sel].to(device)
                    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                        p = head(vit(x)).argmax(1)
                    vc += (p == y).sum().item(); vn += len(sel)
            vacc = vc / max(1, vn)
            print(f"epoch {ep}: train_acc={correct/max(1,seen):.4f} val_acc={vacc:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
            if vacc > best_val:
                best_val = vacc
                best_state = {k: v.detach().cpu().clone() for k, v in vit.state_dict().items()}
        if best_state:
            vit.load_state_dict(best_state)
        os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
        torch.save({"vit": vit.state_dict(), "val_acc": best_val}, args.ckpt)
        print(f"saved fine-tuned backbone (best val_acc={best_val:.4f}) -> {args.ckpt}", flush=True)
    else:
        ck = torch.load(args.ckpt, map_location=device)
        vit.load_state_dict(ck["vit"])
        print(f"loaded fine-tuned backbone (val_acc={ck.get('val_acc')})", flush=True)

    # ---- re-extract embeddings from the fine-tuned backbone ----
    os.makedirs(args.out_cache, exist_ok=True)
    vit.eval()
    inv = {v: k for k, v in img_map.items()}
    n = len(img_map)
    out = np.lib.format.open_memmap(os.path.join(args.out_cache, "embeddings.f16.npy"),
                                    mode="w+", dtype=np.float16, shape=(n, 768))
    with torch.no_grad():
        for i in range(0, n, 256):
            rows_i = np.arange(i, min(i + 256, n))
            x = batch_to_gpu(rows_i)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                f = vit(x)
            out[rows_i] = f.float().cpu().numpy().astype(np.float16)
            if i % 25600 == 0:
                print(f"  extract {i:,}/{n:,}", flush=True)
    out.flush()
    json.dump(img_map, open(os.path.join(args.out_cache, "paths.json"), "w"))
    print(f"DONE: fine-tuned embeddings -> {args.out_cache}", flush=True)


if __name__ == "__main__":
    main()
