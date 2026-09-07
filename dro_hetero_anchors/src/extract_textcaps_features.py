"""ONE-TIME (GPU box): cache FROZEN ResNet18 visual features for the FULL HF TextCaps
dataset, so the anchor sweep trains on cached features (fast, GPU-efficient, full data) —
consistent with the paper's frozen-backbone design (cf. EMBED frozen ViT).

Caches, per split (train/val):
  <out>/<split>_visual.f16.npy   (N, 512) frozen ResNet18 features
  <out>/<split>_meta.json        [{caption, label, image_idx}, ...]
  <out>/classes.json             class_to_idx (top-k most frequent image_classes[0])

The text modality stays raw (caption string) — CharCNN is cheap to train on the fly.

Usage:
  python -m dro_hetero_anchors.src.extract_textcaps_features --inspect
  python -m dro_hetero_anchors.src.extract_textcaps_features --num-classes 10 --out datasets/textcaps/feat_cache
"""
from __future__ import annotations
import argparse, json, os
from collections import Counter
import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def caption_of(item):
    for k in ("caption", "caption_str", "reference_strs", "captions"):
        v = item.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, (list, tuple)) and v:
            return str(v[0])
    return ""


def label_name_of(item):
    ic = item.get("image_classes") or []
    return ic[0] if ic else None


def build_resnet(device):
    import torchvision
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def preprocess(img, size=224):
    from PIL import Image
    img = img.convert("RGB").resize((size, size))
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return x


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--out", default="datasets/textcaps/feat_cache")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("lmms-lab/TextCaps")
    if args.inspect:
        it = ds["train"][0]
        print("KEYS:", list(it.keys()))
        for k, v in it.items():
            s = str(v)[:80] if not hasattr(v, "size") else f"<image {getattr(v,'size',None)}>"
            print(f"  {k}: {s}")
        print("caption_of ->", caption_of(it)[:80])
        print("label_name_of ->", label_name_of(it))
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    # class map from top-k most frequent primary class on train
    cnt = Counter()
    for it in ds["train"]:
        n = label_name_of(it)
        if n:
            cnt[n] += 1
    top = [c for c, _ in cnt.most_common(args.num_classes)]
    class_to_idx = {c: i for i, c in enumerate(top)}
    json.dump(class_to_idx, open(f"{args.out}/classes.json", "w"))
    print(f"classes ({len(class_to_idx)}): {top}")

    resnet = build_resnet(device)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    for split in ["train", "val"]:
        feats, meta = [], []
        buf, buf_meta = [], []

        def flush():
            if not buf:
                return
            x = torch.stack(buf).to(device)
            x = (x - mean) / std
            f = resnet(x).float().cpu().numpy().astype(np.float16)
            for row, m in zip(f, buf_meta):
                m["row"] = len(feats)
                feats.append(row); meta.append(m)
            buf.clear(); buf_meta.clear()

        for i, it in enumerate(ds[split]):
            name = label_name_of(it)
            if name not in class_to_idx:
                continue
            try:
                buf.append(preprocess(it["image"]))
            except Exception:
                continue
            buf_meta.append({"caption": caption_of(it), "label": class_to_idx[name]})
            if len(buf) >= args.batch:
                flush()
            if (i + 1) % 5000 == 0:
                print(f"  {split}: {i+1} scanned, {len(feats)} cached")
        flush()
        np.save(f"{args.out}/{split}_visual.f16.npy", np.stack(feats).astype(np.float16))
        json.dump(meta, open(f"{args.out}/{split}_meta.json", "w"))
        print(f"{split}: cached {len(feats)} images -> {args.out}/{split}_visual.f16.npy")


if __name__ == "__main__":
    main()
