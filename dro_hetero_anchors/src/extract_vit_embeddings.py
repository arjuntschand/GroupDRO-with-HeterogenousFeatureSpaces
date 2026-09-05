"""ONE-TIME (runs on the GPU box): encode every EMBED image referenced by the
6-group index with a FROZEN ViT-Base and cache its 768-d CLS embedding.

Output (portable, ~unique_images * 768 * 2 bytes ~= tens of MB):
  <out>/embeddings.f16.npy   (N, 768) float16
  <out>/paths.json           {relative_dicom_path -> row index}

Everything downstream (model, R*_g, all 7 methods) then runs on these cached
vectors on any machine — no GPU, no DICOMs. Resumable: re-running skips rows
already present in paths.json.

Usage (on box):
  python -m dro_hetero_anchors.src.extract_vit_embeddings \
      --index datasets/embed/index_xenia_6group.parquet \
      --images-root /data/embed --out datasets/embed/vit_cache --batch 64
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
import torch

from .datasets_embed import _load_dicom_image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def collect_paths(index_path: str) -> list:
    idx = pd.read_parquet(index_path)
    seen = {}
    for paths in idx["paths"]:
        # parquet may store the dict as a python dict already
        d = paths if isinstance(paths, dict) else dict(paths)
        for _view, rel in d.items():
            if isinstance(rel, str):
                seen[rel] = True
    return sorted(seen.keys())


def build_vit(device: str):
    import timm
    m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="datasets/embed/index_xenia_6group.parquet")
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out", default="datasets/embed/vit_cache")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    emb_path = os.path.join(args.out, "embeddings.f16.npy")
    map_path = os.path.join(args.out, "paths.json")

    rels = collect_paths(args.index)
    print(f"unique images referenced: {len(rels)}  device={device}")

    # resume: load existing
    done = {}
    embs = []
    if os.path.exists(map_path) and os.path.exists(emb_path):
        done = json.load(open(map_path))
        embs = list(np.load(emb_path).astype(np.float16))
        print(f"resuming: {len(done)} already cached")

    todo = [r for r in rels if r not in done]
    print(f"to encode: {len(todo)}")
    vit = build_vit(device)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    buf_imgs, buf_rels = [], []

    def flush():
        if not buf_imgs:
            return
        x = torch.from_numpy(np.stack(buf_imgs)).to(device)      # (B,224,224) gray [0,1]
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)                    # 3-channel
        x = (x - mean) / std
        feat = vit(x).float().cpu().numpy().astype(np.float16)   # (B,768) CLS
        for r, f in zip(buf_rels, feat):
            done[r] = len(embs)
            embs.append(f)
        buf_imgs.clear(); buf_rels.clear()

    for i, rel in enumerate(todo):
        full = os.path.join(args.images_root, rel)
        try:
            img = _load_dicom_image(full, size=args.size)       # (224,224) float32 [0,1]
        except Exception as e:
            print(f"  SKIP {rel}: {e}")
            continue
        buf_imgs.append(img); buf_rels.append(rel)
        if len(buf_imgs) >= args.batch:
            flush()
        if (i + 1) % 2000 == 0:
            np.save(emb_path, np.stack(embs).astype(np.float16))
            json.dump(done, open(map_path, "w"))
            print(f"  {i+1}/{len(todo)} encoded, {len(embs)} cached")
    flush()
    np.save(emb_path, np.stack(embs).astype(np.float16))
    json.dump(done, open(map_path, "w"))
    print(f"DONE: {len(embs)} embeddings -> {emb_path}  ({os.path.getsize(emb_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
