"""STREAMING full-EMBED feature extraction — solves the 1.8 TB disk problem.

The full 6-group EMBED index references ~363k DICOMs (~1.8 TB), but the frozen-ViT cache
we actually need is only ~558 MB. So we never store the images: for each chunk we
  download -> decode -> frozen ViT -> append embedding -> DELETE the DICOMs -> next chunk.
Peak disk stays a few GB. Fully resumable: already-cached paths are skipped on restart.

Run on the GPU box once EMBED S3 access is restored:
  python -m dro_hetero_anchors.src.stream_embed_features \
      --index datasets/embed/index_xenia_6group.parquet \
      --out datasets/embed/vit_cache_full --chunk 2000 --workers 16

Then train locally on the cache (minutes) with train_embed_xenia.py.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch

BUCKET = "s3://embed-dataset-open"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def all_paths(index_path):
    idx = pd.read_parquet(index_path)
    seen = {}
    for d in idx["paths"]:
        dd = d if isinstance(d, dict) else dict(d)
        for p in dd.values():
            if isinstance(p, str):
                seen[p] = True
    return sorted(seen)


def build_vit(device):
    import timm
    m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def fetch_one(rel, tmpdir, profile=None):
    """Download a single key from S3 into tmpdir. Returns local path or None."""
    dst = os.path.join(tmpdir, rel.replace("/", "__"))
    cmd = ["aws", "s3", "cp", f"{BUCKET}/{rel}", dst, "--region", "us-west-2", "--quiet"]
    if profile:
        cmd += ["--profile", profile]
    r = subprocess.run(cmd, capture_output=True)
    return dst if r.returncode == 0 and os.path.exists(dst) else None


def decode(path, size=224):
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    from PIL import Image
    ds = pydicom.dcmread(path)
    arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size)),
                      dtype=np.float32) / 255.0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="datasets/embed/index_xenia_6group.parquet")
    ap.add_argument("--out", default="datasets/embed/vit_cache_full")
    ap.add_argument("--chunk", type=int, default=2000, help="images downloaded per chunk")
    ap.add_argument("--workers", type=int, default=16, help="parallel S3 downloads")
    ap.add_argument("--batch", type=int, default=128, help="ViT batch size")
    ap.add_argument("--tmp", default="/tmp/embed_stream")
    ap.add_argument("--profile", default="emory-embed", help="AWS CLI profile for EMBED")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    emb_path = os.path.join(args.out, "embeddings.f16.npy")
    map_path = os.path.join(args.out, "paths.json")

    done, embs = {}, []
    if os.path.exists(map_path) and os.path.exists(emb_path):
        done = json.load(open(map_path))
        embs = list(np.load(emb_path).astype(np.float16))
        print(f"resuming: {len(done)} already cached", flush=True)

    rels = all_paths(args.index)
    todo = [r for r in rels if r not in done]
    print(f"total referenced: {len(rels)}   to fetch: {len(todo)}   device={device}", flush=True)
    if not todo:
        print("nothing to do — cache complete"); return

    vit = build_vit(device)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    for ci in range(0, len(todo), args.chunk):
        chunk = todo[ci:ci + args.chunk]
        shutil.rmtree(args.tmp, ignore_errors=True)
        os.makedirs(args.tmp, exist_ok=True)
        # 1) parallel download
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            local = list(ex.map(lambda r: (r, fetch_one(r, args.tmp, args.profile)), chunk))
        got = [(r, p) for r, p in local if p]
        # 2) decode + ViT in batches
        buf_img, buf_rel = [], []

        def flush():
            if not buf_img:
                return
            x = torch.from_numpy(np.stack(buf_img)).to(device).unsqueeze(1).repeat(1, 3, 1, 1)
            x = (x - mean) / std
            f = vit(x).float().cpu().numpy().astype(np.float16)
            for r, v in zip(buf_rel, f):
                done[r] = len(embs); embs.append(v)
            buf_img.clear(); buf_rel.clear()

        for rel, path in got:
            try:
                buf_img.append(decode(path)); buf_rel.append(rel)
            except Exception:
                continue
            if len(buf_img) >= args.batch:
                flush()
        flush()
        # 3) delete the DICOMs — disk never grows
        shutil.rmtree(args.tmp, ignore_errors=True)
        # 4) checkpoint
        np.save(emb_path, np.stack(embs).astype(np.float16))
        json.dump(done, open(map_path, "w"))
        pct = 100.0 * (ci + len(chunk)) / len(todo)
        print(f"  [{pct:5.1f}%] chunk {ci//args.chunk+1}: fetched {len(got)}/{len(chunk)}, "
              f"cached total {len(embs)}", flush=True)

    print(f"DONE: {len(embs)} embeddings -> {emb_path} "
          f"({os.path.getsize(emb_path)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
