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
import argparse, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch

BUCKET = "s3://embed-dataset-open"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def all_paths(index_path, priority_by_group=True):
    """All image paths referenced by the index.

    With priority_by_group (default), paths are ordered SMALLEST GROUP FIRST. This matters:
    the tail groups are what the method is actually about, and together they need only a
    few hundred images, while the two head groups account for >99% of the download. Ordering
    this way means the scientifically critical data is complete within minutes, and because
    the extractor checkpoints every chunk you can stop at any point with a usable cache
    (all tails + however much head data you have time for).
    """
    idx = pd.read_parquet(index_path)

    def imgs_of(row):
        d = row if isinstance(row, dict) else dict(row)
        return [p for p in d.values() if isinstance(p, str)]

    if not priority_by_group or "group" not in idx.columns:
        seen = {}
        for d in idx["paths"]:
            for p in imgs_of(d):
                seen[p] = True
        return sorted(seen)

    order = idx.group.value_counts().sort_values().index.tolist()   # smallest group first
    out, seen = [], set()
    for g in order:
        for d in idx[idx.group == g]["paths"]:
            for p in imgs_of(d):
                if p not in seen:
                    seen.add(p); out.append(p)
    return out


def build_vit(device):
    import timm
    m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


_S3 = None
BUCKET_NAME = "embed-dataset-open"


def get_s3(profile=None, pool=64):
    """One shared boto3 client with a large connection pool. MUCH faster than spawning an
    `aws s3 cp` process per file (process+auth+TLS setup per object was the bottleneck:
    ~6k images/hour vs ~100k+/hour here)."""
    global _S3
    if _S3 is None:
        import boto3
        from botocore.config import Config
        sess = boto3.Session(profile_name=profile) if profile else boto3.Session()
        _S3 = sess.client("s3", region_name="us-west-2",
                          config=Config(max_pool_connections=pool,
                                        retries={"max_attempts": 3, "mode": "standard"}))
    return _S3


def fetch_bytes(rel, profile=None):
    """Download a key straight into memory (no disk write at all)."""
    try:
        obj = get_s3(profile).get_object(Bucket=BUCKET_NAME, Key=rel)
        return obj["Body"].read()
    except Exception:
        return None


def decode(src, size=224):
    """src: filesystem path OR raw DICOM bytes."""
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    from PIL import Image
    if isinstance(src, (bytes, bytearray)):
        import io as _io
        ds = pydicom.dcmread(_io.BytesIO(src))
    else:
        ds = pydicom.dcmread(src)
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
    ap.add_argument("--no-priority", action="store_true",
                    help="do not order tail groups first (default is tails first)")
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

    rels = all_paths(args.index, priority_by_group=not args.no_priority)
    todo = [r for r in rels if r not in done]
    print(f"total referenced: {len(rels)}   to fetch: {len(todo)}   device={device}", flush=True)
    if not todo:
        print("nothing to do — cache complete"); return

    vit = build_vit(device)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    get_s3(args.profile, pool=max(64, args.workers * 2))   # warm the shared client
    t0 = time.time()

    def fetch_and_decode(rel):
        """download bytes -> decode, entirely in memory (never touches disk)."""
        b = fetch_bytes(rel, args.profile)
        if b is None:
            return None
        try:
            return (rel, decode(b))
        except Exception:
            return None

    for ci in range(0, len(todo), args.chunk):
        chunk = todo[ci:ci + args.chunk]
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

        ok = 0
        # download+decode in parallel, feed the GPU as results stream in
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(fetch_and_decode, chunk):
                if res is None:
                    continue
                ok += 1
                buf_rel.append(res[0]); buf_img.append(res[1])
                if len(buf_img) >= args.batch:
                    flush()
        flush()
        # checkpoint (resumable)
        np.save(emb_path, np.stack(embs).astype(np.float16))
        json.dump(done, open(map_path, "w"))
        n_done = ci + len(chunk)
        pct = 100.0 * n_done / len(todo)
        rate = n_done / max(1e-9, time.time() - t0)
        eta = (len(todo) - n_done) / max(1e-9, rate) / 3600
        print(f"  [{pct:5.1f}%] chunk {ci//args.chunk+1}: {ok}/{len(chunk)} ok, "
              f"cached {len(embs)}, {rate:.0f} img/s, ETA {eta:.1f}h", flush=True)

    print(f"DONE: {len(embs)} embeddings -> {emb_path} "
          f"({os.path.getsize(emb_path)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
