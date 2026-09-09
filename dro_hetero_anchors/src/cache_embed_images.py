"""Cache decoded EMBED images as 224x224 uint8 so the ViT backbone can be FINE-TUNED.

Why this exists. The frozen-backbone pipeline only needs one 768-d vector per image, so
stream_embed_features.py downloads a DICOM, encodes it, and deletes it. Fine-tuning needs
the pixels on every epoch, and the raw DICOMs are 1.7 TB, which does not fit. But the model
only ever sees a 224x224 grayscale crop, and storing those as uint8 costs

    363,098 x 224 x 224 x 1 byte = 18.2 GB

which fits comfortably. So we stream the DICOMs once more, keep the decoded arrays, and
throw the DICOMs away again.

Output:
  <out>/images.u8.npy   (N, 224, 224) uint8, memory-mapped so it never loads fully into RAM
  <out>/paths.json      {relative_dicom_path -> row index}

Resumable: re-running skips paths already present in paths.json.

Usage (on the GPU box):
  python -m dro_hetero_anchors.src.cache_embed_images \
      --index datasets/embed/index_production.parquet \
      --out datasets/embed/img_cache --workers 8
"""
from __future__ import annotations
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# NOTE: deliberately self-contained. An earlier version imported these helpers from
# stream_embed_features, which imports torch at module level; the CPU-only workers do not
# have torch, so every worker died on ImportError and then hit `shutdown` without uploading
# anything. Nothing here needs torch, so the functions are inlined.
import pandas as pd

BUCKET_NAME = "embed-dataset-open"
_S3 = None
_PROFILE = None


def all_paths(index_path, priority_by_group=True):
    """Every image path in the index, ordered smallest-group-first so the scientifically
    important tail groups complete early and the run can be stopped at any point."""
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

    order = idx.group.value_counts().sort_values().index.tolist()
    out, seen = [], set()
    for g in order:
        for d in idx[idx.group == g]["paths"]:
            for p in imgs_of(d):
                if p not in seen:
                    seen.add(p); out.append(p)
    return out


def get_s3(profile=None, pool=64):
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
    try:
        return get_s3(profile).get_object(Bucket=BUCKET_NAME, Key=rel)["Body"].read()
    except Exception:
        return None


def decode(src, size=224):
    """DICOM bytes -> 224x224 uint8, VOI-LUT applied and MONOCHROME1 inverted."""
    import io as _io
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    from PIL import Image
    ds = pydicom.dcmread(_io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
    arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size)),
                      dtype=np.uint8)


def _init(profile):
    global _PROFILE
    _PROFILE = profile
    get_s3(profile, pool=8)


def _fetch_decode(rel):
    b = fetch_bytes(rel, _PROFILE)
    if b is None:
        return None
    try:
        return (rel, decode(b))          # returns uint8 (224,224)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="datasets/embed/index_production.parquet")
    ap.add_argument("--out", default="datasets/embed/img_cache")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--profile", default="emory-embed")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker's shard index; lets several machines split the work")
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img_path = os.path.join(args.out, "images.u8.npy")
    map_path = os.path.join(args.out, "paths.json")

    rels = all_paths(args.index, priority_by_group=True)   # tails first, as before
    if args.num_shards > 1:
        # Deterministic stride so every machine gets an interleaved slice of the same
        # ordering. Each shard writes its own npy + paths.json; they are merged afterwards.
        rels = rels[args.shard::args.num_shards]
        print(f"shard {args.shard}/{args.num_shards}: {len(rels):,} of this machine's images",
              flush=True)
    n_total = len(rels)
    done = json.load(open(map_path)) if os.path.exists(map_path) else {}

    # memory-mapped destination sized for the whole index; rows are filled in as we go
    if os.path.exists(img_path):
        arr = np.lib.format.open_memmap(img_path, mode="r+")
        if arr.shape[0] < n_total:       # index grew; start fresh
            arr = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8,
                                            shape=(n_total, args.size, args.size))
            done = {}
    else:
        arr = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8,
                                        shape=(n_total, args.size, args.size))

    todo = [r for r in rels if r not in done]
    print(f"total {n_total:,}  cached {len(done):,}  to fetch {len(todo):,}  "
          f"({n_total * args.size * args.size / 1e9:.1f} GB when complete)", flush=True)
    if not todo:
        print("nothing to do"); return

    t0 = time.time()
    nxt = len(done)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.profile,)) as pool:
        for ci in range(0, len(todo), args.chunk):
            chunk = todo[ci:ci + args.chunk]
            ok = 0
            for res in pool.map(_fetch_decode, chunk, chunksize=8):
                if res is None:
                    continue
                rel, img = res
                arr[nxt] = img
                done[rel] = nxt
                nxt += 1; ok += 1
            arr.flush()
            json.dump(done, open(map_path, "w"))
            n_done = ci + len(chunk)
            rate = n_done / max(1e-9, time.time() - t0)
            eta = (len(todo) - n_done) / max(1e-9, rate) / 3600
            print(f"  [{100.0*n_done/len(todo):5.1f}%] {ok}/{len(chunk)} ok, "
                  f"stored {nxt:,}, {rate:.0f} img/s, ETA {eta:.1f}h", flush=True)

    arr.flush(); json.dump(done, open(map_path, "w"))
    print(f"DONE: {nxt:,} images -> {img_path} "
          f"({os.path.getsize(img_path)/1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
