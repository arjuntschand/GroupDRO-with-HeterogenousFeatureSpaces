"""Merge the per-machine image-cache shards into one array.

The image cache is produced by several machines at once (the GPU box plus a few CPU-only
workers), each handling a disjoint stride of the same ordered path list. Every machine
writes its own images.u8.npy and paths.json. This pulls them together into a single
memory-mapped array with one unified path -> row map, deduplicating by path.

Usage (on the GPU box, after the workers have uploaded to S3):
  python -m dro_hetero_anchors.src.merge_image_shards \
      --local datasets/embed/img_cache \
      --s3 s3://YOUR-WORK-BUCKET/shards \
      --out datasets/embed/img_cache_merged
"""
from __future__ import annotations
import argparse, json, os, subprocess, tempfile
import numpy as np


def load_shard(npy, pjson):
    if not (os.path.exists(npy) and os.path.exists(pjson)):
        return None, None
    return np.load(npy, mmap_mode="r"), json.load(open(pjson))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default="datasets/embed/img_cache",
                    help="this machine's own shard directory")
    ap.add_argument("--s3", default=None, help="s3 prefix holding the other machines' shards")
    ap.add_argument("--out", default="datasets/embed/img_cache_merged")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--keep-download", action="store_true")
    args = ap.parse_args()

    shards = []
    a, p = load_shard(os.path.join(args.local, "images.u8.npy"),
                      os.path.join(args.local, "paths.json"))
    if a is not None:
        shards.append(("local", a, p))
        print(f"local shard: {len(p):,} paths", flush=True)

    tmp = None
    if args.s3:
        tmp = tempfile.mkdtemp(prefix="shards_")
        subprocess.run(["aws", "s3", "cp", args.s3, tmp, "--recursive",
                        "--region", "us-west-2"], check=False)
        names = sorted({f.split("_")[0] for f in os.listdir(tmp) if f.startswith("shard")})
        for nm in names:
            a, p = load_shard(os.path.join(tmp, f"{nm}_images.npy"),
                              os.path.join(tmp, f"{nm}_paths.json"))
            if a is not None:
                shards.append((nm, a, p))
                print(f"{nm}: {len(p):,} paths", flush=True)

    if not shards:
        print("no shards found"); return

    # unified path -> row, first writer wins (shards are disjoint by construction, so
    # duplicates only occur where a machine's slice was narrowed mid-run)
    merged, total = {}, 0
    for nm, arr, pmap in shards:
        for rel in pmap:
            if rel not in merged:
                merged[rel] = (nm, pmap[rel]); total += 1
    print(f"merged unique paths: {total:,}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    out = np.lib.format.open_memmap(os.path.join(args.out, "images.u8.npy"), mode="w+",
                                    dtype=np.uint8, shape=(total, args.size, args.size))
    by_shard = {nm: (arr, pmap) for nm, arr, pmap in shards}
    final_map = {}
    for i, (rel, (nm, row)) in enumerate(merged.items()):
        out[i] = by_shard[nm][0][row]
        final_map[rel] = i
        if i % 25000 == 0:
            print(f"  copied {i:,}/{total:,}", flush=True)
    out.flush()
    json.dump(final_map, open(os.path.join(args.out, "paths.json"), "w"))
    print(f"DONE: {total:,} images -> {args.out} "
          f"({os.path.getsize(os.path.join(args.out,'images.u8.npy'))/1e9:.1f} GB)", flush=True)
    if tmp and not args.keep_download:
        subprocess.run(["rm", "-rf", tmp], check=False)


if __name__ == "__main__":
    main()
