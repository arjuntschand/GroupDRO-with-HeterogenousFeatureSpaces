"""Selective downloader for EMBED DICOMs referenced by our built index.

We never need the full ~1.9 TB image set at once — only the images our index
actually references. This fetches exactly those keys from s3://embed-dataset-open.

Two uses:
  * Prototype (small):   --per-group 12 --max-images 150      → a few hundred MB
  * Full run (on EC2):   (no caps)                            → everything in index

Paths in the index (`path_*` columns) are already full S3 keys under the bucket
(e.g. "images/cohort_2/.../x.dcm"); local dest mirrors that under --data-root.

Example:
  python -m dro_hetero_anchors.src.download_embed \
      --index datasets/embed/index_perbreast.parquet \
      --data-root datasets/embed --per-group 12 --max-images 150 --workers 8
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .datasets_embed import VIEW_TYPES

BUCKET = "s3://embed-dataset-open"
REGION = "us-west-2"
PATH_COLS = [f"path_{v}" for v in VIEW_TYPES]


def collect_keys(index: pd.DataFrame, per_group: int | None, seed: int = 0) -> list:
    """Pick a (optionally per-group-capped) set of breast rows, return unique DICOM keys."""
    if per_group is not None:
        parts = []
        for g, grp in index.groupby("group"):
            n = min(per_group, len(grp))
            parts.append(grp.sample(n=n, random_state=seed))
        index = pd.concat(parts, ignore_index=True)
    keys = set()
    for col in PATH_COLS:
        keys.update(index[col].dropna().tolist())
    return sorted(keys)


def _fetch(key: str, data_root: str, aws: str) -> tuple:
    dest = os.path.join(data_root, key)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return key, "cached"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    r = subprocess.run(
        [aws, "s3", "cp", f"{BUCKET}/{key}", dest, "--region", REGION, "--quiet"],
        capture_output=True, text=True,
    )
    return key, ("ok" if r.returncode == 0 else f"ERR {r.stderr.strip()[:80]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="datasets/embed/index_perbreast.parquet")
    ap.add_argument("--data-root", default="datasets/embed")
    ap.add_argument("--per-group", type=int, default=None,
                    help="cap breasts sampled per modality group (prototype)")
    ap.add_argument("--max-images", type=int, default=None,
                    help="hard cap on total images downloaded (prototype safety)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aws", default=".venv/bin/aws")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    index = pd.read_parquet(args.index)
    keys = collect_keys(index, args.per_group, args.seed)
    if args.max_images is not None:
        keys = keys[: args.max_images]
    print(f"[download] {len(keys)} unique DICOM keys selected "
          f"(per_group={args.per_group}, max_images={args.max_images})")
    if args.dry_run:
        for k in keys[:10]:
            print("  ", k)
        print("  ... (dry run, nothing downloaded)")
        return

    ok = cached = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_fetch, k, args.data_root, args.aws): k for k in keys}
        for i, fut in enumerate(as_completed(futs), 1):
            key, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "cached":
                cached += 1
            else:
                err += 1
                print(f"  [{i}/{len(keys)}] {status}  {key}", file=sys.stderr)
            if i % 25 == 0 or i == len(keys):
                print(f"  progress {i}/{len(keys)}  ok={ok} cached={cached} err={err}")
    print(f"[download] done: ok={ok} cached={cached} err={err} -> {args.data_root}")


if __name__ == "__main__":
    main()
