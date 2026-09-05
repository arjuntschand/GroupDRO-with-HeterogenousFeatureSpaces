"""Build the 6-group EMBED index for Xenia's spec, from the metadata + clinical tables.

Group is assigned from each breast's TRUE view-set (all of full metadata). Density label
is joined PER-EXAM (acc_anon) — the clinical `side` column is per-finding (~48% NaN) and
must not be used for the density join.

Modes:
  --mode full     : all labeled 6-group breast-rows            -> index_xenia_6group.parquet
  --mode offline  : keep only breasts whose ENTIRE true view-set has a cached ViT
                    embedding (present in <cache>/paths.json)  -> index_xenia_offline.parquet

Usage:
  python -m dro_hetero_anchors.src.tools.build_embed_xenia_index --mode offline \
      --data-root datasets/embed --cache datasets/embed/vit_cache
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd

GROUP_OF_VSET = {"M3": "g1", "M1+M3": "g2", "M4": "g3",
                 "M3+M4": "g4", "M1+M3+M4": "g5", "M1+M2+M3+M4": "g6"}
ORDER = ["g1", "g2", "g3", "g4", "g5", "g6"]


def build(data_root: str, mode: str, cache: str):
    meta = pd.read_csv(os.path.join(data_root, "tables", "EMBED_OpenData_metadata_reduced.csv"),
        low_memory=False, usecols=["empi_anon", "acc_anon", "FinalImageType", "ViewPosition",
                                    "ImageLateralityFinal", "spot_mag", "anon_dicom_path"])
    it = meta.FinalImageType.astype(str).str.strip().str.lower()
    vw = meta.ViewPosition.astype(str).str.strip().str.upper()
    lat = meta.ImageLateralityFinal.astype(str).str.strip().str.upper()
    spot = meta.spot_mag.notna() & (meta.spot_mag.astype(str).str.strip() != "") & (meta.spot_mag.astype(str).str.lower() != "nan")
    keep = (~spot) & vw.isin(["CC", "MLO"]) & it.isin(["2d", "cview"]) & lat.isin(["L", "R"])
    m = meta[keep].copy()
    m["Mtype"] = np.where(it[keep].eq("cview") & vw[keep].eq("CC"), "M1",
                 np.where(it[keep].eq("cview") & vw[keep].eq("MLO"), "M2",
                 np.where(it[keep].eq("2d") & vw[keep].eq("CC"), "M3", "M4")))
    m["side"] = lat[keep]
    m = m.sort_values("anon_dicom_path").drop_duplicates(["acc_anon", "side", "Mtype"])
    grp = m.groupby(["acc_anon", "side"])
    gid = grp["Mtype"].apply(lambda s: "+".join(sorted(set(s)))).map(GROUP_OF_VSET)
    paths = grp.apply(lambda d: dict(zip(d.Mtype, d.anon_dicom_path)), include_groups=False)
    empi = grp["empi_anon"].first()
    idx = pd.DataFrame({"group": gid, "empi_anon": empi, "paths": paths}).dropna(subset=["group"]).reset_index()

    if mode == "offline":
        present = set(json.load(open(os.path.join(cache, "paths.json"))).keys())
        idx = idx[idx["paths"].apply(
            lambda d: all(isinstance(p, str) and p in present for p in (d if isinstance(d, dict) else dict(d)).values())
        )].reset_index(drop=True)

    cl = pd.read_csv(os.path.join(data_root, "tables", "EMBED_OpenData_clinical.csv"),
                     low_memory=False, usecols=["acc_anon", "tissueden"])
    cl = cl.dropna(subset=["tissueden"]); cl = cl[cl.tissueden.isin([1, 2, 3, 4])].drop_duplicates(["acc_anon"])
    idx = idx.merge(cl, on="acc_anon", how="inner"); idx["label"] = idx.tissueden.astype(int) - 1

    out = os.path.join(data_root, "index_xenia_offline.parquet" if mode == "offline" else "index_xenia_6group.parquet")
    idx.to_parquet(out)
    print(f"[{mode}] {len(idx)} rows, {idx.empi_anon.nunique()} patients -> {out}")
    print("  per-group:", idx.group.value_counts().reindex(ORDER).fillna(0).astype(int).to_dict())
    print("  labels:", idx.label.value_counts().sort_index().to_dict())
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "offline"], default="offline")
    ap.add_argument("--data-root", default="datasets/embed")
    ap.add_argument("--cache", default="datasets/embed/vit_cache")
    args = ap.parse_args()
    build(args.data_root, args.mode, args.cache)


if __name__ == "__main__":
    main()
