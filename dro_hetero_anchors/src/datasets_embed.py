"""EMBED mammography dataset loader — modality-availability heterogeneity.

Benchmark target: the REMIND paper (arXiv 2603.00046), which frames medical
high-modality learning under missingness as a long-tailed problem over MODALITY
COMBINATIONS and runs GroupDRO on top. We map our method (per-group encoders →
shared latent + Gaussian anchors + GroupDRO) onto the same setting.

Task: breast-density classification (BI-RADS density A/B/C/D → 4 classes).
Groups: defined by WHICH imaging modalities/views an exam has. The 4 canonical
"modalities" (following REMIND's EMBED setup) are the crossing of:
    image type  ∈ {FFDM (full-field digital mammography, "2D"), C-View (synthetic 2D)}
    projection  ∈ {CC (cranio-caudal), MLO (medio-lateral oblique)}
  → FFDM_CC, FFDM_MLO, CVIEW_CC, CVIEW_MLO   (indices 0..3)
An exam-breast that is missing some of these views belongs to a smaller modality
combination; rare combinations (< tail_threshold frequency) are "tail" groups.

Data: EMBED AWS Open Data 20% subset (bucket s3://embed-dataset-open, us-west-2).
  tables/EMBED_OpenData_metadata_reduced.csv  → one row per image (view info + path)
  tables/EMBED_OpenData_clinical.csv          → one row per finding (density label)
  images/cohort_{1,2}/<patient>/<study>/<sop>.dcm

────────────────────────────────────────────────────────────────────────────────
!!! BLIND SCAFFOLD (written 2026-07-21 before we had download access) !!!
Every assumption about column names / value tokens is centralized in the CONFIG
block below and tagged `# VERIFY`. Once the real CSVs are downloaded, run
    python -m dro_hetero_anchors.src.datasets_embed --inspect
to print the actual columns / value counts, then fix any mismatches in one place.
────────────────────────────────────────────────────────────────────────────────
"""

from typing import Tuple, List, Optional, Dict
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

# ── Schema (VERIFIED 2026-07-31 against Emory-HITI/EMBED_Open_Data notebooks) ──
# Confirmed via resources/notebooks/{Sample_Notebook,Data_Curation,DCM_to_PNG}.ipynb
# and resources/AWS_Open_Data_Clinical_Legend.csv in the official repo. The column
# names and value tokens below match their published curation code exactly.
# Metadata CSV (one row per image)
COL_PATH        = "anon_dicom_path"        # relative dicom path under images/
COL_IMAGE_TYPE  = "FinalImageType"         # values "2D" (FFDM) vs "cview" (synthetic)
COL_VIEW        = "ViewPosition"           # "CC" / "MLO" (their curation: .isin(["CC","MLO"]))
COL_LATERALITY  = "ImageLateralityFinal"   # "L" / "R"
COL_EMPI        = "empi_anon"              # patient id (both tables)
COL_ACC         = "acc_anon"               # exam/accession id (both tables)
# Clinical CSV (one row per finding)
COL_SIDE        = "side"                    # VERIFY: laterality of finding (legend lists "bside"/"side")
COL_DENSITY     = "tissueden"              # 1..4 → BI-RADS A..D; 5="Normal male", NaN → dropped

# Exact value tokens (their official filters use pandas .isin with these literals:
#   FinalImageType.isin(['2D','cview'])  and  ViewPosition.isin(['CC','MLO']) ).
# We match case-insensitively on the stripped string but require EXACT equality so
# lateral/spot-mag views (XCCL, XCCM, spot mag, etc.) are correctly excluded.
FFDM_TOKENS  = ("2d",)                      # FinalImageType == "2D"
CVIEW_TOKENS = ("cview",)                   # FinalImageType == "cview"
CC_TOKENS    = ("cc",)                      # ViewPosition == "CC"
MLO_TOKENS   = ("mlo",)                     # ViewPosition == "MLO"

# The 4 canonical view-types (order defines the modality mask bit positions)
VIEW_TYPES = ["FFDM_CC", "FFDM_MLO", "CVIEW_CC", "CVIEW_MLO"]
NUM_VIEW_TYPES = len(VIEW_TYPES)
NUM_CLASSES = 4  # BI-RADS density A/B/C/D

# Density value → class index. tissueden 1=fat(A), 2=scattered(B), 3=hetero(C),
# 4=extremely dense(D); 5="Normal male" and NaN are excluded (not mapped).
DENSITY_MAP = {1: 0, 2: 1, 3: 2, 4: 3}


# ── View-type classification ──────────────────────────────────────────────────

def _classify_view_type(image_type: str, view: str) -> Optional[int]:
    """Map (FinalImageType, ViewPosition) → index in VIEW_TYPES, or None if not a
    canonical CC/MLO view we model. Uses EXACT (case-insensitive) token equality to
    match EMBED's official curation (FinalImageType.isin(['2D','cview']),
    ViewPosition.isin(['CC','MLO'])) — so lateral/spot views like XCCL are excluded."""
    it = str(image_type).strip().lower()
    vw = str(view).strip().lower()
    is_cview = it in CVIEW_TOKENS
    is_ffdm = it in FFDM_TOKENS
    if not (is_cview or is_ffdm):
        return None
    is_cc = vw in CC_TOKENS
    is_mlo = vw in MLO_TOKENS
    if not (is_cc or is_mlo):
        return None
    name = ("CVIEW_" if is_cview else "FFDM_") + ("CC" if is_cc else "MLO")
    return VIEW_TYPES.index(name)


def _norm_side(x) -> Optional[str]:
    s = str(x).strip().upper()
    return s if s in ("L", "R") else None


# ── Build the exam-breast table ────────────────────────────────────────────────

def build_embed_index(
    data_root: str,
    tail_threshold: float = 0.15,
    metadata_file: str = "EMBED_OpenData_metadata_reduced.csv",
    clinical_file: str = "EMBED_OpenData_clinical.csv",
    require_min_views: int = 1,
) -> Tuple[pd.DataFrame, Dict]:
    """Join metadata + clinical, produce one row per (exam, breast side) with:
        empi, acc, side, density label, present-view mask, per-view dicom path, group id.

    A sample's GROUP is its modality-combination (which of the 4 view-types are
    present). Combinations with frequency < tail_threshold are flagged tail.
    """
    tables = os.path.join(data_root, "tables")
    meta = pd.read_csv(os.path.join(tables, metadata_file), low_memory=False)
    clin = pd.read_csv(os.path.join(tables, clinical_file), low_memory=False)

    # --- density label per (acc, side) from clinical ---
    # Density is recorded PER-EXAM, not per-breast. VERIFIED on real data (2026-08):
    # every acc_anon has exactly ONE tissueden value; the clinical `side` column is
    # ~48% NaN because it marks per-FINDING laterality, not density. So we key density
    # on acc_anon alone and broadcast it to both breasts of the exam.
    clin = clin[[COL_ACC, COL_DENSITY]].copy()
    clin["density_cls"] = clin[COL_DENSITY].map(DENSITY_MAP)
    clin = clin.dropna(subset=["density_cls"])
    clin["density_cls"] = clin["density_cls"].astype(int)
    dens = (clin.groupby(COL_ACC)["density_cls"]
            .agg(lambda s: int(s.mode().iloc[0])).reset_index())

    # --- per-image view type + side from metadata ---
    meta = meta.copy()
    meta["view_type"] = [
        _classify_view_type(it, vw)
        for it, vw in zip(meta[COL_IMAGE_TYPE], meta[COL_VIEW])
    ]
    meta["side_norm"] = meta[COL_LATERALITY].map(_norm_side)
    meta = meta.dropna(subset=["view_type", "side_norm"])
    meta["view_type"] = meta["view_type"].astype(int)

    # For each (acc, side, view_type) keep one representative image path
    meta = meta.sort_values(COL_PATH)
    img = (meta.groupby([COL_ACC, "side_norm", "view_type"], as_index=False)
           .first()[[COL_EMPI, COL_ACC, "side_norm", "view_type", COL_PATH]])

    # Assemble rows: one per (acc, side)
    rows = []
    for (acc, side), grp in img.groupby([COL_ACC, "side_norm"]):
        paths = [None] * NUM_VIEW_TYPES
        mask = [0] * NUM_VIEW_TYPES
        for _, r in grp.iterrows():
            vt = int(r["view_type"])
            paths[vt] = r[COL_PATH]
            mask[vt] = 1
        if sum(mask) < require_min_views:
            continue
        rows.append({
            "empi": grp.iloc[0][COL_EMPI], "acc": acc, "side": side,
            "mask": tuple(mask),
            **{f"path_{VIEW_TYPES[i]}": paths[i] for i in range(NUM_VIEW_TYPES)},
        })
    index = pd.DataFrame(rows)

    # Join per-exam density onto each (acc, side) breast sample
    index = index.merge(dens.rename(columns={COL_ACC: "acc"}), on="acc", how="inner")
    index = index.rename(columns={"density_cls": "label"}).reset_index(drop=True)

    # --- assign group ids from modality-combination masks ---
    mask_to_gid: Dict[tuple, int] = {}
    counts = index["mask"].value_counts()
    # Order groups by frequency descending → gid 0 is the most common (head)
    for m in counts.index:
        mask_to_gid[m] = len(mask_to_gid)
    index["group"] = index["mask"].map(mask_to_gid)

    n = len(index)
    group_info = []
    for m, gid in mask_to_gid.items():
        c = int(counts[m])
        views = [VIEW_TYPES[i] for i in range(NUM_VIEW_TYPES) if m[i]]
        group_info.append({
            "gid": gid, "mask": m, "name": "+".join(views),
            "count": c, "freq": c / max(1, n),
            "is_tail": (c / max(1, n)) < tail_threshold,
        })

    meta_out = {
        "num_groups": len(mask_to_gid),
        "num_classes": NUM_CLASSES,
        "n_total": n,
        "tail_threshold": tail_threshold,
        "groups": group_info,
        "group_names": [gi["name"] for gi in sorted(group_info, key=lambda x: x["gid"])],
        "tail_gids": [gi["gid"] for gi in group_info if gi["is_tail"]],
        "head_gids": [gi["gid"] for gi in group_info if not gi["is_tail"]],
    }
    return index, meta_out


def _recompute_groups(index: pd.DataFrame, tail_threshold: float) -> Tuple[pd.DataFrame, Dict]:
    """(Re)assign contiguous modality-combination group ids by descending frequency
    and build the head/tail group_info dict. Used after any row filtering."""
    index = index.copy()
    counts = index["mask"].value_counts()
    mask_to_gid = {m: i for i, m in enumerate(counts.index)}
    index["group"] = index["mask"].map(mask_to_gid)
    n = len(index)
    groups = []
    for m, gid in mask_to_gid.items():
        c = int(counts[m]); views = [VIEW_TYPES[i] for i in range(NUM_VIEW_TYPES) if m[i]]
        groups.append({"gid": gid, "mask": m, "name": "+".join(views),
                       "count": c, "freq": c / max(1, n), "is_tail": (c / max(1, n)) < tail_threshold})
    info = {"num_groups": len(mask_to_gid), "num_classes": NUM_CLASSES, "n_total": n,
            "tail_threshold": tail_threshold, "groups": groups,
            "group_names": [g["name"] for g in sorted(groups, key=lambda x: x["gid"])],
            "tail_gids": [g["gid"] for g in groups if g["is_tail"]],
            "head_gids": [g["gid"] for g in groups if not g["is_tail"]]}
    return index, info


# ── DICOM → tensor ─────────────────────────────────────────────────────────────

def _load_dicom_image(path: str, size: int = 224) -> np.ndarray:
    """Load a mammography DICOM, apply VOI LUT windowing, normalize to [0,1],
    resize to (size, size). Returns a single-channel float32 array."""
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    except ImportError as e:
        raise ImportError("EMBED needs pydicom: pip install pydicom pylibjpeg "
                          "pylibjpeg-libjpeg gdcm") from e
    from PIL import Image
    ds = pydicom.dcmread(path)
    arr = apply_voi_lut(ds.pixel_array, ds)
    arr = arr.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr  # invert so higher = brighter tissue
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    im = Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size))
    return np.asarray(im, dtype=np.float32) / 255.0


class EMBEDDataset(Dataset):
    """Returns (views, mask, y, g):
        views: (NUM_VIEW_TYPES, 3, H, W) — absent views are zeros
        mask:  (NUM_VIEW_TYPES,) float — 1 where a view is present
        y:     density class (0..3)
        g:     modality-combination group id
    ImageNet normalization is applied so pretrained backbones work out of the box.
    """
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, index: pd.DataFrame, images_root: str, size: int = 224,
                 augment: bool = False):
        self.index = index.reset_index(drop=True)
        self.images_root = images_root
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.index)

    def _load_view(self, rel_path) -> torch.Tensor:
        if rel_path is None or (isinstance(rel_path, float) and np.isnan(rel_path)):
            return torch.zeros(3, self.size, self.size)
        full = os.path.join(self.images_root, str(rel_path))
        gray = _load_dicom_image(full, self.size)               # (H, W)
        t = torch.from_numpy(gray).unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        return (t - self.IMAGENET_MEAN) / self.IMAGENET_STD

    def __getitem__(self, i):
        row = self.index.iloc[i]
        mask = torch.tensor(row["mask"], dtype=torch.float32)
        views = torch.stack([
            self._load_view(row[f"path_{VIEW_TYPES[v]}"]) if mask[v] > 0
            else torch.zeros(3, self.size, self.size)
            for v in range(NUM_VIEW_TYPES)
        ], dim=0)  # (V, 3, H, W)
        return views, mask, int(row["label"]), int(row["group"])


def collate_embed(batch):
    views, masks, ys, gs = zip(*batch)
    return (torch.stack(views), torch.stack(masks),
            torch.tensor(ys, dtype=torch.long), torch.tensor(gs, dtype=torch.long))


class StratifiedGroupSampler(Sampler):
    """Maintain group proportions per batch (so tail groups appear).
    Mirrors the NHANES sampler contract."""
    def __init__(self, groups: np.ndarray, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_groups = int(groups.max()) + 1
        self.group_indices = [np.where(groups == g)[0] for g in range(self.num_groups)]
        self.total = len(groups)
        pi = [len(gi) / self.total for gi in self.group_indices]
        # samples per group per batch (>=1 for present groups)
        spg, remaining = [], batch_size
        for g in range(self.num_groups - 1):
            k = max(1, int(round(pi[g] * batch_size)))
            k = min(k, remaining - (self.num_groups - 1 - g))
            spg.append(max(0, k)); remaining -= spg[-1]
        spg.append(max(0, remaining))
        self.samples_per_group = spg

    def __iter__(self):
        pools = [list(gi) for gi in self.group_indices]
        if self.shuffle:
            for p in pools:
                np.random.shuffle(p)
        pos = [0] * self.num_groups
        order = []
        while True:
            batch, done = [], False
            for g in range(self.num_groups):
                k = self.samples_per_group[g]
                if pos[g] + k > len(pools[g]):
                    done = True; break
                batch += pools[g][pos[g]:pos[g] + k]; pos[g] += k
            if done:
                break
            order += batch
        return iter(order)

    def __len__(self):
        return self.total


# ── Public API ─────────────────────────────────────────────────────────────────

def build_embed_loaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    train_frac: float = 0.8,
    data_split_seed: int = 42,
    tail_threshold: float = 0.15,
    stratified: bool = True,
    group_max_train_samples: Optional[List[Optional[int]]] = None,
    require_local_images: bool = False,
    min_group_size: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Build EMBED train/test loaders. Split is patient-level (no exam leakage).

    require_local_images: keep only rows whose present-mask DICOMs exist on disk (for
        prototyping on a small downloaded subset before the full set is on the box).
    min_group_size: drop rows in modality-combos with fewer than this many samples so
        GroupDRO groups are statistically meaningful (the micro-combos are <1% total).
    """
    index, info = build_embed_index(data_root, tail_threshold=tail_threshold)
    # `anon_dicom_path` already includes the "images/" prefix, so join against
    # data_root directly (not data_root/images) → data_root/images/cohort_*/...
    images_root = data_root

    if require_local_images:
        def _all_present_local(row):
            for v in range(NUM_VIEW_TYPES):
                if row["mask"][v]:
                    p = row[f"path_{VIEW_TYPES[v]}"]
                    if not (isinstance(p, str) and os.path.exists(os.path.join(images_root, p))):
                        return False
            return True
        index = index[index.apply(_all_present_local, axis=1)].reset_index(drop=True)
        if len(index) == 0:
            raise RuntimeError("require_local_images: no rows have all their DICOMs on disk")
        index, info = _recompute_groups(index, tail_threshold)
        print(f"[require_local_images] kept {len(index)} breast-samples across "
              f"{info['num_groups']} groups present on disk")

    if min_group_size > 0:
        counts = index["mask"].value_counts()
        keep_masks = set(counts[counts >= min_group_size].index)
        dropped = len(index) - int(index["mask"].isin(keep_masks).sum())
        index = index[index["mask"].isin(keep_masks)].reset_index(drop=True)
        index, info = _recompute_groups(index, tail_threshold)
        print(f"[min_group_size={min_group_size}] dropped {dropped} rows in tiny combos "
              f"-> {info['num_groups']} groups, {len(index)} samples")

    # Patient-level split to avoid leakage across train/test
    rng = np.random.default_rng(data_split_seed)
    patients = index["empi"].unique()
    rng.shuffle(patients)
    n_train = int(len(patients) * train_frac)
    train_pat = set(patients[:n_train])
    is_train = index["empi"].isin(train_pat).values

    train_idx = index[is_train].reset_index(drop=True)
    test_idx = index[~is_train].reset_index(drop=True)

    # Optional per-group training caps (imbalance experiments)
    if group_max_train_samples:
        keep = []
        for g in range(info["num_groups"]):
            gi = np.where(train_idx["group"].values == g)[0]
            cap = group_max_train_samples[g] if g < len(group_max_train_samples) else None
            if cap is not None and len(gi) > cap:
                gi = rng.choice(gi, size=cap, replace=False)
            keep.extend(gi.tolist())
        train_idx = train_idx.iloc[sorted(keep)].reset_index(drop=True)

    train_ds = EMBEDDataset(train_idx, images_root, size=image_size, augment=True)
    test_ds = EMBEDDataset(test_idx, images_root, size=image_size, augment=False)

    def _counts(df):
        return [int((df["group"].values == g).sum()) for g in range(info["num_groups"])]
    train_label_counts = [int((train_idx["label"].values == c).sum())
                          for c in range(info["num_classes"])]
    info = {**info,
            "train_total": len(train_idx), "test_total": len(test_idx),
            "train_group_counts": _counts(train_idx),
            "test_group_counts": _counts(test_idx),
            "train_label_counts": train_label_counts,
            "image_size": image_size}

    if stratified:
        sampler = StratifiedGroupSampler(train_idx["group"].values, batch_size)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, collate_fn=collate_embed,
                                  pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, collate_fn=collate_embed,
                                  pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_embed,
                             pin_memory=True)
    return train_loader, test_loader, info


def print_embed_summary(info: Dict):
    print("=" * 68)
    print("EMBED Dataset Summary (modality-combination groups)")
    print("=" * 68)
    print(f"Groups: {info['num_groups']} | Classes: {info['num_classes']} (density A-D)")
    print(f"Total exam-breasts: {info['n_total']} | "
          f"train={info.get('train_total','?')} test={info.get('test_total','?')}")
    print(f"Tail threshold: {info['tail_threshold']:.0%}")
    print("-" * 68)
    for gi in sorted(info["groups"], key=lambda x: x["gid"]):
        tag = "TAIL" if gi["is_tail"] else "head"
        print(f"  G{gi['gid']:2d} [{tag}] {gi['freq']:6.1%} n={gi['count']:6d}  {gi['name']}")
    print("=" * 68)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "datasets", "embed"))
    ap.add_argument("--inspect", action="store_true",
                    help="Print real CSV columns / value counts to VERIFY schema assumptions")
    args = ap.parse_args()
    root = os.path.abspath(args.data_root)

    if args.inspect:
        tables = os.path.join(root, "tables")
        for f in ["EMBED_OpenData_metadata_reduced.csv", "EMBED_OpenData_clinical.csv"]:
            p = os.path.join(tables, f)
            if not os.path.isfile(p):
                print(f"[missing] {p}"); continue
            df = pd.read_csv(p, nrows=5000, low_memory=False)
            print(f"\n=== {f} ===\ncolumns: {list(df.columns)}")
            for c in [COL_IMAGE_TYPE, COL_VIEW, COL_LATERALITY, COL_SIDE, COL_DENSITY]:
                if c in df.columns:
                    print(f"  {c} values: {df[c].value_counts(dropna=False).head(10).to_dict()}")
    else:
        index, info = build_embed_index(root)
        print_embed_summary(info)
