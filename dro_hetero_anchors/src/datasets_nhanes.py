"""NHANES CVD dataset loader with natural feature-availability heterogeneity.

NHANES (National Health and Nutrition Examination Survey) participants complete
different assessment components (questionnaire, physical exam, lab work), creating
groups with genuinely different feature spaces:

- Group 0 (survey_only): Demographics + smoking questionnaire (~10 features)
- Group 1 (exam): Survey + body measures (BMI/weight/height) (~13 features)
- Group 2 (vitals_labs): Survey + body + blood pressure + lab values (~20 features)

Task: Binary CVD prediction (coronary heart disease OR heart attack OR stroke).
Data: NHANES 2017-March 2020 Pre-Pandemic + 2021-2023 Post-Pandemic.

Reference: Preprocessing follows Xenia Konti's Colab notebook.
"""

from typing import Tuple, List, Optional, Dict
import os
import urllib.request
import ssl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split

NUM_GROUPS = 3  # default; overridden to 4 in "4group" mode
GROUP_NAMES = ["survey_only", "exam", "vitals_labs"]
GROUP_NAMES_4 = ["survey_only", "exam_labs", "exam_bp", "full_workup"]

# NHANES sentinel/refusal codes → NaN
_SENTINEL_VALUES = {7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999}

# ── XPT file URLs ──────────────────────────────────────────────────────────
# Pre-pandemic (2017-March 2020): P_ prefix
_PRE_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
_PRE_FILES = {
    "demo": "P_DEMO.xpt",
    "bmx": "P_BMX.xpt",
    "bpxo": "P_BPXO.xpt",
    "mcq": "P_MCQ.xpt",
    "smq": "P_SMQ.xpt",
    "ghb": "P_GHB.xpt",
    "hdl": "P_HDL.xpt",
    "tchol": "P_TCHOL.xpt",
    "trigly": "P_TRIGLY.xpt",
    # Additional questionnaire files for expanded features
    "diq": "P_DIQ.xpt",
    "bpq": "P_BPQ.xpt",
    "paq": "P_PAQ.xpt",
}

# Post-pandemic (2021-2023): _L suffix
_POST_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles"
_POST_FILES = {
    "demo": "DEMO_L.xpt",
    "bmx": "BMX_L.xpt",
    "bpxo": "BPXO_L.xpt",
    "mcq": "MCQ_L.xpt",
    "smq": "SMQ_L.xpt",
    "ghb": "GHB_L.xpt",
    "hdl": "HDL_L.xpt",
    "tchol": "TCHOL_L.xpt",
    "trigly": "TRIGLY_L.xpt",
    "diq": "DIQ_L.xpt",
    "bpq": "BPQ_L.xpt",
    "paq": "PAQ_L.xpt",
}

# ── Feature definitions ─────────────────────────────────────────────────────
# Survey features (available for all groups)
_SURVEY_COLS = ["RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR", "SMQ020"]
# Extra questionnaire features (expanded/disjoint modes)
_EXTRA_SURVEY_COLS = ["DIQ010", "BPQ020", "BPQ080", "PAQ605", "PAQ650"]
# Body measure features (G1, G2)
_BMX_COLS = ["BMXBMI", "BMXWT", "BMXHT"]
# Blood pressure features (G2 only) — we compute mean of available readings
_BP_SYS_COLS = ["BPXOSY1", "BPXOSY2", "BPXOSY3"]
_BP_DIA_COLS = ["BPXODI1", "BPXODI2", "BPXODI3"]
# Lab features (G2 only)
_LAB_COLS = ["LBXGH", "LBDHDD", "LBXTC", "LBXTR", "LBDLDL"]
# CVD label columns (from MCQ)
_CVD_COLS = ["MCQ160C", "MCQ160E", "MCQ160F"]

# ── Feature mode configurations ────────────────────────────────────────────
# "nested": Original — G0 ⊂ G1 ⊂ G2 (10/13/20 features)
# "expanded": More survey features, still nested — G0 ⊂ G1 ⊂ G2 (15/18/25 features)
# "disjoint": Each group has unique features — G0 ∩ G1 ∩ G2 = shared only (15/15/15 features)

def _get_feature_config(mode: str = "nested"):
    """Return (group_feat_counts: dict, max_features: int, indices: dict) for the given mode."""
    if mode == "nested":
        gfc = {0: 10, 1: 13, 2: 20}
        indices = {g: list(range(n)) for g, n in gfc.items()}
        return gfc, 20, indices
    elif mode == "expanded":
        gfc = {0: 15, 1: 18, 2: 25}
        indices = {g: list(range(n)) for g, n in gfc.items()}
        return gfc, 25, indices
    elif mode == "disjoint":
        gfc = {0: 15, 1: 15, 2: 15}
        indices = {
            0: list(range(10)) + list(range(10, 15)),
            1: list(range(10)) + list(range(15, 20)),
            2: list(range(10)) + list(range(20, 25)),
        }
        return gfc, 25, indices
    elif mode == "partition":
        # zero-overlap: each group sees only its own block of the disjoint layout
        gfc = {0: 10, 1: 5, 2: 5}
        indices = {0: list(range(0, 10)), 1: list(range(15, 20)), 2: list(range(20, 25))}
        return gfc, 25, indices
    elif mode == "4group":
        # 4 groups with NATURAL non-nesting:
        # G0 (survey_only): expanded questionnaire only = 15 features
        # G1 (exam+labs):   survey + body + labs (NO BP) = 23 features
        # G2 (exam+bp):     survey + body + BP (NO labs) = 20 features
        # G3 (full):        survey + body + BP + labs    = 25 features
        #
        # G1 and G2 are NON-NESTED: G1 has labs, G2 has BP — neither is a subset
        #
        # Layout: [0-14] survey(15), [15-17] body(3), [18-19] BP(2), [20-24] labs(5)
        gfc = {0: 15, 1: 23, 2: 20, 3: 25}
        indices = {
            0: list(range(15)),                          # survey only
            1: list(range(18)) + list(range(20, 25)),   # survey + body + labs (skip BP)
            2: list(range(20)),                          # survey + body + BP (skip labs)
            3: list(range(25)),                          # everything
        }
        return gfc, 25, indices
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

# Default (backward compatible)
FEATURES_G0 = 10
FEATURES_G1 = 13
FEATURES_G2 = 20
MAX_FEATURES = FEATURES_G2
FEATURE_INDICES = {
    0: list(range(FEATURES_G0)),
    1: list(range(FEATURES_G1)),
    2: list(range(FEATURES_G2)),
}


# ── Download helpers ─────────────────────────────────────────────────────────

def _download_xpt(url: str, dest: str):
    """Download a single XPT file from CDC with retry logic."""
    if os.path.isfile(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # CDC may require a browser-like User-Agent
    headers = {"User-Agent": "Mozilla/5.0 (Python/NHANES-Loader)"}
    req = urllib.request.Request(url, headers=headers)
    # Create SSL context that works on macOS
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx) as resp, open(dest, "wb") as f:
            f.write(resp.read())
    except Exception as e:
        # Try without SSL verification as fallback
        ctx = ssl._create_unverified_context()
        try:
            with urllib.request.urlopen(req, context=ctx) as resp, open(dest, "wb") as f:
                f.write(resp.read())
        except Exception as e2:
            raise RuntimeError(
                f"Failed to download {url} → {dest}.\n"
                f"Error: {e2}\n"
                f"You can manually download NHANES XPT files from:\n"
                f"  https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx\n"
                f"and place them in: {os.path.dirname(dest)}"
            ) from e2


def _download_nhanes(data_dir: str, use_post_pandemic: bool = True) -> str:
    """Download all required NHANES XPT files."""
    pre_dir = os.path.join(data_dir, "pre_pandemic")
    os.makedirs(pre_dir, exist_ok=True)
    for key, fname in _PRE_FILES.items():
        url = f"{_PRE_BASE}/{fname}"
        dest = os.path.join(pre_dir, fname)
        _download_xpt(url, dest)

    if use_post_pandemic:
        post_dir = os.path.join(data_dir, "post_pandemic")
        os.makedirs(post_dir, exist_ok=True)
        for key, fname in _POST_FILES.items():
            url = f"{_POST_BASE}/{fname}"
            dest = os.path.join(post_dir, fname)
            _download_xpt(url, dest)

    return data_dir


def _load_xpt(path: str) -> pd.DataFrame:
    """Load an XPT (SAS Transport) file."""
    return pd.read_sas(path, format="xport", encoding="utf-8")


# ── Data loading and preprocessing ──────────────────────────────────────────

_nhanes_cache: Dict[tuple, tuple] = {}


def _load_era(data_dir: str, era: str) -> pd.DataFrame:
    """Load and merge all component files for one era."""
    if era == "pre":
        base_dir = os.path.join(data_dir, "pre_pandemic")
        files = _PRE_FILES
    else:
        base_dir = os.path.join(data_dir, "post_pandemic")
        files = _POST_FILES

    # Load demographics as base
    demo = _load_xpt(os.path.join(base_dir, files["demo"]))
    df = demo.copy()

    # Merge other components on SEQN
    for key in ["bmx", "bpxo", "mcq", "smq", "ghb", "hdl", "tchol", "trigly",
                "diq", "bpq", "paq"]:
        if key not in files:
            continue
        fpath = os.path.join(base_dir, files[key])
        if os.path.isfile(fpath):
            comp = _load_xpt(fpath)
            df = df.merge(comp, on="SEQN", how="left", suffixes=("", f"_{key}"))

    # Replace sentinel values with NaN
    for val in _SENTINEL_VALUES:
        df = df.replace(val, np.nan)

    df["ERA"] = era
    return df


def _create_cvd_label(df: pd.DataFrame) -> pd.Series:
    """Create binary CVD label from MCQ columns.
    CVD = 1 if any of: coronary heart disease, heart attack, or stroke.
    MCQ coding: 1=yes, 2=no → map to 1.0/0.0.
    """
    cvd = pd.Series(np.nan, index=df.index, dtype=float)
    for col in _CVD_COLS:
        if col in df.columns:
            mapped = df[col].map({1.0: 1.0, 2.0: 0.0})
            cvd = cvd.fillna(0).combine(mapped.fillna(0), max)
    return cvd


def _assign_groups(df: pd.DataFrame, mode: str = "nested") -> pd.Series:
    """Assign feature-availability groups.

    3-group modes (nested/expanded/disjoint):
        0: survey_only, 1: exam, 2: vitals_labs

    4-group mode:
        0: survey_only, 1: exam+labs (no BP), 2: exam+BP (no labs), 3: full
        G1 and G2 are NON-NESTED (G1 has labs, G2 has BP)
    """
    has_mcq = df[_CVD_COLS].notna().any(axis=1)
    has_smq = df["SMQ020"].notna() if "SMQ020" in df.columns else pd.Series(False, index=df.index)
    has_survey = has_mcq & has_smq

    has_bmx = df["BMXBMI"].notna() if "BMXBMI" in df.columns else pd.Series(False, index=df.index)
    has_bpx = (df["BPXOSY1"].notna() & df["BPXODI1"].notna()) if "BPXOSY1" in df.columns else pd.Series(False, index=df.index)

    lab_available = pd.Series(0, index=df.index)
    for col in _LAB_COLS:
        if col in df.columns:
            lab_available += df[col].notna().astype(int)
    has_labs = lab_available >= 3

    group = pd.Series(-1, index=df.index, dtype=int)

    if mode == "4group":
        # G3: full workup (survey + BMX + BPX + labs)
        group[has_survey & has_bmx & has_bpx & has_labs] = 3
        # G2: exam + BP, NO labs (survey + BMX + BPX, no labs)
        group[(group == -1) & has_survey & has_bmx & has_bpx & ~has_labs] = 2
        # G1: exam + labs, NO BP (survey + BMX + labs, no BPX)
        group[(group == -1) & has_survey & has_bmx & has_labs & ~has_bpx] = 1
        # G0: survey only (no BMX)
        group[(group == -1) & has_survey & ~has_bmx] = 0
        # Unassigned: exam only (BMX but no labs and no BPX) → drop
    else:
        # 3-group modes
        mask_g2 = has_survey & has_bmx & has_bpx & has_labs
        group[mask_g2] = 2
        mask_g1 = has_survey & has_bmx & ~mask_g2
        group[mask_g1] = 1
        mask_g0 = has_survey & ~has_bmx
        group[mask_g0] = 0

    return group


def _extract_basic_survey(df, features, offset=0):
    """Extract 10 basic survey features starting at `offset`."""
    # age, education and income-to-poverty are left as NaN here and filled further down with the
    # median of the TRAINING rows, once the split is known (they used the all-rows median before)
    features[:, offset + 0] = df["RIDAGEYR"].values
    features[:, offset + 1] = (df["RIAGENDR"].fillna(1.0).values == 2).astype(np.float32)
    race = df["RIDRETH3"].fillna(df["RIDRETH3"].mode().iloc[0] if len(df["RIDRETH3"].mode()) > 0 else 3.0)
    for i, cat in enumerate([2, 3, 4, 6, 7]):
        features[:, offset + 2 + i] = (race.values == cat).astype(np.float32)
    edu = df["DMDEDUC2"].copy()
    edu = edu.where(edu.isin([1, 2, 3, 4, 5]), np.nan)
    features[:, offset + 7] = edu.values
    features[:, offset + 8] = df["INDFMPIR"].values
    smk = df["SMQ020"].map({1.0: 1.0, 2.0: 0.0})
    features[:, offset + 9] = smk.fillna(0.0).values


def _extract_extra_survey(df, features, offset):
    """Extract 5 extra questionnaire features at `offset`.
    diabetes_dx, high_bp_dx, high_chol_dx, vigorous_activity, moderate_activity
    """
    def _binary(col, yes=1.0, no=2.0, default=0.0):
        if col in df.columns:
            return df[col].map({yes: 1.0, no: 0.0}).fillna(default).values
        return np.full(len(df), default, dtype=np.float32)

    # DIQ010: diabetes diagnosis (1=yes, 2=no, 3=borderline→0.5)
    if "DIQ010" in df.columns:
        features[:, offset + 0] = df["DIQ010"].map({1.0: 1.0, 2.0: 0.0, 3.0: 0.5}).fillna(0.0).values
    # BPQ020: high blood pressure diagnosis
    features[:, offset + 1] = _binary("BPQ020")
    # BPQ080: high cholesterol diagnosis
    features[:, offset + 2] = _binary("BPQ080")
    # PAQ605: vigorous recreational activities
    features[:, offset + 3] = _binary("PAQ605")
    # PAQ650: moderate recreational activities
    features[:, offset + 4] = _binary("PAQ650")


def _extract_body_measures(df, features, offset):
    """Extract 3 body measure features at `offset`."""
    features[:, offset + 0] = df["BMXBMI"].fillna(0.0).values
    features[:, offset + 1] = df["BMXWT"].fillna(0.0).values
    features[:, offset + 2] = df["BMXHT"].fillna(0.0).values


def _extract_bp(df, features, offset):
    """Extract 2 blood pressure features at `offset`."""
    sys_cols = [c for c in _BP_SYS_COLS if c in df.columns]
    if sys_cols:
        features[:, offset + 0] = df[sys_cols].mean(axis=1, skipna=True).fillna(0.0).values
    dia_cols = [c for c in _BP_DIA_COLS if c in df.columns]
    if dia_cols:
        features[:, offset + 1] = df[dia_cols].mean(axis=1, skipna=True).fillna(0.0).values


def _extract_labs(df, features, offset, cols=None):
    """Extract lab features at `offset`."""
    if cols is None:
        cols = ["LBXGH", "LBDHDD", "LBXTC", "LBXTR", "LBDLDL"]
    for i, col in enumerate(cols):
        if col in df.columns:
            features[:, offset + i] = df[col].fillna(0.0).values


def _preprocess_features(df: pd.DataFrame, feature_mode: str = "nested") -> np.ndarray:
    """Extract and preprocess features into a fixed-width matrix.

    Feature modes:
        "nested": G0(10) ⊂ G1(13) ⊂ G2(20)
        "expanded": G0(15) ⊂ G1(18) ⊂ G2(25) — adds 5 questionnaire features
        "disjoint": G0(15), G1(15), G2(15) — each has 10 shared + 5 unique
    """
    _, max_features, _ = _get_feature_config(feature_mode)
    n = len(df)
    features = np.zeros((n, max_features), dtype=np.float32)

    if feature_mode == "nested":
        # [0-9]: basic survey, [10-12]: body, [13-14]: BP, [15-19]: labs
        _extract_basic_survey(df, features, offset=0)
        _extract_body_measures(df, features, offset=10)
        _extract_bp(df, features, offset=13)
        _extract_labs(df, features, offset=15)

    elif feature_mode == "expanded":
        # [0-9]: basic survey, [10-14]: extra survey, [15-17]: body,
        # [18-19]: BP, [20-24]: labs
        _extract_basic_survey(df, features, offset=0)
        _extract_extra_survey(df, features, offset=10)
        _extract_body_measures(df, features, offset=15)
        _extract_bp(df, features, offset=18)
        _extract_labs(df, features, offset=20)

    elif feature_mode in ("disjoint", "partition"):
        # [0-9]: shared survey (partition: G0 only)
        # [10-14]: G0 unique (extra questionnaire)
        # [15-19]: G1 unique (body + 2 labs: BMI, weight, height, HbA1c, HDL)
        # [20-24]: G2 unique (BP + 3 labs: systolic, diastolic, total_chol, trig, LDL)
        _extract_basic_survey(df, features, offset=0)
        _extract_extra_survey(df, features, offset=10)
        _extract_body_measures(df, features, offset=15)
        # G1 unique slots 18,19 = HbA1c, HDL
        _extract_labs(df, features, offset=18, cols=["LBXGH", "LBDHDD"])
        # G2 unique: BP at 20,21 + labs at 22,23,24
        _extract_bp(df, features, offset=20)
        _extract_labs(df, features, offset=22, cols=["LBXTC", "LBXTR", "LBDLDL"])

    return features


def _load_and_preprocess_nhanes(
    data_dir: str,
    train_frac: float = 0.8,
    val_frac: float = 0.0,        # fraction OF TRAIN held out for model selection
    seed: int = 42,
    use_post_pandemic: bool = True,
    min_group_size: int = 50,
    feature_mode: str = "nested",
) -> tuple:
    """Load, preprocess, and split NHANES data.

    Args:
        feature_mode: "nested" (G0⊂G1⊂G2), "expanded" (more survey features, nested),
                      or "disjoint" (each group has unique features).
    Returns:
        features, labels, groups, splits, group_feature_counts, feature_indices, max_features
    """
    cache_key = (os.path.abspath(data_dir), train_frac, seed, use_post_pandemic,
                 feature_mode, val_frac)
    if cache_key in _nhanes_cache:
        return _nhanes_cache[cache_key]

    # Download data
    _download_nhanes(data_dir, use_post_pandemic=use_post_pandemic)

    # Load and merge eras
    dfs = []
    pre_dir = os.path.join(data_dir, "pre_pandemic")
    dfs.append(_load_era(data_dir, "pre"))
    if use_post_pandemic:
        post_dir = os.path.join(data_dir, "post_pandemic")
        if os.path.isdir(post_dir):
            dfs.append(_load_era(data_dir, "post"))
    df = pd.concat(dfs, ignore_index=True)
    print(f"[NHANES] Loaded {len(df)} total participants ({len(dfs)} era(s))")

    # Create CVD label
    df["CVD"] = _create_cvd_label(df)
    # Drop participants without a definable CVD label
    has_label = df["CVD"].notna()
    print(f"[NHANES] Participants with CVD label: {has_label.sum()} / {len(df)}")
    df = df[has_label].copy()
    df["CVD"] = df["CVD"].astype(int)

    # Assign groups
    df["GROUP"] = _assign_groups(df, mode=feature_mode)
    df = df[df["GROUP"] >= 0].copy()

    # Determine number of groups and names for this mode
    n_groups = 4 if feature_mode == "4group" else 3
    g_names = GROUP_NAMES_4 if feature_mode == "4group" else GROUP_NAMES

    print(f"[NHANES] After group assignment: {len(df)} participants")
    for g in range(n_groups):
        n = (df["GROUP"] == g).sum()
        print(f"  G{g} ({g_names[g]}): {n}")

    # Drop groups with too few samples
    for g in range(n_groups):
        if (df["GROUP"] == g).sum() < min_group_size:
            print(f"  WARNING: G{g} has < {min_group_size} samples, dropping")
            df = df[df["GROUP"] != g]

    # Get feature configuration for this mode
    group_feature_counts, max_feat, feat_indices = _get_feature_config(feature_mode)
    print(f"[NHANES] Feature mode: {feature_mode} (features per group: {group_feature_counts}, max={max_feat})")

    # Extract features
    features = _preprocess_features(df, feature_mode=feature_mode)
    labels = df["CVD"].values.astype(np.int64)
    groups = df["GROUP"].values.astype(np.int64)

    # Train/test split, stratified by group × class
    strat_key = groups * 10 + labels  # combined stratification key
    indices = np.arange(len(features))
    try:
        train_idx, test_idx = train_test_split(
            indices, test_size=1.0 - train_frac, train_size=train_frac,
            random_state=seed, shuffle=True, stratify=strat_key,
        )
    except ValueError:
        # If stratification fails (too few samples in a stratum), fall back
        train_idx, test_idx = train_test_split(
            indices, test_size=1.0 - train_frac, train_size=train_frac,
            random_state=seed, shuffle=True,
        )

    splits = np.array(["test_"] * len(features), dtype="U6")
    splits[train_idx] = "train"
    splits[test_idx] = "test"

    # Carve a VALIDATION split out of train. Without one there is nothing to select the epoch
    # on except the test set, which is what the runners were doing: they reported the epoch
    # with the best test worst-group accuracy, making every tabular number a best-of-forty on
    # test. Model selection has to happen on data the test set never sees.
    if val_frac and val_frac > 0:
        tr_strat = strat_key[train_idx]
        try:
            tr2_idx, val_idx = train_test_split(
                train_idx, test_size=val_frac, random_state=seed, shuffle=True,
                stratify=tr_strat)
        except ValueError:
            tr2_idx, val_idx = train_test_split(
                train_idx, test_size=val_frac, random_state=seed, shuffle=True)
        splits[tr2_idx] = "train"
        splits[val_idx] = "val"
        train_idx = tr2_idx

    # Median imputation fitted on TRAINING rows only (see _extract_basic_survey)
    _tr_rows = np.where(splits == "train")[0]
    for _j in range(features.shape[1]):
        _col = features[:, _j]
        if np.isnan(_col).any():
            _med = np.nanmedian(_col[_tr_rows]) if np.isfinite(_col[_tr_rows]).any() else 0.0
            _col[np.isnan(_col)] = _med

    # Per-group normalization using TRAIN stats only
    for g in sorted(feat_indices.keys()):
        g_idx = np.where(groups == g)[0]
        g_train_idx = np.where((groups == g) & (splits == "train"))[0]
        if len(g_train_idx) == 0:
            continue
        fi = feat_indices[g]
        train_feats = features[np.ix_(g_train_idx, fi)]
        mean = train_feats.mean(axis=0)
        std = train_feats.std(axis=0) + 1e-9
        features[np.ix_(g_idx, fi)] = (features[np.ix_(g_idx, fi)] - mean) / std
        unused_idx = sorted(set(range(max_feat)) - set(fi))
        if unused_idx:
            features[np.ix_(g_idx, unused_idx)] = 0.0

    # Print class distribution
    cvd_pos = labels.sum()
    cvd_neg = len(labels) - cvd_pos
    print(f"[NHANES] CVD distribution: negative={cvd_neg} ({100*cvd_neg/len(labels):.1f}%), "
          f"positive={cvd_pos} ({100*cvd_pos/len(labels):.1f}%)")
    print(f"[NHANES] Train: {(splits=='train').sum()}, Test: {(splits=='test').sum()}")

    result = (features, labels, groups, splits, group_feature_counts, feat_indices, max_feat)
    _nhanes_cache[cache_key] = result
    return result


# ── Dataset classes ──────────────────────────────────────────────────────────

class NHANESDataset(Dataset):
    """Single NHANES dataset that returns (x, y, g) tuples.

    Args:
        features: Full feature matrix (N, MAX_FEATURES)
        labels: Label array (N,)
        groups: Group array (N,)
        splits: Split array (N,)
        train: Whether to use train or test split
        group_max_samples: Optional per-group sample caps
        subsample_seed: Seed for subsampling
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        splits: np.ndarray,
        train: bool = True,
        group_max_samples: Optional[List[Optional[int]]] = None,
        subsample_seed: Optional[int] = None,
        class_balanced: bool = False,
    ):
        self.train = train
        # `train` accepts a split NAME as well as a bool, so a validation split can be
        # requested without changing every existing call site.
        if isinstance(train, str):
            split_name = train
            self.train = (train == "train")
        else:
            split_name = "train" if train else "test"
        mask = splits == split_name

        self.features = torch.from_numpy(features[mask])
        self.labels = torch.from_numpy(labels[mask])
        self.groups = torch.from_numpy(groups[mask])

        # Optional per-group subsampling (train only)
        if train and group_max_samples is not None:
            rng = np.random.default_rng(subsample_seed if subsample_seed is not None else 42)
            keep = []
            n_groups = int(self.groups.max().item()) + 1
            for g in range(n_groups):
                g_indices = (self.groups == g).nonzero(as_tuple=True)[0].numpy()
                if g < len(group_max_samples) and group_max_samples[g] is not None:
                    cap = min(len(g_indices), group_max_samples[g])
                    g_indices = rng.choice(g_indices, size=cap, replace=False)
                keep.extend(g_indices.tolist())
            keep = sorted(keep)
            self.features = self.features[keep]
            self.labels = self.labels[keep]
            self.groups = self.groups[keep]

        # Class-balanced oversampling (train only): oversample minority class
        if train and class_balanced:
            rng = np.random.default_rng(subsample_seed if subsample_seed is not None else 42)
            class_counts = [int((self.labels == c).sum()) for c in range(2)]
            max_count = max(class_counts)
            new_features, new_labels, new_groups = [self.features], [self.labels], [self.groups]
            for c in range(2):
                if class_counts[c] < max_count:
                    deficit = max_count - class_counts[c]
                    c_indices = (self.labels == c).nonzero(as_tuple=True)[0].numpy()
                    oversample_idx = rng.choice(c_indices, size=deficit, replace=True)
                    new_features.append(self.features[oversample_idx])
                    new_labels.append(self.labels[oversample_idx])
                    new_groups.append(self.groups[oversample_idx])
            self.features = torch.cat(new_features, dim=0)
            self.labels = torch.cat(new_labels, dim=0)
            self.groups = torch.cat(new_groups, dim=0)
            # Shuffle
            perm = rng.permutation(len(self.labels))
            self.features = self.features[perm]
            self.labels = self.labels[perm]
            self.groups = self.groups[perm]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.groups[idx].item()

    @property
    def num_groups(self):
        return int(self.groups.max().item()) + 1 if len(self.groups) > 0 else 0

    def get_group_counts(self) -> List[int]:
        ng = self.num_groups
        return [int((self.groups == g).sum()) for g in range(ng)]

    def get_class_counts(self) -> List[int]:
        return [int((self.labels == 0).sum()), int((self.labels == 1).sum())]

    def get_group_class_counts(self) -> List[List[int]]:
        ng = self.num_groups
        counts = [[0, 0] for _ in range(ng)]
        for g in range(ng):
            g_mask = self.groups == g
            for c in range(2):
                counts[g][c] = int(((self.labels == c) & g_mask).sum())
        return counts


class StratifiedGroupSampler(Sampler):
    """Sampler that maintains group proportions pi in each batch."""

    def __init__(self, dataset: NHANESDataset, batch_size: int, group_counts: List[int],
                 shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_counts = group_counts
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.num_groups = len(group_counts)
        self.total_samples = sum(group_counts)

        # Compute group proportions pi and samples per group per batch
        self.pi = [c / self.total_samples for c in group_counts]
        self.samples_per_group = []
        remaining = batch_size
        for g in range(self.num_groups - 1):
            n = max(1, int(round(self.pi[g] * batch_size)))
            n = min(n, remaining - (self.num_groups - 1 - g))
            self.samples_per_group.append(n)
            remaining -= n
        self.samples_per_group.append(remaining)

        # Build index lists per group
        self.group_indices = [[] for _ in range(self.num_groups)]
        for idx in range(len(self.dataset)):
            _, _, g = self.dataset[idx]
            if isinstance(g, torch.Tensor):
                g = g.item()
            self.group_indices[g].append(idx)

    def __iter__(self):
        group_indices = []
        for g in range(self.num_groups):
            indices = self.group_indices[g].copy()
            if self.shuffle:
                np.random.shuffle(indices)
            group_indices.append(indices)

        group_pos = [0] * self.num_groups
        batches = []
        while True:
            batch = []
            can_continue = True
            for g in range(self.num_groups):
                n_samples = self.samples_per_group[g]
                start = group_pos[g]
                end = start + n_samples
                if end > len(group_indices[g]):
                    can_continue = False
                    break
                batch.extend(group_indices[g][start:end])
                group_pos[g] = end
            if not can_continue:
                if len(batch) > 0 and not self.drop_last:
                    batches.append(batch)
                break
            if len(batch) == self.batch_size:
                batches.append(batch)
            elif not self.drop_last and len(batch) > 0:
                batches.append(batch)
            if all(group_pos[g] >= len(group_indices[g]) for g in range(self.num_groups)):
                break

        if self.shuffle:
            np.random.shuffle(batches)
        for batch in batches:
            yield from batch

    def __len__(self):
        if self.drop_last:
            return (self.total_samples // self.batch_size) * self.batch_size
        return self.total_samples


def collate_nhanes(batch):
    """Collate function for NHANES batches."""
    xs, ys, gs = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack([yi if isinstance(yi, torch.Tensor) else torch.tensor(yi) for yi in ys], dim=0)
    g = torch.tensor(gs, dtype=torch.long)
    return x, y, g


# ── Public API ───────────────────────────────────────────────────────────────

def build_nhanes_loaders(
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    stratified: bool = True,
    data_root: Optional[str] = None,
    train_frac: float = 0.8,
    val_frac: float = 0.0,
    use_post_pandemic: bool = True,
    group_max_train_samples: Optional[List[Optional[int]]] = None,
    data_split_seed: Optional[int] = None,
    subsample_seed: Optional[int] = None,
    feature_mode: str = "nested",
    class_balanced: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Build train and test DataLoaders for NHANES CVD prediction.

    Args:
        batch_size: Batch size
        num_workers: DataLoader workers
        seed: Random seed
        stratified: Use stratified group sampling
        data_root: Directory for NHANES data (default: datasets/nhanes)
        train_frac: Train fraction (default 0.8)
        use_post_pandemic: Include 2021-2023 data
        group_max_train_samples: Per-group training sample caps
        data_split_seed: Fixed seed for train/test split
        subsample_seed: Seed for subsampling (varies per run)

    Returns:
        train_loader, test_loader, dataset_info dict
    """
    # Do NOT reset global RNG here — only use seed for data split via train_test_split.
    # The caller (train_nhanes.py) manages the model/training RNG via set_seed().

    if data_root is None:
        data_root = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "nhanes")
        data_root = os.path.abspath(data_root)

    split_seed = data_split_seed if data_split_seed is not None else seed

    features, labels, groups, splits, group_feature_counts, feat_indices, max_feat = \
        _load_and_preprocess_nhanes(
            data_dir=data_root,
            train_frac=train_frac,
            seed=split_seed,
            use_post_pandemic=use_post_pandemic,
            feature_mode=feature_mode,
            val_frac=val_frac,
        )

    sub_seed = subsample_seed if subsample_seed is not None else seed

    train_dataset = NHANESDataset(
        features, labels, groups, splits, train=True,
        group_max_samples=group_max_train_samples,
        subsample_seed=sub_seed,
        class_balanced=class_balanced,
    )
    test_dataset = NHANESDataset(
        features, labels, groups, splits, train=False,
    )
    val_dataset = (NHANESDataset(features, labels, groups, splits, train="val")
                   if (val_frac and (splits == "val").any()) else None)

    # Statistics
    train_group_counts = train_dataset.get_group_counts()
    test_group_counts = test_dataset.get_group_counts()
    train_class_counts = train_dataset.get_class_counts()
    test_class_counts = test_dataset.get_class_counts()
    train_group_class_counts = train_dataset.get_group_class_counts()
    test_group_class_counts = test_dataset.get_group_class_counts()

    num_groups = train_dataset.num_groups
    g_names = GROUP_NAMES_4 if feature_mode == "4group" else GROUP_NAMES

    dataset_info = {
        "num_groups": num_groups,
        "num_classes": 2,
        "train_total": sum(train_group_counts),
        "test_total": sum(test_group_counts),
        "train_group_counts": train_group_counts,
        "test_group_counts": test_group_counts,
        "train_class_counts": train_class_counts,
        "test_class_counts": test_class_counts,
        "train_group_class_counts": train_group_class_counts,
        "test_group_class_counts": test_group_class_counts,
        "group_proportions": [c / max(1, sum(train_group_counts)) for c in train_group_counts],
        "group_names": g_names,
        "group_feature_counts": group_feature_counts,
        "feature_indices": feat_indices,
        "max_features": max_feat,
        "feature_mode": feature_mode,
    }

    # Build loaders
    if stratified and all(c > 0 for c in train_group_counts):
        train_sampler = StratifiedGroupSampler(
            train_dataset, batch_size, train_group_counts, shuffle=True, drop_last=False
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, collate_fn=collate_nhanes, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_nhanes, pin_memory=True,
        )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_nhanes, pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_nhanes, pin_memory=True,
        )
    dataset_info["val_total"] = len(val_dataset) if val_dataset is not None else 0
    dataset_info["val_loader"] = val_loader
    return train_loader, test_loader, dataset_info


def print_nhanes_summary(dataset_info: Dict):
    """Pretty-print NHANES dataset statistics."""
    print("=" * 60)
    print("NHANES CVD Dataset Summary")
    print("=" * 60)
    print(f"Number of groups: {dataset_info['num_groups']}")
    print(f"Number of classes: {dataset_info['num_classes']} (binary: no CVD / CVD)")
    print(f"Feature mode: {dataset_info.get('feature_mode', 'nested')}")
    print()
    print(f"Training samples: {dataset_info['train_total']}")
    print(f"Test samples: {dataset_info['test_total']}")
    print()
    gfc = dataset_info["group_feature_counts"]
    g_names = dataset_info.get("group_names", GROUP_NAMES)
    print("Per-group distribution (training):")
    for g, count in enumerate(dataset_info["train_group_counts"]):
        pct = 100 * count / max(1, dataset_info["train_total"])
        name = g_names[g] if g < len(g_names) else f"G{g}"
        n_feat = gfc.get(g, "?")
        print(f"  G{g} ({name}): {count:5d} samples ({pct:5.1f}%), {n_feat} features")
    print()
    print("Per-class distribution (training):")
    for c, count in enumerate(dataset_info["train_class_counts"]):
        pct = 100 * count / max(1, dataset_info["train_total"])
        label = "No CVD" if c == 0 else "CVD"
        print(f"  Class {c} ({label}): {count:5d} samples ({pct:5.1f}%)")
    print()
    print("Per-group-per-class breakdown (training):")
    for g, class_counts in enumerate(dataset_info["train_group_class_counts"]):
        print(f"  G{g}: Class 0={class_counts[0]}, Class 1={class_counts[1]}")
    print()
    print(f"Group proportions pi: {[f'{p:.3f}' for p in dataset_info['group_proportions']]}")
    print("=" * 60)


if __name__ == "__main__":
    print("Testing NHANES CVD dataset loader...")
    train_loader, test_loader, info = build_nhanes_loaders(batch_size=64, stratified=True)
    print_nhanes_summary(info)

    print("\nTesting batches:")
    for i, (x, y, g) in enumerate(train_loader):
        if i >= 3:
            break
        print(f"Batch {i}: x.shape={x.shape}, y.shape={y.shape}, g.shape={g.shape}")
        print(f"  Group dist: {[int((g == gid).sum()) for gid in range(info['num_groups'])]}")
        print(f"  Class dist: {[int((y == cid).sum()) for cid in range(2)]}")
