# EMBED experiment design (benchmark vs REMIND)

Verified against the real downloaded tables (2026-08-18). This is both the recap of
what the dataset is and the plan for the ablation + hyperparameter study.

## 1. What's in the dataset

EMBED open subset = **23,256 patients → 72,770 mammogram exams → 480,323 images (~1.9 TB DICOM)**.
Every image is a breast X-ray. Acquisition is mostly HOLOGIC (92%), some GE / FUJIFILM.

### The "modalities" (input feature spaces)
An image is defined by two axes we care about, crossed = **4 canonical view-types**:

| axis | values |
|------|--------|
| **image type** | **FFDM** ("2D", real full-field digital mammogram, 364,564 imgs) · **C-View** ("cview", synthetic 2D from tomosynthesis, 115,759 imgs) |
| **projection** | **CC** (cranio-caudal, top-down, 217,546) · **MLO** (medio-lateral-oblique, angled, 232,782) |

→ **FFDM_CC, FFDM_MLO, CVIEW_CC, CVIEW_MLO.** (Other view codes — XCCL, ML, LM, spot-mag —
are non-standard lateral/magnification views and are excluded, matching EMBED's own curation
`ViewPosition.isin(['CC','MLO'])`.) Each also has a **laterality** L/R (≈50/50); density is
per-exam so we form one sample per breast.

**Why this is a "heterogeneous feature space" problem:** not every exam has all 4 view-types.
The *modality combination present* defines the group, and the distribution is long-tailed:

| tier | share | groups |
|------|-------|--------|
| HEAD | ~96% | FFDM only (CC+MLO) ~55% · FFDM+C-View (all 4) ~41% |
| TAIL | ~4%  | incomplete exams — FFDM-CC-only, FFDM-MLO-only, partial C-View combos |

### The outputs (what we classify)
**Breast density — BI-RADS `tissueden`, per exam:**

| class | code | meaning | share |
|-------|------|---------|-------|
| A | 1 | almost entirely fatty | 9.9% |
| B | 2 | scattered fibroglandular | 41.9% |
| C | 3 | heterogeneously dense | 42.8% |
| D | 4 | extremely dense | 5.4% |

- **Primary task = 4-class (A/B/C/D)** — matches REMIND, metric = accuracy (head/tail/overall).
- Excluded: `tissueden=5` ("Normal male", 131 exams) and NaN (451).
- **Optional binary task**: dense (C/D, 48%) vs non-dense (A/B, 52%) — the clinically actionable
  split; near-balanced. Good secondary result if we want it.

## 2. Our method's three components (the ablation axes)

Our contribution is a representation-learning story with three parts. The study turns each
on/off to show each earns its place and that they're **synergistic** (the finding from our
tabular datasets: anchors + GroupDRO together >> either alone).

| axis | OFF (control) | ON (ours) |
|------|---------------|-----------|
| **A. Encoder** | shared multi-view encoder (one ResNet + per-view tag) | **per-group encoders** — a dedicated ResNet per view-type (`per_view_encoders`) |
| **B. Anchors** | plain latent | **class-conditional Gaussian anchors** + W₂-fit + separation losses (`lambda_fit/lambda_sep`) |
| **C. GroupDRO** | ERM (uniform) | **GroupDRO** min-max reweighting of tail modality-combos (`groupdro_enabled`) |

All modes project into ONE shared latent so anchors + GroupDRO operate on aligned features
regardless of which views an exam has.

## 3. Ablation matrix (`run_embed_ablation.py`)

| cell | encoder | anchors | GroupDRO | role |
|------|:-------:|:-------:|:--------:|------|
| `erm_shared` | shared | – | – | plain baseline (the "normal model" control) |
| `gdro_shared` | shared | – | ✓ | GroupDRO alone |
| `anchors_shared` | shared | ✓ | – | anchors alone |
| `anchors_gdro_shared` | shared | ✓ | ✓ | **anchor × GroupDRO synergy** (key cell) |
| `erm_pergroup` | per-group | – | – | per-group encoder alone |
| `gdro_pergroup` | per-group | – | ✓ | per-group + GroupDRO |
| `full` | per-group | ✓ | ✓ | **our full method** |

`--cells core` runs the 4 most informative (`erm_shared`, `gdro_shared`, `anchors_gdro_shared`,
`full`); `--cells all` runs the full factorial. External comparators (report from REMIND's
Table 1, we don't rerun them): REMIND 80.7 overall / 82.8 tail, FlexMoE 78.6, plain GroupDRO.

## 4. Metrics & protocol
- **Accuracy per head group / tail group / overall** (REMIND's metric) + worst-group acc.
- **Patient-level** train/test split (no exam leakage), fixed `data_split_seed`.
- **≥3 seeds** for the final numbers (start with 2 to find promising cells, scale to 10 —
  our standard practice). Report mean ± std.
- Class-weighted loss (`class_weight: auto`) for the A/D imbalance.
- `min_group_size: 30` so GroupDRO groups are statistically meaningful.

## 5. Hyperparameter search (on the winning cells only)
Coarse grid, tune on validation head/tail, then lock for the seed sweep:
- `lr` ∈ {5e-5 (REMIND default), 1e-4}
- `backbone` ∈ {resnet18 (fast iterate), resnet50 (final headline)}
- `groupdro_eta` ∈ {0.5, 1.0, 2.0}   (tail up-weighting strength)
- `lambda_fit`, `lambda_sep` ∈ {1e-3, 1e-2}   (anchor strength)
- `latent_dim` ∈ {128, 256}

## 6. Order of operations on the GPU box
1. `--quick` smoke of `run_embed_ablation.py` (subset, 2 epochs) — sanity.
2. Download a large subset, run `--cells core --seeds 42 1337 --epochs 20` — find the story.
3. HP sweep on `full` + `anchors_gdro_shared`.
4. `--cells all --seeds 42 1337 7 --epochs 20` at chosen HPs (optionally resnet50) — headline.
5. Compare to REMIND Table 1; write up.
