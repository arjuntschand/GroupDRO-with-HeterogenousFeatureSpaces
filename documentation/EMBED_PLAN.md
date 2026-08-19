# EMBED Experiment Plan (benchmark vs. REMIND)

Status as of 2026-07-21: **code scaffolded, blocked on data access.** Access form
submitted (3-business-day decision). Everything below runs the moment credentials land.

## Why EMBED

Xenia flagged **REMIND** (arXiv 2603.00046, Wu/Shuai/Shen, U-Michigan): same problem
as us — groups defined by *which modalities a patient has*, GroupDRO on top — but
their solution is a Mixture-of-Experts with group-specific routing and **no
representation learning**. Our angle is exactly the representation-learning story
they leave out (per-group encoders → shared latent + Gaussian anchors + GroupDRO).
EMBED is the one of their three datasets with a public slice, so we start there.

## Task & groups (matching REMIND's EMBED setup)

- **Task:** breast-density classification, BI-RADS **A/B/C/D → 4 classes** (`tissueden`).
- **4 canonical modalities:** {FFDM, C-View} × {CC, MLO} = FFDM_CC, FFDM_MLO, CVIEW_CC, CVIEW_MLO.
- **Groups = modality combinations** present for an exam-breast. **Tail = combos with <15% frequency.**
- **Metric = accuracy**, reported **per head group / per tail group / overall** (their Table 1).

**REMIND's EMBED numbers to beat:** head 75.8, tail 82.8, overall 80.7 (vs FlexMoE 78.6).
They also list plain **GroupDRO** as a baseline → clean apples-to-apples for our GroupDRO+anchors.

## How our method maps

One **multi-view encoder** (`mammo_multiview`, ResNet backbone) encodes each present
view; a per-view-type embedding tags which of the 4 views it is; present-view latents
are **masked-mean-pooled** → exam latent. This natively handles missing modalities
(the "heterogeneous feature space"). Class anchors align density classes in the shared
latent; **GroupDRO upweights tail modality-combinations**. See `train_embed.py`.

## Files (all written, tested on fake data)

| File | Role |
|------|------|
| `dro_hetero_anchors/src/datasets_embed.py` | Join metadata+clinical CSVs → exam-breast index, modality-combo groups, tail flags, DICOM loading |
| `dro_hetero_anchors/src/encoders/embed_encoder.py` | `MammoMultiViewEncoder` (masked multi-view pooling) |
| `dro_hetero_anchors/src/train_embed.py` | Training loop; reports head/tail/overall accuracy |
| `experiments/embed_baseline.yaml` | ERM baseline |
| `experiments/embed_groupdro.yaml` | Full method (anchors + GroupDRO) |

## Schema verification — mostly RESOLVED 2026-07-31 (no data download needed)

Verified against the **official Emory-HITI/EMBED_Open_Data repo** (sample notebooks +
`AWS_Open_Data_Clinical_Legend.csv`), which contain their real column names and
curation code. Confirmed and baked into `datasets_embed.py`:
- ✅ Column names all correct: `anon_dicom_path`, `FinalImageType`, `ViewPosition`,
  `ImageLateralityFinal`, `empi_anon`, `acc_anon`, `tissueden`.
- ✅ `FinalImageType` tokens: their curation filters `FinalImageType.isin(['2D','cview'])`
  → `'2D'`=FFDM, `'cview'`=synthetic C-View. Matched exactly.
- ✅ `tissueden` encoding: 1=fat(A), 2=scattered(B), 3=heterogeneous(C), 4=extremely
  dense(D), **5="Normal male"**, NaN=unknown. Map 1-4→0-3; drop 5 and NaN. (5 is a
  documented male exclusion, worth a line in the paper's data section.)
- ✅ `ViewPosition` "CC substring catches XCCL" **BUG FIXED**: switched to exact
  equality to match their `ViewPosition.isin(['CC','MLO'])`; XCCL/XCCM/ML/spot-mag
  now correctly excluded (smoke-tested).

Still to confirm once tables download (run `--inspect`):
- ⏳ Clinical-side laterality column: legend lists both `side` and `bside` — check
  which carries the per-finding L/R for keying density on (acc, side).
- ⏳ Sample granularity: we key on (exam, side); confirm REMIND uses per-exam-breast
  vs per-exam to match their head/tail numbers exactly (decide with Xenia).

## Runbook once access lands

```bash
pip install -r dro_hetero_anchors/requirements.txt        # adds pydicom + jpeg plugins
aws configure                                             # keys from EMBED access grant
# 1) tables only (small) — validate cohort/groups before any images:
aws s3 cp s3://embed-dataset-open/tables/ datasets/embed/tables/ --recursive --region us-west-2
python -m dro_hetero_anchors.src.datasets_embed --inspect          # VERIFY schema
python -m dro_hetero_anchors.src.datasets_embed                    # print group/tail distribution
# 2) fix any VERIFY mismatches, then pull only the images referenced by our index
#    (helper TODO: filter metadata → dicom paths → aws s3 cp per file/prefix)
# 3) train:
python -m dro_hetero_anchors.src.train_embed --config experiments/embed_baseline.yaml
python -m dro_hetero_anchors.src.train_embed --config experiments/embed_groupdro.yaml
```

## Open design questions (decide with Xenia)

1. **Sample granularity:** we use exam-breast (exam × laterality) since density is
   per-breast. REMIND may use per-exam — confirm to match their numbers exactly.
2. **Image download volume:** full open subset is 480k images / >2TB. We only need the
   views in our index; add a helper to fetch just those (step 2 above) to keep it small.
3. **Per-group projection heads:** currently the encoder is shared (mask-driven). If we
   want a stronger "per-group encoder" claim, add optional per-group projection after pooling.
4. **DUA constraint:** EMBED forbids releasing model weights / embedding spaces trained on
   EMBED. Publish results + code only; do NOT release EMBED-trained checkpoints.
```
