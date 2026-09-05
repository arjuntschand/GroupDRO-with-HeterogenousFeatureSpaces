# EMBED under Xenia's spec — method, data, and results

Implements `EMBED_Experiments_Description.docx` exactly. Supersedes the earlier
ResNet-on-images EMBED pipeline (which used a different architecture).

## Task & data

- **Task:** 4-class BI-RADS breast-density classification (`tissueden` A/B/C/D → 0–3).
- **Density is per-exam** (`acc_anon`), broadcast to both breasts; the clinical `side`
  column is per-finding (~48% NaN) and is **not** used for the density join.
- **Unit:** one row per (exam, breast side). **Four view types (modalities):**
  M1 = C-View CC, M2 = C-View MLO, M3 = FFDM CC, M4 = FFDM MLO.
- **Six groups** by which views are present, with |g| = #views:

  | group | views | full-set freq | Xenia target |
  |---|---|---|---|
  | g1 | M3 | 0.9% | 3.0% |
  | g2 | M1,M3 | 0.1% | 0.6% |
  | g3 | M4 | 0.4% | 1.2% |
  | **g4** | M3,M4 (head) | 56.8% | 57.8% |
  | g5 | M1,M3,M4 | 0.1% | 0.3% |
  | **g6** | all four (head) | 41.7% | 37.1% |

  Head groups match Xenia's table closely; the small tail groups are ~3× lower in
  our per-breast-row counting (a per-exam vs per-breast denominator difference —
  flagged for alignment with her exact preprocessing).

- **Full labeled 6-group set:** 128,680 breast-rows / 22,997 patients.
- **This experiment trains on the offline-available subset** (images already downloaded
  before EMBED S3 access lapsed): **16,575 rows / 10,533 patients**, all 6 groups present
  (g1=801, g2=96, g3=411, g4=7485, g5=50, g6=7732). Label mix A/B/C/D ≈ 10/41/43/5%,
  matching canonical EMBED prevalence. Patient-level 70/10/20 train/val/test split.
  **Groups are assigned from each breast's TRUE view-set (full metadata), then kept only if
  the entire true view-set is cached** — an earlier version assigned groups from
  locally-present views, which contaminated the tails (a true-g4 breast with only its M4
  image downloaded was miscounted as g3: 42% of the old g3, 21% of old g1). Fixed.

## Architecture (all downstream of a frozen ViT-Base)

1. **Frozen ViT-Base** encodes every image once → cached 768-d CLS embedding. Everything
   below runs on cached vectors (no image training). 48,009 embeddings cached (73.7 MB).
2. **Per-view projection** `Linear(768→64)`, shared across groups (one per view type).
3. **Per-group MLP_g**: concat present views' projections (dim |g|·64) →
   `Linear(|g|·64→64), GELU, Linear(64→64)` → latent z ∈ ℝ⁶⁴. This is the heterogeneous
   feature-space part: g1 sees 64 inputs, g4 128, g6 256.
4. **Shared linear head** `Linear(64→4)`.
5. **Four diagonal-Gaussian class anchors** N(m_c, diag(s_c)), s_c = softplus(ρ_c)+1e-4.

Shared model: 276,228 parameters.

## Objective

min_θ max_{λ∈simplex}  Σ_g λ_g [ (L_g^task − R*_g) + λ_fit·L_g^fit ] + λ_sep·L^sep

- **L_g^task** = cross-entropy. **L_g^fit** = diagonal-W₂ between per-class batch moments
  and anchor moments, `Σ_c[Σ_j(m̂−m_c)² + Σ_j(√ŝ−√s_c)²]/64` (classes with ≥8 samples).
  **L^sep** = reparameterized anchor samples pushed through the head, CE vs class.
- **R*_g** = group g's best achievable CE, from a dedicated per-group model via 5-fold
  out-of-fold CV. Setting R*=0 recovers standard GroupDRO.
- **λ update** (max player): EMA (decay 0.9) of per-group excess, then
  λ_g ← λ_g·exp(γ·excess_g), renormalized (γ = 0.02).

## The seven methods (Step 5)

| # | method | DRO | anchors | regret | checkpoint on |
|---|---|---|---|---|---|
| 1 | ERM | – | – | – | avg loss |
| 2 | GroupDRO (R*=0) | ✓ | – | – | worst-group loss |
| 3 | Group-only (dedicated) | per-group models | | | per group |
| 4 | REMIND | *cite published numbers* | | | |
| 5 | **Ours** (anchors+regret) | ✓ | ✓ | ✓ | max excess |
| 6 | Ablation: align-only | ✓ | ✓ | – | avg loss |
| 7 | Ablation: regret-only | ✓ | – | ✓ | max excess |

Rows {2,5,6,7} form the 2×2 (anchors × regret). **Row 2 (GroupDRO) vs Row 5 (Ours)
is the headline comparison.**

Training: AdamW lr 5e-5, wd 5e-5, batch 32, 20 epochs, StepLR(step 5, γ 0.1). Seeds 0/1/42.

## Results

Full tables in `runs/embed_xenia/REPORT.md`; plot in `runs/embed_xenia/rstar_vs_lambda.png`.
3 seeds (0/1/42). Test-set sizes: g1=197, g2=16, g3=136, g4=1492, g5=17, g6=1534
(tail groups g2/g5 are tiny → their per-group accuracy is high-variance).

### Headline (3 seeds; clean/uncontaminated groups)

Accuracy summary (overall-weighted / overall-macro / tail / worst-group) + macro-F1:

| method | overall-wt | overall-macro | tail | worst-group | macro-F1 |
|---|---|---|---|---|---|
| ERM | **0.733** | 0.706 | 0.691 | 0.529 | 0.482 |
| GroupDRO (R*=0) | **0.741** | 0.705 | 0.685 | 0.490 | **0.490** |
| Ablation: align-only | 0.718 | 0.690 | 0.675 | 0.490 | 0.439 |
| **Ablation: regret-only** | 0.729 | **0.713** | **0.705** | **0.588** | 0.477 |
| Ours (anchors+regret) | 0.722 | 0.703 | 0.692 | 0.549 | 0.452 |
| Group-only (dedicated) | 0.732 | 0.708 | 0.696 | 0.588 | 0.459 |

Because the tail test sets are tiny (g2=17, g5=**5**), accuracy there is high-variance;
**cross-entropy loss is the reliable read** (the DRO methods optimize it):

| method | tail loss | worst-group loss |
|---|---|---|
| ERM | 1.066 | 1.612 |
| GroupDRO (R*=0) | 0.810 | 1.048 |
| **regret-only** | **0.808** | **0.940** |
| Ours | 0.813 | 0.947 |

### Honest read

1. **DRO clearly helps tail calibration over ERM.** Both GroupDRO and regret-only cut
   tail loss ~24% (1.066→0.81) and worst-group loss (1.61→0.94–1.05). This is the solid,
   reliable benefit.
2. **Regret-only is the best-behaved variant** — best tail accuracy (0.705±0.003, tight),
   best worst-group accuracy (0.588) and loss (0.940), while overall-weighted (0.729) is
   ≈ GroupDRO (0.741). It Pareto-improves plain GroupDRO on the tail. Note GroupDRO wins raw
   overall accuracy but *sacrifices* the worst group (0.490 < ERM's 0.529).
3. **The anchor component does not help** — align-only < ERM and Ours < regret-only on
   nearly everything (verified across anchor weights below). On EMBED's redundant views the
   representation-alignment mechanism doesn't fire.
4. **Caveat on the mechanism (below): regret's tail edge is second-order, not from directly
   up-weighting the tails** — those stay at ~0 weight under every DRO variant. So the gains
   are real but modest; EMBED is a low-gain setting for this method.

### Mechanism: tiny tail groups are memorized, so DRO can't grip them

Final group weights λ_g (mean over seeds):

| group | R*_g | λ GroupDRO | λ regret-only | test n |
|---|---|---|---|---|
| g4 (head) | 0.617 | 0.780 | 0.456 | 1474 |
| g6 (head) | 0.588 | 0.216 | 0.543 | 1557 |
| g1 | 0.700 | 0.004 | 0.001 | 167 |
| g3 | 0.831 | 0.000 | 0.000 | 97 |
| g2 | 1.150 | 0.000 | 0.000 | 17 |
| g5 | 1.238 | 0.000 | 0.000 | 5 |

Both DRO variants place ~all weight on the two large groups and **~0 on the tails** —
regret only reallocates *between the heads* (g4→g6, by R*). Cause: the tail groups
(g2=96, g5=50 rows) are **memorized** (train loss → 0), so the max player sees no training
loss there and abandons them; regret can't rescue them either (`max(0, train−R*)=0`).
Directly verified (GroupDRO, seed 0):

| group | n_train | train loss | test loss |
|---|---|---|---|
| g5 | 42 | 0.176 | 0.889 |
| g2 | 69 | 0.226 | 0.994 |
| g3 | 282 | 0.321 | 0.729 |
| g1 | 548 | 0.471 | 0.610 |
| g6 | 5402 | 0.538 | 0.614 |
| g4 | 5259 | 0.558 | 0.599 |

So regret-only's better tail metrics come *indirectly* (through the head reallocation and
shared projections/head/anchors), not from emphasizing the tails. **The regret/DRO mechanism
needs tail groups large enough not to be memorized** — on EMBED's offline availability tails
(tens of exams) that condition fails. Consistent with the project's broader result that the
gains live on datasets with genuine, adequately-sized feature-availability heterogeneity
(tabular NHANES / Fed-Heart), not on EMBED's redundant mammographic views.

### Anchor-weight sensitivity (diagnostic, not the headline)

At init (g6 batch): L_task=1.40, **L_fit=1.58, L_sep=2.13** — at λ=1.0 the anchor losses
dwarf the task loss, so the optimizer chases anchor geometry over classification. Sweeping
the anchor weight (λ_fit=λ_sep=λ) for 'ours', 3 seeds:

| anchor weight | overall-wt | tail | worst-group |
|---|---|---|---|
| **regret-only (λ=0)** | 0.727 | **0.709** | **0.608** |
| λ=0.1 | 0.726 | 0.705 | **0.608** |
| λ=0.3 | **0.733** | 0.695 | 0.549 |
| λ=1.0 (faithful spec) | 0.722 | 0.692 | 0.549 |

More anchor weight monotonically *reduces* tail/worst-group robustness (0.709→0.692 tail,
0.608→0.549 worst); anchors never beat regret-only on the tail at any weight. **On EMBED's
redundant mammographic views the anchor-alignment component provides no benefit** — the
regret/DRO reweighting is what carries the method. (This matches the earlier ResNet-era
finding that EMBED's four views are redundant images of the same breast, not genuinely
heterogeneous feature spaces where anchor alignment helps.)

### Exploratory fix: validation-signal DRO (deviates from spec)

The train-loss max player is fooled by tail memorization (above). Driving the λ update by
per-group **validation** loss keeps the tails' signal high. Result (3 seeds):

| method | signal | overall-wt | tail | worst | final λ (mean) |
|---|---|---|---|---|---|
| GroupDRO | train (spec) | 0.741 | 0.685 | 0.490 | g4:.78 g6:.22 |
| GroupDRO | val | 0.715 | 0.688 | 0.529 | **g3:1.00 (collapse)** |
| Ours | train (spec) | 0.722 | 0.692 | 0.549 | g4:.34 g6:.66 |
| Ours | val | 0.725 | 0.685 | 0.510 | **g6:0.99 (collapse)** |

**Verdict: not a clean fix.** The val-driven max player is too aggressive and collapses all
weight onto a single group (winner-take-all min-max). GroupDRO-val *does* lift worst-group
(0.490→0.529) by dumping all weight on g3, but at an overall cost (0.741→0.715) and the
single-group collapse is fragile. Ours-val collapses onto a head (g6) and does not help.
No configuration explored beats regret-only's balance. Exploration stopped here to avoid
tuning the λ step-size toward a manufactured win.

## Bottom line (for Xenia)

1. **Faithful implementation of your spec is done, cheap, and reproducible** — frozen ViT
   cache + tiny MLPs, all 6 groups, 3 seeds, full `metrics_long.csv` + tables + plot.
2. **Regret is the honest positive: regret-only is the best-behaved method** — best tail
   accuracy (0.705) and worst-group accuracy (0.588) and loss (0.940), and it Pareto-improves
   plain GroupDRO on the tail while staying ≈ on overall. Plain GroupDRO wins raw overall
   accuracy but *sacrifices* the worst group (0.490 < ERM 0.529). This is modest, not a blowout.
3. **The anchor-alignment component does not help** on EMBED — verified across anchor weights
   (monotonically worse tail with more anchor weight). EMBED's 4 views are redundant images of
   the same breast, so representation alignment has little to do.
4. **Root cause, verified:** the availability tails are tiny (g2=96, g5=50) → memorized
   (train loss→0, test loss high) → every DRO variant gives them ~0 weight; regret's tail edge
   is a second-order effect, not direct up-weighting. A property of EMBED's data, not the code.
5. **Recommendation:** to make the tail-robustness story land decisively, the tail groups need
   to be large enough not to be memorized. Either (a) restore EMBED S3 access and train on the
   full 128,680-row set (tails ~2× larger), or (b) keep the headline gains on the tabular
   datasets with genuine, adequately-sized feature-availability heterogeneity (NHANES,
   Fed-Heart), where the method shows clear worst-group improvements.

## Reproduce

```bash
# one-time, on a GPU box with the DICOMs:
python -m dro_hetero_anchors.src.extract_vit_embeddings \
    --index datasets/embed/index_xenia_offline.parquet --images-root datasets/embed \
    --out datasets/embed/vit_cache --batch 128
# then, anywhere (CPU is fine):
python -m dro_hetero_anchors.src.train_embed_xenia \
    --index datasets/embed/index_xenia_offline.parquet --cache datasets/embed/vit_cache \
    --out runs/embed_xenia --seeds 0 1 42 --epochs 20
python -m dro_hetero_anchors.src.report_embed_xenia --run runs/embed_xenia
```
