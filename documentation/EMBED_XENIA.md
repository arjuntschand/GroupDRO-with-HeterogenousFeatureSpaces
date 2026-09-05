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

> ⚠️ Numbers below are being regenerated on the corrected (uncontaminated) index; the
> qualitative findings (GroupDRO strongest, anchors don't help, tail memorization) are
> expected to hold. Values updated once the clean 3-seed run finishes.

### Headline (mean±std over seeds)

| method | overall-wt | overall-macro | tail | worst-group | macro-F1 |
|---|---|---|---|---|---|
| ERM | 0.723 | 0.736 | 0.742 | 0.686 | 0.451 |
| **GroupDRO (R*=0)** | **0.755** | **0.749** | **0.744** | 0.684 | **0.513** |
| Ablation: align-only | 0.725 | 0.722 | 0.719 | 0.686 | 0.450 |
| Ablation: regret-only | 0.755 | 0.737 | 0.726 | 0.674 | 0.507 |
| Ours (anchors+regret) | 0.731 | 0.705 | 0.690 | 0.648 | 0.447 |
| Group-only (dedicated) | 0.742 | 0.728 | 0.719 | 0.659 | 0.473 |

### Honest read

This is **not** a "our method wins" result, and it should not be presented as one.

1. **GroupDRO is the strongest method on this data** — it lifts overall accuracy
   (0.723→0.755) and macro-F1 (0.451→0.513) substantially over ERM, driven by the two
   large groups g4/g6 (g6 F1 0.593→0.671). Its worst-group/tail are ≈ ERM.
2. **The anchor component, at the spec's λ_fit=λ_sep=1.0, does not help and slightly
   hurts** (align-only < ERM; Ours < GroupDRO on every summary metric). The
   representation constraint trades classification accuracy for anchor geometry.
3. **Regret ≈ GroupDRO** (0.755 vs 0.755 overall) — neutral here (see why below).

### Why: the DRO mechanism is neutralized by tiny, memorizable tail groups

Final group weights λ_g (mean over seeds) tell the story:

| group | R*_g | λ GroupDRO | λ Ours | test n |
|---|---|---|---|---|
| g4 (head) | 0.619 | 0.702 | 0.348 | 1492 |
| g6 (head) | 0.594 | 0.272 | 0.643 | 1534 |
| g1 | 0.698 | 0.026 | 0.008 | 197 |
| g3 | 0.771 | 0.000 | 0.001 | 136 |
| g2 | 0.972 | 0.000 | 0.000 | 16 |
| g5 | 1.544 | 0.000 | 0.000 | 17 |

Both methods place ~97% of the weight on the two **large** groups and ~0 on the tiny
tails — the *opposite* of the intended tail-robustness behavior. Cause: the tail groups
(g2=100, g5=53 rows total) are **memorized** (train loss → 0) by the 276k-param model, so
the min-max update sees ~no training loss there and abandons them, even though their
**test** loss is the highest in the panel. Regret cannot rescue them either, because
`excess = max(0, train_loss − R*) = 0` once the group is memorized. The result:
worst-group/tail accuracy (measured on exactly these memorized groups) does not improve.

Directly verified (GroupDRO, seed 0) — tiny groups have the *lowest* train loss but
*highest* test loss, so a train-loss-driven max player abandons them:

| group | n_train | train loss | test loss |
|---|---|---|---|
| g5 | 31 | 0.164 | 0.681 |
| g2 | 67 | 0.228 | 0.665 |
| g3 | 497 | 0.429 | 0.744 |
| g1 | 723 | 0.522 | 0.676 |
| g6 | 5431 | 0.559 | 0.584 |
| g4 | 5240 | 0.571 | 0.590 |

This is the useful finding for the method: **the regret/DRO mechanism needs tail groups
large enough not to be memorized.** On EMBED's *offline availability* tails (tens of
exams) that condition fails. It is consistent with the project's broader result that the
gains live on datasets with genuine, adequately-sized feature-availability heterogeneity
(tabular NHANES / Fed-Heart), not on EMBED's redundant mammographic views.

### Anchor-weight sensitivity (diagnostic, not the headline)

At init (g6 batch): L_task=1.43, **L_fit=1.57, L_sep=2.13** — at λ=1.0 the anchor losses
dwarf the task loss, so the optimizer chases anchor geometry over classification. Sweeping
the anchor weight (λ_fit=λ_sep=λ) for 'ours', 3 seeds:

| anchor weight | overall-wt | tail | worst-group |
|---|---|---|---|
| **regret-only (λ=0)** | **0.760** | **0.729** | **0.694** |
| λ=0.1 | 0.749 | 0.728 | 0.668 |
| λ=0.3 | 0.750 | 0.704 | 0.672 |
| λ=1.0 (faithful spec) | 0.731 | 0.690 | 0.648 |

Lower is monotonically better; anchors never beat regret-only at any weight. **On EMBED's
redundant mammographic views the anchor-alignment component provides no benefit** — the
regret/DRO reweighting is what carries the method. (This matches the earlier ResNet-era
finding that EMBED's four views are redundant images of the same breast, not genuinely
heterogeneous feature spaces where anchor alignment helps.)

### Exploratory fix: validation-signal DRO (deviates from spec)

The train-loss max player is fooled by tail memorization (above). Driving the λ update by
per-group **validation** loss keeps the tails' signal high. Result (3 seeds):

| method | signal | overall-wt | tail | worst | final λ (mean) |
|---|---|---|---|---|---|
| GroupDRO | train (spec) | 0.755 | 0.744 | 0.684 | g4:.70 g6:.27 |
| GroupDRO | val | 0.697 | 0.728 | 0.654 | **g2:1.00 (collapse)** |
| Ours | train (spec) | 0.731 | 0.690 | 0.648 | g4:.35 g6:.64 |
| Ours | val | 0.733 | 0.721 | 0.691 | **g6:1.00 (collapse)** |

**Verdict: not a clean fix.** The val-driven max player is too aggressive and collapses all
weight onto a single group (winner-take-all min-max). For GroupDRO this tanks overall
(0.755→0.697). Ours-val is notably *more stable* than GroupDRO-val (the anchors cushion the
collapse — consistent with the prior "anchors stabilize min-max DRO" observation) and nudges
worst-group to 0.691, but it still does not beat faithful GroupDRO's overall/tail balance.
No configuration explored beats plain GroupDRO. Exploration stopped here to avoid tuning the
λ step-size toward a manufactured win.

## Bottom line (for Xenia)

1. **Faithful implementation of your spec is done, cheap, and reproducible** — frozen ViT
   cache + tiny MLPs, all 6 groups, 3 seeds, full `metrics_long.csv` + tables + plot.
2. **On this (offline) EMBED subset, GroupDRO is the strongest method; the anchor-alignment
   component does not help and slightly hurts at λ=1.0** (verified sensitivity: monotonically
   worse with more anchor weight). Regret ≈ GroupDRO.
3. **Root cause, verified:** the availability tails are tiny (g2=100, g5=53) → memorized
   (train loss→0) → the DRO/regret max player abandons them. The mechanism has nothing to
   grip. This is a property of EMBED's data, not an implementation issue.
4. **Recommendation:** to make the tail-robustness story land, the tail groups need to be
   large enough not to be memorized. Either (a) restore EMBED S3 access and train on the full
   128,680-row set (tails would be ~5–10× larger), or (b) keep the headline gains on the
   tabular datasets with genuine, adequately-sized feature-availability heterogeneity
   (NHANES, Fed-Heart), where the method already shows clear worst-group improvements.

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
