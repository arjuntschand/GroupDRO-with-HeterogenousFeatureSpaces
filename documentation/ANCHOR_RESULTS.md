# Class-conditional Gaussian anchors — do they help? (the crux result)

**Question.** The paper's novel component is the class-conditional Gaussian anchor module
(per-group encoders map into a shared latent aligned by class anchors). Across the datasets
the anchors had appeared not to help. This document settles whether they actually do.

**Key discovery.** Every headline/ablation config in the repo set the anchor weights to
`lambda_fit = lambda_sep = 0.001` — i.e. **essentially off**. The anchors were never actually
tested at a meaningful weight. When turned on properly, they produce a significant
worst-group gain **on genuinely disjoint feature spaces**.

## Headline result (validated, 10 seeds, paired)

**NHANES, `feature_mode: disjoint`** (each group has 10 shared + 5 unique features → genuinely
different input spaces; per-group encoders + GroupDRO held fixed, only the anchor weight varies):

| anchor weight (λ_fit=λ_sep) | worst-group acc | balanced acc |
|---|---|---|
| 0.001 (≈ off) | 73.22 ± 1.35 | 76.87 ± 1.21 |
| **0.1 (on)** | **75.87 ± 1.70** | **79.03 ± 1.67** |
| 0.3 (on) | 75.31 ± 2.02 | 79.49 ± 2.20 |

**Anchors on vs off: +2.65 pts worst-group, 9/10 seeds improve, paired t-test p = 0.0022,
Wilcoxon p = 0.0059, Cohen's d = 1.33 (large).** Balanced accuracy also improves +2.16.

Per-seed paired Δ (λ0.1 − λ0.001): +2.48, +2.62, +3.85, +2.44, +0.41, +2.07, −0.51, +2.56,
+6.83, +3.74.

## Specificity — anchors fire on *disjoint* heterogeneity, not redundant features

The mechanism predicts anchors help most when groups' feature spaces are genuinely different,
and little when features overlap/are redundant. Evidence:

- **Fed-Heart** (feature-*drop* heterogeneity, mostly overlapping 11–13 features): anchors flat,
  best Δ +0.55 within noise (5 seeds).
- **EMBED** (four views = redundant images of the same breast): anchors do not help at any weight
  (see `EMBED_XENIA.md`).
- **NHANES disjoint** (genuinely unique features per group): **+2.65, p=0.002.**

<!-- SPECIFICITY: NHANES nested/expanded (overlapping/nested features) vs disjoint -->

## Which anchor loss drives it? (fit vs sep decomposition)

<!-- DECOMP: fit-only vs sep-only vs both -->

## Interpretation for the paper

Per-group encoders handle heterogeneous inputs; GroupDRO handles imbalance; **the anchor
alignment provides a further, significant worst-group gain specifically when the groups'
feature spaces are genuinely disjoint.** This gives a clean when-it-works / when-it-doesn't
characterization (works: NHANES-disjoint; null: Fed-Heart overlapping, EMBED redundant views),
which is a stronger and more honest contribution than a single leaderboard number.

## Reproduce

```bash
python run_anchor_sweep.py --dataset nhanes_disjoint --lams 0.001 0.1 0.3 \
    --seeds 42 1337 7 2024 31337 11 22 33 44 55 --out runs/anchor_val_nhanes_disjoint
# fit vs sep:
python run_anchor_sweep.py --dataset nhanes_disjoint --arms 0.1,0 0,0.1 0.1,0.1 \
    --seeds <10 seeds> --out runs/anchor_decomp_nhanes
# specificity across feature modes:
python run_anchor_sweep.py --dataset nhanes_disjoint --base experiments/nhanes_expanded_pergroup_gdro.yaml \
    --lams 0.001 0.1 --seeds <10 seeds> --out runs/anchor_spec_expanded
```
