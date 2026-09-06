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

## Specificity — anchors help where the worst group is genuinely struggling

Anchors on (λ=0.1) vs off (≈0), worst-group, paired over seeds. Figure:
`documentation/figures/anchor_specificity.png`.

| dataset / mode | off | on | Δ | seeds win | p |
|---|---|---|---|---|---|
| Fed-Heart (overlapping features) | 72.45 | 71.70 | −0.75 | 0/5 | 0.178 (ns) |
| NHANES expanded (nested, 15/18/25f, high baseline) | 76.81 | 77.52 | +0.71 | 6/10 | 0.265 (ns) |
| **NHANES nested (10/13/20f)** | 70.21 | 73.92 | **+3.72** | 9/10 | **0.008** |
| **NHANES disjoint (unique features/group)** | 73.22 | 75.87 | **+2.65** | 9/10 | **0.002** |

**Anchors give a significant worst-group gain on two of the four settings** — NHANES nested
(+3.72) and disjoint (+2.65), both p<0.01 with 9/10 seeds improving. They are neutral where
the worst-group baseline is already strong (expanded, 76.8%) or the feature spaces heavily
overlap (Fed-Heart feature-drop). EMBED's four redundant views (same breast) also show no
anchor benefit at any weight (`EMBED_XENIA.md`). The pattern: **anchors help when a group's
representation is genuinely under-aligned and has headroom, not when it is already well-served.**

## Which anchor loss drives it? (fit vs sep decomposition)

NHANES disjoint, 10 seeds, worst-group. Figure: `documentation/figures/anchor_fit_vs_sep.png`.

| arm | worst-group | Δ vs off | p |
|---|---|---|---|
| off (λ=0.001) | 73.22 ± 1.35 | — | — |
| **fit only (alignment, λ_fit=0.1, λ_sep=0)** | **75.66 ± 2.50** | **+2.44** | 0.032 |
| sep only (λ_fit=0, λ_sep=0.1) | 71.27 ± 0.67 | −1.95 | 0.0009 |
| both (λ_fit=λ_sep=0.1) | 75.87 ± 1.70 | +2.65 | 0.002 |

**The gain comes entirely from the anchor-*fit* (W₂ alignment) loss** — pulling each group's
per-class latents onto the shared class anchors. The separation loss *alone* is
counterproductive (it spreads anchors without aligning groups), but is harmless combined with
fit (both ≈ fit-only). Mechanistically: **class-conditional latent alignment is the active
ingredient.**

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
