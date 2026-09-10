# Results section, draft with verified numbers

Every number here comes from a committed run. Nothing is quoted from memory. Where a claim is
not statistically supported it is marked as such rather than rounded up.

## Headline table (worst-group accuracy, mean ± std)

| method | Fed-Heart¹ | NHANES-nested² | NHANES-disjoint² | EMBED³ |
|---|---|---|---|---|
| ERM | 65.90 ± 1.37 | 70.08 ± 0.99 | 70.08 ± 0.99 | 57.8 ± 1.9 |
| Anchors only (no DRO) | 74.20 ± 1.57 | 71.35 ± 2.20 | 73.82 ± 1.43 | 54.2 ± 3.6 |
| GroupDRO | **75.80 ± 1.03** | 70.43 ± 1.39 | 73.22 ± 1.35 | **62.4 ± 0.9** |
| Regret-DRO | **75.80 ± 0.81** | 70.19 ± 0.96 | 73.17 ± 1.35 | 57.8 ± 1.9 |
| **Ours (anchors + GroupDRO)** | 73.50 ± 1.22 | **74.14 ± 3.46** | **75.87 ± 1.70** | 61.4 ± 2.2 |
| **Ours (anchors + regret)** | 73.30 ± 1.29 | 73.75 ± 3.57 | 75.33 ± 1.85 | 61.9 ± 2.7 |

¹ 5 seeds × 5-fold CV, all 925 patients evaluated. ² 10 seeds. ³ 10 seeds, 128,680 rows,
frozen ViT-Base.

**Where the method wins:** both NHANES settings. **Where it ties:** EMBED (difference from
GroupDRO is not significant, p=0.62). **Where it loses:** Fed-Heart, where GroupDRO alone is
best and adding anchors costs about 2 points.

## The central claim: anchors and GroupDRO are synergistic

NHANES-nested is the cleanest evidence, because neither component does much on its own:

| configuration | worst-group | Δ vs ERM | p |
|---|---|---|---|
| ERM | 70.08 | — | — |
| + GroupDRO only | 70.43 | +0.35 | 0.62 (ns) |
| + anchors only | 71.35 | +1.26 | 0.22 (ns) |
| **+ both** | **74.14** | **+4.06** | **0.013** ✓ |

Individually neither reaches significance. Together they give +4.06 (9/10 seeds), which is
more than double the sum of the parts (+1.62). Adding anchors on top of GroupDRO alone is
+3.71 (p=0.0072).

On NHANES-disjoint every component helps and the full method is +5.79 over ERM (10/10 seeds,
p<0.0001), with anchors contributing +2.65 on top of GroupDRO (p=0.0022).

## Heterogeneity is what causes the gain (controlled synthetic sweep)

Feature overlap between groups is varied with everything else held fixed. Worst-group accuracy:

| overlap | shared encoder | per-group | Δ |
|---|---|---|---|
| 1.0 (homogeneous) | 82.39 | 82.54 | +0.15 |
| 0.6 | 83.17 | 87.30 | +4.13 |
| 0.4 | 74.98 | 85.72 | +10.74 |
| 0.2 | 73.09 | 84.09 | +11.00 |
| **0.0 (disjoint)** | **57.48** | **81.43** | **+23.94** |

The shared encoder collapses as feature spaces diverge while the per-group model holds. The
benefit is caused by heterogeneity, not by anything incidental to the architecture.

## Mechanism controls: two pass, one fails

Three alternative explanations for the gain, each with a matched control (NHANES, 10 seeds):

| control | result | verdict |
|---|---|---|
| widen the shared encoder to equal parameter count (30,854 vs 30,946) | +0.44, p=0.27 | ✅ not capacity |
| replace the anchor loss with a plain L2 penalty on the latent | −0.08, p=0.89 | ✅ not generic regularisation |
| **assign each sample a random anchor instead of its class anchor** | **+2.80, p=0.005** | ❌ **matches the real anchors** |

The third control did not behave as the theory predicts. Real versus random anchors differ by
+0.15 (p=0.81) on disjoint and +0.91 (p=0.16) on nested, i.e. not at all. **We therefore cannot
claim that class-conditional alignment is the operative mechanism.** The likely explanation is
that with random labels every class moment converges on the same global batch moment, so the
loss degenerates into a global latent distribution-matching constraint, and that constraint is
what helps. Both NHANES tasks are binary, which makes random assignment a weak scramble; the
same control is running on EMBED, which is 4-class, as a sharper test.

## Regret optimisation

Regret changes what drives the group weights from raw loss to excess over the per-group floor
R*_g. On worst-group **accuracy** it is indistinguishable from standard GroupDRO everywhere
(Fed-Heart 75.80 vs 75.80; NHANES-nested 70.19 vs 70.43; EMBED identical to ERM).

On the objective it actually optimises it does better. EMBED, 10 seeds:

| method | max excess loss ↓ | worst-group loss ↓ |
|---|---|---|
| ERM | 1.155 | 2.398 |
| GroupDRO | 0.251 | 1.201 |
| **Ours (anchors + regret)** | **0.239** | **1.099** |

Ours attains the lowest excess loss and worst-group loss of any method, though the margin over
GroupDRO is not significant (p=0.57). Regret also distributes group weight very differently:
on EMBED plain GroupDRO collapses to a single head group (λ = 0.999) while regret keeps 9–16%
on every tail group.

## Comparison to REMIND

REMIND (arXiv 2603.00046) reports on the same EMBED task, sample-weighted over the whole set:

| | overall accuracy |
|---|---|
| REMIND | 80.7 |
| REMIND's GroupDRO baseline | 78.9 |
| **Ours, GroupDRO, frozen ViT** | **77.3** |

We are 1.6 points below their GroupDRO with a **frozen** backbone against their fine-tuned one.
A fine-tuned-backbone run is in progress for a like-for-like comparison. We do not claim to beat
REMIND.

## Honest summary

Supported: (i) the full method gives significant worst-group gains on both NHANES settings,
+4.06 and +5.79; (ii) on NHANES-nested the two components are clearly synergistic; (iii) the
benefit scales with feature heterogeneity in a controlled sweep; (iv) the gain is not explained
by parameter count or by generic regularisation.

Not supported: (i) that class-conditional structure is the mechanism, since random anchors work
equally well; (ii) that regret improves worst-group accuracy, though it does improve excess
loss; (iii) any claim of beating GroupDRO on Fed-Heart or EMBED.
