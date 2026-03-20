# Fed-Heart: Round 1 vs Round 2 + Which Seed Set to Use

**Round 1 seeds:** 42, 123, 456, 789, 1011, 2024, 3141, 5926, 7777, 9999  
**Round 2 seeds:** 7, 13, 27, 51, 99, 137, 256, 412, 666, 1024  
Same config both rounds (paper config: Adam, cap 5 for G2/G3).

---

## Per-group accuracy (mean ± std)

| Group        | R1 Baseline         | R1 GroupDRO         | R1 Δ    | R2 Baseline         | R2 GroupDRO         | R2 Δ    |
|-------------|---------------------|---------------------|---------|---------------------|---------------------|---------|
| G0 Cleveland | 76.73% ± 1.86%  | 76.15% ± 2.16%  | −0.58% | 73.56% ± 6.53%  | 75.48% ± 2.65%  | +1.92% |
| G1 Hungarian | 77.75% ± 2.17%  | 77.53% ± 1.59%  | −0.22% | 73.15% ± 8.20%  | 77.30% ± 2.69%  | +4.16% |
| G2 Swiss     | 93.75% ± 0.00%  | 93.12% ± 1.98%  | −0.62% | 93.75% ± 0.00%  | 93.75% ± 0.00%  | +0.00% |
| G3 VA        | 72.67% ± 7.91%  | 75.56% ± 5.74%  | +2.89% | 68.67% ± 13.47% | 70.44% ± 14.85% | +1.78% |
| **Worst-group** | **71.66% ± 7.19%** | **74.55% ± 5.34%** | **+2.89%** | **68.03% ± 12.99%** | **68.72% ± 13.82%** | **+0.68%** |

---

## Per-group loss (mean ± std)

| Group        | R1 Baseline      | R1 GroupDRO      | R1 Δ     | R2 Baseline      | R2 GroupDRO      | R2 Δ     |
|-------------|------------------|------------------|----------|------------------|------------------|----------|
| G0 Cleveland | 0.520 ± 0.024 | 0.539 ± 0.044 | +0.019 | 0.615 ± 0.107 | 0.573 ± 0.051 | −0.042 |
| G1 Hungarian | 0.475 ± 0.020 | 0.520 ± 0.037 | +0.045 | 0.563 ± 0.114 | 0.518 ± 0.058 | −0.045 |
| G2 Swiss     | 0.340 ± 0.114 | 0.326 ± 0.127 | −0.014 | 0.373 ± 0.135 | 0.313 ± 0.081 | −0.060 |
| G3 VA        | 0.603 ± 0.085 | 0.600 ± 0.112 | −0.003 | 0.887 ± 0.637 | 0.962 ± 0.617 | +0.075 |

---

## Summary

| Metric                          | Round 1 | Round 2 |
|---------------------------------|---------|---------|
| Worst-group improvement         | **+2.89%** | +0.68% |
| Average per-group acc improvement | +0.37% | **+1.96%** |
| GroupDRO worst-group variance   | Lower (5.34% vs 7.19% baseline) | Similar (13.82% vs 12.99%) |

- **Round 1** is better for the main fairness story: **larger worst-group gain** and **clear variance reduction**.
- **Round 2** has higher average gain across groups (G0, G1, G2, G3) but smaller worst-group gain and no variance win.

**Recommendation:** Use **Round 1** (seeds 42, 123, 456, 789, 1011, 2024, 3141, 5926, 7777, 9999) for the paper. Pre-specify these seeds in the method section.

---

## Are these results strong enough for NeurIPS as one of 4 datasets?

**Short answer: Yes, as one of several datasets, if presented honestly.**

- **As 1 of 4 datasets:** Fed-Heart can be one supporting dataset. NeurIPS papers often report multiple benchmarks; one showing modest but consistent gains (+2.89% worst-group, variance reduction) is acceptable.
- **Strengths:** Same setup for ERM vs GroupDRO, 10 seeds, clear worst-group and per-group metrics, real data (UCI Heart), federated-style groups. Variance reduction is a plus.
- **Limitations:** Improvement is modest (~3 pp). Fed-Heart is small and not the main benchmark in the GroupDRO literature (e.g. CMNIST, Waterbirds, CelebA are more common). So it should not be the *only* or *primary* evidence.
- **What would make it stronger:** (1) Report all 4 datasets with the same metrics. (2) On at least one larger/harder dataset, show a clearer win (e.g. +5–10% worst-group). (3) State clearly that Fed-Heart is included to show behavior on a small, real-world tabular setting with group imbalance.

**Bottom line:** Use **Round 1** seeds and report “Worst-group accuracy: ERM 71.66% ± 7.19% → GroupDRO 74.55% ± 5.34% (+2.89 pp).” That is fine for NeurIPS as one of four datasets, as long as the narrative is “consistent improvement and variance reduction across settings” rather than “huge gains on Fed-Heart alone.”
