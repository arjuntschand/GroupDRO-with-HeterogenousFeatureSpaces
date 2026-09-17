# Fed-Heart: 10-Seed Results (Paper Config)

**Config:** Adam, lr=0.001, batch_size=64, `group_max_train_samples=[null, null, 5, 5]`  
**Seeds:** 42, 123, 456, 789, 1011, 2024, 3141, 5926, 7777, 9999

---

## Summary: Mean ± Std (10 seeds)

| Metric | Baseline | GroupDRO | Δ (percentage points) |
|--------|----------|----------|------------------------|
| **Worst-group accuracy** | **71.66% ± 7.19%** | **74.55% ± 5.34%** | **+2.89%** |
| G0 Cleveland | 76.73% ± 1.86% | 76.15% ± 2.16% | −0.58% |
| G1 Hungarian | 77.75% ± 2.17% | 77.53% ± 1.59% | −0.22% |
| G2 Swiss | 93.75% ± 0.00% | 93.12% ± 1.98% | −0.62% |
| G3 VA | 72.67% ± 7.91% | 75.56% ± 5.74% | **+2.89%** |

- **Worst group** is always G3 (VA) or occasionally G2; GroupDRO improves worst-group by **+2.89 pp** on average.
- **Variance:** GroupDRO reduces std of worst-group accuracy (7.19% → 5.34%), i.e. more stable across seeds.

---

## Per-group loss (mean ± std)

| Group | Baseline | GroupDRO |
|-------|----------|----------|
| G0 Cleveland | 0.520 ± 0.024 | 0.539 ± 0.044 |
| G1 Hungarian | 0.475 ± 0.020 | 0.520 ± 0.037 |
| G2 Swiss | 0.340 ± 0.114 | 0.326 ± 0.127 |
| G3 VA | 0.603 ± 0.085 | 0.600 ± 0.112 |

---

## Per-seed worst-group accuracy

| Seed | Baseline | GroupDRO | Δ |
|------|----------|----------|---|
| 42 | 60.00% | 75.56% | +15.56% |
| 123 | 77.78% | 77.78% | 0.00% |
| 456 | 73.33% | 75.96% | +2.63% |
| 789 | 76.40% | 77.53% | +1.12% |
| 1011 | 75.56% | 77.78% | +2.22% |
| 2024 | 57.78% | 60.00% | +2.22% |
| 3141 | 68.89% | 73.33% | +4.44% |
| 5926 | 76.40% | 77.53% | +1.12% |
| 7777 | 76.40% | 75.96% | −0.44% |
| 9999 | 74.04% | 74.04% | 0.00% |

**Paper-ready line:**  
*Worst-group accuracy: ERM 71.66% ± 7.19% → GroupDRO 74.55% ± 5.34% (+2.89 pp, reduced variance).*
