# Fed-Heart Round 3: 10 New Seeds

**Seeds:** 17, 33, 88, 111, 222, 333, 444, 555, 888, 1234  
**Config:** Same paper config (Adam, group_max_train_samples=[null,null,5,5])

---

## Worst-group accuracy (mean ± std)

|           | Baseline         | GroupDRO         | Δ      |
|-----------|------------------|------------------|--------|
| Worst-grp | 74.00% ± 4.53%   | 74.94% ± 4.68%   | +0.93% |

---

## Per-group accuracy (mean ± std)

| Group         | Baseline         | GroupDRO         | Δ (pp) |
|---------------|------------------|------------------|--------|
| G0 Cleveland  | 76.83% ± 2.81%   | 77.31% ± 3.90%   | +0.48% |
| G1 Hungarian  | 76.97% ± 2.20%   | 77.75% ± 2.30%   | +0.79% |
| G2 Swiss      | 92.50% ± 3.95%   | 93.75% ± 0.00%   | +1.25% |
| G3 VA         | 75.11% ± 4.89%   | 76.22% ± 4.45%   | +1.11% |

---

## Per-group loss (mean ± std)

| Group         | Baseline       | GroupDRO       | Δ      |
|---------------|----------------|----------------|--------|
| G0 Cleveland  | 0.533 ± 0.031  | 0.569 ± 0.046  | +0.036 |
| G1 Hungarian  | 0.489 ± 0.037  | 0.530 ± 0.062  | +0.041 |
| G2 Swiss      | 0.359 ± 0.144  | 0.382 ± 0.092  | +0.023 |
| G3 VA         | 0.583 ± 0.056  | 0.749 ± 0.348  | +0.165 |

---

## Per-seed worst-group accuracy

| Seed | Baseline | GroupDRO | Δ      |
|------|----------|----------|--------|
| 17   | 74.04%   | 75.96%   | +1.92% |
| 33   | 75.56%   | 77.53%   | +1.97% |
| 88   | 62.22%   | 64.44%   | +2.22% |
| 111  | 72.12%   | 68.27%   | −3.85% |
| 222  | 73.33%   | 78.65%   | +5.32% |
| 333  | 75.28%   | 76.40%   | +1.12% |
| 444  | 77.78%   | 76.40%   | −1.37% |
| 555  | 76.92%   | 77.53%   | +0.61% |
| 888  | 75.00%   | 76.40%   | +1.40% |
| 1234 | 77.78%   | 77.78%   | 0.00%  |

---

**Summary:** Round 3 gives +0.93% worst-group on average; all four groups improve in accuracy. Variance is moderate (baseline 4.53%, GroupDRO 4.68%).
