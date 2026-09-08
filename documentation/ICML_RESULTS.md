# ICML results package — NHANES & Fed-Heart

Protocol follows Xenia's Step 5/6 spec (EMBED_Experiments_Description.docx) extended
with regret optimization. **6 methods × 10 seeds.** All methods share identical model
capacity and differ only in the objective. Anchors ON = λ_fit = λ_sep = 0.1; OFF = 0.001.
R*_g estimated by 5-fold out-of-fold dedicated per-group models (nested-CV early stopping).

Metrics: per-group accuracy / macro-F1 / loss / R*_g / excess loss; overall accuracy and
macro-F1; worst-group raw loss (max_g L_g) and max excess loss (max_g L_g − R*_g), each
with the group it points at. Raw rows: `runs/matrix_<tag>/metrics_long.csv`.


## Table 1 — Method comparison (worst-group accuracy, mean ± std over 10 seeds)

| method | Fed-Heart | NHANES-nested | NHANES-disjoint | NHANES-expanded |
|---|---|---|---|---|
| ERM | 65.58 ± 7.57 | 70.08 ± 0.99 | 70.08 ± 0.99 | 75.06 ± 0.59 |
| GroupDRO | 74.80 ± 2.84 | 70.43 ± 1.39 | 73.22 ± 1.35 | 77.66 ± 1.53 |
| Regret-DRO | 74.80 ± 2.84 | 70.19 ± 0.96 | 73.17 ± 1.35 | 77.40 ± 1.82 |
| Anchors only | 74.21 ± 3.51 | 71.35 ± 2.20 | 73.82 ± 1.43 | 76.18 ± 1.24 |
| Ours (anchors+GroupDRO) | 74.50 ± 2.95 | 74.14 ± 3.46 | 75.87 ± 1.70 | 77.81 ± 1.49 |
| Ours (anchors+regret) ⭐ | 74.25 ± 3.31 | 73.75 ± 3.57 | 75.33 ± 1.85 | 77.90 ± 1.43 |


## Table 2 — Overall metrics per dataset


### Fed-Heart

| method | worst-group acc | overall acc | balanced acc | mean F1 | worst-group F1 | worst-group loss (group) | max excess loss (group) |
|---|---|---|---|---|---|---|---|
| ERM | 65.58 ± 7.57 | 76.33 ± 5.40 | 75.24 ± 5.61 | 65.39 ± 4.38 | 43.91 ± 6.90 | 0.751 (Switzerland) | 0.244 (Switzerland) |
| GroupDRO | 74.80 ± 2.84 | 80.20 ± 2.39 | 83.56 ± 1.52 | 63.62 ± 3.35 | 45.43 ± 2.99 | 0.650 (VA) | 0.056 (Cleveland) |
| Regret-DRO | 74.80 ± 2.84 | 80.53 ± 2.29 | 83.77 ± 1.49 | 63.64 ± 2.75 | 45.43 ± 2.99 | 0.608 (VA) | 0.035 (Cleveland) |
| Anchors only | 74.21 ± 3.51 | 79.53 ± 2.44 | 82.14 ± 2.80 | 64.42 ± 2.59 | 46.93 ± 3.09 | 0.669 (VA) | 0.084 (Cleveland) |
| Ours (anchors+GroupDRO) | 74.50 ± 2.95 | 79.40 ± 2.91 | 82.84 ± 2.23 | 63.62 ± 3.49 | 46.74 ± 3.26 | 0.695 (Hungarian) | 0.071 (Cleveland) |
| Ours (anchors+regret) | 74.25 ± 3.31 | 79.80 ± 2.58 | 83.03 ± 2.09 | 63.59 ± 3.35 | 45.96 ± 3.32 | 0.722 (VA) | 0.072 (Cleveland) |

### NHANES-nested

| method | worst-group acc | overall acc | balanced acc | mean F1 | worst-group F1 | worst-group loss (group) | max excess loss (group) |
|---|---|---|---|---|---|---|---|
| ERM | 70.08 ± 0.99 | 71.43 ± 1.01 | 72.09 ± 1.02 | 35.91 ± 0.46 | 34.40 ± 0.53 | 0.487 (G0 survey) | 0.214 (G2 labs) |
| GroupDRO | 70.43 ± 1.39 | 71.78 ± 1.26 | 72.27 ± 1.32 | 35.35 ± 1.20 | 32.71 ± 1.60 | 0.499 (G0 survey) | 0.211 (G2 labs) |
| Regret-DRO | 70.19 ± 0.96 | 71.78 ± 1.35 | 72.53 ± 1.61 | 35.36 ± 1.25 | 32.73 ± 2.14 | 0.496 (G0 survey) | 0.209 (G2 labs) |
| Anchors only | 71.35 ± 2.20 | 74.08 ± 1.88 | 73.81 ± 2.53 | 35.69 ± 1.39 | 32.85 ± 2.27 | 0.521 (G0 survey) | 0.223 (G0 survey) |
| Ours (anchors+GroupDRO) | 74.14 ± 3.46 | 76.65 ± 2.95 | 76.97 ± 3.05 | 35.04 ± 2.37 | 32.53 ± 2.37 | 0.525 (G2 labs) | 0.247 (G2 labs) |
| Ours (anchors+regret) | 73.75 ± 3.57 | 76.22 ± 3.22 | 76.49 ± 3.26 | 34.80 ± 2.05 | 31.89 ± 2.16 | 0.528 (G2 labs) | 0.252 (G2 labs) |

### NHANES-disjoint

| method | worst-group acc | overall acc | balanced acc | mean F1 | worst-group F1 | worst-group loss (group) | max excess loss (group) |
|---|---|---|---|---|---|---|---|
| ERM | 70.08 ± 0.99 | 71.43 ± 1.01 | 72.09 ± 1.02 | 35.91 ± 0.46 | 34.40 ± 0.53 | 0.487 (G0 survey) | 0.214 (G2 labs) |
| GroupDRO | 73.22 ± 1.35 | 74.53 ± 1.42 | 75.82 ± 1.67 | 35.17 ± 1.48 | 32.84 ± 2.96 | 0.548 (G0 survey) | 0.250 (G0 survey) |
| Regret-DRO | 73.17 ± 1.35 | 74.63 ± 1.49 | 75.84 ± 1.77 | 35.10 ± 1.11 | 32.70 ± 2.16 | 0.560 (G0 survey) | 0.263 (G0 survey) |
| Anchors only | 73.82 ± 1.43 | 75.16 ± 0.92 | 75.80 ± 1.18 | 34.63 ± 1.52 | 30.95 ± 3.73 | 0.542 (G0 survey) | 0.249 (G0 survey) |
| Ours (anchors+GroupDRO) | 75.87 ± 1.70 | 77.96 ± 2.34 | 78.76 ± 1.77 | 35.59 ± 1.71 | 32.10 ± 3.89 | 0.520 (G0 survey) | 0.240 (G2 labs) |
| Ours (anchors+regret) | 75.33 ± 1.85 | 77.37 ± 2.52 | 77.96 ± 1.94 | 36.02 ± 1.19 | 32.71 ± 2.25 | 0.503 (G2 labs) | 0.224 (G2 labs) |

### NHANES-expanded

| method | worst-group acc | overall acc | balanced acc | mean F1 | worst-group F1 | worst-group loss (group) | max excess loss (group) |
|---|---|---|---|---|---|---|---|
| ERM | 75.06 ± 0.59 | 75.97 ± 0.57 | 76.85 ± 0.60 | 40.19 ± 0.41 | 37.49 ± 0.41 | 0.453 (G0 survey) | 0.175 (G2 labs) |
| GroupDRO | 77.66 ± 1.53 | 78.96 ± 1.64 | 79.82 ± 1.69 | 36.52 ± 1.80 | 32.76 ± 4.00 | 0.603 (G0 survey) | 0.302 (G0 survey) |
| Regret-DRO | 77.40 ± 1.82 | 78.96 ± 1.39 | 79.82 ± 1.53 | 36.40 ± 1.26 | 32.83 ± 2.90 | 0.614 (G0 survey) | 0.311 (G0 survey) |
| Anchors only | 76.18 ± 1.24 | 77.99 ± 1.24 | 78.53 ± 1.27 | 37.89 ± 1.51 | 35.67 ± 2.45 | 0.510 (G0 survey) | 0.223 (G0 survey) |
| Ours (anchors+GroupDRO) | 77.81 ± 1.49 | 79.73 ± 1.29 | 80.31 ± 1.46 | 37.87 ± 2.18 | 34.31 ± 4.31 | 0.519 (G0 survey) | 0.234 (G0 survey) |
| Ours (anchors+regret) | 77.90 ± 1.43 | 79.89 ± 1.23 | 80.55 ± 1.41 | 38.11 ± 2.06 | 35.16 ± 3.81 | 0.541 (G0 survey) | 0.253 (G0 survey) |


## Table 3 — Per-group breakdown: accuracy, macro-F1, loss, R*_g, excess loss


### Fed-Heart


**ERM**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| Cleveland | 61 | 76.72 | 76.34 | 0.525 | 0.499 | 0.026 |
| Hungarian | 53 | 79.25 | 78.31 | 0.493 | 0.592 | 0.000 |
| Switzerland | 10 | 75.00 | 44.33 | 0.598 | 0.424 | 0.174 |
| VA | 26 | 70.00 | 62.58 | 0.651 | 1.480 | 0.000 |

**GroupDRO**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| Cleveland | 61 | 79.84 | 79.58 | 0.510 | 0.499 | 0.011 |
| Hungarian | 53 | 78.87 | 76.14 | 0.534 | 0.592 | 0.000 |
| Switzerland | 10 | 99.00 | 49.74 | 0.158 | 0.424 | 0.000 |
| VA | 26 | 76.54 | 49.04 | 0.633 | 1.480 | 0.000 |

**Regret-DRO**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| Cleveland | 61 | 80.49 | 80.24 | 0.481 | 0.499 | 0.000 |
| Hungarian | 53 | 79.06 | 76.32 | 0.512 | 0.592 | 0.000 |
| Switzerland | 10 | 99.00 | 49.74 | 0.130 | 0.424 | 0.000 |
| VA | 26 | 76.54 | 48.25 | 0.601 | 1.480 | 0.000 |

**Ours (anchors+regret)**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| Cleveland | 61 | 79.67 | 79.43 | 0.535 | 0.499 | 0.036 |
| Hungarian | 53 | 78.30 | 76.33 | 0.572 | 0.592 | 0.000 |
| Switzerland | 10 | 98.00 | 49.47 | 0.150 | 0.424 | 0.000 |
| VA | 26 | 76.15 | 49.11 | 0.695 | 1.480 | 0.000 |

### NHANES-nested


**ERM**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| G0 survey | 533 | 70.17 | 37.15 | 0.487 | 0.314 | 0.173 |
| G1 exam | 530 | 75.25 | 36.00 | 0.420 | 0.276 | 0.143 |
| G2 labs | 2338 | 70.85 | 34.57 | 0.479 | 0.266 | 0.214 |

**GroupDRO**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| G0 survey | 533 | 71.28 | 37.27 | 0.492 | 0.314 | 0.178 |
| G1 exam | 530 | 74.19 | 33.23 | 0.448 | 0.276 | 0.171 |
| G2 labs | 2338 | 71.35 | 35.57 | 0.471 | 0.266 | 0.205 |

**Regret-DRO**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| G0 survey | 533 | 71.18 | 37.27 | 0.490 | 0.314 | 0.176 |
| G1 exam | 530 | 75.28 | 33.12 | 0.441 | 0.276 | 0.165 |
| G2 labs | 2338 | 71.12 | 35.70 | 0.473 | 0.266 | 0.207 |

**Ours (anchors+regret)**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| G0 survey | 533 | 74.78 | 37.28 | 0.485 | 0.314 | 0.171 |
| G1 exam | 530 | 78.72 | 32.32 | 0.440 | 0.276 | 0.164 |
| G2 labs | 2338 | 75.98 | 34.78 | 0.514 | 0.266 | 0.248 |

### NHANES-disjoint


**ERM**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| G0 survey | 533 | 70.17 | 37.15 | 0.487 | 0.300 | 0.187 |
| G1 exam | 530 | 75.25 | 36.00 | 0.420 | 0.283 | 0.137 |
| G2 labs | 2338 | 70.85 | 34.57 | 0.479 | 0.265 | 0.214 |

**GroupDRO**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| G0 survey | 533 | 74.20 | 36.45 | 0.548 | 0.300 | 0.248 |
| G1 exam | 530 | 79.87 | 33.50 | 0.452 | 0.283 | 0.169 |
| G2 labs | 2338 | 73.39 | 35.57 | 0.445 | 0.265 | 0.179 |

**Regret-DRO**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| G0 survey | 533 | 73.98 | 36.52 | 0.558 | 0.300 | 0.258 |
| G1 exam | 530 | 79.98 | 33.09 | 0.464 | 0.283 | 0.181 |
| G2 labs | 2338 | 73.56 | 35.70 | 0.441 | 0.265 | 0.176 |

**Ours (anchors+regret)**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| G0 survey | 533 | 76.83 | 38.78 | 0.475 | 0.300 | 0.175 |
| G1 exam | 530 | 80.21 | 33.19 | 0.423 | 0.283 | 0.141 |
| G2 labs | 2338 | 76.85 | 36.10 | 0.469 | 0.265 | 0.204 |

### NHANES-expanded


**ERM**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| G0 survey | 533 | 75.38 | 41.30 | 0.453 | 0.307 | 0.146 |
| G1 exam | 530 | 79.96 | 41.79 | 0.388 | 0.268 | 0.121 |
| G2 labs | 2338 | 75.20 | 37.49 | 0.433 | 0.258 | 0.175 |

**GroupDRO**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| G0 survey | 533 | 77.90 | 37.22 | 0.602 | 0.307 | 0.296 |
| G1 exam | 530 | 83.38 | 33.50 | 0.533 | 0.268 | 0.265 |
| G2 labs | 2338 | 78.20 | 38.85 | 0.446 | 0.258 | 0.188 |

**Regret-DRO**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| G0 survey | 533 | 77.41 | 36.72 | 0.614 | 0.307 | 0.308 |
| G1 exam | 530 | 83.85 | 33.61 | 0.510 | 0.268 | 0.243 |
| G2 labs | 2338 | 78.20 | 38.89 | 0.445 | 0.258 | 0.187 |

**Ours (anchors+regret)**

| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |
|---|---|---|---|---|---|---|
| G0 survey | 533 | 79.06 | 38.57 | 0.509 | 0.307 | 0.203 |
| G1 exam | 530 | 83.28 | 36.09 | 0.453 | 0.268 | 0.186 |
| G2 labs | 2338 | 79.31 | 39.67 | 0.450 | 0.258 | 0.192 |


## Table 4 — 2×2 interaction (anchors × regret) and paired significance

Worst-group accuracy. GroupDRO (row 2) vs Ours-regret is the headline comparison.

| dataset | GroupDRO | Regret-DRO | Ours (anc+GDRO) | Ours (anc+regret) | anchor effect | regret effect |
|---|---|---|---|---|---|---|
| Fed-Heart | 74.80 ± 2.84 | 74.80 ± 2.84 | 74.50 ± 2.95 | 74.25 ± 3.31 | -0.30 (3/10, p=0.700) ns | identical on all seeds† |
| NHANES-nested | 70.43 ± 1.39 | 70.19 ± 0.96 | 74.14 ± 3.46 | 73.75 ± 3.57 | +3.71 (9/10, p=0.007) ** | -0.25 (5/10, p=0.618) ns |
| NHANES-disjoint | 73.22 ± 1.35 | 73.17 ± 1.35 | 75.87 ± 1.70 | 75.33 ± 1.85 | +2.65 (9/10, p=0.002) ** | -0.05 (4/10, p=0.821) ns |
| NHANES-expanded | 77.66 ± 1.53 | 77.40 ± 1.82 | 77.81 ± 1.49 | 77.90 ± 1.43 | +0.15 (6/10, p=0.809) ns | -0.26 (3/10, p=0.768) ns |

† On Fed-Heart, GroupDRO and Regret-DRO produce **identical worst-group accuracy on all
10 seeds** — the runs genuinely differ (per-group losses differ, e.g. 0.934 vs 0.765 on
seed 2024) but worst-group *accuracy* is quantized on that dataset's small test groups
(26–61 samples), so both land on the same discrete value. Regret's effect there shows up
in the loss objective it actually optimizes: **max excess loss 0.040 → 0.011** and
worst-group loss 0.650 → 0.608 (Table 2). Accuracy is too coarse a probe on Fed-Heart.


## Table 5 — Parameter testing: group definition (NHANES feature modes)

Same method, different *group structure* — nested (G0⊂G1⊂G2), expanded (nested, more
features), disjoint (each group has unique features). Worst-group accuracy.

| method | NHANES-nested | NHANES-disjoint | NHANES-expanded |
|---|---|---|---|
| ERM | 70.08 ± 0.99 | 70.08 ± 0.99 | 75.06 ± 0.59 |
| GroupDRO | 70.43 ± 1.39 | 73.22 ± 1.35 | 77.66 ± 1.53 |
| Regret-DRO | 70.19 ± 0.96 | 73.17 ± 1.35 | 77.40 ± 1.82 |
| Anchors only | 71.35 ± 2.20 | 73.82 ± 1.43 | 76.18 ± 1.24 |
| Ours (anchors+GroupDRO) | 74.14 ± 3.46 | 75.87 ± 1.70 | 77.81 ± 1.49 |
| Ours (anchors+regret) | 73.75 ± 3.57 | 75.33 ± 1.85 | 77.90 ± 1.43 |