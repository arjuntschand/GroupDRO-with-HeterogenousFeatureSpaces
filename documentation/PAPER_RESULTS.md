# Paper results — full ablation package

2×2×2 ablation: encoder {shared, per-group} × GroupDRO {off, on} × anchors {off, on}.
Anchors on = λ_fit=λ_sep=0.1 (validated). 10 seeds, mean±std.


## fedheart_fedheart

| config | worst-group | overall | balanced |
|---|---|---|---|
| Shared ERM | 66.47 ± 1.49 | 72.15 ± 1.77 | 73.33 ± 1.39 |
| Shared ERM +anc | 68.05 ± 1.86 | 71.47 ± 1.90 | 73.06 ± 1.97 |
| Shared GDRO | 69.20 ± 1.01 | 72.07 ± 1.35 | 75.31 ± 2.10 |
| Shared GDRO +anc | 69.82 ± 1.17 | 72.33 ± 0.95 | 74.38 ± 1.31 |
| Per-group ERM | 71.70 ± 0.00 | 75.93 ± 0.20 | 78.88 ± 0.12 |
| Per-group ERM +anc | 69.81 ± 0.00 | 74.93 ± 0.74 | 78.59 ± 0.88 |
| Per-group GDRO | 72.26 ± 1.21 | 74.93 ± 0.61 | 78.28 ± 0.41 |
| Per-group GDRO +anc (Ours) | 71.70 ± 0.00 | 75.67 ± 0.61 | 78.66 ± 0.55 |

- **anchors on the full method (per-group GDRO): 72.26 → 71.70 = -0.57 pts worst-group**
- per-group figure: `figures/pergroup_fedheart_fedheart.png`

## nhanes_disjoint

| config | worst-group | overall | balanced |
|---|---|---|---|
| Shared ERM | 70.08 ± 0.99 | 71.43 ± 1.01 | 72.09 ± 1.02 |
| Shared ERM +anc | 74.10 ± 2.84 | 75.36 ± 2.98 | 75.84 ± 2.88 |
| Shared GDRO | 72.16 ± 1.50 | 73.38 ± 1.67 | 73.90 ± 1.54 |
| Shared GDRO +anc | 76.07 ± 2.58 | 77.64 ± 2.31 | 77.94 ± 2.33 |
| Per-group ERM | 72.82 ± 1.39 | 73.84 ± 1.56 | 74.96 ± 1.94 |
| Per-group ERM +anc | 73.82 ± 1.43 | 75.16 ± 0.92 | 75.80 ± 1.18 |
| Per-group GDRO | 73.22 ± 1.35 | 74.53 ± 1.42 | 75.82 ± 1.67 |
| Per-group GDRO +anc (Ours) | 75.87 ± 1.70 | 77.96 ± 2.34 | 78.76 ± 1.77 |

- **anchors on the full method (per-group GDRO): 73.22 → 75.87 = +2.65 pts worst-group**
- per-group figure: `figures/pergroup_nhanes_disjoint.png`

## nhanes_expanded

| config | worst-group | overall | balanced |
|---|---|---|---|
| Shared ERM | 75.06 ± 0.59 | 75.97 ± 0.57 | 76.85 ± 0.60 |
| Shared ERM +anc | 77.31 ± 1.86 | 78.24 ± 1.90 | 79.02 ± 1.80 |
| Shared GDRO | 77.07 ± 1.69 | 78.07 ± 1.72 | 78.95 ± 1.61 |
| Shared GDRO +anc | 79.08 ± 1.48 | 80.10 ± 1.48 | 80.78 ± 1.48 |
| Per-group ERM | 75.74 ± 2.75 | 77.77 ± 2.62 | 78.16 ± 2.76 |
| Per-group ERM +anc | 76.18 ± 1.24 | 77.99 ± 1.24 | 78.53 ± 1.27 |
| Per-group GDRO | 77.66 ± 1.53 | 78.96 ± 1.64 | 79.82 ± 1.69 |
| Per-group GDRO +anc (Ours) | 77.81 ± 1.49 | 79.73 ± 1.29 | 80.31 ± 1.46 |

- **anchors on the full method (per-group GDRO): 77.66 → 77.81 = +0.15 pts worst-group**
- per-group figure: `figures/pergroup_nhanes_expanded.png`

## nhanes_nested

| config | worst-group | overall | balanced |
|---|---|---|---|
| Shared ERM | 70.08 ± 0.99 | 71.43 ± 1.01 | 72.09 ± 1.02 |
| Shared ERM +anc | 74.10 ± 2.84 | 75.36 ± 2.98 | 75.84 ± 2.88 |
| Shared GDRO | 72.16 ± 1.50 | 73.38 ± 1.67 | 73.90 ± 1.54 |
| Shared GDRO +anc | 76.07 ± 2.58 | 77.64 ± 2.31 | 77.94 ± 2.33 |
| Per-group ERM | 68.97 ± 1.79 | 71.37 ± 1.94 | 71.17 ± 1.90 |
| Per-group ERM +anc | 71.35 ± 2.20 | 74.08 ± 1.88 | 73.81 ± 2.53 |
| Per-group GDRO | 70.43 ± 1.39 | 71.78 ± 1.26 | 72.27 ± 1.32 |
| Per-group GDRO +anc (Ours) | 74.14 ± 3.46 | 76.65 ± 2.95 | 76.97 ± 3.05 |

- **anchors on the full method (per-group GDRO): 70.43 → 74.14 = +3.71 pts worst-group**
- per-group figure: `figures/pergroup_nhanes_nested.png`

## Figures
- `figures/ablation_worst_group.png` (worst-group by cell, all datasets)