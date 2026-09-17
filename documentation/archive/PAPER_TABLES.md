# Complete paper results — all tables

All numbers are **worst-group / overall / balanced accuracy (%)**, mean ± std over 10 seeds
(TextCaps: 3 seeds). Anchors ON = λ_fit = λ_sep = 0.1; OFF = 0.001 (exactly 0.0 is
numerically unstable — see ANCHOR_RESULTS.md).


## Table 1 — Headline: full method vs naive baseline (worst-group accuracy)

| dataset | Shared ERM (naive) | Per-group GroupDRO | **+ anchors (Ours)** | **gain vs naive** |
|---|---|---|---|---|
| Fed-Heart | 66.47 ± 1.49 | 72.26 ± 1.21 | 71.70 ± 0.00 | **+5.79** |
| NHANES-nested | 70.08 ± 0.99 | 70.43 ± 1.39 | 74.14 ± 3.46 | **+4.06** |
| NHANES-disjoint | 70.08 ± 0.99 | 73.22 ± 1.35 | 75.87 ± 1.70 | **+5.79** |
| NHANES-expanded | 75.06 ± 0.59 | 77.66 ± 1.53 | 77.81 ± 1.49 | **+2.75** |
| TextCaps (full data) | — | 63.03 ± 0.48 | 63.85 ± 0.33 | **+0.82** |


## Table 2 — Full 2×2×2 ablation (encoder × GroupDRO × anchors)


### Fed-Heart

| config | worst-group | overall | balanced |
|---|---|---|---|
| Shared ERM | 66.47 ± 1.49 | 72.15 ± 1.77 | 73.33 ± 1.39 |
| Shared ERM + anchors | 68.05 ± 1.86 | 71.47 ± 1.90 | 73.06 ± 1.97 |
| Shared GroupDRO | 69.20 ± 1.01 | 72.07 ± 1.35 | 75.31 ± 2.10 |
| Shared GroupDRO + anchors | 69.82 ± 1.17 | 72.33 ± 0.95 | 74.38 ± 1.31 |
| Per-group ERM | 71.70 ± 0.00 | 75.93 ± 0.20 | 78.88 ± 0.12 |
| Per-group ERM + anchors | 69.81 ± 0.00 | 74.93 ± 0.74 | 78.59 ± 0.88 |
| Per-group GroupDRO | 72.26 ± 1.21 | 74.93 ± 0.61 | 78.28 ± 0.41 |
| Per-group GroupDRO + anchors (Ours) ⭐ | 71.70 ± 0.00 | 75.67 ± 0.61 | 78.66 ± 0.55 |

### NHANES-nested

| config | worst-group | overall | balanced |
|---|---|---|---|
| Shared ERM | 70.08 ± 0.99 | 71.43 ± 1.01 | 72.09 ± 1.02 |
| Shared ERM + anchors | 74.10 ± 2.84 | 75.36 ± 2.98 | 75.84 ± 2.88 |
| Shared GroupDRO | 72.16 ± 1.50 | 73.38 ± 1.67 | 73.90 ± 1.54 |
| Shared GroupDRO + anchors | 76.07 ± 2.58 | 77.64 ± 2.31 | 77.94 ± 2.33 |
| Per-group ERM | 68.97 ± 1.79 | 71.37 ± 1.94 | 71.17 ± 1.90 |
| Per-group ERM + anchors | 71.35 ± 2.20 | 74.08 ± 1.88 | 73.81 ± 2.53 |
| Per-group GroupDRO | 70.43 ± 1.39 | 71.78 ± 1.26 | 72.27 ± 1.32 |
| Per-group GroupDRO + anchors (Ours) ⭐ | 74.14 ± 3.46 | 76.65 ± 2.95 | 76.97 ± 3.05 |

### NHANES-disjoint

| config | worst-group | overall | balanced |
|---|---|---|---|
| Shared ERM | 70.08 ± 0.99 | 71.43 ± 1.01 | 72.09 ± 1.02 |
| Shared ERM + anchors | 74.10 ± 2.84 | 75.36 ± 2.98 | 75.84 ± 2.88 |
| Shared GroupDRO | 72.16 ± 1.50 | 73.38 ± 1.67 | 73.90 ± 1.54 |
| Shared GroupDRO + anchors | 76.07 ± 2.58 | 77.64 ± 2.31 | 77.94 ± 2.33 |
| Per-group ERM | 72.82 ± 1.39 | 73.84 ± 1.56 | 74.96 ± 1.94 |
| Per-group ERM + anchors | 73.82 ± 1.43 | 75.16 ± 0.92 | 75.80 ± 1.18 |
| Per-group GroupDRO | 73.22 ± 1.35 | 74.53 ± 1.42 | 75.82 ± 1.67 |
| Per-group GroupDRO + anchors (Ours) ⭐ | 75.87 ± 1.70 | 77.96 ± 2.34 | 78.76 ± 1.77 |

### NHANES-expanded

| config | worst-group | overall | balanced |
|---|---|---|---|
| Shared ERM | 75.06 ± 0.59 | 75.97 ± 0.57 | 76.85 ± 0.60 |
| Shared ERM + anchors | 77.31 ± 1.86 | 78.24 ± 1.90 | 79.02 ± 1.80 |
| Shared GroupDRO | 77.07 ± 1.69 | 78.07 ± 1.72 | 78.95 ± 1.61 |
| Shared GroupDRO + anchors | 79.08 ± 1.48 | 80.10 ± 1.48 | 80.78 ± 1.48 |
| Per-group ERM | 75.74 ± 2.75 | 77.77 ± 2.62 | 78.16 ± 2.76 |
| Per-group ERM + anchors | 76.18 ± 1.24 | 77.99 ± 1.24 | 78.53 ± 1.27 |
| Per-group GroupDRO | 77.66 ± 1.53 | 78.96 ± 1.64 | 79.82 ± 1.69 |
| Per-group GroupDRO + anchors (Ours) ⭐ | 77.81 ± 1.49 | 79.73 ± 1.29 | 80.31 ± 1.46 |


## Table 3 — Per-group accuracy and loss (baseline vs Ours)


### Fed-Heart

| group | ERM acc | Ours acc | Δ acc | ERM loss | Ours loss |
|---|---|---|---|---|---|
| Cleveland | 71.95 | 76.39 | +4.44 | 0.640 | 0.609 |
| Hungarian | 73.17 | 71.70 | -1.47 | 0.622 | 0.643 |
| Switzerland | 81.11 | 90.00 | +8.89 | 0.431 | 0.345 |
| VA | 67.09 | 76.54 | +9.44 | 0.813 | 0.537 |

### NHANES-nested

| group | ERM acc | Ours acc | Δ acc | ERM loss | Ours loss |
|---|---|---|---|---|---|
| G0 survey | 70.17 | 75.01 | +4.84 | 0.487 | 0.489 |
| G1 exam | 75.25 | 79.53 | +4.28 | 0.420 | 0.441 |
| G2 labs | 70.85 | 76.38 | +5.53 | 0.479 | 0.509 |

### NHANES-disjoint

| group | ERM acc | Ours acc | Δ acc | ERM loss | Ours loss |
|---|---|---|---|---|---|
| G0 survey | 70.17 | 77.26 | +7.09 | 0.487 | 0.496 |
| G1 exam | 75.25 | 81.75 | +6.51 | 0.420 | 0.436 |
| G2 labs | 70.85 | 77.25 | +6.40 | 0.479 | 0.467 |

### NHANES-expanded

| group | ERM acc | Ours acc | Δ acc | ERM loss | Ours loss |
|---|---|---|---|---|---|
| G0 survey | 75.38 | 78.95 | +3.56 | 0.453 | 0.486 |
| G1 exam | 79.96 | 82.77 | +2.81 | 0.388 | 0.437 |
| G2 labs | 75.20 | 79.22 | +4.02 | 0.433 | 0.444 |


## Table 4 — Anchor contribution (paired, same seeds)

Effect of turning anchors ON, holding encoder + GroupDRO fixed.

| dataset | base config | anchors off | anchors on | Δ | seeds won | p |
|---|---|---|---|---|---|---|
| Fed-Heart | Per-group GDRO | 72.26 | 71.70 | **-0.57** | 1/10 | 0.1934 ns |
| Fed-Heart | Shared GDRO | 69.20 | 69.82 | **+0.62** | 6/10 | 0.0403 * |
| NHANES-nested | Per-group GDRO | 70.43 | 74.14 | **+3.71** | 9/10 | 0.0072 ** |
| NHANES-nested | Shared GDRO | 72.16 | 76.07 | **+3.91** | 10/10 | 0.0015 ** |
| NHANES-disjoint | Per-group GDRO | 73.22 | 75.87 | **+2.65** | 9/10 | 0.0022 ** |
| NHANES-disjoint | Shared GDRO | 72.16 | 76.07 | **+3.91** | 10/10 | 0.0015 ** |
| NHANES-expanded | Per-group GDRO | 77.66 | 77.81 | **+0.15** | 6/10 | 0.8094 ns |
| NHANES-expanded | Shared GDRO | 77.07 | 79.08 | **+2.01** | 8/10 | 0.0402 * |
| TextCaps (full) | GroupDRO | 63.03 | 63.85 | **+0.82** | 3/3 | 0.2601 ns |


## Table 5 — NHANES clinical metrics (AUROC, disjoint mode)

| config | overall AUROC | worst-group balanced acc |
|---|---|---|
| Shared ERM | 79.61 ± 0.20 | 71.11 ± 1.18 |
| Shared ERM + anchors | 79.65 ± 0.35 | 69.75 ± 1.98 |
| Shared GroupDRO | 79.05 ± 0.42 | 69.91 ± 1.79 |
| Shared GroupDRO + anchors | 79.32 ± 0.40 | 68.08 ± 2.25 |
| Per-group ERM | 78.46 ± 1.07 | 65.46 ± 2.69 |
| Per-group ERM + anchors | 78.67 ± 0.88 | 65.58 ± 4.13 |
| Per-group GroupDRO | 78.56 ± 1.20 | 66.16 ± 3.06 |
| Per-group GroupDRO + anchors (Ours) | 78.25 ± 1.39 | 64.21 ± 3.84 |


## Table 6 — Anchor mechanism: which loss drives the gain (NHANES-disjoint, 10 seeds)

| arm | worst-group | Δ vs off | p |
|---|---|---|---|
| off (λ=0.001) | 73.22 ± 1.35 | — | — |
| fit only (alignment) | 75.66 ± 2.50 | +2.44 | 0.0319 |
| sep only | 71.27 ± 0.67 | -1.95 | 0.0009 |
| both | 75.87 ± 1.70 | +2.65 | 0.0022 |