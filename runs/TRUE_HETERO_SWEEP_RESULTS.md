# True Heterogeneous Feature Spaces: Per-Group Encoder Results

Each encoder has a genuinely different input_dim matching its available features.


## aggressive_no_cap (gdro)

### Per-Group Accuracy

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 76.83% +/- 2.41% | 76.44% +/- 2.52% | -0.38% |
| G1 Hungarian | 76.85% +/- 1.83% | 77.42% +/- 1.91% | +0.56% |
| G2 Switzerland | 96.88% +/- 3.12% | 96.88% +/- 3.12% | +0.00% |
| G3 VA Long Beach | 77.11% +/- 1.74% | 77.33% +/- 1.33% | +0.22% |
| **Worst-group** | 75.62% +/- 1.65% | 75.70% +/- 1.89% | **+0.07%** |

### Per-Group Loss

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 0.596 +/- 0.123 | 0.562 +/- 0.065 | -0.034 |
| G1 Hungarian | 0.538 +/- 0.084 | 0.533 +/- 0.047 | -0.005 |
| G2 Switzerland | 0.211 +/- 0.155 | 0.227 +/- 0.151 | +0.015 |
| G3 VA Long Beach | 0.628 +/- 0.190 | 0.547 +/- 0.092 | -0.081 |

## mild_realistic_cap_15_25 (gdro)

### Per-Group Accuracy

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 78.08% +/- 2.23% | 77.79% +/- 2.45% | -0.29% |
| G1 Hungarian | 80.45% +/- 2.20% | 80.34% +/- 1.69% | -0.11% |
| G2 Switzerland | 96.88% +/- 3.12% | 96.88% +/- 3.12% | +0.00% |
| G3 VA Long Beach | 77.11% +/- 2.00% | 77.78% +/- 0.00% | +0.67% |
| **Worst-group** | 76.36% +/- 2.08% | 76.93% +/- 1.19% | **+0.57%** |

### Per-Group Loss

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 0.537 +/- 0.074 | 0.558 +/- 0.076 | +0.021 |
| G1 Hungarian | 0.442 +/- 0.056 | 0.457 +/- 0.046 | +0.015 |
| G2 Switzerland | 0.188 +/- 0.108 | 0.215 +/- 0.191 | +0.027 |
| G3 VA Long Beach | 0.626 +/- 0.214 | 0.751 +/- 0.419 | +0.125 |

## mild_realistic_cap_20_30 (gdro)

### Per-Group Accuracy

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 77.79% +/- 1.52% | 78.37% +/- 1.44% | +0.58% |
| G1 Hungarian | 80.56% +/- 2.80% | 80.56% +/- 2.56% | +0.00% |
| G2 Switzerland | 96.88% +/- 3.12% | 96.88% +/- 3.12% | +0.00% |
| G3 VA Long Beach | 77.78% +/- 0.00% | 78.22% +/- 1.33% | +0.44% |
| **Worst-group** | 77.10% +/- 0.95% | 77.45% +/- 0.43% | **+0.35%** |

### Per-Group Loss

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 0.535 +/- 0.039 | 0.536 +/- 0.043 | +0.000 |
| G1 Hungarian | 0.463 +/- 0.032 | 0.469 +/- 0.040 | +0.006 |
| G2 Switzerland | 0.211 +/- 0.097 | 0.173 +/- 0.133 | -0.039 |
| G3 VA Long Beach | 0.514 +/- 0.031 | 0.515 +/- 0.038 | +0.000 |

## mild_realistic_cap_20_30 (strong_eta)

### Per-Group Accuracy

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 77.79% +/- 1.52% | 77.98% +/- 1.32% | +0.19% |
| G1 Hungarian | 80.56% +/- 2.80% | 80.67% +/- 3.17% | +0.11% |
| G2 Switzerland | 96.88% +/- 3.12% | 96.88% +/- 3.12% | +0.00% |
| G3 VA Long Beach | 77.78% +/- 0.00% | 78.22% +/- 1.33% | +0.44% |
| **Worst-group** | 77.10% +/- 0.95% | 77.25% +/- 0.71% | **+0.15%** |

### Per-Group Loss

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 0.535 +/- 0.039 | 0.537 +/- 0.040 | +0.002 |
| G1 Hungarian | 0.463 +/- 0.032 | 0.469 +/- 0.038 | +0.006 |
| G2 Switzerland | 0.211 +/- 0.097 | 0.171 +/- 0.124 | -0.040 |
| G3 VA Long Beach | 0.514 +/- 0.031 | 0.515 +/- 0.037 | +0.000 |

## mild_realistic_cap_20_30 (with_kl)

### Per-Group Accuracy

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 77.79% +/- 1.52% | 78.37% +/- 1.44% | +0.58% |
| G1 Hungarian | 80.56% +/- 2.80% | 80.56% +/- 2.56% | +0.00% |
| G2 Switzerland | 96.88% +/- 3.12% | 96.88% +/- 3.12% | +0.00% |
| G3 VA Long Beach | 77.78% +/- 0.00% | 78.22% +/- 1.33% | +0.44% |
| **Worst-group** | 77.10% +/- 0.95% | 77.45% +/- 0.43% | **+0.35%** |

### Per-Group Loss

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 0.535 +/- 0.039 | 0.536 +/- 0.043 | +0.000 |
| G1 Hungarian | 0.463 +/- 0.032 | 0.469 +/- 0.040 | +0.006 |
| G2 Switzerland | 0.211 +/- 0.097 | 0.173 +/- 0.133 | -0.039 |
| G3 VA Long Beach | 0.514 +/- 0.031 | 0.515 +/- 0.038 | +0.000 |

## mild_realistic_no_cap (gdro)

### Per-Group Accuracy

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 78.27% +/- 1.93% | 77.79% +/- 1.99% | -0.48% |
| G1 Hungarian | 80.00% +/- 2.60% | 80.11% +/- 2.41% | +0.11% |
| G2 Switzerland | 96.88% +/- 3.12% | 96.88% +/- 3.12% | +0.00% |
| G3 VA Long Beach | 77.56% +/- 0.67% | 77.78% +/- 0.99% | +0.22% |
| **Worst-group** | 77.17% +/- 0.99% | 76.86% +/- 1.21% | **-0.30%** |

### Per-Group Loss

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 0.560 +/- 0.099 | 0.549 +/- 0.035 | -0.011 |
| G1 Hungarian | 0.447 +/- 0.039 | 0.481 +/- 0.043 | +0.034 |
| G2 Switzerland | 0.191 +/- 0.124 | 0.171 +/- 0.124 | -0.020 |
| G3 VA Long Beach | 0.573 +/- 0.230 | 0.509 +/- 0.033 | -0.064 |

## moderate_cap_15_25 (gdro)

### Per-Group Accuracy

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 77.40% +/- 2.16% | 77.98% +/- 2.37% | +0.58% |
| G1 Hungarian | 81.35% +/- 3.90% | 81.35% +/- 3.56% | +0.00% |
| G2 Switzerland | 96.88% +/- 3.12% | 96.88% +/- 3.12% | +0.00% |
| G3 VA Long Beach | 77.56% +/- 1.20% | 78.22% +/- 1.33% | +0.67% |
| **Worst-group** | 76.24% +/- 1.63% | 76.78% +/- 1.49% | **+0.55%** |

### Per-Group Loss

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 0.542 +/- 0.068 | 0.555 +/- 0.042 | +0.013 |
| G1 Hungarian | 0.467 +/- 0.059 | 0.483 +/- 0.047 | +0.016 |
| G2 Switzerland | 0.251 +/- 0.099 | 0.183 +/- 0.113 | -0.068 |
| G3 VA Long Beach | 0.656 +/- 0.294 | 0.721 +/- 0.393 | +0.065 |

## moderate_cap_20_30 (gdro)

### Per-Group Accuracy

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 77.98% +/- 2.21% | 78.17% +/- 3.22% | +0.19% |
| G1 Hungarian | 80.45% +/- 2.98% | 81.69% +/- 3.22% | +1.24% |
| G2 Switzerland | 96.88% +/- 3.12% | 96.88% +/- 3.12% | +0.00% |
| G3 VA Long Beach | 77.56% +/- 1.85% | 77.56% +/- 1.85% | +0.00% |
| **Worst-group** | 77.07% +/- 2.02% | 76.88% +/- 2.20% | **-0.19%** |

### Per-Group Loss

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 0.543 +/- 0.065 | 0.542 +/- 0.072 | -0.001 |
| G1 Hungarian | 0.455 +/- 0.044 | 0.459 +/- 0.068 | +0.005 |
| G2 Switzerland | 0.228 +/- 0.151 | 0.204 +/- 0.168 | -0.024 |
| G3 VA Long Beach | 0.594 +/- 0.050 | 0.598 +/- 0.056 | +0.004 |

## moderate_no_cap (gdro)

### Per-Group Accuracy

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 78.17% +/- 2.92% | 77.69% +/- 2.46% | -0.48% |
| G1 Hungarian | 79.66% +/- 1.91% | 79.89% +/- 2.77% | +0.22% |
| G2 Switzerland | 96.88% +/- 3.12% | 96.88% +/- 3.12% | +0.00% |
| G3 VA Long Beach | 78.22% +/- 0.89% | 77.56% +/- 1.56% | -0.67% |
| **Worst-group** | 77.08% +/- 1.74% | 76.77% +/- 1.67% | **-0.31%** |

### Per-Group Loss

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 0.538 +/- 0.054 | 0.553 +/- 0.076 | +0.015 |
| G1 Hungarian | 0.468 +/- 0.071 | 0.488 +/- 0.061 | +0.020 |
| G2 Switzerland | 0.240 +/- 0.166 | 0.220 +/- 0.164 | -0.020 |
| G3 VA Long Beach | 0.552 +/- 0.013 | 0.571 +/- 0.069 | +0.019 |

## non_overlapping_no_cap (gdro)

### Per-Group Accuracy

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 74.13% +/- 2.87% | 73.65% +/- 2.89% | -0.48% |
| G1 Hungarian | 79.33% +/- 3.19% | 79.33% +/- 3.77% | -0.00% |
| G2 Switzerland | 96.25% +/- 3.06% | 96.25% +/- 3.06% | +0.00% |
| G3 VA Long Beach | 79.33% +/- 3.59% | 78.67% +/- 2.67% | -0.67% |
| **Worst-group** | 73.92% +/- 2.53% | 73.35% +/- 2.49% | **-0.57%** |

### Per-Group Loss

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 0.613 +/- 0.078 | 0.600 +/- 0.056 | -0.014 |
| G1 Hungarian | 0.467 +/- 0.078 | 0.464 +/- 0.069 | -0.003 |
| G2 Switzerland | 0.175 +/- 0.134 | 0.180 +/- 0.142 | +0.005 |
| G3 VA Long Beach | 0.578 +/- 0.195 | 0.547 +/- 0.130 | -0.031 |

## progressive_cap_15_25 (gdro)

### Per-Group Accuracy

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 77.88% +/- 2.19% | 77.88% +/- 1.72% | +0.00% |
| G1 Hungarian | 78.88% +/- 2.18% | 80.00% +/- 3.17% | +1.12% |
| G2 Switzerland | 96.88% +/- 3.12% | 96.88% +/- 3.12% | +0.00% |
| G3 VA Long Beach | 77.33% +/- 1.33% | 77.78% +/- 0.99% | +0.44% |
| **Worst-group** | 76.55% +/- 1.65% | 77.09% +/- 1.62% | **+0.54%** |

### Per-Group Loss

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 0.542 +/- 0.067 | 0.575 +/- 0.108 | +0.033 |
| G1 Hungarian | 0.465 +/- 0.028 | 0.512 +/- 0.075 | +0.047 |
| G2 Switzerland | 0.225 +/- 0.113 | 0.212 +/- 0.144 | -0.012 |
| G3 VA Long Beach | 0.645 +/- 0.220 | 0.770 +/- 0.430 | +0.126 |

## progressive_cap_20_30 (gdro)

### Per-Group Accuracy

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 77.50% +/- 1.93% | 76.73% +/- 2.05% | -0.77% |
| G1 Hungarian | 79.44% +/- 2.75% | 80.34% +/- 3.38% | +0.90% |
| G2 Switzerland | 95.62% +/- 4.00% | 96.88% +/- 3.12% | +1.25% |
| G3 VA Long Beach | 77.11% +/- 1.42% | 77.33% +/- 1.33% | +0.22% |
| **Worst-group** | 76.37% +/- 1.60% | 76.40% +/- 1.94% | **+0.03%** |

### Per-Group Loss

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 0.537 +/- 0.049 | 0.567 +/- 0.047 | +0.029 |
| G1 Hungarian | 0.484 +/- 0.061 | 0.522 +/- 0.050 | +0.038 |
| G2 Switzerland | 0.275 +/- 0.148 | 0.267 +/- 0.097 | -0.009 |
| G3 VA Long Beach | 0.543 +/- 0.022 | 0.547 +/- 0.018 | +0.004 |

## progressive_cap_20_30 (strong_eta)

### Per-Group Accuracy

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 77.50% +/- 1.93% | 77.02% +/- 1.85% | -0.48% |
| G1 Hungarian | 79.44% +/- 2.75% | 79.78% +/- 2.51% | +0.34% |
| G2 Switzerland | 95.62% +/- 4.00% | 96.88% +/- 3.12% | +1.25% |
| G3 VA Long Beach | 77.11% +/- 1.42% | 77.33% +/- 1.33% | +0.22% |
| **Worst-group** | 76.37% +/- 1.60% | 76.60% +/- 1.68% | **+0.22%** |

### Per-Group Loss

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 0.537 +/- 0.049 | 0.559 +/- 0.057 | +0.022 |
| G1 Hungarian | 0.484 +/- 0.061 | 0.495 +/- 0.059 | +0.011 |
| G2 Switzerland | 0.275 +/- 0.148 | 0.224 +/- 0.130 | -0.052 |
| G3 VA Long Beach | 0.543 +/- 0.022 | 0.568 +/- 0.048 | +0.025 |

## progressive_cap_20_30 (with_kl)

### Per-Group Accuracy

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 77.50% +/- 1.93% | 76.73% +/- 2.05% | -0.77% |
| G1 Hungarian | 79.44% +/- 2.75% | 80.34% +/- 3.38% | +0.90% |
| G2 Switzerland | 95.62% +/- 4.00% | 96.88% +/- 3.12% | +1.25% |
| G3 VA Long Beach | 77.11% +/- 1.42% | 77.33% +/- 1.33% | +0.22% |
| **Worst-group** | 76.37% +/- 1.60% | 76.40% +/- 1.94% | **+0.03%** |

### Per-Group Loss

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 0.537 +/- 0.049 | 0.567 +/- 0.047 | +0.029 |
| G1 Hungarian | 0.484 +/- 0.061 | 0.522 +/- 0.050 | +0.038 |
| G2 Switzerland | 0.275 +/- 0.148 | 0.267 +/- 0.097 | -0.009 |
| G3 VA Long Beach | 0.543 +/- 0.022 | 0.547 +/- 0.018 | +0.004 |

## progressive_no_cap (gdro)

### Per-Group Accuracy

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 77.98% +/- 2.04% | 77.02% +/- 1.74% | -0.96% |
| G1 Hungarian | 79.21% +/- 2.85% | 79.66% +/- 2.10% | +0.45% |
| G2 Switzerland | 96.88% +/- 3.12% | 96.88% +/- 3.12% | +0.00% |
| G3 VA Long Beach | 78.00% +/- 0.67% | 77.78% +/- 0.00% | -0.22% |
| **Worst-group** | 77.05% +/- 1.56% | 76.76% +/- 1.53% | **-0.29%** |

### Per-Group Loss

| Group | Baseline (ERM) | GroupDRO | Change |
|-------|----------------|----------|--------|
| G0 Cleveland | 0.515 +/- 0.041 | 0.536 +/- 0.041 | +0.021 |
| G1 Hungarian | 0.456 +/- 0.045 | 0.472 +/- 0.059 | +0.016 |
| G2 Switzerland | 0.227 +/- 0.117 | 0.207 +/- 0.124 | -0.020 |
| G3 VA Long Beach | 0.534 +/- 0.019 | 0.537 +/- 0.017 | +0.002 |
