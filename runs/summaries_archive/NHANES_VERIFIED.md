# NHANES CVD — verified results (from nhanes_all_results.json)

Worst-group / Overall accuracy, mean±std over seeds. Supersedes the poster's
mislabeled 83.5% (that value was balanced-accuracy, not worst-group).


## Nested feature mode

| config | worst-group | overall |
|---|---|---|
| Shared ERM | 68.97 ± 1.74 | 71.80 ± 2.02 |
| Shared GroupDRO | 71.37 ± 0.80 | 73.04 ± 0.35 |
| Per-group ERM | 69.08 ± 1.68 | 71.09 ± 2.30 |
| Per-group GroupDRO | 70.11 ± 0.66 | 71.29 ± 0.57 |

- GDRO gain (per-group): +1.03 pts | full-stack (shared ERM→per-group GDRO): +1.14 pts

## Expanded feature mode

| config | worst-group | overall |
|---|---|---|
| Shared ERM | 74.30 ± 0.98 | 77.68 ± 0.97 |
| Shared GroupDRO | 76.91 ± 1.03 | 77.94 ± 1.02 |
| Per-group ERM | 76.67 ± 1.69 | 78.28 ± 1.74 |
| Per-group GroupDRO | 76.86 ± 2.20 | 77.94 ± 2.24 |

- GDRO gain (per-group): +0.19 pts | full-stack (shared ERM→per-group GDRO): +2.56 pts

## Disjoint feature mode

| config | worst-group | overall |
|---|---|---|
| Shared ERM | 70.92 ± 1.84 | 72.73 ± 1.51 |
| Shared GroupDRO | 72.28 ± 1.22 | 73.75 ± 1.06 |
| Per-group ERM | 71.83 ± 1.90 | 73.17 ± 2.03 |
| Per-group GroupDRO | 72.69 ± 1.32 | 73.57 ± 1.02 |

- GDRO gain (per-group): +0.87 pts | full-stack (shared ERM→per-group GDRO): +1.77 pts