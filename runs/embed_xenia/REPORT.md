# EMBED (Xenia spec) — results

seeds: [np.int64(0), np.int64(1), np.int64(42)]   groups: ['g1', 'g2', 'g3', 'g4', 'g5', 'g6']
n_params (shared model): 276,228
test-set size per group: g1=167, g2=17, g3=97, g4=1474, g5=5, g6=1557
(tail groups g2/g5 are tiny → their per-group accuracy is high-variance)


## Accuracy (per group; overall/tail/worst = mean±std over seeds)

| method | g1 | g2 | g3 | g4 | g5 | g6 | overall | tail | worst |
|---|---|---|---|---|---|---|---|---|---|
| ERM | 0.731 | 0.569 | 0.691 | 0.760 | 0.800 | 0.729 | 0.713±0.015 | 0.697±0.023 | 0.569±0.068 |
| GroupDRO (R*=0) | 0.741 | 0.529 | 0.687 | 0.759 | 0.800 | 0.729 | 0.708±0.003 | 0.689±0.004 | 0.529±0.000 |
| Ablation: align-only | 0.725 | 0.588 | 0.701 | 0.729 | 0.733 | 0.720 | 0.699±0.023 | 0.687±0.036 | 0.580±0.087 |
| Ablation: regret-only | 0.733 | 0.510 | 0.691 | 0.748 | 0.800 | 0.717 | 0.700±0.009 | 0.683±0.011 | 0.510±0.034 |
| Ours (anchors+regret) | 0.725 | 0.588 | 0.701 | 0.730 | 0.733 | 0.720 | 0.700±0.023 | 0.687±0.036 | 0.580±0.087 |
| Group-only (dedicated) | 0.747 | 0.588 | 0.715 | 0.750 | 0.733 | 0.716 | 0.708±0.021 | 0.696±0.031 | 0.588±0.059 |

## Macro-F1 (per group; overall/tail/worst = mean±std over seeds)

| method | g1 | g2 | g3 | g4 | g5 | g6 | overall | tail | worst |
|---|---|---|---|---|---|---|---|---|---|
| ERM | 0.466 | 0.286 | 0.454 | 0.641 | 0.450 | 0.652 | 0.492±0.011 | 0.414±0.018 | 0.652±0.005 |
| GroupDRO (R*=0) | 0.477 | 0.268 | 0.429 | 0.631 | 0.450 | 0.659 | 0.486±0.005 | 0.406±0.006 | 0.659±0.018 |
| Ablation: align-only | 0.405 | 0.293 | 0.429 | 0.531 | 0.422 | 0.635 | 0.452±0.003 | 0.387±0.008 | 0.635±0.009 |
| Ablation: regret-only | 0.411 | 0.264 | 0.415 | 0.608 | 0.450 | 0.643 | 0.465±0.017 | 0.385±0.016 | 0.647±0.030 |
| Ours (anchors+regret) | 0.405 | 0.293 | 0.429 | 0.534 | 0.422 | 0.635 | 0.453±0.003 | 0.387±0.008 | 0.635±0.010 |
| Group-only (dedicated) | 0.433 | 0.289 | 0.416 | 0.577 | 0.411 | 0.624 | 0.459±0.014 | 0.387±0.022 | 0.624±0.027 |

## Cross-entropy loss (per group; overall/tail/worst = mean±std over seeds)

| method | g1 | g2 | g3 | g4 | g5 | g6 | overall | tail | worst |
|---|---|---|---|---|---|---|---|---|---|
| ERM | 0.626 | 0.762 | 0.684 | 0.590 | 0.842 | 0.610 | 0.686±0.013 | 0.729±0.020 | 0.876±0.040 |
| GroupDRO (R*=0) | 0.632 | 0.772 | 0.683 | 0.595 | 0.837 | 0.617 | 0.689±0.015 | 0.731±0.020 | 0.871±0.040 |
| Ablation: align-only | 0.741 | 0.951 | 0.801 | 0.716 | 0.828 | 0.716 | 0.792±0.021 | 0.830±0.029 | 0.956±0.025 |
| Ablation: regret-only | 0.650 | 0.770 | 0.708 | 0.613 | 0.830 | 0.634 | 0.701±0.011 | 0.740±0.016 | 0.852±0.046 |
| Ours (anchors+regret) | 0.742 | 0.947 | 0.802 | 0.716 | 0.825 | 0.716 | 0.791±0.021 | 0.829±0.029 | 0.952±0.023 |
| Group-only (dedicated) | 0.641 | 1.387 | 0.752 | 0.622 | 1.573 | 0.644 | 0.936±0.080 | 1.088±0.123 | 1.574±0.325 |

## Headline (dual reporting)

(overall-wt = sample-weighted 'entire dataset'; overall-macro = mean over the 6 groups; tail = mean over tail groups.)

### A. Xenia spec — 3 seeds  (seeds [0, 1, 42])

| method | overall-wt | overall-macro | tail | worst-group |
|---|---|---|---|---|
| ERM | 0.741 ± 0.000 | 0.713 ± 0.015 | 0.697 ± 0.023 | 0.569 ± 0.068 |
| GroupDRO (R*=0) | 0.741 ± 0.003 | 0.708 ± 0.003 | 0.689 ± 0.004 | 0.529 ± 0.000 |
| Ablation: align-only | 0.723 ± 0.005 | 0.699 ± 0.023 | 0.687 ± 0.036 | 0.580 ± 0.087 |
| Ablation: regret-only | 0.730 ± 0.006 | 0.700 ± 0.009 | 0.683 ± 0.011 | 0.510 ± 0.034 |
| Ours (anchors+regret) | 0.724 ± 0.005 | 0.700 ± 0.023 | 0.687 ± 0.036 | 0.580 ± 0.087 |
| Group-only (dedicated) | 0.732 ± 0.004 | 0.708 ± 0.021 | 0.696 ± 0.031 | 0.588 ± 0.059 |

## Per-method detail (all seeds)

(overall-macro = mean over the 6 groups; overall-weighted = sample-weighted 'entire dataset', dominated by heads g4/g6; tail = mean over tail groups.)

- **ERM**: overall-weighted 0.741±0.000, overall-macro 0.713±0.015, tail 0.697±0.023, worst-group 0.569±0.068
- **GroupDRO (R*=0)**: overall-weighted 0.741±0.003, overall-macro 0.708±0.003, tail 0.689±0.004, worst-group 0.529±0.000
- **Ablation: align-only**: overall-weighted 0.723±0.005, overall-macro 0.699±0.023, tail 0.687±0.036, worst-group 0.580±0.087
- **Ablation: regret-only**: overall-weighted 0.730±0.006, overall-macro 0.700±0.009, tail 0.683±0.011, worst-group 0.510±0.034
- **Ours (anchors+regret)**: overall-weighted 0.724±0.005, overall-macro 0.700±0.023, tail 0.687±0.036, worst-group 0.580±0.087
- **Group-only (dedicated)**: overall-weighted 0.732±0.004, overall-macro 0.708±0.021, tail 0.696±0.031, worst-group 0.588±0.059