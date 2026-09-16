# EMBED (Xenia spec) — results

seeds: [np.int64(0), np.int64(1), np.int64(7), np.int64(11), np.int64(22), np.int64(33), np.int64(42), np.int64(1337), np.int64(2024), np.int64(31337)]   groups: ['g1', 'g2', 'g3', 'g4', 'g5', 'g6']
n_params (shared model): 276,228
test-set size per group: g1=209, g2=37, g3=128, g4=14665, g5=14, g6=10579
(tail groups g2/g5 are tiny → their per-group accuracy is high-variance)


## Accuracy (per group; overall/tail/worst = mean±std over seeds)

| method | g1 | g2 | g3 | g4 | g5 | g6 | overall | tail | worst |
|---|---|---|---|---|---|---|---|---|---|
| ERM | 0.712 | 0.578 | 0.680 | 0.749 | 0.664 | 0.755 | 0.690±0.008 | 0.659±0.011 | 0.578±0.019 |
| Anchors only (no DRO) | 0.696 | 0.603 | 0.660 | 0.746 | 0.543 | 0.754 | 0.667±0.010 | 0.625±0.010 | 0.542±0.036 |
| GroupDRO (raw loss) | 0.724 | 0.624 | 0.699 | 0.772 | 0.700 | 0.778 | 0.716±0.007 | 0.687±0.010 | 0.624±0.009 |
| Ours: anchors + GroupDRO | 0.712 | 0.622 | 0.686 | 0.763 | 0.657 | 0.773 | 0.702±0.010 | 0.669±0.014 | 0.614±0.022 |
| Regret-DRO (no anchors) | 0.717 | 0.578 | 0.683 | 0.752 | 0.671 | 0.759 | 0.694±0.008 | 0.662±0.012 | 0.578±0.019 |
| Ours: anchors + regret-DRO | 0.701 | 0.619 | 0.711 | 0.748 | 0.693 | 0.758 | 0.705±0.011 | 0.681±0.013 | 0.619±0.027 |
| Group-only (dedicated) | 0.679 | 0.595 | 0.662 | 0.729 | 0.550 | 0.742 | 0.659±0.010 | 0.622±0.015 | 0.545±0.040 |

## Macro-F1 (per group; overall/tail/worst = mean±std over seeds)

| method | g1 | g2 | g3 | g4 | g5 | g6 | overall | tail | worst |
|---|---|---|---|---|---|---|---|---|---|
| ERM | 0.537 | 0.259 | 0.527 | 0.634 | 0.381 | 0.689 | 0.504±0.012 | 0.426±0.016 | 0.689±0.012 |
| Anchors only (no DRO) | 0.458 | 0.272 | 0.530 | 0.618 | 0.312 | 0.682 | 0.479±0.025 | 0.393±0.025 | 0.682±0.024 |
| GroupDRO (raw loss) | 0.610 | 0.277 | 0.601 | 0.692 | 0.405 | 0.723 | 0.551±0.008 | 0.473±0.011 | 0.723±0.009 |
| Ours: anchors + GroupDRO | 0.547 | 0.286 | 0.549 | 0.665 | 0.378 | 0.713 | 0.523±0.013 | 0.440±0.019 | 0.713±0.009 |
| Regret-DRO (no anchors) | 0.536 | 0.259 | 0.534 | 0.642 | 0.386 | 0.696 | 0.509±0.014 | 0.429±0.017 | 0.696±0.011 |
| Ours: anchors + regret-DRO | 0.415 | 0.281 | 0.544 | 0.628 | 0.400 | 0.695 | 0.494±0.026 | 0.410±0.029 | 0.695±0.016 |
| Group-only (dedicated) | 0.371 | 0.265 | 0.432 | 0.558 | 0.316 | 0.653 | 0.433±0.010 | 0.346±0.010 | 0.653±0.017 |

## Cross-entropy loss (per group; overall/tail/worst = mean±std over seeds)

| method | g1 | g2 | g3 | g4 | g5 | g6 | overall | tail | worst |
|---|---|---|---|---|---|---|---|---|---|
| ERM | 0.650 | 1.817 | 0.958 | 0.579 | 2.398 | 0.563 | 1.161±0.022 | 1.456±0.032 | 2.398±0.149 |
| Anchors only (no DRO) | 0.743 | 1.131 | 0.903 | 0.629 | 1.334 | 0.612 | 0.892±0.021 | 1.028±0.042 | 1.340±0.104 |
| GroupDRO (raw loss) | 0.659 | 0.999 | 0.790 | 0.533 | 1.200 | 0.516 | 0.783±0.021 | 0.912±0.032 | 1.200±0.159 |
| Ours: anchors + GroupDRO | 0.698 | 1.073 | 0.817 | 0.571 | 1.202 | 0.554 | 0.819±0.018 | 0.947±0.025 | 1.211±0.116 |
| Regret-DRO (no anchors) | 0.642 | 1.618 | 0.879 | 0.573 | 2.162 | 0.556 | 1.072±0.024 | 1.325±0.035 | 2.162±0.145 |
| Ours: anchors + regret-DRO | 0.723 | 1.004 | 0.785 | 0.627 | 1.099 | 0.610 | 0.808±0.018 | 0.903±0.022 | 1.099±0.082 |
| Group-only (dedicated) | 0.726 | 1.325 | 0.780 | 0.634 | 1.927 | 0.600 | 0.999±0.057 | 1.190±0.088 | 1.927±0.281 |

## Headline (dual reporting)

(overall-wt = sample-weighted 'entire dataset'; overall-macro = mean over the 6 groups; tail = mean over tail groups.)

### A. Xenia spec — 3 seeds  (seeds [0, 1, 42])

| method | overall-wt | overall-macro | tail | worst-group |
|---|---|---|---|---|
| ERM | 0.752 ± 0.002 | 0.686 ± 0.008 | 0.653 ± 0.012 | 0.559 ± 0.016 |
| Anchors only (no DRO) | 0.748 ± 0.012 | 0.668 ± 0.004 | 0.627 ± 0.002 | 0.546 ± 0.040 |
| GroupDRO (raw loss) | 0.772 ± 0.003 | 0.717 ± 0.006 | 0.689 ± 0.008 | 0.622 ± 0.000 |
| Ours: anchors + GroupDRO | 0.768 ± 0.003 | 0.693 ± 0.004 | 0.655 ± 0.006 | 0.601 ± 0.026 |
| Regret-DRO (no anchors) | 0.755 ± 0.001 | 0.689 ± 0.008 | 0.655 ± 0.012 | 0.559 ± 0.016 |
| Ours: anchors + regret-DRO | 0.747 ± 0.002 | 0.699 ± 0.008 | 0.674 ± 0.012 | 0.622 ± 0.027 |
| Group-only (dedicated) | 0.732 ± 0.006 | 0.655 ± 0.003 | 0.616 ± 0.006 | 0.524 ± 0.041 |

### B. Extended — 10 seeds  (seeds [0, 1, 7, 11, 22, 33, 42, 1337, 2024, 31337])

| method | overall-wt | overall-macro | tail | worst-group |
|---|---|---|---|---|
| ERM | 0.751 ± 0.003 | 0.690 ± 0.008 | 0.659 ± 0.011 | 0.578 ± 0.019 |
| Anchors only (no DRO) | 0.748 ± 0.011 | 0.667 ± 0.010 | 0.625 ± 0.010 | 0.542 ± 0.036 |
| GroupDRO (raw loss) | 0.773 ± 0.004 | 0.716 ± 0.007 | 0.687 ± 0.010 | 0.624 ± 0.009 |
| Ours: anchors + GroupDRO | 0.766 ± 0.004 | 0.702 ± 0.010 | 0.669 ± 0.014 | 0.614 ± 0.022 |
| Regret-DRO (no anchors) | 0.754 ± 0.003 | 0.694 ± 0.008 | 0.662 ± 0.012 | 0.578 ± 0.019 |
| Ours: anchors + regret-DRO | 0.752 ± 0.010 | 0.705 ± 0.011 | 0.681 ± 0.013 | 0.619 ± 0.027 |
| Group-only (dedicated) | 0.733 ± 0.004 | 0.659 ± 0.010 | 0.622 ± 0.015 | 0.545 ± 0.040 |

_Both are reported: **A** matches the protocol in `EMBED_Experiments_Description.docx` exactly (seeds 0/1/42); **B** adds 7 more seeds for tighter error bars, matching the 10-seed rigor used on the tabular datasets. Prefer B for any claim of significance; A for direct comparison to the spec._

## Per-method detail (all seeds)

(overall-macro = mean over the 6 groups; overall-weighted = sample-weighted 'entire dataset', dominated by heads g4/g6; tail = mean over tail groups.)

- **ERM**: overall-weighted 0.751±0.003, overall-macro 0.690±0.008, tail 0.659±0.011, worst-group 0.578±0.019
- **Anchors only (no DRO)**: overall-weighted 0.748±0.011, overall-macro 0.667±0.010, tail 0.625±0.010, worst-group 0.542±0.036
- **GroupDRO (raw loss)**: overall-weighted 0.773±0.004, overall-macro 0.716±0.007, tail 0.687±0.010, worst-group 0.624±0.009
- **Ours: anchors + GroupDRO**: overall-weighted 0.766±0.004, overall-macro 0.702±0.010, tail 0.669±0.014, worst-group 0.614±0.022
- **Regret-DRO (no anchors)**: overall-weighted 0.754±0.003, overall-macro 0.694±0.008, tail 0.662±0.012, worst-group 0.578±0.019
- **Ours: anchors + regret-DRO**: overall-weighted 0.752±0.010, overall-macro 0.705±0.011, tail 0.681±0.013, worst-group 0.619±0.027
- **Group-only (dedicated)**: overall-weighted 0.733±0.004, overall-macro 0.659±0.010, tail 0.622±0.015, worst-group 0.545±0.040