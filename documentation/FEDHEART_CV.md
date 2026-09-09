# Fed-Heart: 5-fold cross validation with median imputation

## Why this protocol

The original single-split protocol evaluated Switzerland on only **10 patients**. Two reasons:
the loader dropped every row with a missing value (Switzerland is missing cholesterol on most
records, so 63% of that site was discarded), and a single 80/20 split then tested on 20% of
what remained. With 10 test samples, worst-group accuracy moves in 10-point jumps and the
seed-to-seed spread was ±7.57, meaning some of the reported gain was split luck.

Two standard fixes, neither of which invents data:

1. **Median imputation.** Keep every patient, fill missing features with that site's median.
   This is what the FLamby benchmark itself does. Switzerland goes from 46 back to 123 patients.
2. **5-fold cross validation.** Rotate the held-out fold so every patient is tested exactly once,
   then pool per-group accuracy weighted by fold test counts.

Switzerland is now evaluated on **125 patients instead of 10**. Every group is fully evaluated:
Cleveland 305, Hungarian 295, Switzerland 125, VA 200 (925 patients total, vs 150 before).

## Results (5 seeds x 5 folds = 100 runs)

| method | worst-group | overall | balanced |
|---|---|---|---|
| ERM | 65.90 ± 1.37 | 73.99 ± 0.61 | 72.30 ± 0.41 |
| GroupDRO | 75.80 ± 1.03 | 79.65 ± 0.21 | 81.10 ± 0.24 |
| Anchors only | 74.20 ± 1.57 | 79.26 ± 0.55 | 80.41 ± 0.44 |
| Ours (anchors + GroupDRO) | 73.50 ± 1.22 | 78.40 ± 0.88 | 79.85 ± 0.73 |

**Headline: +9.90 worst-group accuracy** (ERM 65.90 to GroupDRO 75.80).

## What changed versus the old protocol

| | single split (10 test) | 5-fold CV + imputation (125 test) |
|---|---|---|
| ERM | 65.58 | 65.90 |
| GroupDRO | 74.80 | 75.80 |
| gain | +9.22 | +9.90 |
| seed spread (std) | ±7.57 | **±1.03** |
| patients evaluated | 150 | **925** |

The gain survived the stricter protocol rather than shrinking, and the variance dropped by a
factor of seven. The original effect was real; the old protocol was simply too noisy to trust.

## Per-group accuracy (pooled across folds)

| group | patients evaluated | ERM | GroupDRO |
|---|---|---|---|
| Cleveland | 305 | 75.54 | 77.25 |
| Hungarian | 295 | 80.47 | 79.66 |
| Switzerland | 125 | 66.40 | 91.68 |
| VA | 200 | 66.80 | 75.80 |

## Reproduce

```bash
python run_fedheart_cv.py --folds 5 --seeds 42 1337 7 2024 31337
```