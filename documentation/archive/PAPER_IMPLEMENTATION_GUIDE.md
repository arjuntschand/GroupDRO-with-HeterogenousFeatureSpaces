# Paper / Formula Implementation Guide

This document preserves the detailed code ↔ math walkthrough formerly in IMPLEMENTATION_GUIDE.md. It is now secondary to the root README.

## Algorithm Skeleton
Refer to README Section 2 for pipeline diagram; core mathematical correspondences remain:
- Encoders φ_g: `dro_hetero_anchors/src/encoders/`
- Anchors μ_c: `dro_hetero_anchors/src/model/anchors.py`
- W₂ distance: `dro_hetero_anchors/src/model/wasserstein.py`
- Fit & separation losses: `dro_hetero_anchors/src/model/losses.py`
- GroupDRO: `dro_hetero_anchors/src/model/groupdro.py`

## Notes
See archived COMPLETE_GUIDE.md and REPO_OVERVIEW.md if historical context required.

## Key Components (Structure Overview)
This guide assumes the current structure:
- Root `README.md`: entry point (purpose, configuration, usage).
- `dro_hetero_anchors/src/`: encoders, loaders, training loop, model pieces (anchors, losses, GroupDRO, utilities).
- `experiments/`: YAML specs + `INDEX.csv`.
- `runs/`: metrics JSONL/CSV, checkpoints (`last.ckpt`, `best.ckpt`), aggregated indices, optional SQLite.
- `data/`: datasets (MNIST, USPS, etc.).
- `dro_hetero_anchors/testFiles/`: manual debug helpers.
- `dro_hetero_anchors/tests/`: unit tests (expand over time).

Config parameter `majority_frac` (float in [0,1]) adjusts majority sampling after balanced test split; default 0.8 when omitted.
Per-epoch metrics include `metrics_version` (currently `"1.0"`).
Future extension targets: additional tests, config validation, visualization utilities.

## Quick Reminder: How Training Uses Configs
Each invocation of `python -m dro_hetero_anchors.src.train --config <file>`:
1. Parses the YAML into a fresh `cfg` dict.
2. Constructs brand-new encoder modules, classification head, and anchor parameters (weights start from initialization unless a checkpoint is loaded manually).
3. Builds loaders according to flags (`use_skewed_mnist_usps`, etc.) and (now) `majority_frac` if provided.
4. Trains for `epochs`, logging per-epoch metrics and saving `last.ckpt` plus `best.ckpt` (by worst-group accuracy).

No test file (e.g., `test_wasserstein.py`) is executed automatically during training; tests run only when explicitly invoked.

### Majority Fraction (Skew) Mechanics

Contract:
- Input: `majority_frac` (float in [0,1]) specifying desired proportion of majority-class TRAIN examples post-balanced split.
- Output: DataLoaders where TRAIN set is skewed accordingly; TEST set remains perfectly balanced per class.
- Guarantees: Class presence in TEST is never reduced; TRAIN majority sampling clamps to dataset size boundaries.
- Failure Modes: Extreme small minority size can lead to high variance in worst-group metrics; warn if minority count < 50.

Edge Cases Considered:
1. `majority_frac=0.0` -> Uniform sampling of majority classes (effectively no majority boost).
2. `majority_frac=1.0` -> All possible majority examples included (upper bound limited by original train pool).
3. Very small minority class (< 10 samples) -> model may overfit; aggregation scripts still log metrics_version consistently.
4. Repeated runs with same seed produce identical splits (deterministic torch + numpy seed setting assumed upstream).
5. Non-integer target counts -> floor rounding applied; residual difference < number_of_majority_classes.

### YAML Example Snippet (Annotated)
```yaml
dataset: mnist_usps
use_skewed_mnist_usps: true
majority_frac: 0.9          # Increase majority class presence to 90% of TRAIN set
batch_size: 128
epochs: 30
encoder: cnn28
anchor_distance: wasserstein
groupdro: false             # Set true to enable GroupDRO weighting logic
log_every: 1
```

If `majority_frac` is omitted, default (0.8) is applied inside loader construction. Aggregated run indices will not explicitly list the parameter unless you add it to the metrics schema in future versions.


