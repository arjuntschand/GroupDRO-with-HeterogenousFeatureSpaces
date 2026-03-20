# Fed-Heart Disease: Experiment Results Summary

**Dataset**: 486 train / 254 test, 4 groups (hospitals), binary classification.

## Best Results (worst-group accuracy)

| Config | Seed | Worst-group acc | Balanced acc | Epoch (best) |
|--------|------|-----------------|--------------|--------------|
| fedheart_groupdro_earlystop | 1337 | **77.78%** | 82.02% | 6 |
| fedheart_groupdro_earlystop | 1339 | **77.78%** | 82.98% | 4 |
| fedheart_groupdro | 1337 | **77.78%** | 82.02% | 6 |
| fedheart_baseline_strong | 1339 | **77.78%** | 82.26% | 3 |
| fedheart_groupdro_strong | 1339 | **77.78%** | 82.98% | 3 |
| fedheart_baseline | 1337 | 77.53% | 81.74% | 6 |
| fedheart_baseline | 1339 | 77.53% | 82.46% | 3 |

## Multi-seed summary (seeds 1337, 1338, 1339)

| Config | Worst-group (mean) | Worst-group (min–max) | Balanced (mean) |
|--------|--------------------|----------------------|-----------------|
| **fedheart_groupdro_earlystop** | **76.85%** | 75.00–77.78% | 81.73% |
| fedheart_baseline | 76.78% | 75.28–77.53% | 81.63% |
| fedheart_groupdro_strong | 76.77% | 75.28–77.78% | 81.49% |
| fedheart_baseline_strong | 76.67% | 75.28–77.78% | 81.20% |

## Findings

1. **Best single run**: **77.78%** worst-group (multiple configs/seeds).
2. **GroupDRO vs baseline**: GroupDRO (earlystop) and baseline are close; GroupDRO can match or slightly exceed baseline on worst-group and balanced acc depending on seed.
3. **Strong regularization** (weight_decay 0.001, dropout 0.15) often gave **75.96%** for seed 1337—worse than lighter reg. On this small dataset, heavier reg can hurt.
4. **Early stopping** (patience 15) works well; best checkpoint is often at epochs 3–6.
5. **Cosine LR** and **reduce_on_plateau** did not improve over the original earlystop setup for seed 1337.

## Recommended configs for reporting

- **Baseline**: `experiments/fedheart_baseline.yaml` (or `fedheart_baseline_strong.yaml` if you want cosine LR + early stop).
- **GroupDRO**: `experiments/fedheart_groupdro_earlystop.yaml` (eta 0.1, KL 0.1, early stop 15, cosine LR).
- **Seeds**: Run 3 seeds (e.g. 1337, 1338, 1339) and report **mean ± std** worst-group and balanced accuracy.

## Commands to reproduce

```bash
# From repo root
python -m dro_hetero_anchors.src.train_fedheart --config experiments/fedheart_baseline.yaml
python -m dro_hetero_anchors.src.train_fedheart --config experiments/fedheart_groupdro_earlystop.yaml

# Multi-seed sweep (creates run_dir_seed1338, run_dir_seed1339)
python scripts/run_fedheart_sweep.py --configs fedheart_baseline fedheart_groupdro_earlystop --seeds 1337 1338 1339
python scripts/run_fedheart_sweep.py --list-only   # print all results
```

## New configs added during sweep

- `fedheart_baseline_strong.yaml` – cosine LR, early stop 15, weight_decay 0.001, dropout 0.15
- `fedheart_groupdro_strong.yaml` – GroupDRO eta 0.2, KL 0.05, same strong reg
- `fedheart_groupdro_strong_kl02.yaml` – KL 0.02
- `fedheart_groupdro_strong_eta05.yaml` – eta 0.5
- `fedheart_groupdro_long.yaml` – 80 epochs, no early stop
- `fedheart_groupdro_plateau.yaml` – reduce_on_plateau LR
