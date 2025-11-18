GroupDRO with Heterogeneous Feature Spaces & Latent Anchors
===========================================================

Unified README consolidating prior guides (README.md, REPO_OVERVIEW.md, COMPLETE_GUIDE.md, MAPPING.md). This is now the single entry point for purpose, architecture, directory map, configuration, and usage.

## 1. Purpose

Train a robust classifier across heterogeneous feature spaces (e.g., MNIST 28×28 vs USPS 16×16, optionally MNIST 32×32) using:
- Per‑group encoders φ_g mapping raw inputs to a shared latent space ℝ^k.
- Class Gaussian anchors μ_c = N(m_c, S_c) (learned means m_c and low-rank factors L_c → S_c = L_c L_c^T + εI).
- Three losses: (1) classification (optionally GroupDRO reweighted), (2) anchor fit (Gaussian W₂ / Bures distance), (3) anchor separation (classifier synthetic samples or W₂ margin).
- Objective emphasizes worst‑group performance (robustness) while aligning latent distributions across domains.

## 2. High-Level Flow

```
Input batch (mixed groups)
  → Split by group g
    → Per-group encoder φ_g
      → Concatenate latent vectors z (B × latent_dim)
        → Classification head ψ → logits
          → CE (standard or GroupDRO-weighted)
          → Batch per-class latent moments (m̂_c, Ŝ_c)
          → Anchor fit loss: W₂( (m̂_c,Ŝ_c), (m_c,S_c) )
          → Anchor separation loss (synthetic CE or W₂ margin)
            → Total loss = CE + λ_fit * L_fit + λ_sep * L_sep
              → Backprop + optimizer step (+ GroupDRO weight update)
```

## 3. Directory Tree (Essential Parts)

```
.
├── README.md                # This file
├── documentation/           # Remaining detailed docs & archives
│   ├── PAPER_IMPLEMENTATION_GUIDE.md  # Deep code ↔ formula walkthrough
│   ├── archive/             # Archived legacy docs & cleanup notes
│   │   ├── CLEANUP_NOTES.md
│   │   ├── COMPLETE_GUIDE.md
│   │   ├── MAPPING.md
│   │   └── REPO_OVERVIEW.md
├── dro_hetero_anchors/      # Core package
│   ├── experiments/         # YAML configs (INDEX.csv auto-generated)
│   ├── src/
│   │   ├── train.py         # Training loop + evaluation + logging
│   │   ├── eval.py          # Standalone checkpoint evaluation
│   │   ├── datasets.py      # Loader builders + split & skew helper
│   │   ├── encoders/        # cnn28, cnn32 registry
│   │   ├── model/           # anchors, losses, head, groupdro, wasserstein
│   │   ├── results_logger.py# JSONL + CSV writer
│   │   ├── utils.py         # Seed, meters, console
│   │   └── tools/           # aggregate_runs, index_experiments
│   ├── testFiles/           # Relocated debug & test utility scripts
│   └── tests/               # Formal unit tests (to expand)
├── runs/                    # Aggregated run outputs (metrics, ckpts, index.csv, metrics.sqlite)
├── data/                    # Downloaded datasets (MNIST, USPS, etc.)
├── requirements.txt         # Python dependencies
└── .venv/                   # Primary virtual environment (keep only one)
```

## 4. Virtual Environments

Keep a single `.venv/` at the repository root. A nested `dro_hetero_anchors/.venv` existed and is redundant—remove it to avoid PATH confusion.

## 5. Installation & Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# List experiments
python -m dro_hetero_anchors.src.tools.index_experiments
cat dro_hetero_anchors/experiments/INDEX.csv | head

# Run a config
python -m dro_hetero_anchors.src.train --config dro_hetero_anchors/experiments/mnist_usps_25k_2k.yaml

# Evaluate latest checkpoint
python -m dro_hetero_anchors.src.eval --config dro_hetero_anchors/experiments/mnist_usps_25k_2k.yaml --ckpt runs/last.ckpt

# Aggregate runs catalog (CSV + SQLite)
python -m dro_hetero_anchors.src.tools.aggregate_runs
```

## 6. Configuration Schema (Representative Keys)

Data / Groups:
- Flags: `use_skewed_mnist_usps`, `use_skewed_mnist_usps_mnist32`, `use_usps_only_balanced`
- Sizes: `mnist_size`, `usps_size`, `mnist28_size`, `mnist32_size`
- Majority classes: `mnist_majority`, `usps_majority`, `mnist28_majority`, `mnist32_majority`
- (Future) `majority_frac` – parameterize skew (currently implicit 0.8)

Model:
- `latent_dim`, `num_classes`, `head_hidden` (0 → linear head, >0 → MLP)

Anchors & Separation:
- `anchor_eps`, `sep_samples_per_class`, `sep_method` in {`classifier`, `w2_margin`}, `sep_margin`
- Loss weights: `lambda_fit`, `lambda_sep`

Optimization & Logging:
- `lr`, `weight_decay`, `epochs`, `grad_clip`, `log_interval`, `save_every`, `seed`

GroupDRO:
- `groupdro_enabled`, `groupdro_eta`

## 7. Metrics & Logging

Per‑epoch JSONL record includes (typical fields):
```json
{
  "epoch": 5,
  "train_loss": 0.913,
  "train_acc": 0.842,
  "test_acc": 0.857,
  "worst_group_acc": 0.733,
  "per_group_acc": [0.733, 0.958],
  "per_class_acc": {...},
  "per_group_by_class_acc": {...},
  "metrics_version": "1.0"
}
```
Flattened CSV mirrors JSONL for spreadsheet ingestion. Run catalog: `runs/index.csv` & `runs/metrics.sqlite` (generated by `aggregate_runs.py`).

## 8. Core Modules & Paper Mapping

| Paper Concept | Code Location |
|---------------|---------------|
| Group encoder φ_g | `src/encoders/` (cnn28, cnn32) |
| Shared latent | Output of encoders (concatenated per batch) |
| Classification head ψ | `src/model/head.py` |
| Gaussian anchors μ_c | `src/model/anchors.py` |
| Anchor fit (W₂) | `src/model/losses.py`, `src/model/wasserstein.py` |
| Anchor separation | `src/model/losses.py` (`anchor_sep_loss`) |
| GroupDRO weighting | `src/model/groupdro.py` |
| Per-class moments | `src/model/losses.py` (`per_class_batch_moments`) |

## 9. Debug & Utility Scripts (now in `testFiles/`)

Moved for clarity:
- `debug_batches.py`: Inspect early batch class composition & detect missing classes.
- `debug_predictions.py`: Reconstruct train/test split, evaluate confusion per USPS class.
- `debug_usps.py`: Minimal binary USPS classification sanity check (class 1 vs 5) to validate feature learnability.
- `test_train.py`: Smoke test: purge `runs/`, execute a short training run, assert checkpoint creation.

These are development aids, not part of the core library API.

## 10. Duplicate Directories & Cleanup Recommendations

Virtual environments:
- Keep only root `.venv/`. Remove `dro_hetero_anchors/.venv` manually to eliminate ambiguity (`rm -rf dro_hetero_anchors/.venv`).

Runs:
- Root `runs/` is canonical (contains metrics, checkpoints, aggregated index). The nested `dro_hetero_anchors/runs/` had a stray event file; recommend deleting or merging its contents.

Data:
- Root `data/` centralizes datasets; no need for nested package `data/` copies unless isolating experiments. Consolidate to root.

## 11. Planned Enhancements

Short-term:
- Parameterize skew ratio via `majority_frac` in configs.
- Add unit tests: split invariants, padding correctness, small anchor loss numerical stability.
- Add config validation (pydantic) to catch missing keys early.

Medium-term:
- Extend to new heterogeneous dataset (e.g., TextCaps) with resolution-based groups.
- Add simple visualization CLI for worst‑group trend comparisons.

## 12. Glossary

- Worst‑group accuracy: Minimum accuracy across groups per epoch (robust objective).
- Anchor fit loss: W₂ distance aligning batch latent moments to anchor Gaussians.
- Anchor separation loss: Encourages distinct anchors (classification of synthetic samples or enforced W₂ margin).
- GroupDRO: Adaptive group reweighting focusing optimization on underperforming groups.

## 13. FAQ

Q: Why anchors instead of prototypes?  
A: Anchors model covariance, letting W₂ capture shape differences, not just mean shifts.

Q: Why pad inputs?  
A: Mixed resolutions in one batch require uniform tensor shape; we pad smaller images to the max (instead of resizing all up/down uniformly).

Q: Can I add a new dataset group?  
A: Implement a loader returning `(train_loader, test_loader)` with `(x,y,g)` tuples, add a config flag, and update `train.py` to branch on it.

## 14. Attribution & License

Include a license file (e.g., MIT) and citation block here in a future pass.

---
Single-source README established. Legacy docs archived under `documentation/archive/` for historical reference.
