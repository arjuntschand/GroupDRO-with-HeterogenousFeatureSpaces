GroupDRO with Heterogeneous Feature Spaces & Latent Anchors
===========================================================

Authoritative README for purpose, architecture, directory map, configuration, and usage.

## 1. Purpose

Train a robust classifier across heterogeneous feature spaces using:
- Per‑group encoders φ_g mapping raw inputs to a shared latent space ℝ^k.
- Class Gaussian anchors μ_c = N(m_c, S_c) (learned means m_c and low-rank factors L_c → S_c = L_c L_c^T + εI).
- Three losses: (1) classification (optionally GroupDRO reweighted), (2) anchor fit (Gaussian W₂ / Bures distance), (3) anchor separation (classifier synthetic samples or W₂ margin).
- Objective emphasizes worst‑group performance (robustness) while aligning latent distributions across domains.

**Supported Datasets:**
- **MNIST/USPS**: Image classification with heterogeneous resolutions (28×28 vs 16×16)
- **Fed-Heart Disease**: Tabular binary classification across 4 hospitals (from FLamby)

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
│   ├── experiments/         # (moved to repo root)
│   ├── src/
│   │   ├── train.py         # Training loop + evaluation + logging
│   │   ├── eval.py          # Standalone checkpoint evaluation
│   │   ├── datasets.py      # Loader builders + split & skew helper
│   │   ├── encoders/        # cnn28, cnn32 registry
│   │   ├── model/           # anchors, losses, head, groupdro, wasserstein
│   │   ├── results_logger.py# JSONL + CSV writer
│   │   ├── utils.py         # Seed, meters, console
│   │   └── tools/           # aggregate_runs, index_experiments
│   ├── requirements.txt     # Python dependencies
│   ├── testFiles/           # Relocated debug & test utility scripts
│   └── tests/               # Formal unit tests (to expand)
├── runs/                    # Aggregated run outputs (metrics, ckpts, index.csv, metrics.sqlite)
├── datasets/                # Downloaded datasets (MNIST, USPS, etc.)
└── .venv/                   # Primary virtual environment (keep only one)
```

## 4. Virtual Environments

Keep a single `.venv/` at the repository root. A nested `dro_hetero_anchors/.venv` existed before and was removed to avoid PATH confusion.

## 5. Installation & Quickstart

**All commands below are run from the repository root** (`GroupDRO-with-HeterogenousFeatureSpaces/`).

```bash
# Create and activate venv (repo root)
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies (requirements live in dro_hetero_anchors/)
pip install -r dro_hetero_anchors/requirements.txt

# Fed-Heart Disease: FLamby is optional. If you don't install it, the loader
# will auto-download UCI Heart Disease data to datasets/fed_heart_disease/
# (no extra deps). To use FLamby instead (e.g. for exact FLamby splits):
#   pip install git+https://github.com/owkin/FLamby.git

# Optional: use venv Python explicitly without activating (from repo root)
# .venv/bin/python -m dro_hetero_anchors.src.train --config experiments/...

# List experiments
python -m dro_hetero_anchors.src.tools.index_experiments
cat experiments/INDEX.csv | head

# Run a config (MNIST/USPS)
python -m dro_hetero_anchors.src.train --config experiments/mnist_usps_25k_2k.yaml

# Run Fed-Heart Disease baseline
python -m dro_hetero_anchors.src.train_fedheart --config experiments/fedheart_baseline.yaml

# Run Fed-Heart Disease with GroupDRO
python -m dro_hetero_anchors.src.train_fedheart --config experiments/fedheart_groupdro.yaml

# Evaluate latest checkpoint
python -m dro_hetero_anchors.src.eval --config experiments/mnist_usps_25k_2k.yaml --ckpt runs/last.ckpt

# Aggregate runs catalog (CSV + SQLite)
python -m dro_hetero_anchors.src.tools.aggregate_runs
```

## 5.1 Fed-Heart Disease Dataset

Fed-Heart Disease is a federated binary classification dataset from [FLamby](https://github.com/owkin/FLamby):
- **4 groups**: Each hospital is a different group with potentially different data distributions
- **13 features**: Tabular patient data after preprocessing
- **Binary classification**: Heart disease (positive) vs no heart disease (negative)

**Key features of our implementation:**
1. **Stratified batch sampling**: Each batch maintains the group proportions π from the training set
2. **GroupDRO with KL penalty**: Weights q are initialized to π and regularized via KL(q||π) to prevent over-focusing on minority groups
3. **No warmup, no weight clamping**: Weights can freely adapt from the start
4. **Comprehensive logging**: Per-group, per-class, and per-group-per-class metrics

**Configuration options:**
```yaml
# GroupDRO settings
groupdro_enabled: true
groupdro_eta: 0.1            # Step size for weight updates
groupdro_gamma: 0.9          # Smoothing factor
groupdro_update_mode: exp_smooth
groupdro_objective: weighted
groupdro_kl_lambda: 0.1      # KL(q||π) penalty - tune this!
```

**Hyperparameter tuning:**
- `groupdro_kl_lambda`: Start with 0.1; lower (0.01-0.05) allows more focus on worst group, higher (0.5) keeps closer to original distribution
- `groupdro_eta`: Controls how fast weights shift; 0.1 is reasonable default

## 6. Configuration Schema (Representative Keys)

Data / Groups:
- Flags: `use_skewed_mnist_usps`, `use_skewed_mnist_usps_mnist32`, `use_usps_only_balanced`
- Sizes: `mnist_size`, `usps_size`, `mnist28_size`, `mnist32_size`
- Majority classes: `mnist_majority`, `usps_majority`, `mnist28_majority`, `mnist32_majority`
- `majority_frac`: fraction of TRAIN samples drawn from specified majority classes after creating a balanced per-class TEST split (omit to use default 0.8)

Model:
- `latent_dim`, `num_classes`, `head_hidden` (0 → linear head, >0 → MLP)

Anchors & Separation:
- `anchor_eps`, `sep_samples_per_class`, `sep_method` in {`classifier`, `w2_margin`}, `sep_margin`
- Loss weights: `lambda_fit`, `lambda_sep`

Optimization & Logging:
- `lr`, `weight_decay`, `epochs`, `grad_clip`, `log_interval`, `save_every`, `seed`

GroupDRO:
- `groupdro_enabled`: Enable/disable GroupDRO (default: false)
- `groupdro_eta`: Step size for weight updates (default: 0.1)
- `groupdro_gamma`: Smoothing factor for exp_smooth mode (default: 1.0)
- `groupdro_update_mode`: Weight update strategy (`exp`, `softmax`, `exp_smooth`)
- `groupdro_objective`: Loss aggregation (`weighted`, `max`, `logsumexp`)
- `groupdro_kl_lambda`: KL(q||π) penalty coefficient (default: 0.0, recommended: 0.1)
- Note: Weights are initialized proportional to group sample sizes (π)

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
| Group encoder φ_g | `src/encoders/` (cnn28, cnn32, mlp_tabular) |
| Shared latent | Output of encoders (concatenated per batch) |
| Classification head ψ | `src/model/head.py` |
| Gaussian anchors μ_c | `src/model/anchors.py` |
| Anchor fit (W₂) | `src/model/losses.py`, `src/model/wasserstein.py` |
| Anchor separation | `src/model/losses.py` (`anchor_sep_loss`) |
| GroupDRO weighting | `src/model/groupdro.py` |
| KL penalty | `src/model/groupdro.py` (`compute_kl_divergence`) |
| Per-class moments | `src/model/losses.py` (`per_class_batch_moments`) |
| Fed-Heart Disease | `src/datasets_fedheart.py`, `src/train_fedheart.py` |

## 9. Canonical Directories

Virtual environment: root `.venv/`.
Runs output: root `runs/` (metrics, checkpoints, indices).
Datasets: root `datasets/`.

## 10. Glossary

- Worst‑group accuracy: Minimum accuracy across groups per epoch (robust objective).
- Anchor fit loss: W₂ distance aligning batch latent moments to anchor Gaussians.
- Anchor separation loss: Encourages distinct anchors (classification of synthetic samples or enforced W₂ margin).
- GroupDRO: Adaptive group reweighting focusing optimization on underperforming groups.

## 11. FAQ

Q: Why anchors instead of prototypes?  
A: Anchors model covariance, letting W₂ capture shape differences, not just mean shifts.

Q: Why pad inputs?  
A: Mixed resolutions in one batch require uniform tensor shape; we pad smaller images to the max (instead of resizing all up/down uniformly).

Q: Can I add a new dataset group?  
A: Implement a loader returning `(train_loader, test_loader)` with `(x,y,g)` tuples, add a config flag, and update `train.py` to branch on it.

---
End of README.
