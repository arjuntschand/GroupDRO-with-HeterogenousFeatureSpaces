# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project implementing GroupDRO (Group Distributionally Robust Optimization) across heterogeneous feature spaces. Per-group encoders map different input modalities (mixed-resolution images, tabular data, text+visual) into a shared latent space, where class-conditional Gaussian anchors and GroupDRO reweighting ensure worst-group robustness.

## Commands

### Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r dro_hetero_anchors/requirements.txt
```

### Training
```bash
# MNIST/USPS (mixed resolution images)
python -m dro_hetero_anchors.src.train --config experiments/<config>.yaml

# Fed-Heart Disease (tabular, 4 hospitals)
python -m dro_hetero_anchors.src.train_fedheart --config experiments/<config>.yaml

# TextCaps (multi-modal: visual + text)
python -m dro_hetero_anchors.src.train_textcaps --config experiments/<config>.yaml

# NHANES CVD (tabular, 3 feature-availability groups)
python -m dro_hetero_anchors.src.train_nhanes --config experiments/<config>.yaml
python run_nhanes_experiments.py --seeds 42 1337 7 --only core
```

### Evaluation & Tools
```bash
python -m dro_hetero_anchors.src.eval --config <cfg.yaml> --ckpt <ckpt.pt>
python -m dro_hetero_anchors.src.tools.index_experiments    # generate experiments/INDEX.csv
python -m dro_hetero_anchors.src.tools.aggregate_runs       # aggregate into runs/index.csv + runs/metrics.sqlite
```

### Tests
```bash
python -m pytest dro_hetero_anchors/tests/
python -m pytest dro_hetero_anchors/tests/test_wasserstein.py::test_gaussian_w2_symmetry_nonneg -v
```

## Architecture

### Training Pipeline
Configuration (YAML) → group-aware data loaders → per-group encoders φ_g → shared latent space → classification head + anchor module → total loss (λ_ce·CE + λ_fit·W₂_fit + λ_sep·L_sep) → optional GroupDRO reweighting → backward pass.

### Key Modules (`dro_hetero_anchors/src/`)

- **`train.py` / `train_fedheart.py` / `train_textcaps.py`**: Dataset-specific training loops. Each builds models via `build_models()`, runs train/eval epochs, logs metrics.
- **`datasets.py` / `datasets_fedheart.py` / `datasets_textcaps.py`**: Dataset-specific loaders with group-aware batching. MNIST/USPS uses `pad_to_max_collate()` for mixed resolutions.
- **`model/anchors.py`**: `AnchorModule` — per-class Gaussian anchors N(m_c, S_c) where S_c = L_c·L_c^T + ε·I.
- **`model/losses.py`**: Anchor fit loss (W₂ between batch moments and anchor moments), anchor separation loss (classifier or W₂ margin methods), focal loss, label smoothing.
- **`model/groupdro.py`**: `GroupDRO` class — maintains weight vector q on simplex, supports update modes (`exp`, `softmax`, `exp_smooth`), objective modes (`weighted`, `max`, `logsumexp`), optional KL(q‖π) penalty.
- **`model/wasserstein.py`**: `gaussian_w2()` for Bures-Wasserstein distance, `psd_sqrt()` for stable matrix square roots.
- **`encoders/__init__.py`**: `ENCODER_REGISTRY` maps string names to encoder classes. Add new encoders here.
- **`encoders/`**: CNN28/CNN32 (images), ResNetVisualEncoder (3-channel), CharCNN/Transformer (text), MLPTabularEncoder variants (tabular).

### Experiment Configuration
100+ YAML files in `experiments/`. Key config sections: data (loader flags, sizes, skew), groups (per-group encoder/shape), model (latent_dim, num_classes), anchors, losses (lambda_fit, lambda_sep), optimization, GroupDRO settings, logging.

### Output Structure
Each run writes to its `run_dir`: checkpoint (.pt), metrics JSONL, flattened CSV, TensorBoard events. `aggregate_runs` tool indexes all runs into `runs/index.csv` and `runs/metrics.sqlite`.

## Key Patterns

- **Encoder registry**: Look up encoder classes by string name from YAML config via `ENCODER_REGISTRY`.
- **Per-group processing**: Training loops split batches by group ID, apply group-specific encoders, then merge latent vectors for shared head/anchors.
- **Batch moments**: `per_class_batch_moments()` in `losses.py` computes running mean and covariance per class from latent embeddings each batch.
- **Datasets are in `datasets/`** (gitignored) and auto-download on first run.

## Fed-Heart Disease: Heterogeneous Feature Config Options

Key YAML config fields for the Fed-Heart heterogeneous feature experiments:

- **`common_encoder: true`**: All groups share one encoder (baseline). Default is per-group encoders.
- **`feature_mask`**: Per-group list of feature indices to keep. `null` = keep all 13. E.g. `[null, [0,1,2,4,5,6,7,8,9,10,11,12], ...]` drops chol (idx 3) for G1.
- **`true_hetero_input_dim: true`**: Each per-group encoder gets `input_dim` matching its actual feature count (not 13 with zeros). Required for genuine heterogeneity experiments.
- **`group_max_train_samples`**: Cap training samples per group. E.g. `[null, null, 10, 15]` caps G2 to 10 and G3 to 15.
- **`data_split_seed`**: Fixed seed for train/test split (like FLamby). When set, the experiment `seed` only controls model init and subsampling, not the data partition. This produces stable, comparable results across seeds.

### Feature Index Reference (after preprocessing)
`0:age, 1:sex, 2:trestbps, 3:chol, 4:fbs, 5:thalach, 6:exang, 7:oldpeak, 8-10:cp(one-hot), 11-12:restecg(one-hot)`

Note: The original UCI columns for slope, ca, and thal are dropped during preprocessing (line 77 of `datasets_fedheart.py`).

### Batch Experiment Runner
`run_experiments.py` runs multiple configs x seeds and collects results. See `runs/HETERO_EXPERIMENT_RESULTS.md` for latest results.

## NHANES CVD: Natural Feature-Availability Heterogeneity

**Task:** Binary CVD prediction from NHANES 2017-2020 + 2021-2023 survey data.
**Data:** 17,005 participants, 10.5% CVD prevalence (auto class-weighted loss).

### Groups (by assessment completion)
- **G0 (survey_only, 2.1k):** Demographics + smoking (10 features)
- **G1 (exam, 2.1k):** Survey + body measures (13 features)
- **G2 (vitals_labs, 9.4k):** Survey + body + BP + labs (20 features)

Features are nested: G0 ⊂ G1 ⊂ G2. Data downloads automatically from CDC on first run.

### Feature Index Reference (after preprocessing)
`0:age, 1:gender, 2-6:race(one-hot), 7:education, 8:income_poverty, 9:ever_smoked, 10:bmi, 11:weight, 12:height, 13:mean_systolic_bp, 14:mean_diastolic_bp, 15:hba1c, 16:hdl, 17:total_cholesterol, 18:triglycerides, 19:ldl`

### Key Config Options
- **`class_weight: "auto"`**: Inverse-frequency class weighting (default, important for 10.5% CVD rate)
- **`use_post_pandemic: true`**: Include 2021-2023 data (doubles dataset size)
- **`data_split_seed: 100`**: Fixed train/test split; model `seed` varies initialization only

### Feature Modes
- **`feature_mode: "nested"`** (default): G0 ⊂ G1 ⊂ G2 (10/13/20 features)
- **`feature_mode: "expanded"`**: More survey features, still nested (15/18/25)
- **`feature_mode: "disjoint"`**: Each group has 10 shared + 5 unique features (15/15/15). Per-group encoders essential.

### Experiment Runners
- `run_nhanes_experiments.py` — Original nested-only suite
- `run_nhanes_all.py` — Full suite: nested × expanded × disjoint × shared/pergroup × ERM/GDRO
- Results: `runs/NHANES_EXPERIMENT_RESULTS.md`, `runs/nhanes_all_results.json`
