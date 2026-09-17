# CLAUDE.md

Guidance for working in this repository.

## Project Overview

Research code for GroupDRO (Group Distributionally Robust Optimization) across heterogeneous feature spaces. Per-group encoders map different input feature sets into a shared latent space; per-class Gaussian anchors align the groups; a worst-group (GroupDRO) or excess-risk (regret-DRO) objective protects the worst group. Paper datasets: Fed-Heart Disease (4 hospitals), NHANES CVD (3 feature-availability groups), EMBED mammography (6 view-availability groups). MNIST/USPS, TextCaps, and the first image-based EMBED track were early exploration and live under `legacy/` (not maintained, not in the paper).

## Commands

Everything runs from the repository root.

### Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Reporting (reads committed results under runs/)
```bash
python preflight_check.py      # protocol invariants across arms
python final_report.py         # -> runs/FINAL_TABLES.txt
python build_site.py --build   # -> site/index.html (no flag = serve on :8000); make_shareable.py -> site/shareable.html
python make_paper_figures.py   # figs/paper/fig1, fig2, table1.tex
python plot_mechanism.py       # figs/paper/fig3, fig4, fig5
```

### Training
```bash
# single config, one seed
python -m dro_hetero_anchors.src.train_fedheart --config experiments/fedheart_exp_paper_hetagg_gdro.yaml
python -m dro_hetero_anchors.src.train_nhanes  --config experiments/nhanes_pergroup_gdro.yaml

# paper protocol
python estimate_rstar_v2.py --dataset fedheart|nhanes
python run_fedheart_cv.py --out runs/fedheart_cv
python run_method_matrix.py --dataset nhanes --base experiments/nhanes_pergroup_gdro.yaml --rstar runs/rstar_nhanes_nested.json --tag nhanes_nested
python run_baselines_tabular.py --dataset fedheart --folds 5      # Reweigh / FlexMoE / REMIND
python run_sweep.py --dataset nhanes                               # equal-budget HP sweep
python run_dynamics_fedheart.py ; python run_dynamics_nhanes.py    # fig5 inputs

# EMBED (needs EMBED access; runs on cached frozen-ViT features)
python -m dro_hetero_anchors.src.tools.build_embed_xenia_index --mode full
python -m dro_hetero_anchors.src.extract_vit_embeddings --images-root datasets/embed
python -m dro_hetero_anchors.src.train_embed_xenia --out runs/embed_xenia_production
python run_baselines_embed.py
```

### Tests
```bash
python -m pytest dro_hetero_anchors/tests -q
```

## Architecture

Configuration (YAML) → group-aware data loaders → per-group encoders φ_g → shared latent space → classification head + anchor module → total loss (CE + λ_fit·W₂_fit + λ_sep·L_sep) → GroupDRO / regret-DRO reweighting → backward pass.

### Key Modules (`dro_hetero_anchors/src/`)
- **`train_fedheart.py` / `train_nhanes.py`**: tabular training loops; build models via `build_models()`, run train/eval epochs, log metrics per epoch to `metrics.csv` (+ `metrics.jsonl`, checkpoints, TensorBoard, all gitignored).
- **`train_embed_xenia.py`** + **`model/embed_xenia.py`**: EMBED pipeline on cached ViT features (per-view projections → per-group MLP → shared head → diagonal anchors; R*_g by 5-fold OOF; 7 methods). Writes `metrics_long.csv`, `rstar.json`, `curve_*.json`.
- **`datasets_fedheart.py` / `datasets_nhanes.py` / `datasets_embed.py`**: loaders with group-aware batching. Fed-Heart and NHANES raw files are committed under `datasets/`.
- **`model/anchors.py`**: `AnchorModule`, per-class Gaussians N(m_c, L_c L_cᵀ + εI).
- **`model/losses.py`**: anchor fit (W₂ between batch moments and anchors, pooled or per-group), anchor separation, focal loss, label smoothing.
- **`model/groupdro.py`**: `GroupDRO` weights on the simplex; update modes `exp`, `softmax`, `exp_smooth`; objectives `weighted`, `max`, `logsumexp`; optional KL(q‖π).
- **`model/wasserstein.py`**: `gaussian_w2()` (Bures–Wasserstein), `psd_sqrt()`.
- **`model/baselines.py`**: reimplementations of Reweigh, FlexMoE, REMIND from the REMIND paper.
- **`encoders/__init__.py`**: `ENCODER_REGISTRY` (tabular MLP encoders). Add new encoders here.

### Runners and reporting (repo root)
`run_method_matrix.py` (7-method matrix, emits `metrics_long.csv`), `run_fedheart_cv.py` (5-fold CV protocol), `run_baselines_tabular.py` / `run_baselines_embed.py`, `estimate_rstar_v2.py`, `run_sweep.py`, `run_capacity_sweep.py`, `run_fedheart_lamfit.py`, `run_dynamics_*.py`, the mechanism studies (`run_anchor_sweep.py`, `run_anchor_control_v2.py`, `run_latent_diagnostic.py`, `run_pergroup_fit_test.py`, `run_mechanism_controls.py`, `run_synthetic_scaling.py`), `preflight_check.py`. Reporting: `build_site.py` (+ `make_figures.py`, `make_shareable.py`), `final_report.py`, `make_paper_figures.py`, `plot_mechanism.py`, `plot_training_dynamics.py`.

### Results layout (`runs/`)
Git tracks results only: `metrics_long.csv` / `results.json` per run family, `rstar*.json`, per-epoch `metrics.csv` per run, `FINAL_TABLES.txt`. Canonical families: `fedheart_cv`, `fedheart_uncapped`, `matrix_nhanes_nested`, `baselines_*`, `embed_fix_final` (EMBED), `dynamics_*`. Suffixed twins (`_VALSEL`, `_OLDDRO`, `_G064`, `_TESTSEL`, `_NOVAL`, `_SINGLESPLIT`) are superseded-protocol snapshots kept so each correction can be quantified (see `documentation/MORNING_BRIEF.md`). `runs/embed_*` is an allowlist in `.gitignore`: the EMBED data use agreement forbids releasing weights, embeddings, or indexes, so only named metric files are admitted. Older summaries are in `runs/summaries_archive/`.

## Working conventions
- Commit messages carry no tool attribution or trailers.
- Never `git add -A`; add explicit paths. Checkpoints, logs, TensorBoard, `metrics.jsonl`, and everything under `datasets/` except the committed Fed-Heart/NHANES raw files are ignored.
- Configs referenced by the runners live in `experiments/`; all other historical configs are under `legacy/*/configs/`.

## Fed-Heart Disease: Heterogeneous Feature Config Options

- **`common_encoder: true`**: all groups share one encoder (baseline). Default is per-group encoders.
- **`feature_mask`**: per-group list of feature indices to keep; `null` keeps all 13.
- **`true_hetero_input_dim: true`**: each per-group encoder's `input_dim` matches its actual feature count.
- **`group_max_train_samples`**: cap training samples per group, e.g. `[null, null, 20, 25]` (the "capped" protocol); `fedheart_uncapped.yaml` removes the cap.
- **`data_split_seed`**: fixed train/test split; the experiment `seed` then only controls model init.

Feature indices after preprocessing: `0:age, 1:sex, 2:trestbps, 3:chol, 4:fbs, 5:thalach, 6:exang, 7:oldpeak, 8-10:cp(one-hot), 11-12:restecg(one-hot)`. The UCI columns slope, ca, thal are dropped in `datasets_fedheart.py`.

## NHANES CVD

Binary CVD prediction from NHANES 2017-2020 + 2021-2023 (17,005 participants, 10.5% prevalence, `class_weight: "auto"`). Groups by assessment completion: G0 survey only (10 features), G1 + body measures (13), G2 + BP and labs (20); nested. Feature indices: `0:age, 1:gender, 2-6:race(one-hot), 7:education, 8:income_poverty, 9:ever_smoked, 10:bmi, 11:weight, 12:height, 13:mean_systolic_bp, 14:mean_diastolic_bp, 15:hba1c, 16:hdl, 17:total_cholesterol, 18:triglycerides, 19:ldl`. Key options: `use_post_pandemic: true`, `data_split_seed: 100`, `feature_mode: nested|expanded|disjoint` (disjoint and expanded were cut from the paper).
