# GroupDRO with Heterogeneous Feature Spaces

Worst-group robust training when different groups of patients have genuinely
different input features.

Standard training assumes every example has the same input columns. In
practice four hospitals each run a different subset of the same clinical
workup, some survey participants give blood and others only answer questions,
and some mammograms have four views while others have one. The usual fix is to
drop everything the groups do not share and train one model on the common
columns. Averaging the loss then lets the large, well-measured groups dominate.

This repository trains one encoder per group into a shared latent space, with
a single classification head, per-class Gaussian anchors aligned by the
2-Wasserstein distance, and a worst-group (GroupDRO) or excess-risk
(regret-DRO) objective over groups. It contains the code, configurations, and
per-seed results behind the paper's three datasets:

| Dataset | Task | Groups | Protocol |
|---|---|---|---|
| Fed-Heart (UCI heart disease, 4 sites) | binary heart disease | Cleveland, Hungarian, Switzerland, VA; each site records a different subset of tests | 10 seeds x 5-fold CV, median imputation; also an uncapped variant |
| NHANES 2017-2023 (CDC) | binary cardiovascular disease, 10.5% positive | survey-only, +exam, +blood pressure and labs (10 / 13 / 20 features, nested) | 10 seeds, fixed split |
| EMBED (Emory mammography) | BI-RADS density, 4 classes | 6 groups by which of four views a breast has | 10 seeds, frozen ViT-Base features |

## Layout

```
dro_hetero_anchors/src/      the package
  model/                     anchors, losses, Wasserstein distance, GroupDRO, heads,
                             REMIND-paper baselines (Reweigh, FlexMoE, REMIND), EMBED model
  encoders/                  tabular encoders (registry in __init__.py)
  datasets_fedheart.py       loaders; datasets_nhanes.py; datasets_embed.py
  train_fedheart.py          trainers; train_nhanes.py; train_embed_xenia.py
  tools/build_embed_xenia_index.py, extract_vit_embeddings.py, stream_embed_features.py,
                             download_embed.py, report_embed_xenia.py   (EMBED data pipeline)
experiments/                 the five YAML configs the runners start from
run_*.py, estimate_rstar_v2.py, preflight_check.py, run_overnight.sh
                             experiment runners (see below)
build_site.py, final_report.py, make_paper_figures.py, plot_mechanism.py,
plot_training_dynamics.py, make_figures.py, make_shareable.py
                             reporting: tables, figures, results site
runs/                        results: metrics_long.csv / results.json per run family,
                             per-epoch metrics.csv per run, R* estimates, FINAL_TABLES.txt
figs/                        current paper figures
datasets/                    Fed-Heart (UCI) and NHANES (CDC) raw files are committed;
                             EMBED must be obtained separately (see below)
documentation/               method notes, protocol audits, meeting briefs, archive/
legacy/                      archived exploration (MNIST/USPS, TextCaps, the first
                             EMBED track, superseded tabular runners); not maintained
```

Every table and figure is derived from the `metrics_long.csv` files under
`runs/`, one row per method x seed x group, so a number in a figure cannot
disagree with the same number in a table. Checkpoints, TensorBoard logs and
console logs are not tracked; the tabular checkpoints are attached to the
GitHub release `results-v1` as per-family tarballs.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Everything runs from the repository root. The Fed-Heart and NHANES raw files
are in the repo, so the tabular experiments need no download. (Both loaders
can also fetch the files themselves from UCI and the CDC.)

## Reproducing the reported numbers from the committed results

```bash
python preflight_check.py          # every arm on a dataset measured under the same protocol
python final_report.py             # prints runs/FINAL_TABLES.txt
python build_site.py --build       # site/index.html, one tab per dataset (omit --build to serve it locally)
python make_shareable.py           # site/shareable.html, single self-contained page
python make_paper_figures.py       # figs/paper/fig1_ladder, fig2_efficiency, table1.tex
python plot_mechanism.py           # figs/paper/fig3 (lambda vs R*), fig4 (loss curves), fig5
```

`final_report.py` reads only `metrics_long.csv` files; `plot_mechanism.py` and
`plot_training_dynamics.py` additionally read the per-epoch `metrics.csv` of
the `fedheart_cv`, `matrix_nhanes_nested`, and `dynamics_*` runs.

## Retraining

Tabular (Fed-Heart, NHANES). Seeds default to the ten used in the paper.

```bash
# R*_g reference losses (per-group Bayes-risk estimate), used by regret-DRO
python estimate_rstar_v2.py --dataset fedheart
python estimate_rstar_v2.py --dataset nhanes

# Fed-Heart: 5-fold CV with median imputation, every arm of the method matrix
python run_fedheart_cv.py --out runs/fedheart_cv

# NHANES: the method matrix (ERM, GroupDRO, regret-DRO, anchors, ours, group-only)
python run_method_matrix.py --dataset nhanes \
    --base experiments/nhanes_pergroup_gdro.yaml \
    --rstar runs/rstar_nhanes_nested.json --tag nhanes_nested

# REMIND-paper baselines (Reweigh, FlexMoE, REMIND) at released and capacity-matched sizes
python run_baselines_tabular.py --dataset fedheart --folds 5
python run_baselines_tabular.py --dataset nhanes
python run_baselines_tabular.py --dataset nhanes --capacity-matched

# equal-budget hyperparameter sweep, validation-selected
python run_sweep.py --dataset nhanes

# training-dynamics runs behind fig5 (per-epoch group weights logged)
python run_dynamics_fedheart.py
python run_dynamics_nhanes.py

# a single config, one seed
python -m dro_hetero_anchors.src.train_fedheart --config experiments/fedheart_exp_paper_hetagg_gdro.yaml
python -m dro_hetero_anchors.src.train_nhanes  --config experiments/nhanes_pergroup_gdro.yaml
```

`run_overnight.sh` chains the baseline and capacity-matched runs for both
tabular datasets. The mechanism studies referenced in the documentation are
`run_anchor_sweep.py`, `run_anchor_control_v2.py`, `run_latent_diagnostic.py`,
`run_pergroup_fit_test.py`, `run_mechanism_controls.py`,
`run_capacity_sweep.py`, `run_fedheart_lamfit.py`, and
`run_synthetic_scaling.py`; each has a docstring explaining the question it
answers.

### EMBED

EMBED is distributed by Emory through the AWS Open Data programme
(`s3://embed-dataset-open`, us-west-2) under a research use agreement that
must be signed first. The agreement forbids redistributing the images, any
index derived from the tables, cached embeddings, or trained weights, so none
of those are in this repository. Only metric summaries are committed
(`runs/embed_*/metrics_long.csv`, `rstar.json`, and the per-epoch
`curve_*.json` files).

To rebuild the EMBED results once you have access:

```bash
# 1. tables + images (or stream them, see stream_embed_features.py)
aws s3 cp s3://embed-dataset-open/tables/ datasets/embed/tables/ --recursive

# 2. the 6-group breast-level index from the metadata and clinical tables
python -m dro_hetero_anchors.src.tools.build_embed_xenia_index --mode full

# 3. frozen ViT-Base CLS embeddings for every referenced image (GPU)
python -m dro_hetero_anchors.src.extract_vit_embeddings --images-root datasets/embed

# 4. all methods x seeds on the cached features, then the report
python -m dro_hetero_anchors.src.train_embed_xenia --out runs/embed_xenia_production --seeds 0 1 42 7 11 22 33 1337 2024 31337
python -m dro_hetero_anchors.src.report_embed_xenia --run runs/embed_xenia_production
python run_baselines_embed.py --index datasets/embed/index_xenia_6group.parquet --cache datasets/embed/vit_cache
```

The model is a per-view linear projection of the frozen 768-d features, a
per-group MLP over the concatenated views a breast has, a shared head, and
diagonal Gaussian anchors, trained with the same GroupDRO / regret-DRO
objective as the tabular experiments.

## Tests

```bash
python -m pytest dro_hetero_anchors/tests -q
```

## Documentation

`documentation/START_HERE.md` is the plain-language overview.
`CLAIM_AUDIT.md` maps each claim to the run that tests it, `MORNING_BRIEF.md`
records the protocol corrections and the superseded result snapshots kept for
comparison (`runs/*_VALSEL`, `*_OLDDRO`, `*_TESTSEL`, ...), `FEDHEART_CV.md`
and `SWEEP_AUDIT.md` describe the evaluation protocol, and
`PAPER_RESULTS_DRAFT.md` is the written results section. Older documents are
in `documentation/archive/`.
