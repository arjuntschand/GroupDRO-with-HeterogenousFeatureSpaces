# GroupDRO with Heterogeneous Feature Spaces

Worst-group robust training when different groups of patients have genuinely
different input features.

Standard training assumes every example has the same input columns. In
practice four hospitals each run a different subset of the same clinical
workup, some survey participants give blood and others only answer questions,
and some mammograms have four views while others have one. The usual fix is
to drop everything the groups do not share and train one model on the common
columns; averaging the loss then lets the large, well-measured groups
dominate. This repository trains one encoder per group into a shared latent
space, with a single classification head, per-class Gaussian anchors aligned
by the 2-Wasserstein distance, and a worst-group (GroupDRO) or excess-risk
(regret-DRO) objective over groups. `docs/method.md` explains the method and
states plainly which claims the evidence supports.

## The three datasets

Each has its own folder with a README covering the data, the groups, the
protocol, the scripts, and where its results live.

| folder | dataset | task | groups |
|---|---|---|---|
| [`fedheart/`](fedheart/README.md) | UCI heart disease, 4 sites, 925 patients | binary heart disease | each site records a different subset of tests |
| [`nhanes/`](nhanes/README.md) | CDC NHANES 2017-2023, 17,005 adults | binary cardiovascular disease | survey only, plus exam, plus labs (10 / 13 / 20 features, nested) |
| [`embed/`](embed/README.md) | Emory EMBED mammography, 128,680 records | 4-class breast density | which of four imaging views a breast has |

The Fed-Heart and NHANES raw files are committed, so the tabular experiments
need no download. EMBED must be obtained under its research use agreement;
the folder README gives the steps and explains what cannot be published.

## Layout

```
dro_hetero_anchors/src/      the package: model/ (anchors, losses, Wasserstein, GroupDRO,
                             baselines, EMBED model), encoders/, loaders, trainers,
                             tools/ (EMBED index and feature pipeline)
experiments/                 the five YAML configs the runners start from
run_*.py, estimate_rstar_v2.py, preflight_check.py, run_overnight.sh
                             experiment runners; each docstring says which question it answers
build_site.py, final_report.py, make_paper_figures.py, plot_mechanism.py,
plot_training_dynamics.py, make_figures.py, make_shareable.py
                             reporting: tables, figures, results site
runs/                        results: metrics_long.csv / results.json per run family,
                             per-epoch metrics.csv per run, R* estimates, FINAL_TABLES.txt
figs/                        current paper figures
datasets/                    Fed-Heart (UCI) and NHANES (CDC) raw files
docs/                        method.md, results.md, the poster and paper drafts
fedheart/  nhanes/  embed/   per-dataset guides
legacy/                      archived early exploration (MNIST/USPS, TextCaps, the first
                             EMBED track, superseded runners); not maintained
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Everything runs from the repository root.

## Reproducing the reported numbers

All tables and figures are derived from the `metrics_long.csv` files under
`runs/`, one row per method, seed and group, so nothing can drift between a
table and a figure. `docs/results.md` explains the protocol, the metric
choice, and the corrections made along the way.

```bash
python preflight_check.py          # every arm on a dataset measured under the same protocol
python final_report.py             # prints runs/FINAL_TABLES.txt
python build_site.py --build       # site/index.html, one tab per dataset (omit --build to serve it locally)
python make_shareable.py           # site/shareable.html, single self-contained page
python make_paper_figures.py       # figs/paper/fig1_ladder, fig2_efficiency, table1.tex
python plot_mechanism.py           # figs/paper/fig3 (lambda against R*), fig4 (loss curves), fig5 (dynamics)
```

## Retraining

The per-dataset READMEs list the exact commands. In short: `estimate_rstar_v2.py`
computes the per-group reference losses, `run_fedheart_cv.py` and
`run_method_matrix.py` run every arm of the ablation grid on Fed-Heart and
NHANES, `run_baselines_tabular.py` and `run_baselines_embed.py` run the
REMIND-paper baselines, and `train_embed_xenia.py` runs the EMBED pipeline
on cached ViT features. Checkpoints, TensorBoard logs and console logs are
not tracked; the tabular checkpoints are attached to the GitHub release
`results-v1` as per-family tarballs.

## Tests

```bash
python -m pytest dro_hetero_anchors/tests -q
```
