# Results: how they are produced and how to read them

## Where the numbers come from

Every table and figure is built from the `metrics_long.csv` files under
`runs/`, one row per method, seed and group, so a number in a figure cannot
disagree with the same number in a table. Nothing is quoted from memory.

```bash
python preflight_check.py      # asserts every arm on a dataset was measured the same way
python final_report.py         # prints the tables; the committed copy is runs/FINAL_TABLES.txt
python build_site.py --build   # site/index.html, one tab per dataset
python make_paper_figures.py   # figs/paper/fig1_ladder, fig2_efficiency, table1.tex
python plot_mechanism.py       # figs/paper/fig3 (lambda against R*), fig4 (loss curves), fig5 (dynamics)
```

`runs/FINAL_TABLES.txt` is the authoritative summary. It is regenerated
whenever a run changes, so if this page and that file ever disagree, the
file wins.

## Protocol, per dataset

| | Fed-Heart | NHANES | EMBED |
|---|---|---|---|
| seeds | 10 | 10 | 10 |
| split | 5-fold CV, every patient tested once, median imputation | fixed 80/20 with a validation slice | patient-level 70/10/20 |
| epoch selection | validation | validation | validation |
| groups | 4 sites | 3 assessment levels | 6 view sets |
| features | 8 to 10 per site, differing | 10 / 13 / 20, nested | 1 to 4 views |
| baselines | Reweigh, Flex-MoE, REMIND (128 experts), released and matched sizes | same | same, on the same cached features |

Every comparison is between arms trained on the same rows and evaluated on
the same rows; `preflight_check.py` verifies that before any table is built.

## Which metric leads, and why

Three metrics are reported for every arm: worst-group accuracy, worst-group
loss, and maximum excess loss (loss above the group's own reference R*).

The theory behind the method never mentions accuracy. Its propositions are
about Bayes risk and about worst-group excess risk, and the objective is a
sum of risks. Excess loss is therefore the quantity the theory predicts, and
reporting it is testing a stated prediction rather than choosing the metric
after the fact. Worst-group loss leads over maximum excess loss because it
needs no R*, and R* remains the least certain estimate in the pipeline.

At the time of writing (September 2026) the tally of our full method
against the three baselines on the three datasets is:

| metric | wins | ties | losses |
|---|---|---|---|
| worst-group accuracy | 0 | 9 | 3 |
| worst-group loss | 9 | 3 | 0 |

Read it as: on accuracy we match the baselines, with one significant loss
(Fed-Heart against REMIND); on loss we are ahead everywhere, at roughly a
quarter of REMIND's parameter count. Within our own grid, per-group encoders
carry Fed-Heart, the anchors carry NHANES, and on EMBED the anchors cost
accuracy while improving worst-group loss.

## Protocol corrections, and the snapshots kept to quantify them

Several comparisons at one point silently mixed a method difference with a
protocol difference. Each was fixed, and the superseded results were kept
rather than deleted so the size of each correction is on record.

- **Fed-Heart baselines were scored on a fifth of the data.** Our arms used
  5-fold CV over all 925 patients; the baselines used a single 20% split.
  Snapshot: `runs/baselines_fedheart_SINGLESPLIT`.
- **Fed-Heart's ten seeds were one initialisation ten times.** The loader
  reseeded torch after the experiment seed was set. Fixed by reseeding after
  the loaders. Snapshot: `runs/baselines_fedheart_NOSEEDVAR`.
- **The reported epoch was chosen on the test set.** Both tabular datasets
  now carry a validation split and select on it. Snapshots:
  `runs/fedheart_cv_VALSEL`, `runs/matrix_nhanes_nested_VALSEL`, and the
  `_TESTSEL` / `_NOVAL` baseline variants.
- **The shared encoder only ever saw common features.** With the shared
  encoder given the same padded features, the per-group advantage on NHANES
  shrinks to insignificance. "ERM, common features" is still reported
  because it is what a practitioner would build.
- **REMIND was running at 4 experts.** Its paper specifies 128. Snapshot:
  `runs/baselines_embed` before `_remind128`.
- **R* was measured on our own capped training set.** R* is a property of
  the group's distribution and cannot depend on how many samples we trained
  on. The estimator was rewritten (`estimate_rstar_v2.py`); the old values
  are `runs/rstar_*_OLD.json`.
- **The GroupDRO update rule was brought to the specification** (exponential
  step on an EMA of excess loss, step size 0.02). Snapshots:
  `runs/*_OLDDRO` (old rule) and `runs/*_G064` (a larger step size).

## Hyperparameter search

The equal-budget sweep (`run_sweep.py`) gives every method the same number
of configurations, selected on validation. On NHANES validation barely ranks
configurations the way test does, so the sweep's gains there are not
trustworthy and the reported arms keep their defaults. Its one solid finding
is that Fed-Heart's best configuration sets the anchor weight to 0.01, which
independently reproduces the direct result that anchors hurt on Fed-Heart.
The capacity sweep (`run_capacity_sweep.py`) widens our model to the
baselines' parameter counts and does not change the picture.

## Figures

`figs/paper/` holds the current paper figures: the contribution ladder per
dataset (fig1), worst-group loss against parameter count (fig2), final group
weight against R* per method (fig3), per-group loss curves with R* drawn as
a floor (fig4), and the training dynamics of the two full-method
configurations (fig5). `figs/dynamics/` holds the per-group panels behind
fig5.
