# Learning across heterogeneous feature spaces with group robustness

Worst-group robust training when different groups of patients have different
input features.

Standard training assumes every example has the same input columns. In
practice, four hospitals each record a different subset of the same clinical
workup, some survey participants give blood and others only answer questions,
and some mammograms have four views while others have one. The usual fix is
to drop everything the groups do not share and train one model on the common
columns, or to impute what is missing. This repository instead trains one
encoder per group into a shared latent space with a single classification
head. Per-class Gaussian anchors align the groups in that space, and the
objective is the worst-group excess loss: each group's loss minus an estimate
of the best loss it can reach, so that groups whose features are simply less
informative are not upweighted for that reason alone.

The method, the protocol and every result in the paper are reproducible from
this repository. The EMBED images and embeddings are not included, because
the data use agreement does not allow redistributing them; the per-run metric
files are.

## Datasets

| folder | dataset | task | groups |
|---|---|---|---|
| [`fedheart/`](fedheart/README.md) | UCI heart disease, 4 hospitals, 920 patients | binary heart disease | each hospital records a different subset of 13 clinical features |
| [`nhanes/`](nhanes/README.md) | CDC NHANES 2017 to 2023, 17,005 adults | binary cardiovascular disease | survey only, plus body measures, plus blood pressure and labs (10 / 13 / 20 features, nested) |
| [`embed/`](embed/README.md) | Emory EMBED mammography, 128,680 records from 22,997 patients | 4-class breast density | which of four imaging views a record has (6 groups) |

For each dataset there is also a design with fully disjoint feature spaces,
in which no feature is shared by two groups. The Fed-Heart and NHANES raw
files are committed under `datasets/`, so the tabular experiments need no
download. EMBED must be obtained under its research use agreement; its folder
README gives the steps.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest dro_hetero_anchors/tests -q
```

Everything runs from the repository root.

## Reproducing the paper

The numbers in the paper come from protocol v4 (`documentation/PROTOCOL_V4_2026-09-22.md`,
with its amendments): one trainer for our arms and the published baselines,
natural-proportion batches, a fixed budget of 10 reported epochs, 10 seeds,
5 folds on Fed-Heart, the step size chosen on validation. The per-epoch logs
of every run live under `runs/v4c/`, and everything else is derived from them.

```bash
python report_v4.py runs/v4c          # per-seed summary and curves from the per-epoch logs
python build_site.py --build          # site/index.html, the results website (omit --build to serve it)
python make_shareable.py              # site/shareable.html, one self-contained page
python make_paper_figures_v4.py       # figs/paper/fig_dynamics_v4 (main) and _appx
python make_diagram_figures.py        # figs/paper/fig_intro_problem, fig_setup_architecture
python plot_latent_scatter.py         # figs/paper/fig11_latent_scatter_*  (latent space with and without anchors)
python compute_latent_metrics.py      # the latent-space table
```

## Retraining

Reference losses first, then the tabular arms, then EMBED.

```bash
# per-group reference losses (nested cross-validation; EMBED estimates its own inside its trainer)
python estimate_rstar_v3.py --dataset nhanes   --base experiments/nhanes_v3.yaml
python estimate_rstar_v3.py --dataset fedheart --base experiments/fedheart_v3.yaml --n-folds 5 --fold 0   # and folds 1 to 4

# tabular: every method, one code path (see the docstring for --methods, --steps, --seeds, --shard)
python train_v4_tabular.py --dataset nhanes   --base experiments/nhanes_v3.yaml   --rstar runs/rstar_v3/nhanes/nested.json \
    --out runs/v4c/nhanes   --sampler proportional --schedule cosine --epochs 30 --sep-stopgrad
python train_v4_tabular.py --dataset fedheart --base experiments/fedheart_v3.yaml --rstar runs/rstar_v3/fedheart --folds 5 \
    --out runs/v4c/fedheart --sampler proportional --schedule cosine --epochs 30 --sep-stopgrad

# the designs with disjoint feature spaces use experiments/nhanes_partition2_v3.yaml (NHANES, designed)
# and experiments/fedheart_disjoint_v3.yaml with the matching runs/rstar_v3/ references

# EMBED, on cached frozen-ViT features (needs EMBED access; see embed/README.md)
python -m dro_hetero_anchors.src.tools.build_embed_xenia_index --mode full
python -m dro_hetero_anchors.src.extract_vit_embeddings --images-root datasets/embed
python -m dro_hetero_anchors.src.train_embed_xenia --index datasets/embed/index_production.parquet \
    --cache datasets/embed/vit_cache_full --sep-stopgrad --out runs/v4c/embed_g2.0        # add --disjoint for the disjoint views
python run_baselines_embed.py --v4 --dro-gamma 0.02 --methods Reweigh FlexMoE REMIND --out runs/v4c/embed_bl
```

Checkpoints, TensorBoard logs and console logs are not tracked. Earlier
tabular checkpoints are attached to the GitHub release `results-v1`.

## Layout

```
dro_hetero_anchors/src/      the package: model/ (anchors, losses, Wasserstein distance, GroupDRO,
                             baselines, EMBED model), encoders/, data loaders, the EMBED trainer,
                             tools/ (EMBED index and feature pipeline)
train_v4_tabular.py          protocol v4 trainer for NHANES and Fed-Heart, ours and the baselines
report_v4.py, site_v4.py     per-seed summaries and the V4 tab of the results website
estimate_rstar_v3.py         per-group reference losses
run_baselines_embed.py       Reweigh, Flex-MoE and REMIND on EMBED
experiments/                 YAML configs; the *_v3.yaml files are the ones the paper uses
runs/                        results: per-epoch logs and summaries (v4c is the paper), reference
                             losses (rstar_v3), earlier protocol snapshots kept for comparison
figs/paper/                  paper figures
documentation/               the protocol and its amendments, review logs, figure notes
docs/                        method and results notes, the poster and paper drafts
fedheart/  nhanes/  embed/   per-dataset guides
legacy/                      archived early exploration (MNIST/USPS, TextCaps, the first EMBED
                             track, superseded runners); not maintained
```

The remaining `run_*.py` and `plot_*.py` scripts are the mechanism studies
and earlier-protocol runners; each docstring says which question it answers
and which `runs/` family it writes.
