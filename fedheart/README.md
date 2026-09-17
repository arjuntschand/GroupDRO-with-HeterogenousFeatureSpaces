# Fed-Heart

Binary heart-disease prediction across four cardiology sites from the UCI
heart disease archive: Cleveland, Hungarian, Switzerland, and the VA. Each
site ran a different subset of the same clinical workup, so the columns
genuinely differ between sites rather than being withheld. Switzerland, for
example, lacks serum cholesterol on most of its records.

## Data

The four raw UCI files are committed under `datasets/fed_heart_disease/`
(48 KB), so nothing needs downloading. The loader in
`dro_hetero_anchors/src/datasets_fedheart.py` keeps 13 features after
preprocessing:

```
0 age  1 sex  2 trestbps  3 chol  4 fbs  5 thalach  6 exang  7 oldpeak
8-10 cp (one-hot)  11-12 restecg (one-hot)
```

The UCI columns slope, ca and thal are dropped. Each site is one group and
sees only the features it records, set by `feature_mask` in the config, and
each per-group encoder's input width matches its real feature count
(`true_hetero_input_dim: true`).

| group | patients | features |
|---|---|---|
| Cleveland | 305 | 10 |
| Hungarian | 295 | 8 |
| Switzerland | 125 | 8 |
| VA | 200 | 9 |

## Protocol

Every patient is evaluated exactly once under 5-fold cross validation with
median imputation, which is what the FLamby benchmark does. Without
imputation the loader dropped any row with a missing value, which discarded
63% of Switzerland; a single 80/20 split then tested that site on 10
patients. Ten seeds, five folds, per-group accuracy pooled by fold test
counts. Model initialisation is reseeded after the loaders are built, so the
seeds vary the model and not only the resampling.

Two configurations are reported:

- **Capped** (`fedheart_exp_paper_hetagg_gdro.yaml`): Switzerland is held
  to 20 training patients and the VA to 25, a scarcity we impose to simulate
  small sites. This is the paper's main Fed-Heart setting.
- **Uncapped** (`fedheart_uncapped.yaml`): the same with no cap, as a
  robustness check that the pattern is not an artefact of the cap.

Architecture and optimisation: one MLP with layer norm per site, latent
dimension 64, head hidden size 32, Adam at learning rate 0.001, batch 64, up
to 100 epochs with early stopping at 30. GroupDRO uses the specification's
step size 0.02 on an exponential moving average (decay 0.9) of the group
losses. The anchor weight, when on, is 0.1 for both fit and separation.

## Scripts

Run from the repository root.

```bash
python estimate_rstar_v2.py --dataset fedheart      # R*_g per site -> runs/rstar_fedheart.json
python run_fedheart_cv.py --out runs/fedheart_cv    # every arm of the ablation grid, 10 seeds x 5 folds
python run_fedheart_cv.py --base experiments/fedheart_uncapped.yaml --out runs/fedheart_uncapped
python run_baselines_tabular.py --dataset fedheart --folds 5                     # Reweigh, Flex-MoE, REMIND
python run_baselines_tabular.py --dataset fedheart --folds 5 --capacity-matched
python run_fedheart_lamfit.py                       # anchor-weight sweep under CV
python run_dynamics_fedheart.py                     # per-epoch group weights for the dynamics figure
```

## Results

| folder | what it holds |
|---|---|
| `runs/fedheart_cv/` | the capped grid: `metrics_long.csv` (method x seed x group), `results.json`, and each fold's per-epoch `metrics.csv` |
| `runs/fedheart_uncapped/` | the uncapped grid |
| `runs/baselines_fedheart*/` | the three baselines at released and matched sizes; `_uncapped` variants |
| `runs/fedheart_cv_lam*/` | anchor weight 0.05, 0.1, 0.3, 0.5 |
| `runs/dynamics_fedheart/` | the two full-method runs with group weights logged each epoch |
| `runs/rstar_fedheart*.json` | R*_g estimates |
| `runs/fedheart_cv_{G064,VALSEL,OLDDRO}/`, `runs/baselines_fedheart_{SINGLESPLIT,NOSEEDVAR,TESTSEL,NOVAL}/` | superseded-protocol snapshots kept so each correction can be quantified |

The current numbers are in `runs/FINAL_TABLES.txt`. Per-group encoders are
the large effect on this dataset (several points of worst-group accuracy over
the common-features baseline); adding the anchors costs accuracy, which is
consistent with the sites' features overlapping heavily.
