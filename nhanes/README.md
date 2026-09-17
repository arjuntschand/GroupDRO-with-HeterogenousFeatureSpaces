# NHANES

Binary cardiovascular-disease prediction for 17,005 US adults from the CDC's
National Health and Nutrition Examination Survey, cycles 2017-2020 and
2021-2023. Prevalence is 10.5%, so the loss is inverse-frequency class
weighted (`class_weight: "auto"`).

## Data

The 24 raw CDC files (`.xpt`, 33 MB) are committed under
`datasets/nhanes/pre_pandemic/` and `datasets/nhanes/post_pandemic/`, so
nothing needs downloading. Preprocessing is in
`dro_hetero_anchors/src/datasets_nhanes.py` and runs in memory.

Groups are defined by how much of the assessment a participant completed,
which is a logistical fact about how the survey runs rather than a split we
chose. Nobody gives blood without first answering the questionnaire, so the
feature sets nest.

| group | who | participants | features |
|---|---|---|---|
| G0, survey only | questionnaire | 2,666 | 10 |
| G1, exam | plus body measures | 2,648 | 13 |
| G2, vitals and labs | plus blood pressure and labs | 11,691 | 20 |

Feature indices after preprocessing:

```
0 age  1 gender  2-6 race (one-hot)  7 education  8 income_poverty  9 ever_smoked
10 bmi  11 weight  12 height
13 mean_systolic_bp  14 mean_diastolic_bp  15 hba1c  16 hdl  17 total_cholesterol
18 triglycerides  19 ldl
```

G2 is larger than the other two combined, which is why ordinary training
neglects G0 and G1.

Two other feature modes exist in the loader and were cut from the paper:
`expanded` (15/18/25 features, still nested, tests nothing new) and
`disjoint` (10 shared plus 5 private features per group, a partition we
constructed whose private sets turned out to be substitutes for each other).
Their runs are kept for an appendix.

## Protocol

Ten seeds on a fixed train/validation/test split (`data_split_seed: 100`,
80/20 with a 15% validation slice of the training portion), so the seed
varies model initialisation only. Stratified batching preserves group
proportions. Architecture and optimisation match Fed-Heart: one MLP with
layer norm per group, latent dimension 64, head hidden size 32, Adam at
learning rate 0.001, batch 128, up to 100 epochs with early stopping at 25,
GroupDRO step size 0.02 on an exponential moving average of the group
losses.

## Scripts

Run from the repository root.

```bash
python estimate_rstar_v2.py --dataset nhanes                 # R*_g -> runs/rstar_nhanes_nested.json
python run_method_matrix.py --dataset nhanes \
    --base experiments/nhanes_pergroup_gdro.yaml \
    --rstar runs/rstar_nhanes_nested.json --tag nhanes_nested   # every arm, 10 seeds
python run_baselines_tabular.py --dataset nhanes                      # Reweigh, Flex-MoE, REMIND
python run_baselines_tabular.py --dataset nhanes --capacity-matched
python run_sweep.py --dataset nhanes                                  # equal-budget hyperparameter sweep
python run_capacity_sweep.py                                          # widen our model to the baselines' size
python run_dynamics_nhanes.py                                         # per-epoch group weights for the dynamics figure
```

Mechanism studies, all on NHANES: `run_anchor_sweep.py` (anchor weight),
`run_anchor_control_v2.py` (permuted-label anchor control),
`run_latent_diagnostic.py` (alignment measured in the latent space),
`run_pergroup_fit_test.py` (pooled versus per-group fit loss),
`run_mechanism_controls.py` (capacity, generic regulariser, random target).

## Results

| folder | what it holds |
|---|---|
| `runs/matrix_nhanes_nested/` | the grid: `metrics_long.csv`, `results.json`, per-run per-epoch `metrics.csv` |
| `runs/baselines_nhanes*/` | the three baselines at released and matched sizes |
| `runs/sweep_nhanes/`, `runs/capsweep_nhanes*` | hyperparameter and capacity sweeps |
| `runs/dynamics_nhanes/` | the two full-method runs with group weights logged each epoch |
| `runs/anchor_*`, `runs/anchorctl2/`, `runs/latdiag/`, `runs/pgfit/`, `runs/mechanism_*` and the matching `runs/*.json` | mechanism studies |
| `runs/matrix_nhanes_{disjoint,expanded}*/` | the two cut feature modes |
| `runs/matrix_nhanes_nested_{G064,VALSEL,OLDDRO}/` | superseded-protocol snapshots |
| `runs/rstar_nhanes*.json` | R*_g estimates |

The current numbers are in `runs/FINAL_TABLES.txt`. NHANES is the dataset
where the anchors earn their place: switching them on gives a significant
worst-group accuracy gain over per-group GroupDRO, while per-group encoders
alone are roughly neutral because splitting groups of 2,100 people across
separate encoders costs about as much in sample efficiency as the extra
columns return.
