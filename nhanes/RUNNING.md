# Running the NHANES experiments

Everything runs from the repository root with the project's virtualenv active.
The frozen protocol is `experiments/nhanes_final.yaml` (1/G weight start, held-out
lambda signal refreshed every step at gamma 0.1, eq. 13 floors); see
`documentation/FINAL_PROTOCOL.md` for what it fixes and why.

## 1. Optimal-loss floors (once per configuration)

    python estimate_rstar_v2.py --dataset nhanes --base experiments/nhanes_final.yaml \
        --out runs/rstar_nhanes_eq13.json

Writes one JSON with `rstar` (the eq. 13 floors R~_g = min{group-only fit, constant
predictor} - c_g), `rstar_before_margin`, `margin_c_g`, `constant_predictor_bound` and
`rstar_fit_only`, per group. `--group-cap N0 N1 N2` caps the rows used to fit each
group's floor (this is how the sample-size-vs-R* relationship was measured).

## 2. The ablation matrix (our 10 arms)

    python run_method_matrix.py --dataset nhanes --base experiments/nhanes_final.yaml \
        --rstar runs/rstar_nhanes_eq13.json --tag nhanes_final --out runs/final_nhanes

Trains ERM, PerGroupOnly, Shared_GDRO, GroupDRO, Shared_Anchors, AnchorsOnly,
Shared_Anchors_GDRO, Ours_GDRO, RegretDRO, Ours_Regret (the full method) for 10 seeds
(`--seeds` to change; `--methods` to run a subset). The anchored arms use anchor weight
0.1, the others 0.001 (off). Every run writes `<out>/<arm>_s<seed>/metrics.csv` (one row
per epoch); the runner selects the reported epoch on validation worst-group accuracy and
writes

  - `<out>/metrics_long.csv`  one row per (arm, seed, group): accuracy, macro_f1, loss,
                              R_star, excess_loss, n, n_params   <- what every table reads
  - `<out>/results.json`      per (arm, seed) summary incl. AUROC and per-group vectors

A finished directory is reused, so give a fresh `--out` for a fresh run.

## 3. Published baselines (Reweigh, Flex-MoE, REMIND)

    python run_baselines_tabular.py --dataset nhanes --base experiments/nhanes_final.yaml \
        --out runs/baselines_nhanes                    # published configurations
    python run_baselines_tabular.py --dataset nhanes --base experiments/nhanes_final.yaml \
        --capacity-matched --out runs/baselines_nhanes_matched

Same `metrics_long.csv` schema, same seeds and split as our arms.

## 4. The two NHANES variants

  - No common information (groups share no columns; G0 gets the 10 survey features, G1
    body measures + HbA1c + HDL, G2 blood pressure + the other lipids):
    `experiments/nhanes_partition.yaml` (`feature_mode: partition`, defined in
    `dro_hetero_anchors/src/datasets_nhanes.py`). Floors: `runs/rstar_nhanes_partition.json`;
    results: `runs/matrix_nhanes_partition`, `runs/baselines_nhanes_partition`.
  - Group sizes changed (`group_max_train_samples` caps a group's training rows):
    `experiments/nhanes_capA_infoScarce.yaml` caps the most-informed group G2 at 500;
    `experiments/nhanes_capB_poorScarce.yaml` caps the least-informed group G0 at 500.
    Floors: `runs/rstar_nhanes_capA_infoScarce.json`, `runs/rstar_nhanes_capB_poorScarce.json`;
    results: `runs/matrix_nhanes_cap{A,B}`, `runs/baselines_nhanes_cap{A,B}`.

Run steps 1-3 with `--base` pointing at the variant config and a new `--out`.
The committed result files for these variants were produced under the earlier weight
schedule (gamma 0.02, group-proportion start); rerunning them under `nhanes_final.yaml`'s
settings gives the frozen-protocol numbers.

## 5. Reporting

    python final_report.py > runs/FINAL_TABLES.txt      # text tables
    python build_site.py --build                         # site/index.html
    python xenia_checks.py                               # the tables from the 2026-09-17 checks
