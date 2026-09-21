# Code review of 2026-09-20 (Xenia) and the corrected re-runs

Record of what the review found, what was verified against the code, what was changed, and how
the results moved. Results from before this date ("the earlier pipeline") are in
`runs/final_*`, `runs/baselines_*`, `documentation/FINAL_PROTOCOL.md`; they are superseded for
any dataset listed as corrected below.

## 1. Findings, verified against the code

| # | Finding | Verified | Where |
|---|---|---|---|
| 1 | The "5 folds" on Fed-Heart were five independent 80/20 holdouts (`data_split_seed = 1000+k`), not K-fold | yes: per site 31-38% of patients were never tested, about 27% tested twice or more | `run_fedheart_cv.py`, `datasets_fedheart.py` |
| 2 | Training decisions read the test split: best checkpoint, early stopping (patience 30 / 25), plateau scheduler all used test worst-group accuracy. The reported epoch was validation-selected, but when training stopped was test-driven | yes, in both tabular trainers | `train_fedheart.py`, `train_nhanes.py` |
| 3 | `psd_sqrt` returned a Cholesky factor, not the principal square root, and the ridge was Frobenius-scaled: the full-covariance Bures term was wrong. Diagonal matrices were unaffected | yes; the old tests passed with the bug (they checked L Lᵀ = A) | `model/wasserstein.py` |
| 4 | Anchor covariance and anchor samples came from three different distributions | yes | `model/anchors.py` |
| 5 | Excess loss clamped at zero in the weight update | yes | `model/groupdro.py` |
| 6 | References (R*) fitted on train+test rows; one global file used for every fold; min over ~14 candidates scored on the rows then reported; margin silently 0 when the joint model won; candidates under-trained | yes | `estimate_rstar_v2.py` |
| 7 | Imputation medians computed over all rows (normalisation was already train-only) | yes | `datasets_fedheart.py` |
| 8 | "Anchors off" arms carried an anchor weight of 0.001 | yes | runners |
| 9 | Pooled alignment moments; λ started at group frequencies; η = 0.02 too small; capped config; alignment in the λ signal | the last four were already fixed in `fedheart_final.yaml` (the review read the older `fedheart_exp_paper_hetagg_gdro.yaml`); pooled moments confirmed | configs |

Two statements made earlier were wrong and had been repeated on the site and in the draft:
"every patient is evaluated exactly once" (finding 1), and the implicit claim that the tabular
anchored arms optimised a Wasserstein alignment loss (finding 3). The earlier "draft
architecture is worse" comparison compared a correct diagonal loss against the broken
full-covariance one and is void.

## 2. Changes (commit d84555b92 and after)

- `datasets_fedheart.py`: fixed stratified K-fold per site (`n_folds`, `fold_index`; random state 0,
  independent of the model seed; verified 920/920 patients tested exactly once); imputation
  medians from training rows only.
- `train_fedheart.py`, `train_nhanes.py`: checkpoint / early stopping / scheduler read validation
  only and raise without a validation split (`allow_test_selection: true` reproduces the old
  behaviour); outputs `selected_epoch`, `validation_score_at_selected_epoch`,
  `test_metrics_at_selected_epoch`; anchor weight 0 skips the anchor terms; every weight update
  logged to `lambda_updates.jsonl` (weights, signed payoff, L1 change, entropy).
- `model/wasserstein.py`: principal square root by eigendecomposition; `diagonal_gaussian_w2_squared`;
  tests against `scipy.linalg.sqrtm`.
- `model/anchors.py`: diagonal mode with one softplus scale for covariance and samples.
- `model/groupdro.py`: signed excess by default (`regret_clamp: true` = old), max-subtraction
  exponentiated update, absent groups get the mean payoff.
- `run_fedheart_cv.py`: `--rstar-dir` (one reference file per fold, hard error if missing),
  `--methods`, `--legacy-holdouts`; anchors-off weight 0.0.
- `run_baselines_tabular.py`: same real folds as our arms; `--rstar`, `--legacy-holdouts`.
- `estimate_rstar_v3.py` (Xenia's): adopted. Two marked additions: `--n-folds/--fold` (references
  from one outer fold's training rows, same partition as the trainer) and `--joint-head-hidden`
  (joint candidate's head matches the deployed MLP head); fold id and row hash recorded.
  Open point for Xenia: when the constant predictor is the chosen bound (VA, folds 0 and 4) the
  margin is still bootstrapped from the fitted model's per-sample losses (0.215-0.225 vs about
  0.095 in the other folds); it should arguably come from the constant predictor's losses.
- Config `experiments/fedheart_v3.yaml`: diagonal anchors, per-(group, class) alignment, uniform
  weight start, signed excess, alignment out of the weight signal, MLP head 32, no caps.

## 3. Fed-Heart, corrected

References per fold (`runs/rstar_v3/fedheart/fold<k>.json`): Cleveland 0.37-0.40, Hungarian
0.33-0.41, Switzerland 0.06-0.10, VA 0.34-0.49. Selection bias of the old min-over-candidates
estimator: about 0 except VA (up to 0.10).

η sweep (`runs/v3_eta_sweep_fh`, 3 seeds × 5 folds, 8 cells per arm, chosen on validation
worst-group excess): GroupDRO 10, Regret-DRO 2, Ours(GroupDRO) 0.1, full method 2. At η = 0.1
per epoch the weights move 0.05-0.08 (L1); from η = 2 they move 0.4-1.3. Test worst-group
accuracy is flat in η (70-73).

Matrix (`runs/fedheart_v3`, 10 seeds × 5 folds), worst-group acc | loss | excess | overall acc:

| arm | corrected | earlier pipeline (acc \| loss) |
|---|---|---|
| ERM, common features | 68.64 \| 0.658 \| 0.493 \| 72.3 | 68.28 \| 0.638 |
| shared + GroupDRO | 67.63 \| 0.664 \| 0.447 \| 71.8 | 68.13 \| 0.628 |
| shared + anchors | 66.51 \| 0.666 \| 0.489 \| 71.0 | 68.99 \| 0.620 |
| shared + anchors + GroupDRO | 67.13 \| 0.669 \| 0.450 \| 71.1 | 67.17 \| 0.613 |
| per-group ERM | 74.30 \| 0.590 \| 0.285 \| 78.6 | 72.85 \| 0.577 |
| per-group + GroupDRO | 74.89 \| 0.577 \| 0.209 \| 79.1 | 72.99 \| 0.583 |
| per-group + Regret-DRO | 74.19 \| 0.568 \| 0.259 \| 78.8 | 73.18 \| 0.576 |
| per-group + anchors | 73.24 \| 0.627 \| 0.351 \| 78.5 | 71.20 \| 0.597 |
| per-group + anchors + GroupDRO | 73.94 \| 0.612 \| 0.276 \| 78.5 | 71.15 \| 0.591 |
| full method | 72.93 \| 0.600 \| 0.240 \| 77.9 | 72.25 \| 0.570 |

Paired over seeds: per-group ERM vs common-features ERM +5.66 acc (p < 0.001), loss −0.068
(p = 0.002). Full method vs ERM +4.29 (p < 0.001), excess 0.493 → 0.240. Full method vs per-group
ERM −1.37 acc (p = 0.033), loss tie. Full method vs Regret-DRO without anchors: loss +0.032
(p = 0.018). Regret vs GroupDRO: acc and loss tie; GroupDRO lower on excess (p = 0.028).

Baselines on the same folds (`runs/baselines_v3/fedheart_{released,matched}`), published configs:
Reweigh 72.35 | 0.633 | 0.347; Flex-MoE 72.65 | 0.964 | 0.880; REMIND 72.95 | 0.600 | 0.286.
- Full method vs baselines: accuracy ties (±0.6, n.s.); loss lower vs Flex-MoE only (37.8%,
  p = 0.003), tie vs Reweigh and REMIND; excess 31% lower vs Reweigh (p = 0.004), 73% vs Flex-MoE
  (p < 0.001), 16% vs REMIND (n.s.).
- Per-group + GroupDRO vs baselines: accuracy +2.5 / +2.2 / +1.9 (all p < 0.01); loss 9% / 40% /
  4% lower (first two significant); excess 40% / 76% / 27% lower (all p ≤ 0.001).

Reading: on Fed-Heart the per-group encoders are the contribution. Neither the anchors nor the
regret objective adds to them; the anchors cost worst-group loss. The claim "the full method
has lower worst-group loss than every baseline on Fed-Heart" does not hold under the corrected
pipeline; "per-group encoders with GroupDRO beat all three baselines on worst-group accuracy and
excess" does.

## 4. NHANES, corrected

Loader: age, education and income were imputed with all-rows medians; now training rows only.
References (`runs/rstar_v3/nhanes/nested.json`, train split only, nested): 0.279 / 0.242 / 0.252
(earlier 0.279 / 0.239 / 0.250). Selection bias 0.000 for every group; ordering certificate
holds; 44% of candidate fits hit the 1,500-step cap (every selected family was linear).

η sweep (`runs/v3_eta_sweep_nh`, 3 seeds, validation worst-group excess): GroupDRO 20, Regret-DRO
20, Ours(GroupDRO) 1, full method per-step 0.1. Weights move 0.01-0.06 at η = 0.1, fully from η = 5.

Matrix (`runs/nhanes_v3`, 10 seeds), worst-group acc | loss | excess | AUROC | class-balanced acc:

| arm | corrected | earlier (acc \| loss) |
|---|---|---|
| ERM, common features | 68.80 \| 0.504 \| 0.242 \| 0.796 \| 71.7 | 69.28 \| 0.496 |
| shared + GroupDRO | 73.32 \| 0.479 \| 0.207 \| 0.780 \| 69.9 | 73.08 \| 0.460 |
| shared + anchors | 70.89 \| 0.487 \| 0.221 \| 0.795 \| 70.9 | 72.87 \| 0.463 |
| shared + anchors + GroupDRO | 73.23 \| 0.461 \| 0.188 \| 0.790 \| 69.7 | 77.43 \| 0.441 |
| per-group ERM | 65.54 \| 0.576 \| 0.312 \| 0.793 \| 69.6 | 66.88 \| 0.547 |
| per-group + GroupDRO | 69.05 \| 0.504 \| 0.249 \| 0.797 \| 70.2 | 71.35 \| 0.507 |
| per-group + Regret-DRO | 69.10 \| 0.497 \| 0.242 \| 0.799 \| 70.3 | 70.84 \| 0.527 |
| per-group + anchors | 68.36 \| 0.560 \| 0.289 \| 0.796 \| 69.4 | 68.34 \| 0.548 |
| per-group + anchors + GroupDRO | 71.78 \| 0.536 \| 0.275 \| 0.789 \| 66.8 | 74.30 \| 0.524 |
| full method | 72.06 \| 0.532 \| 0.277 \| 0.792 \| 67.2 | 74.67 \| 0.522 |

Paired: full method vs ERM +3.26 acc (p = 0.001), loss +0.028 (n.s.). Anchors: +2.95 over
Regret-DRO (p = 0.003), +2.73 over GroupDRO (p = 0.002), with loss +0.03 (n.s.) and lower AUROC
and class-balanced accuracy, i.e. partly an operating-point effect, as before. Regret vs
GroupDRO: tie. Per-group ERM vs common-features ERM: −3.26 acc (p = 0.001), loss +0.072.
Full method vs shared + anchors + GroupDRO: acc −1.17 (n.s.), loss +0.071 (p = 0.001).

Baselines (`runs/baselines_v3/nhanes_{released,matched}`): Reweigh 72.24 | 0.667 | 0.413;
Flex-MoE 72.50 | 0.644 | 0.377; REMIND 72.25 | 0.665 | 0.393. Full method vs each: accuracy tie
(±0.5), loss 20.2% / 17.5% / 20.0% lower (all p ≤ 0.005), excess 32.9% / 26.6% / 29.5% lower (all
p ≤ 0.013).

Reading: the comparison against the published baselines survives on NHANES (loss and excess
lower than all three, accuracy tied). The full method's accuracy is 2.6 lower than the earlier
pipeline reported, and the common-features GroupDRO arms remain the strongest rows on loss.

## 5. NHANES with no common information, corrected

`experiments/nhanes_partition_v3.yaml`, references `runs/rstar_v3/nhanes/partition.json` (0.277 /
0.291 / 0.289; the ordering certificate prints VIOLATED, which is expected: these groups are not
nested, so the check does not apply). η as selected on nested NHANES. 10 seeds
(`runs/nhanes_partition_v3`, `runs/baselines_v3/nhanes_partition`).

| arm | per-group acc G0 / G1 / G2 | worst-group acc | worst-group loss | AUROC overall (per group) |
|---|---|---|---|---|
| ERM, common features | 88.0 / 90.0 / 89.7 | 88.0 | 0.681 | 0.500 (0.50 / 0.50 / 0.50) |
| common features + GroupDRO | 88.0 / 90.0 / 89.7 | 88.0 | 0.651 | 0.500 |
| per-group ERM | 66.4 / 72.7 / 72.6 | 66.2 | 0.618 | 0.709 (0.78 / 0.59 / 0.71) |
| per-group + GroupDRO | 70.3 / 81.0 / 71.9 | 69.5 | 0.611 | 0.703 |
| per-group + Regret-DRO | 69.2 / 84.1 / 72.6 | 68.9 | 0.597 | 0.706 |
| per-group + anchors | 70.5 / 79.2 / 71.5 | 69.1 | 0.593 | 0.705 |
| per-group + anchors + GroupDRO | 75.2 / 83.9 / 76.2 | 73.4 | 0.600 | 0.689 |
| full method | 76.9 / 85.2 / 77.4 | 74.7 | 0.588 | 0.687 (0.77 / 0.57 / 0.69) |
| Reweigh | 73.4 / 90.0 / 89.7 | 73.4 | 0.646 | not logged |
| Flex-MoE | 77.2 / 90.0 / 89.7 | 77.2 | 0.683 | not logged |
| REMIND | 71.2 / 90.0 / 89.7 | 71.2 | 0.630 | not logged |

The 88-90% accuracies are the base rate: those models predict the majority class. The shared-
encoder arms collapse on every group (AUROC 0.500); the three baselines collapse on G1 and G2
(exactly 90.0 / 89.7 on every seed) and model only G0. The per-group arms keep a predictor on all
three groups, and here the anchors and regret both help (66.2 → 74.7 worst-group accuracy, loss
0.618 → 0.588). The earlier conclusion survives the corrections.

## 6. EMBED, corrected

EMBED already had diagonal anchors, validation-only selection and train+val references. The one
finding that applies is the clamp (5), and only to the two regret arms: for ERM, anchors-only,
GroupDRO and anchors+GroupDRO the reference is 0, so the clamp was never active; those rows are
unchanged. Signed excess, γ ∈ {0.5, 2.0, 8.0}, 10 seeds (`runs/embed_v3`), γ chosen on validation
max-excess: Regret-DRO 0.5, full method 2.0.

| arm | worst-group acc | loss | excess | overall acc |
|---|---|---|---|---|
| ERM | 62.16 | 1.350 | 0.452 | 75.72 |
| GroupDRO | 62.70 | 1.051 | 0.179 | 77.44 |
| Regret-DRO, clamped (earlier) | 61.02 | 1.344 | 0.470 | 75.66 |
| Regret-DRO, signed | 62.97 | 1.042 | 0.264 | 77.56 |
| anchors + GroupDRO | 61.62 | 1.134 | 0.324 | 77.02 |
| full method, clamped (earlier) | 62.74 | 1.066 | 0.280 | 76.47 |
| full method, signed | 62.20 | 1.067 | 0.273 | 76.19 |

The signed excess repairs Regret-DRO without anchors (loss 1.344 → 1.042, p < 0.001), which under
the clamp never moved its weights; the full method is unchanged (p = 0.93). The three DRO arms
are indistinguishable from one another (full method vs Regret-DRO: acc −0.77, loss +0.025, n.s.).
Against the baselines (Reweigh 59.35 | 2.115; Flex-MoE 62.26 | 1.333; REMIND-128 61.78 | 1.207) the
full method has 49.6% / 20.0% / 11.7% lower worst-group loss and 79.9% / 56.0% / 39.2% lower excess
(all p < 0.005), +1.2 to +6.4 overall accuracy (p ≤ 0.026), and ties on worst-group accuracy.

## 7. What the corrected runs support (for the abstract and the results section)

Full method against the three published baselines, published configurations, paired over 10 seeds:

| | worst-group loss lower by | worst-group excess lower by | worst-group acc |
|---|---|---|---|
| NHANES | 20.2% / 17.5% / 20.0% (all sig.) | 32.9% / 26.6% / 29.5% (all sig.) | tie |
| Fed-Heart | 5.2% (n.s.) / 37.8% (sig.) / 0.0% (n.s.) | 30.9% (sig.) / 72.8% (sig.) / 16.3% (n.s.) | tie |
| EMBED | 49.6% / 20.0% / 11.7% (all sig.) | 79.9% / 56.0% / 39.2% (all sig.) | tie |

- "Lower worst-group loss than every baseline on every dataset" no longer holds: on Fed-Heart the
  full method ties Reweigh and REMIND on loss. Supported: lower worst-group loss than all three
  baselines on NHANES and EMBED (12-50%, all significant), and lower worst-group excess on all
  three datasets (16-80%; 8 of 9 significant, the exception Fed-Heart vs REMIND).
- Worst-group accuracy matches the baselines everywhere; overall accuracy is significantly higher
  on EMBED and tied on the tabular datasets.
- Which component helps depends on the dataset: per-group encoders on Fed-Heart (+5.7 over the
  common-features model; anchors and DRO add nothing), anchors on NHANES (+3 accuracy over
  unanchored DRO, partly an operating-point effect), the group weighting on EMBED. Regret ties
  GroupDRO on accuracy and loss on all three; the signed excess is what makes it work at all on EMBED.
- The strongest unambiguous result is the no-common-information setting: shared-encoder models
  and all three baselines fall to the majority class, the per-group arms do not, and there the
  anchors and regret both help.
- Parameter counts are unchanged (1.4-17x fewer than the baselines on the tabular datasets).
