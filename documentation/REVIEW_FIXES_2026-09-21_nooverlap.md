# No-overlap experiments: what was wrong and the corrected results (2026-09-21)

Prompted by Xenia's question: how can "ERM, common features" be trained when the groups are disjoint?

## What was wrong in the first no-common-information NHANES table
1. `run_method_matrix.py` runs the same arm list on every dataset. On `feature_mode: partition` the
   intersection of the groups' features is empty, so the "common features" arms (ERM, Shared_GDRO,
   Shared_Anchors_GDRO) were built with 0 input features (log line: "Shared encoder: common
   features only (0 features)"). They can only predict a constant; AUROC 0.500 was by construction.
2. `run_baselines_tabular.py` used the nested NHANES block constants for every feature mode. On the
   partition layout (G0 cols 0-9, G1 cols 15-19, G2 cols 20-24) the baselines read all-zero columns
   for G1 and never read G2's columns. Their "collapse to the majority class" was this, not the methods.
   Fixed in `build_blocks` (blocks now come from the loader's own per-group column lists). The
   nested and Fed-Heart block layouts are unchanged, so the main v3 baselines are unaffected.

## Valid comparators added
- `Independent`: per-group encoder AND per-group head (`per_group_head: true`), same trainer.
- Shared encoder on the zero-filled union (`shared_common_features: false`), with ERM / GroupDRO /
  anchors + GroupDRO.
- Flex-MoE is reported as not applicable: as published its first stage trains experts on samples
  with all modalities, and none exist when groups share nothing. REMIND and Reweigh need no change:
  REMIND's paper defines groups as arbitrary modality combinations, fills absent modalities with
  learnable embeddings, and shares experts, routing matrix and head across groups.

## NHANES, no overlap (10 seeds; worst-group acc | loss | excess; p vs full method)
| arm | acc | loss | excess |
|---|---|---|---|
| Full method | 74.7 | 0.588 | 0.301 |
| Per-group + anchors + GroupDRO | 73.4 | 0.600 | 0.315 |
| Per-group + Regret-DRO | 68.9 (p=0.004) | 0.597 | 0.307 |
| Per-group + GroupDRO | 69.5 (p=0.02) | 0.611 | 0.322 |
| Per-group + ERM, shared head | 66.2 (p<0.001) | 0.618 | 0.330 |
| Independent model per group | 65.6 (p<0.001) | 0.621 | 0.332 |
| Shared encoder, zero-filled + anchors + GroupDRO | 74.2 (n.s.) | 0.521 (p=0.01, better) | 0.234 |
| Shared encoder, zero-filled + GroupDRO | 73.2 | 0.552 (p=0.04, better) | 0.263 |
| Shared encoder, zero-filled + ERM | 66.6 | 0.582 | 0.300 |
| REMIND | 72.1 (p=0.10) | 0.563 (n.s.) | 0.277 |
| Reweigh | 71.6 (p=0.06) | 0.567 (n.s.) | 0.281 |

## Fed-Heart, no overlap, rotation 0 (10 seeds x 5 folds)
Blocks fixed in advance: [age, sex, trestbps] / [thalach, exang, oldpeak] / [cp one-hot] /
[chol, fbs, restecg one-hot] -> Cleveland / Hungarian / Switzerland / VA. Step sizes carried over
from the main v3 runs. Independent 63.4 | 0.636; per-group ERM 64.1 | 0.632; GroupDRO 64.6 | 0.632;
REMIND 63.9 | 0.641; Reweigh 63.8 | 0.641; shared zero-filled 60.9-61.5 | 0.65-0.67; full method
62.9 | 0.708 (loss significantly worse than every per-group arm and both baselines). Sharing does
not help here. Rotations 1-3 of the block-to-hospital assignment are queued.

## EMBED, no overlap
`--disjoint`: g1 {M3}, g2 {M1}, g3 {M4}, g6 {M2}; g4 and g5 dropped (four views cannot give six
disjoint groups). Runs in `runs/embed_disjoint_v3`, baselines in `runs/baselines_embed_disjoint`.

## Checks added later on 2026-09-21

**Input-scramble check** (`NHANES_SCRAMBLE_GROUP=<g>`, diagnostic hook in `datasets_nhanes.py`, seed 42):
permuting one group's feature rows sends that group's macro-F1 to chance (about 0.47) for the full
method, Independent, REMIND, Reweigh and the REMIND backbone under ERM, so all of them use every
group's inputs. Flex-MoE does not change on G1/G2 because it already predicts only the majority
class there (acc exactly 90.0 / 89.7).

**REMIND's backbone under plain ERM** (`SoftMoE_ERM`): 67.2 worst-group accuracy, level with
Independent (65.6) and per-group ERM (66.2). REMIND's 72.1 comes from its group reweighting;
Reweigh matches it (71.6).

**AUROC (threshold-free), NHANES no overlap, 10 seeds** (`runs/nhanes_nooverlap_v3/auroc_long.csv`;
baselines re-run with AUROC logging in `runs/baselines_v3/nhanes_partition_auroc_*`):
worst-group AUROC REMIND 0.607, Imputation-ERM 0.596, AnchorsOnly 0.595, REMIND-backbone-ERM 0.595,
Independent 0.594, per-group ERM 0.587, Reweigh 0.583, GroupDRO 0.574, Imputation+anchors+GroupDRO
0.570, Ours_GDRO 0.568, full method 0.567, RegretDRO 0.567, Flex-MoE 0.555. No method differs
significantly from the full method (closest: REMIND, p = 0.07, in REMIND's favour). Per-group AUROC
is about 0.77 / 0.57-0.61 / 0.70 for every method including separate models.

Conclusion: on this split no method, ours included, discriminates better than a separate model per
group. The +9 points of worst-group accuracy of the full method over Independent is an operating-
point shift on a 90%-negative task, not better discrimination; the methods with the highest accuracy
have the lowest AUROC. Majority-class seeds (macro-F1 < 0.48): Flex-MoE 2-3 of 10 per group; our
anchored arms 4 of 10 on G1; REMIND 1; Reweigh 0. Both our arms and the baselines select the
reported epoch on validation worst-group ACCURACY, which itself favours majority-leaning epochs on
this data. The same caution applies to the accuracy gains on the main NHANES table.

## AUROC on the MAIN NHANES table (2026-09-21, late)
Baselines re-run with AUROC logging (`runs/baselines_v3/nhanes_auroc_*`; means reproduce the table
run: worst-group acc within 0.15, worst-group loss identical to 3 decimals). Our arms' AUROC comes
from the trainer's own per-group logging at the validation-selected epoch. `runs/nhanes_v3/auroc_long.csv`.

Worst-group AUROC, 10 seeds: ERM common features 0.785; shared anchors + GroupDRO 0.781; per-group
Regret-DRO 0.776; per-group GroupDRO 0.775; shared GroupDRO 0.772; AnchorsOnly 0.766; per-group ERM
0.765; per-group + anchors + GroupDRO 0.761; full method 0.760; Flex-MoE 0.723; REMIND 0.722;
Reweigh 0.712.

- Against the published baselines the result HOLDS on the threshold-free metric: full method vs
  Flex-MoE p = 0.04, vs REMIND p = 0.002, vs Reweigh p < 0.001.
- Inside our ablation the anchors' +3 worst-group accuracy does NOT reflect better discrimination:
  per-group GroupDRO 0.775 vs full method 0.760 (p = 0.04, anchors lower). It is an operating-point
  effect. ERM on the ten common features has the highest AUROC (better than the full method, p = 0.003).

## EMBED, no overlap: results (10 seeds)
`runs/embed_disjoint_v3/metrics_long.csv` (arms from job g2.0, regret_only from g0.5; group_only's
excess recomputed against the same references), baselines `runs/baselines_embed_disjoint`. Run on
the laptop from the private S3 feature bundle (the GPU instance had no capacity); nothing from the
bundle is tracked by git.

worst-group acc | loss | excess: dedicated model per group 61.3 | 0.927 | 0.097; per-group GroupDRO
59.4 | 0.994 | 0.161; full method 56.4 | 1.016 | 0.184; AnchorsOnly 62.0 | 1.024; Regret-DRO 64.3 |
1.040; anchors + GroupDRO 56.4 | 1.041; per-group ERM 62.7 | 1.492 | 0.660; Flex-MoE 62.3 | 1.508 |
0.748; REMIND 58.9 | 1.970 | 1.204; Reweigh 60.7 | 2.183 | 1.418.

- Full method vs baselines: loss 33-53% lower, excess 75-87% lower (all p < 0.001); worst-group
  accuracy lower than Flex-MoE (p = 0.007) and Reweigh (p = 0.001), tie with REMIND. The worst
  group holds 44 exams, so accuracy moves in steps of 2.3 points.
- Dedicated models beat the full method on worst-group loss (p < 0.001) and accuracy. Sharing does
  not help on disjoint EMBED; the robust objective is what separates the DRO arms from ERM and from
  the baselines. Anchors add nothing over per-group GroupDRO.

Across all three no-overlap experiments: no method that shares parameters, ours included, beats a
dedicated model per group on discrimination or loss (NHANES: equal AUROC; Fed-Heart: equal; EMBED:
dedicated models better).
