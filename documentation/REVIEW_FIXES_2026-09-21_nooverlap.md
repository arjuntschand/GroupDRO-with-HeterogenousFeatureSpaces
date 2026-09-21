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
