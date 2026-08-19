# EMBED results log

## Run 1 — first real GPU run (2026-08-19)  [PRELIMINARY]

**Setup:** AWS g5.xlarge (A10G), subset of ~8,241 breast-samples (per-group cap 2000,
require_local_images), resnet18, 15 epochs, 1 seed (42), class_weight=auto,
min_group_size=30. Core ablation cells. Metric = accuracy.

| cell | encoder | anchors | GroupDRO | tail acc | overall acc |
|------|---------|:-------:|:--------:|:--------:|:-----------:|
| erm_shared (baseline) | shared | – | – | 74.5% | 77.6% |
| gdro_shared | shared | – | ✓ | 76.2% | 77.0% |
| anchors_gdro_shared | shared | ✓ | ✓ | 75.9% | **78.4%** |
| full | per-group | ✓ | ✓ | **76.6%** | 77.3% |

**Signal:** tail (worst-group) accuracy rises as method components are added
(ERM 74.5 -> GDRO 76.2 -> full 76.6; +2.1 pts on the tail) while overall holds ~77-78%.
Consistent with the thesis. REMIND's EMBED reference: head 75.8 / tail 82.8 / overall 80.7.

**Caveats (why this is preliminary, not the headline):**
- Subset over-represents tail groups (per-group cap distorts head/tail proportions vs full data).
- 1 seed, 15 epochs, resnet18, no hyperparameter tuning.
- Need: proportional/full-data run, >=3 seeds, HP sweep, resnet50, then compare to REMIND Table 1.

Cost so far: ~2-3 hrs g5.xlarge (~$2-3) + ~$40/mo for the 500GB EBS while stopped.
