# Legacy code

Everything in this directory is archived exploration that predates the paper's
experimental protocol. None of it produced a number that appears in the paper,
the results site, or `runs/FINAL_TABLES.txt`. It is kept for provenance only
and is not maintained: imports still refer to the old package layout, so these
scripts do not run from here without adjustment. To run them as they were,
check out the tag `pre-cleanup-2026-09-16` (local only; it holds the
pre-cleanup history including all run outputs).

| Directory | What it was |
|---|---|
| `mnist_usps/` | The first testbed: MNIST (28x28) and USPS (16x16) as two groups with different resolutions. Loader, trainer, CNN encoders, the multi-seed suite runners, debug scripts, and the configs that the runners referenced. |
| `textcaps/` | Multimodal exploration on TextCaps (image + OCR text as separate groups). Loaders (HuggingFace and local), trainer, text/visual encoders, a frozen-backbone feature-cache variant, the anchor-weight sweep, and the only TextCaps result file anything ever read. |
| `embed_resnet/` | The first EMBED mammography track: DICOM decoding, a ResNet multi-view encoder trained on images, and eleven runner scripts for its ablation, tuning, and seed-validation sweeps. Superseded in September 2026 by the frozen-ViT pipeline in `dro_hetero_anchors/src/train_embed_xenia.py`, which is what the paper reports. The docs here (`docs/`) describe the AWS setup used for that track. |
| `early_tabular/` | Superseded Fed-Heart and NHANES runners (single-split protocol, the 2x2x2 ablation grid, feature-mask and dropout sweeps), the earlier R* estimators, three generations of paper-package/figure builders, the markdown-stitching results viewer, and their configs. Replaced by `run_method_matrix.py`, `run_fedheart_cv.py`, `estimate_rstar_v2.py`, and `build_site.py` at the repo root. |

The run outputs these scripts wrote (under `runs/mnist*`, `runs/usps*`,
`runs/textcaps_*`, `runs/embed_ablation` and friends, the March 2026
`runs/fedheart_<variant>` and `runs/nhanes_<variant>_s*` directories) are
gitignored and exist only on the machine that produced them. Their summary
markdown and JSON files are in `runs/summaries_archive/`.
