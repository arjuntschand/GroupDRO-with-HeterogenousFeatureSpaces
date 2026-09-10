#!/bin/bash
# Fill in the missing PerGroupOnly arm (per-group encoders, no anchors, no DRO) on every
# tabular dataset. The matrix runner reuses any arm whose run_dir already has metrics.csv,
# so this only trains the new arm.
cd /Users/arjun/Documents/GitHub/GroupDRO-with-HeterogenousFeatureSpaces
PY=.venv/bin/python
set -x
$PY run_method_matrix.py --dataset nhanes --tag nhanes_disjoint \
  --base experiments/nhanes_disjoint_pergroup_gdro.yaml \
  --rstar runs/rstar_nhanes_disjoint.json >> runs/matrix_nhanes_disjoint.log 2>&1
$PY run_method_matrix.py --dataset nhanes --tag nhanes_nested \
  --base experiments/nhanes_pergroup_gdro.yaml \
  --rstar runs/rstar_nhanes_nested.json >> runs/matrix_nhanes_nested.log 2>&1
$PY run_method_matrix.py --dataset fedheart --tag fedheart \
  --base experiments/fedheart_exp_paper_hetagg_gdro.yaml --moving-split \
  --rstar runs/rstar_fedheart.json >> runs/matrix_fedheart.log 2>&1
echo "PERGROUP ARM COMPLETE $(date)"
