#!/bin/bash
cd /Users/arjun/Documents/GitHub/GroupDRO-with-HeterogenousFeatureSpaces
PY=.venv/bin/python
SEEDS="42 1337 7 2024 31337 11 22 33 44 55"
$PY run_baselines_tabular.py --dataset nhanes   --seeds $SEEDS --epochs 60 \
    > runs/baselines_nhanes.log 2>&1
echo "[baselines] nhanes done $(date)"
$PY run_baselines_tabular.py --dataset fedheart --seeds $SEEDS --epochs 60 \
    > runs/baselines_fedheart.log 2>&1
echo "[baselines] fedheart done $(date)"
