#!/bin/bash
cd "$(cd "$(dirname "$0")" && pwd)"
PY=.venv/bin/python
SEEDS="42 1337 7 2024 31337 11 22 33 44 55"
LOG=runs/overnight.log
echo "[overnight] start $(date)" > $LOG

# 1. Baselines at RELEASED defaults (their configuration, as published)
for D in nhanes fedheart; do
  $PY run_baselines_tabular.py --dataset $D --seeds $SEEDS --epochs 60 \
      --out runs/baselines_$D >> $LOG 2>&1
  echo "[overnight] baselines $D released-defaults done $(date)" >> $LOG
done

# 2. Baselines CAPACITY-MATCHED to our model, so the comparison isolates the architecture
for D in nhanes fedheart; do
  $PY run_baselines_tabular.py --dataset $D --seeds $SEEDS --epochs 60 --capacity-matched \
      --out runs/baselines_${D}_matched >> $LOG 2>&1
  echo "[overnight] baselines $D capacity-matched done $(date)" >> $LOG
done

# 3. Our method with the per-dataset best lambda_fit. Our own sweep put Fed-Heart's optimum
#    at 0.3 and we have been running 0.1, which is NHANES's optimum. We tuned the baselines
#    at their published defaults and under-tuned ourselves.
$PY run_fedheart_lamfit.py >> $LOG 2>&1
echo "[overnight] fedheart lambda_fit sweep under CV done $(date)" >> $LOG

# 4. Dynamics reruns so the plots carry train AND test loss
$PY run_dynamics_fedheart.py >> $LOG 2>&1
echo "[overnight] fedheart dynamics done $(date)" >> $LOG
echo "[overnight] ALL DONE $(date)" >> $LOG
