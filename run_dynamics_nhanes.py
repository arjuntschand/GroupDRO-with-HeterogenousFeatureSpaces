"""Retrain the two full-method configs on NHANES with per-epoch group weights logged,
so the training-dynamics plots reflect what actually drove the gradients.

Two configs, exactly the pair Xenia asked for:
  anchors + per-group encoders + GroupDRO   (R* = 0)
  anchors + per-group encoders + regret-DRO (R* from the group-only models)
"""
from __future__ import annotations
import shutil
import copy, json, yaml
from dro_hetero_anchors.src.train_nhanes import train

BASE = "experiments/nhanes_pergroup_gdro.yaml"
RSTAR = "runs/rstar_nhanes_nested.json"
SEED = 42

rs = json.load(open(RSTAR))["rstar"]
rstar = [rs[str(i)] if str(i) in rs else rs.get(i, 0.0) for i in range(len(rs))]
print("R*_g =", [round(x, 4) for x in rstar], flush=True)

for tag, regret in [("anchors_groupdro", False), ("anchors_regretdro", True)]:
    cfg = yaml.safe_load(open(BASE))
    cfg.update(seed=SEED, common_encoder=False, groupdro_enabled=True,
               lambda_fit=0.1, lambda_sep=0.1,        # anchors ON
               use_regret=regret, optimal_losses=rstar,
               run_dir=f"runs/dynamics_nhanes/{tag}")
    shutil.rmtree(cfg["run_dir"], ignore_errors=True)   # fresh header
    print(f"\n=== {tag} (regret={regret}) ===", flush=True)
    train(cfg)
    print(f"=== {tag} done ===", flush=True)
