"""Fed-Heart training-dynamics runs, same two configs as NHANES, with per-epoch group
weights logged so the lambda plots reflect what actually drove the gradients."""
from __future__ import annotations
import shutil
import json, yaml
from dro_hetero_anchors.src.train_fedheart import train

BASE = "experiments/fedheart_exp_paper_hetagg_gdro.yaml"
rs = json.load(open("runs/rstar_fedheart.json"))["rstar"]
rstar = [rs[str(i)] if str(i) in rs else rs.get(i, 0.0) for i in range(len(rs))]
print("R*_g =", [round(x, 4) for x in rstar], flush=True)

for tag, regret in [("anchors_groupdro", False), ("anchors_regretdro", True)]:
    cfg = yaml.safe_load(open(BASE))
    cfg.update(seed=42, common_encoder=False, groupdro_enabled=True,
               lambda_fit=0.1, lambda_sep=0.1, impute_missing=True,
               use_regret=regret, optimal_losses=rstar,
               run_dir=f"runs/dynamics_fedheart/{tag}")
    shutil.rmtree(cfg["run_dir"], ignore_errors=True)   # fresh header
    print(f"\n=== {tag} (regret={regret}) ===", flush=True)
    train(cfg)
