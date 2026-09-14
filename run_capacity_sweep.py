"""Capacity sweep for OUR method, to match the tuning effort the baselines received.

The baselines were run at two configurations, their published defaults and a capacity-matched
resize. Our method has only ever been run at one width (latent 64, hidden 64), which was carried
over from the EMBED spec and never swept for the tabular datasets. That asymmetry is the fair
criticism, not the parameter count itself.

Widening only: same layers, same structure, more units. Deepening would change the architecture
and stop being the same model.

  latent/hidden 32  ->    8,754 params
  latent/hidden 64  ->   30,818   (current)
  latent/hidden 128 ->  114,882
  latent/hidden 256 ->  442,754   (comparable to FlexMoE's published 536,354)
"""
from __future__ import annotations
import copy, json, shutil, sys
import numpy as np, yaml
from dro_hetero_anchors.src.train_nhanes import train as train_nhanes

BASE = "experiments/nhanes_pergroup_gdro.yaml"
SEEDS = [42, 1337, 7, 2024, 31337]
rs = json.load(open("runs/rstar_nhanes_nested.json"))["rstar"]
rstar = [rs[str(i)] if str(i) in rs else rs.get(i, 0.0) for i in range(len(rs))]

results = {}
for width in [32, 64, 128, 256]:
    accs = []
    for seed in SEEDS:
        cfg = yaml.safe_load(open(BASE))
        cfg.update(seed=seed, common_encoder=False, groupdro_enabled=True,
                   lambda_fit=0.1, lambda_sep=0.1, use_regret=True, optimal_losses=rstar,
                   latent_dim=width, head_hidden=max(8, width // 2),
                   run_dir=f"runs/capsweep_nhanes/w{width}_s{seed}")
        for g in cfg["groups"]:
            g["hidden_dim"] = width
        shutil.rmtree(cfg["run_dir"], ignore_errors=True)
        try:
            r = train_nhanes(cfg)
            accs.append(r.get("best_worst_group_acc", float("nan")) * 100)
        except Exception as e:
            print(f"  w{width} s{seed} FAILED: {e}", flush=True)
    if accs:
        results[width] = accs
        print(f"  width {width:>3}: {np.mean(accs):.2f} +- {np.std(accs, ddof=1):.2f}  (n={len(accs)})",
              flush=True)
json.dump(results, open("runs/capsweep_nhanes.json", "w"), indent=2)
print("\n  width   worst-group")
for w, a in results.items():
    print(f"  {w:>5}   {np.mean(a):.2f} +- {np.std(a, ddof=1):.2f}")
