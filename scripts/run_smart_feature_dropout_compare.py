#!/usr/bin/env python3
"""
Compare common vs per-group encoders under a smarter, medical-plausible feature masking scheme.

By default this matches your earlier regime:
  - feature masking is done by zeroing features in the dataset
  - encoders keep `input_dim=13` for all groups

It runs 5 seeds for:
  1) common encoder + ERM
  2) common encoder + GroupDRO
  3) per-group encoder + ERM
  4) per-group encoder + GroupDRO
"""

import os
import sys
import json
import yaml
import shutil
import subprocess
from pathlib import Path
import numpy as np


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
VENV_PY = os.path.join(PROJECT_ROOT, "venv", "bin", "python")
if not os.path.exists(VENV_PY):
    VENV_PY = sys.executable

SEEDS = [7, 13, 27, 51, 99]

# Feature index mapping (Fed-Heart 13 features):
# 0 age, 1 sex, 2 cp, 3 chol, 4 fbs, 5 thalach, 6 exang, 7 oldpeak, 8 slope,
# 9 ca, 10 thal, 11 restecg_0, 12 restecg_1
#
# Smarter masking:
# - age/sex are always kept
# - drop correlated "test bundles" together:
#   - Blood panel: chol+fbs
#   - Exercise stress test bundle: thalach+exang+oldpeak+slope
#   - Resting ECG bundle: restecg_0+restecg_1
#
# Group -> retained features (null means keep all 13):
SMART_FEATURE_MASK = [
    None,  # G0 Cleveland: all 13
    [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12],  # G1 Hungarian: drop chol,fbs
    [0, 1, 2, 3, 4, 9, 10, 11, 12],        # G2 Switzerland: drop exercise bundle
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],    # G3 VA Long Beach: drop resting ECG
]


def write_cfg(tmp_path: str, cfg: dict) -> None:
    with open(tmp_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_one(cfg: dict) -> dict | None:
    run_dir = cfg["run_dir"]
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    cfg_path = f"/tmp/{cfg['run_name']}.yaml"
    write_cfg(cfg_path, cfg)

    cmd = [VENV_PY, "-m", "dro_hetero_anchors.src.train_fedheart", "--config", cfg_path]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or "").strip().split("\n")[-1:]
        print(f"  [FAIL] {cfg['run_name']}: {err[0] if err else 'unknown error'}")
        return None

    metrics_path = Path(run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        print(f"  [FAIL] {cfg['run_name']}: missing metrics.jsonl")
        return None

    # Pick the epoch with highest test_worst_group_acc
    best = None
    for line in metrics_path.read_text().strip().split("\n"):
        m = json.loads(line)
        if best is None or m.get("test_worst_group_acc", 0) > best.get("test_worst_group_acc", 0):
            best = m
    return best


def summarize(seeds_results: list[dict], num_groups: int = 4) -> dict:
    # seeds_results already contain the chosen "best epoch" metrics for each seed.
    per_group_acc = {g: [] for g in range(num_groups)}
    per_group_loss = {g: [] for g in range(num_groups)}
    worst_group_accs = []

    for m in seeds_results:
        per_acc = m.get("test_per_group_acc", [])
        per_loss = m.get("test_per_group_loss", [])
        for g in range(num_groups):
            if g < len(per_acc):
                per_group_acc[g].append(per_acc[g])
            if g < len(per_loss):
                per_group_loss[g].append(per_loss[g])
        worst_group_accs.append(m.get("test_worst_group_acc", 0))

    out = {"n_seeds": len(seeds_results)}
    for g in range(num_groups):
        a = np.array(per_group_acc[g], dtype=float)
        l = np.array(per_group_loss[g], dtype=float)
        out[f"g{g}_acc_mean"] = float(np.mean(a)) if a.size else 0.0
        out[f"g{g}_acc_std"] = float(np.std(a)) if a.size else 0.0
        out[f"g{g}_loss_mean"] = float(np.mean(l)) if l.size else 0.0
        out[f"g{g}_loss_std"] = float(np.std(l)) if l.size else 0.0

    wg = np.array(worst_group_accs, dtype=float)
    out["wg_acc_mean"] = float(np.mean(wg)) if wg.size else 0.0
    out["wg_acc_std"] = float(np.std(wg)) if wg.size else 0.0
    # overall_acc_mean isn't always reliable in metrics; we focus on per-group + worst-group
    return out


def fmt_pct(mean: float, std: float) -> str:
    return f"{mean * 100:.2f}% +/- {std * 100:.2f}%"


def main():
    group_names = ["Cleveland", "Hungarian", "Switzerland", "VA Long Beach"]

    # Shared hyperparameters (match your earlier "paper-style" defaults)
    base = {
        "use_fedheart": True,
        "num_classes": 2,
        "stratified_batching": True,
        "train_frac": 0.66,
        "group_max_train_samples": None,  # no cap (natural imbalance)
        "batch_size": 64,
        "num_workers": 0,
        "epochs": 100,
        "early_stopping_patience": 30,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "optimizer": "adam",
        "grad_clip": 1.0,
        "lr_scheduler": "cosine",
        "lr_min": 1.0e-5,
        "latent_dim": 64,
        "head_hidden": 32,
        "anchor_eps": 0.0001,
        "sep_samples_per_class": 4,
        "sep_method": "classifier",
        "sep_margin": 1.0,
        "lambda_fit": 0.001,
        "lambda_sep": 0.001,
        "log_interval": 20,
        "save_every": 10,
        "feature_mask": SMART_FEATURE_MASK,
    }

    # Config variants
    variants = [
        ("common_erm", True, False),
        ("common_gdro", True, True),
        ("hetero_erm", False, False),
        ("hetero_gdro", False, True),
    ]

    all_summaries = {}

    for variant_name, common_encoder, groupdro_enabled in variants:
        print(f"\n=== Variant: {variant_name} ===")
        seeds_results = []

        for seed in SEEDS:
            run_name = f"smart_{variant_name}_s{seed}"
            run_dir = f"runs/{run_name}"

            cfg = dict(base)
            cfg.update(
                {
                    "seed": seed,
                    "run_name": run_name,
                    "run_dir": run_dir,
                    "common_encoder": common_encoder,
                    "groupdro_enabled": groupdro_enabled,
                    "groups": [
                        {"name": "cleveland", "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": 64, "dropout": 0.1},
                        {"name": "hungarian", "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": 64, "dropout": 0.1},
                        {"name": "switzerland", "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": 64, "dropout": 0.1},
                        {"name": "va_long_beach", "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": 64, "dropout": 0.1},
                    ],
                }
            )

            if groupdro_enabled:
                cfg.update(
                    {
                        "groupdro_eta": 1.0,
                        "groupdro_gamma": 0.9,
                        "groupdro_update_mode": "softmax",
                        "groupdro_objective": "weighted",
                        "groupdro_kl_lambda": 0.0,
                        "groupdro_uniform_init": True,
                    }
                )

            print(f"  Running seed={seed} ...")
            best_metrics = run_one(cfg)
            if best_metrics is not None:
                seeds_results.append(best_metrics)

        if len(seeds_results) != len(SEEDS):
            print(f"  [WARN] variant {variant_name}: only {len(seeds_results)}/{len(SEEDS)} seeds succeeded")

        summary = summarize(seeds_results, num_groups=4) if seeds_results else None
        all_summaries[variant_name] = summary

    # Print side-by-side summary
    print("\n\n================ SMART FEATURE DROPOUT SUMMARY ================\n")
    for variant_name in [v[0] for v in variants]:
        s = all_summaries.get(variant_name)
        if not s:
            continue
        print(f"{variant_name}: WG acc = {fmt_pct(s['wg_acc_mean'], s['wg_acc_std'])}")
        for g in range(4):
            ga_m = s[f"g{g}_acc_mean"]
            ga_s = s[f"g{g}_acc_std"]
            gl_m = s[f"g{g}_loss_mean"]
            gl_s = s[f"g{g}_loss_std"]
            print(
                f"  G{g} {group_names[g]:<13} acc={fmt_pct(ga_m, ga_s)}  loss={gl_m:.3f} +/- {gl_s:.3f}"
            )

        # Show ERM -> GroupDRO gain if applicable
        if variant_name.endswith("gdro"):
            base_name = variant_name.replace("gdro", "erm")
            bs = all_summaries.get(base_name)
            if bs:
                gain = (s["wg_acc_mean"] - bs["wg_acc_mean"]) * 100
                print(f"  WG gain (GDRO - ERM) = {gain:+.2f} percentage points")

        print()

    # Save json
    out_path = "runs/smart_feature_dropout_compare_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

