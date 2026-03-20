#!/usr/bin/env python3
"""
Run a small set of feature-mask schemes and compare:
  - common encoder ERM vs GroupDRO
  - per-group encoders ERM vs GroupDRO

Keeps `input_dim=13` for all groups (matches your screenshot regime).
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

group_names = ["Cleveland", "Hungarian", "Switzerland", "VA Long Beach"]


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

    best = None
    for line in metrics_path.read_text().strip().split("\n"):
        m = json.loads(line)
        if best is None or m.get("test_worst_group_acc", 0) > best.get("test_worst_group_acc", 0):
            best = m
    return best


def summarize(seeds_results: list[dict], num_groups: int = 4) -> dict:
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
    return out


def fmt_pct(mean: float, std: float) -> str:
    return f"{mean * 100:.2f}% +/- {std * 100:.2f}%"


def main():
    # Feature index mapping:
    # 0 age, 1 sex, 2 cp, 3 chol, 4 fbs, 5 thalach, 6 exang, 7 oldpeak, 8 slope,
    # 9 ca, 10 thal, 11 restecg_0, 12 restecg_1

    # Scheme A (your “original-ish” mask from prior configs; closest to your screenshot behavior)
    # G0: all
    # G1: drop chol (idx 3)
    # G2: drop fbs, exang, oldpeak (idx 4,6,7)
    # G3: drop chol, restecg (idx 3,11,12)
    SCHEME_A_ORIGINAL = [
        None,
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 5, 8, 9, 10, 11, 12],
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10],
    ]

    # Scheme B (stronger bundles: blood panel, exercise bundle, resting ECG bundle)
    SCHEME_B_BUNDLED = [
        None,
        [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12],  # drop chol,fbs
        [0, 1, 2, 3, 4, 9, 10, 11, 12],        # drop exercise outputs (thalach+exang+oldpeak+slope)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],    # drop restecg
    ]

    # Scheme C (smart test-group correlation, but not overly aggressive)
    # - Hungary: drop chol only
    # - Switzerland: drop exercise ECG outputs exang+oldpeak+slope (keep thalach)
    # - VA: drop resting ECG
    SCHEME_C_SMART_PARTIAL = [
        None,
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],          # drop chol (idx 3)
        [0, 1, 2, 3, 4, 5, 9, 10, 11, 12],                # drop exang,oldpeak,slope (idx 6,7,8)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],             # drop restecg (idx 11,12)
    ]

    schemes = [
        ("A_originalish", SCHEME_A_ORIGINAL),
        ("B_bundled", SCHEME_B_BUNDLED),
        ("C_smart_partial", SCHEME_C_SMART_PARTIAL),
    ]

    base = {
        "use_fedheart": True,
        "num_classes": 2,
        "stratified_batching": True,
        "train_frac": 0.66,
        "group_max_train_samples": None,  # no cap
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
        # `feature_mask` injected per scheme
        "groups": [
            {"name": "cleveland", "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": 64, "dropout": 0.1},
            {"name": "hungarian", "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": 64, "dropout": 0.1},
            {"name": "switzerland", "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": 64, "dropout": 0.1},
            {"name": "va_long_beach", "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": 64, "dropout": 0.1},
        ],
    }

    variants = [
        ("common_erm", True, False),
        ("common_gdro", True, True),
        ("hetero_erm", False, False),
        ("hetero_gdro", False, True),
    ]

    results = {}

    for scheme_name, feature_mask in schemes:
        print(f"\n\n================ Scheme: {scheme_name} ================")
        results[scheme_name] = {}

        for variant_name, common_encoder, groupdro_enabled in variants:
            print(f"\n=== Variant: {variant_name} ===")
            seeds_results = []

            for seed in SEEDS:
                run_name = f"mask_{scheme_name}_{variant_name}_s{seed}"
                run_dir = f"runs/{run_name}"
                cfg = dict(base)
                cfg.update(
                    {
                        "seed": seed,
                        "run_name": run_name,
                        "run_dir": run_dir,
                        "feature_mask": feature_mask,
                        "common_encoder": common_encoder,
                        "groupdro_enabled": groupdro_enabled,
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
                best = run_one(cfg)
                if best is not None:
                    seeds_results.append(best)

            summary = summarize(seeds_results, num_groups=4) if seeds_results else None
            results[scheme_name][variant_name] = summary

    # Print a compact comparison
    print("\n\n================ COMPACT SUMMARY (WG acc) ================\n")
    for scheme_name, _ in schemes:
        print(f"Scheme: {scheme_name}")
        for variant_name, _, _ in variants:
            s = results[scheme_name].get(variant_name)
            if not s:
                continue
            print(f"  {variant_name:>12}: WG acc {fmt_pct(s['wg_acc_mean'], s['wg_acc_std'])}")
        print()

    out_path = "runs/feature_mask_schemes_compare_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

