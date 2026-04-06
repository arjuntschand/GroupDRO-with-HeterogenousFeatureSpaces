#!/usr/bin/env python3
"""Run ALL NHANES experiment variants: nested, expanded, disjoint × shared/pergroup × ERM/GDRO.

Usage:
    python run_nhanes_all.py --seeds 42 1337 7 13 27 51 99 137 256 412
    python run_nhanes_all.py --only disjoint --seeds 42 1337 7
    python run_nhanes_all.py --collect-only
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# All experiments organized by feature mode
ALL_EXPERIMENTS = {
    # Original nested (10/13/20)
    "nested_shared_erm":       "experiments/nhanes_shared_erm.yaml",
    "nested_shared_gdro":      "experiments/nhanes_shared_gdro.yaml",
    "nested_pergroup_erm":     "experiments/nhanes_pergroup_erm.yaml",
    "nested_pergroup_gdro":    "experiments/nhanes_pergroup_gdro.yaml",
    # Expanded nested (15/18/25)
    "expanded_shared_erm":     "experiments/nhanes_expanded_shared_erm.yaml",
    "expanded_shared_gdro":    "experiments/nhanes_expanded_shared_gdro.yaml",
    "expanded_pergroup_erm":   "experiments/nhanes_expanded_pergroup_erm.yaml",
    "expanded_pergroup_gdro":  "experiments/nhanes_expanded_pergroup_gdro.yaml",
    # Disjoint (15/15/15)
    "disjoint_shared_erm":     "experiments/nhanes_disjoint_shared_erm.yaml",
    "disjoint_shared_gdro":    "experiments/nhanes_disjoint_shared_gdro.yaml",
    "disjoint_pergroup_erm":   "experiments/nhanes_disjoint_pergroup_erm.yaml",
    "disjoint_pergroup_gdro":  "experiments/nhanes_disjoint_pergroup_gdro.yaml",
}

GROUP_NAMES = ["survey_only", "exam", "vitals_labs"]


def run_single(config_path: str, seed: int, run_dir: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = seed
    cfg["run_dir"] = run_dir
    from dro_hetero_anchors.src.train_nhanes import train
    return train(cfg)


def collect_results(run_dir: str) -> dict:
    import torch
    ckpt = Path(run_dir) / "best_worst_group.ckpt"
    if not ckpt.exists():
        ckpt = Path(run_dir) / "last.ckpt"
    if not ckpt.exists():
        return None
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    return data.get("test_metrics", None)


def aggregate(metrics_list):
    if not metrics_list:
        return None
    n = len(metrics_list)
    agg = {"n_seeds": n}

    for k in ["overall_acc", "balanced_acc", "worst_group_acc", "best_group_acc", "overall_auroc"]:
        vals = [m[k] for m in metrics_list if m and k in m]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    for k in ["per_group_acc", "per_group_loss", "per_group_sensitivity",
              "per_group_specificity", "per_group_f1", "per_group_auroc"]:
        lists = [m[k] for m in metrics_list if m and k in m]
        if lists:
            arr = np.array(lists)
            agg[k] = {"mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist()}

    return agg


def print_table(results):
    modes = ["nested", "expanded", "disjoint"]
    print("\n" + "=" * 130)
    print("NHANES CVD — COMPREHENSIVE RESULTS")
    print("=" * 130)

    for mode in modes:
        keys = [k for k in results if k.startswith(mode)]
        if not keys:
            continue

        feat_desc = {"nested": "G0⊂G1⊂G2 (10/13/20)", "expanded": "G0⊂G1⊂G2 (15/18/25)", "disjoint": "Unique per-group (15/15/15)"}
        print(f"\n{'─'*130}")
        print(f"Feature mode: {mode.upper()} — {feat_desc.get(mode, '')}")
        print(f"{'─'*130}")
        print(f"{'Config':<28} {'N':>3} {'Worst-Grp':>14} {'Overall':>14} {'Balanced':>14} {'AUROC':>12} {'G0':>10} {'G1':>10} {'G2':>10}")
        print(f"{'-'*115}")

        for enc_type in ["shared", "pergroup"]:
            for method in ["erm", "gdro"]:
                key = f"{mode}_{enc_type}_{method}"
                agg = results.get(key)
                if not agg:
                    continue
                n = agg["n_seeds"]
                def f(d):
                    if not d: return "N/A"
                    return f"{d['mean']*100:.2f}±{d['std']*100:.2f}" if n > 1 else f"{d['mean']*100:.2f}"
                def fg(d, i):
                    if not d: return "N/A"
                    return f"{d['mean'][i]*100:.1f}±{d['std'][i]*100:.1f}" if n > 1 else f"{d['mean'][i]*100:.1f}"

                label = f"{enc_type}_{method}"
                print(f"{label:<28} {n:>3} {f(agg.get('worst_group_acc')):>14} "
                      f"{f(agg.get('overall_acc')):>14} {f(agg.get('balanced_acc')):>14} "
                      f"{f(agg.get('overall_auroc')):>12} "
                      f"{fg(agg.get('per_group_acc'),0):>10} {fg(agg.get('per_group_acc'),1):>10} {fg(agg.get('per_group_acc'),2):>10}")

    # Clinical metrics table
    print(f"\n{'='*130}")
    print("CLINICAL METRICS (Sensitivity / Specificity / AUROC per group)")
    print(f"{'='*130}")
    for mode in modes:
        keys = [k for k in results if k.startswith(mode)]
        if not keys:
            continue
        print(f"\n--- {mode.upper()} ---")
        print(f"{'Config':<28} {'G0 Sens':>10} {'G0 Spec':>10} {'G0 AUC':>10} {'G1 Sens':>10} {'G1 Spec':>10} {'G1 AUC':>10} {'G2 Sens':>10} {'G2 Spec':>10} {'G2 AUC':>10}")

        for enc_type in ["shared", "pergroup"]:
            for method in ["erm", "gdro"]:
                key = f"{mode}_{enc_type}_{method}"
                agg = results.get(key)
                if not agg: continue
                def fg(d, i):
                    if not d: return "N/A"
                    return f"{d['mean'][i]*100:.1f}"

                label = f"{enc_type}_{method}"
                sens = agg.get("per_group_sensitivity", {})
                spec = agg.get("per_group_specificity", {})
                auc = agg.get("per_group_auroc", {})
                print(f"{label:<28} {fg(sens,0):>10} {fg(spec,0):>10} {fg(auc,0):>10} "
                      f"{fg(sens,1):>10} {fg(spec,1):>10} {fg(auc,1):>10} "
                      f"{fg(sens,2):>10} {fg(spec,2):>10} {fg(auc,2):>10}")

    print(f"\n{'='*130}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 7, 13, 27])
    parser.add_argument("--only", choices=["nested", "expanded", "disjoint", "all"], default="all")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    if args.only != "all":
        exp_keys = [k for k in ALL_EXPERIMENTS if k.startswith(args.only)]
    else:
        exp_keys = list(ALL_EXPERIMENTS.keys())

    all_results = {}
    for exp_key in exp_keys:
        config_path = ALL_EXPERIMENTS[exp_key]
        all_results[exp_key] = []

        for seed in args.seeds:
            run_dir = f"runs/nhanes_{exp_key}_s{seed}"

            if args.collect_only:
                m = collect_results(run_dir)
                if m:
                    all_results[exp_key].append(m)
                    print(f"[COLLECT] {exp_key} s{seed}: wg={m['worst_group_acc']:.4f}")
                continue

            if Path(run_dir).exists() and (Path(run_dir) / "best_worst_group.ckpt").exists():
                m = collect_results(run_dir)
                if m:
                    all_results[exp_key].append(m)
                    print(f"[SKIP] {exp_key} s{seed}: wg={m['worst_group_acc']:.4f}")
                    continue

            print(f"\n{'='*60}\nRUNNING: {exp_key} seed={seed}\n{'='*60}")
            t0 = time.time()
            try:
                run_single(str(PROJECT_ROOT / config_path), seed, run_dir)
                m = collect_results(run_dir)
                if m:
                    all_results[exp_key].append(m)
                print(f"[DONE] {exp_key} s{seed}: {time.time()-t0:.0f}s, wg={m['worst_group_acc']:.4f}")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[ERROR] {exp_key} s{seed}: {e}")

    aggregated = {k: aggregate(v) for k, v in all_results.items() if v}

    out_path = PROJECT_ROOT / "runs" / "nhanes_all_results.json"
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print_table(aggregated)


if __name__ == "__main__":
    main()
