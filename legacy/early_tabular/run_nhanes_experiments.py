#!/usr/bin/env python3
"""
Run the full NHANES CVD experiment suite.

Experiment design (2x2):
  - Encoder: shared vs per-group
  - Optimization: ERM vs GroupDRO
  + Additional tuned variants

Usage:
    python run_nhanes_experiments.py                       # All experiments, seed 42
    python run_nhanes_experiments.py --seeds 42 1337 7     # Multiple seeds
    python run_nhanes_experiments.py --only gdro           # Only GroupDRO experiments
    python run_nhanes_experiments.py --collect-only         # Just collect existing results
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

EXPERIMENTS = {
    # Core 2x2 design
    "shared_erm":       "experiments/nhanes_shared_erm.yaml",
    "shared_gdro":      "experiments/nhanes_shared_gdro.yaml",
    "pergroup_erm":     "experiments/nhanes_pergroup_erm.yaml",
    "pergroup_gdro":    "experiments/nhanes_pergroup_gdro.yaml",
    # Tuned variants
    "pergroup_gdro_v2": "experiments/nhanes_pergroup_gdro_v2.yaml",
    "pergroup_gdro_v3": "experiments/nhanes_pergroup_gdro_v3.yaml",
    "pergroup_gdro_v4": "experiments/nhanes_pergroup_gdro_v4.yaml",
    "pergroup_gdro_v5": "experiments/nhanes_pergroup_gdro_v5.yaml",
    "pergroup_gdro_v6": "experiments/nhanes_pergroup_gdro_v6.yaml",
}

GROUP_NAMES = ["survey_only", "exam", "vitals_labs"]


def run_single(config_path: str, seed: int, run_dir_override: str) -> dict:
    """Run a single experiment and return final metrics."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["seed"] = seed
    cfg["run_dir"] = run_dir_override

    from dro_hetero_anchors.src.train_nhanes import train
    result = train(cfg)
    return result


def collect_results(run_dir: str) -> dict:
    """Read the best checkpoint from a run directory and extract metrics."""
    ckpt_path = Path(run_dir) / "best_worst_group.ckpt"
    if not ckpt_path.exists():
        ckpt_path = Path(run_dir) / "last.ckpt"
    if not ckpt_path.exists():
        return None

    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("test_metrics", None)


def aggregate_seeds(all_metrics: list) -> dict:
    """Aggregate metrics across seeds: compute mean and std."""
    if not all_metrics:
        return None

    keys_scalar = ["overall_acc", "balanced_acc", "worst_group_acc", "best_group_acc"]
    keys_list = ["per_group_acc", "per_group_loss"]

    agg = {}
    for k in keys_scalar:
        vals = [m[k] for m in all_metrics if m and k in m]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    for k in keys_list:
        lists = [m[k] for m in all_metrics if m and k in m]
        if lists:
            arr = np.array(lists)
            agg[k] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
            }

    agg["n_seeds"] = len(all_metrics)
    return agg


def print_results_table(results: dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 110)
    print("NHANES CVD EXPERIMENT RESULTS SUMMARY")
    print("=" * 110)

    conditions = [
        ("Shared encoder", "shared", "Shared"),
        ("Per-group encoder", "pergroup", "Per-group"),
        ("Per-group (deep)", "pergroup_gdro_v2", "Per-group"),
        ("Per-group (focal)", "pergroup_gdro_v3", "Per-group"),
    ]

    # Header
    print(f"\n{'Condition':<22} {'Encoder':<12} {'Method':<8} "
          f"{'Overall':<14} {'Worst-Grp':<14} {'Balanced':<14} "
          f"{'G0(surv)':<12} {'G1(exam)':<12} {'G2(vital)':<12}")
    print("-" * 120)

    for hetero_label, key_prefix, enc_label in conditions:
        methods = []
        if key_prefix.startswith("pergroup_gdro_v"):
            methods = [(key_prefix, "GDRO")]
        else:
            methods = [(f"{key_prefix}_erm", "ERM"), (f"{key_prefix}_gdro", "GDRO")]

        for exp_key, method_label in methods:
            agg = results.get(exp_key)
            if agg is None:
                print(f"{hetero_label:<22} {enc_label:<12} {method_label:<8} {'(not run)'}")
                continue

            n = agg["n_seeds"]
            oa = agg.get("overall_acc", {})
            wa = agg.get("worst_group_acc", {})
            ba = agg.get("balanced_acc", {})
            pga = agg.get("per_group_acc", {})

            def fmt(d):
                if not d:
                    return "N/A"
                m, s = d.get("mean", 0), d.get("std", 0)
                if n == 1:
                    return f"{m*100:.2f}%"
                return f"{m*100:.2f}+/-{s*100:.2f}"

            def fmt_g(pga_dict, idx):
                if not pga_dict:
                    return "N/A"
                m = pga_dict["mean"][idx]
                s = pga_dict["std"][idx]
                if n == 1:
                    return f"{m*100:.1f}%"
                return f"{m*100:.1f}+/-{s*100:.1f}"

            print(f"{hetero_label:<22} {enc_label:<12} {method_label:<8} "
                  f"{fmt(oa):<14} {fmt(wa):<14} {fmt(ba):<14} "
                  f"{fmt_g(pga, 0):<12} {fmt_g(pga, 1):<12} {fmt_g(pga, 2):<12}")

    print("-" * 120)

    # Per-group loss table
    print(f"\n{'Condition':<22} {'Encoder':<12} {'Method':<8} "
          f"{'G0 Loss':<12} {'G1 Loss':<12} {'G2 Loss':<12}")
    print("-" * 75)

    for hetero_label, key_prefix, enc_label in conditions:
        methods = []
        if key_prefix.startswith("pergroup_gdro_v"):
            methods = [(key_prefix, "GDRO")]
        else:
            methods = [(f"{key_prefix}_erm", "ERM"), (f"{key_prefix}_gdro", "GDRO")]

        for exp_key, method_label in methods:
            agg = results.get(exp_key)
            if agg is None:
                continue

            pgl = agg.get("per_group_loss", {})
            n = agg["n_seeds"]

            def fmt_l(pgl_dict, idx):
                if not pgl_dict:
                    return "N/A"
                m = pgl_dict["mean"][idx]
                s = pgl_dict["std"][idx]
                if n == 1:
                    return f"{m:.4f}"
                return f"{m:.3f}+/-{s:.3f}"

            print(f"{hetero_label:<22} {enc_label:<12} {method_label:<8} "
                  f"{fmt_l(pgl, 0):<12} {fmt_l(pgl, 1):<12} {fmt_l(pgl, 2):<12}")

    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(description="Run NHANES CVD experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Seeds to run (default: [42])")
    parser.add_argument("--only", choices=["erm", "gdro", "shared", "pergroup", "core", "all"], default="all",
                        help="Which experiments to run")
    parser.add_argument("--collect-only", action="store_true",
                        help="Only collect existing results, don't run new experiments")
    args = parser.parse_args()

    # Filter experiments
    if args.only == "erm":
        exp_keys = [k for k in EXPERIMENTS if "erm" in k]
    elif args.only == "gdro":
        exp_keys = [k for k in EXPERIMENTS if "gdro" in k]
    elif args.only == "shared":
        exp_keys = [k for k in EXPERIMENTS if "shared" in k]
    elif args.only == "pergroup":
        exp_keys = [k for k in EXPERIMENTS if "pergroup" in k]
    elif args.only == "core":
        exp_keys = ["shared_erm", "shared_gdro", "pergroup_erm", "pergroup_gdro"]
    else:
        exp_keys = list(EXPERIMENTS.keys())

    all_results = {}

    for exp_key in exp_keys:
        config_path = EXPERIMENTS[exp_key]
        all_results[exp_key] = []

        for seed in args.seeds:
            run_dir = f"runs/nhanes_{exp_key}_s{seed}"

            if args.collect_only:
                metrics = collect_results(run_dir)
                if metrics:
                    all_results[exp_key].append(metrics)
                    print(f"[COLLECT] {exp_key} seed={seed}: worst_group={metrics['worst_group_acc']:.4f}")
                else:
                    print(f"[COLLECT] {exp_key} seed={seed}: not found")
            else:
                if Path(run_dir).exists() and (Path(run_dir) / "best_worst_group.ckpt").exists():
                    metrics = collect_results(run_dir)
                    if metrics:
                        all_results[exp_key].append(metrics)
                        print(f"[SKIP] {exp_key} seed={seed}: already done, worst_group={metrics['worst_group_acc']:.4f}")
                        continue

                print(f"\n{'='*60}")
                print(f"RUNNING: {exp_key} seed={seed}")
                print(f"{'='*60}")
                t0 = time.time()
                try:
                    result = run_single(str(PROJECT_ROOT / config_path), seed, run_dir)
                    metrics = collect_results(run_dir)
                    if metrics:
                        all_results[exp_key].append(metrics)
                    elapsed = time.time() - t0
                    print(f"[DONE] {exp_key} seed={seed}: {elapsed:.1f}s, worst_group={metrics['worst_group_acc']:.4f}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[ERROR] {exp_key} seed={seed}: {e}")

    # Aggregate results
    aggregated = {}
    for exp_key, metrics_list in all_results.items():
        if metrics_list:
            aggregated[exp_key] = aggregate_seeds(metrics_list)

    # Save results
    results_path = PROJECT_ROOT / "runs" / "nhanes_experiment_results.json"
    os.makedirs(results_path.parent, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print_results_table(aggregated)


if __name__ == "__main__":
    main()
