#!/usr/bin/env python3
"""
Run the full Fed-Heart heterogeneous feature experiment suite.

Experiment design (2x2x3):
  - Encoder: shared vs per-group
  - Optimization: ERM vs GroupDRO
  - Heterogeneity: none, moderate, aggressive

Usage:
    python run_experiments.py                    # Run all experiments, seed 42
    python run_experiments.py --seeds 42 1337    # Run with 2 seeds
    python run_experiments.py --only hetero      # Only heterogeneous experiments
    python run_experiments.py --collect-only      # Just collect existing results
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import yaml
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


EXPERIMENTS = {
    # Homogeneous features (control)
    "shared_erm":    "experiments/fedheart_exp_shared_erm.yaml",
    "shared_gdro":   "experiments/fedheart_exp_shared_gdro.yaml",
    "pergroup_erm":  "experiments/fedheart_exp_pergroup_erm.yaml",
    "pergroup_gdro": "experiments/fedheart_exp_pergroup_gdro.yaml",
    # Moderate heterogeneity
    "hetmod_erm":    "experiments/fedheart_exp_hetmod_erm.yaml",
    "hetmod_gdro":   "experiments/fedheart_exp_hetmod_gdro.yaml",
    # Aggressive heterogeneity
    "hetagg_erm":    "experiments/fedheart_exp_hetagg_erm.yaml",
    "hetagg_gdro":   "experiments/fedheart_exp_hetagg_gdro.yaml",
    # Severe heterogeneity
    "hetsev_erm":    "experiments/fedheart_exp_hetsev_erm.yaml",
    "hetsev_gdro":   "experiments/fedheart_exp_hetsev_gdro.yaml",
    # Aggressive heterogeneity + imbalance
    "hetagg_imb_erm":  "experiments/fedheart_exp_hetagg_imb_erm.yaml",
    "hetagg_imb_gdro": "experiments/fedheart_exp_hetagg_imb_gdro.yaml",
}

GROUP_NAMES = ["Cleveland", "Hungarian", "Switzerland", "VA Long Beach"]


def run_single(config_path: str, seed: int, run_dir_override: str) -> dict:
    """Run a single experiment and return final metrics."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["seed"] = seed
    cfg["run_dir"] = run_dir_override

    # Import here to avoid top-level torch import
    from dro_hetero_anchors.src.train_fedheart import train
    result = train(cfg)
    return result


def collect_results(run_dir: str) -> dict:
    """Read the best checkpoint from a run directory and extract metrics."""
    ckpt_path = Path(run_dir) / "best_worst_group.ckpt"
    if not ckpt_path.exists():
        # Try last checkpoint
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
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 100)

    # Define the table structure
    conditions = [
        ("Homogeneous (all 13)", "shared", "Shared"),
        ("Homogeneous (all 13)", "pergroup", "Per-group"),
        ("Moderate hetero", "hetmod", "Per-group"),
        ("Aggressive hetero", "hetagg", "Per-group"),
        ("Severe hetero", "hetsev", "Per-group"),
        ("Aggressive+imbalance", "hetagg_imb", "Per-group"),
    ]

    # Header
    print(f"\n{'Condition':<25} {'Encoder':<12} {'Method':<8} "
          f"{'Overall':<14} {'Worst-Grp':<14} {'Balanced':<14} "
          f"{'G0':<12} {'G1':<12} {'G2':<12} {'G3':<12}")
    print("-" * 135)

    for hetero_label, key_prefix, enc_label in conditions:
        for method_suffix, method_label in [("erm", "ERM"), ("gdro", "GDRO")]:
            exp_key = f"{key_prefix}_{method_suffix}"
            agg = results.get(exp_key)
            if agg is None:
                print(f"{hetero_label:<25} {enc_label:<12} {method_label:<8} {'(not run)'}")
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

            print(f"{hetero_label:<25} {enc_label:<12} {method_label:<8} "
                  f"{fmt(oa):<14} {fmt(wa):<14} {fmt(ba):<14} "
                  f"{fmt_g(pga, 0):<12} {fmt_g(pga, 1):<12} {fmt_g(pga, 2):<12} {fmt_g(pga, 3):<12}")

    print("-" * 135)

    # Per-group loss table
    print(f"\n{'Condition':<25} {'Encoder':<12} {'Method':<8} "
          f"{'G0 Loss':<12} {'G1 Loss':<12} {'G2 Loss':<12} {'G3 Loss':<12}")
    print("-" * 85)

    for hetero_label, key_prefix, enc_label in conditions:
        for method_suffix, method_label in [("erm", "ERM"), ("gdro", "GDRO")]:
            exp_key = f"{key_prefix}_{method_suffix}"
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

            print(f"{hetero_label:<25} {enc_label:<12} {method_label:<8} "
                  f"{fmt_l(pgl, 0):<12} {fmt_l(pgl, 1):<12} {fmt_l(pgl, 2):<12} {fmt_l(pgl, 3):<12}")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Run Fed-Heart heterogeneous feature experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Seeds to run (default: [42])")
    parser.add_argument("--only", choices=["homo", "hetero", "all"], default="all",
                        help="Which experiments to run")
    parser.add_argument("--collect-only", action="store_true",
                        help="Only collect existing results, don't run new experiments")
    parser.add_argument("--extra-seeds", nargs="+", type=int, default=None,
                        help="Run extra seeds only for experiments where GDRO shows > threshold improvement")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Worst-group improvement threshold (in %%) to trigger extra seeds")
    args = parser.parse_args()

    # Filter experiments
    if args.only == "homo":
        exp_keys = [k for k in EXPERIMENTS if not k.startswith("het")]
    elif args.only == "hetero":
        exp_keys = [k for k in EXPERIMENTS if k.startswith("het")]
    else:
        exp_keys = list(EXPERIMENTS.keys())

    # Run experiments
    all_results = {}  # exp_key -> list of metrics dicts

    for exp_key in exp_keys:
        config_path = EXPERIMENTS[exp_key]
        all_results[exp_key] = []

        for seed in args.seeds:
            run_dir = f"runs/exp_{exp_key}_s{seed}"

            if args.collect_only:
                metrics = collect_results(run_dir)
                if metrics:
                    all_results[exp_key].append(metrics)
                    print(f"[COLLECT] {exp_key} seed={seed}: worst_group={metrics['worst_group_acc']:.4f}")
                else:
                    print(f"[COLLECT] {exp_key} seed={seed}: not found")
            else:
                # Check if already run
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
                    print(f"[ERROR] {exp_key} seed={seed}: {e}")

    # Aggregate results
    aggregated = {}
    for exp_key, metrics_list in all_results.items():
        if metrics_list:
            aggregated[exp_key] = aggregate_seeds(metrics_list)

    # Check if extra seeds needed
    if args.extra_seeds and not args.collect_only:
        promising = []
        for base in ["hetmod", "hetagg"]:
            erm_key = f"{base}_erm"
            gdro_key = f"{base}_gdro"
            if erm_key in aggregated and gdro_key in aggregated:
                erm_wg = aggregated[erm_key]["worst_group_acc"]["mean"]
                gdro_wg = aggregated[gdro_key]["worst_group_acc"]["mean"]
                improvement = (gdro_wg - erm_wg) * 100
                if improvement > args.threshold:
                    promising.extend([erm_key, gdro_key])
                    print(f"\n[PROMISING] {base}: GroupDRO improves worst-group by {improvement:.2f}% -> running extra seeds")

        if promising:
            for exp_key in promising:
                config_path = EXPERIMENTS[exp_key]
                for seed in args.extra_seeds:
                    run_dir = f"runs/exp_{exp_key}_s{seed}"
                    if Path(run_dir).exists() and (Path(run_dir) / "best_worst_group.ckpt").exists():
                        metrics = collect_results(run_dir)
                        if metrics:
                            all_results[exp_key].append(metrics)
                            continue

                    print(f"\n[EXTRA] Running {exp_key} seed={seed}")
                    try:
                        run_single(str(PROJECT_ROOT / config_path), seed, run_dir)
                        metrics = collect_results(run_dir)
                        if metrics:
                            all_results[exp_key].append(metrics)
                    except Exception as e:
                        print(f"[ERROR] {exp_key} seed={seed}: {e}")

            # Re-aggregate
            for exp_key, metrics_list in all_results.items():
                if metrics_list:
                    aggregated[exp_key] = aggregate_seeds(metrics_list)

    # Save results
    results_path = PROJECT_ROOT / "runs" / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print formatted table
    print_results_table(aggregated)


if __name__ == "__main__":
    main()
