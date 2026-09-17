#!/usr/bin/env python3
"""Run ablation studies + optimization experiments for NHANES and Fed-Heart."""
import json, os, sys, time
from pathlib import Path
import yaml, numpy as np
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

SEEDS = [42, 1337, 7, 13, 27]

# ── Fed-Heart Ablations ─────────────────────────────────────────────────
FEDHEART_ABLATIONS = {
    # Already have from prior runs (reference):
    # "fh_shared_erm":   fedheart_exp_shared_hetagg_erm  → 65.14%
    # "fh_shared_gdro":  fedheart_exp_shared_hetagg_gdro → 71.52%
    # "fh_pergroup_erm": fedheart_exp_paper_hetagg_erm   → 74.05%
    # "fh_pergroup_gdro": fedheart_exp_paper_hetagg_gdro → 74.25%
    "fh_pergroup_gdro_noanchor": "experiments/fedheart_ablation_pergroup_noanchor.yaml",
    "fh_pergroup_anchor_nogdro": "experiments/fedheart_ablation_anchor_nogdro.yaml",
    "fh_shared_gdro_noanchor":   "experiments/fedheart_ablation_shared_noanchor.yaml",
}

# ── NHANES Ablations (expanded mode) ────────────────────────────────────
NHANES_ABLATIONS = {
    "nh_exp_pergroup_gdro_noanchor": "experiments/nhanes_expanded_ablation_noanchor.yaml",
    "nh_exp_pergroup_anchor_nogdro": "experiments/nhanes_expanded_ablation_nogdro.yaml",
    "nh_exp_pergroup_noanchor_nogdro": "experiments/nhanes_expanded_ablation_noanchor_nogdro.yaml",
}

# ── NHANES Optimization ─────────────────────────────────────────────────
NHANES_OPTIMIZATION = {
    "nh_exp_opt_deep": "experiments/nhanes_expanded_opt_deep_gdro.yaml",
    "nh_exp_opt_wide": "experiments/nhanes_expanded_opt_wide_gdro.yaml",
    "nh_exp_opt_balanced": "experiments/nhanes_expanded_opt_balanced_gdro.yaml",
    "nh_dis_opt_deep": "experiments/nhanes_disjoint_opt_deep_gdro.yaml",
}


def run_single(config_path, seed, run_dir, is_fedheart=False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = seed
    cfg["run_dir"] = run_dir
    if is_fedheart:
        from dro_hetero_anchors.src.train_fedheart import train
    else:
        from dro_hetero_anchors.src.train_nhanes import train
    return train(cfg)


def collect(run_dir):
    import torch
    for name in ["best_worst_group.ckpt", "last.ckpt"]:
        p = Path(run_dir) / name
        if p.exists():
            return torch.load(p, map_location="cpu", weights_only=False).get("test_metrics")
    return None


def run_suite(experiments, seeds, prefix, is_fedheart=False):
    results = {}
    for key, cfg_path in experiments.items():
        results[key] = []
        for seed in seeds:
            run_dir = f"runs/{key}_s{seed}"
            if Path(run_dir).exists() and (Path(run_dir) / "best_worst_group.ckpt").exists():
                m = collect(run_dir)
                if m:
                    results[key].append(m)
                    print(f"[SKIP] {key} s{seed}: wg={m['worst_group_acc']:.4f}")
                    continue
            t0 = time.time()
            try:
                run_single(str(PROJECT_ROOT / cfg_path), seed, run_dir, is_fedheart)
                m = collect(run_dir)
                if m:
                    results[key].append(m)
                print(f"[DONE] {key} s{seed}: {time.time()-t0:.0f}s, wg={m['worst_group_acc']:.4f}")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[ERROR] {key} s{seed}: {e}")
    return results


def aggregate(metrics_list):
    if not metrics_list: return None
    n = len(metrics_list)
    agg = {"n_seeds": n}
    for k in ["overall_acc", "balanced_acc", "worst_group_acc", "best_group_acc", "overall_auroc"]:
        vals = [m[k] for m in metrics_list if m and k in m]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    for k in ["per_group_acc", "per_group_loss"]:
        lists = [m[k] for m in metrics_list if m and k in m]
        if lists:
            arr = np.array(lists)
            agg[k] = {"mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist()}
    return agg


def print_section(title, results):
    print(f"\n{'='*100}")
    print(title)
    print(f"{'='*100}")
    print(f"{'Config':<40} {'N':>3} {'Worst-Grp':>14} {'Overall':>14} {'Balanced':>14}")
    print("-" * 85)
    for key, ml in results.items():
        agg = aggregate(ml)
        if not agg: continue
        n = agg["n_seeds"]
        def f(d):
            if not d: return "N/A"
            return f"{d['mean']*100:.2f}±{d['std']*100:.2f}" if n > 1 else f"{d['mean']*100:.2f}"
        print(f"{key:<40} {n:>3} {f(agg.get('worst_group_acc')):>14} "
              f"{f(agg.get('overall_acc')):>14} {f(agg.get('balanced_acc')):>14}")


def main():
    all_results = {}

    print("\n" + "#"*100)
    print("PHASE 1: FED-HEART DISEASE ABLATIONS")
    print("#"*100)
    fh_results = run_suite(FEDHEART_ABLATIONS, SEEDS, "fh", is_fedheart=True)
    all_results.update(fh_results)
    print_section("FED-HEART ABLATIONS", fh_results)

    print("\n" + "#"*100)
    print("PHASE 2: NHANES ABLATIONS")
    print("#"*100)
    nh_abl_results = run_suite(NHANES_ABLATIONS, SEEDS, "nh_abl")
    all_results.update(nh_abl_results)
    print_section("NHANES ABLATIONS (Expanded mode)", nh_abl_results)

    print("\n" + "#"*100)
    print("PHASE 3: NHANES OPTIMIZATION")
    print("#"*100)
    nh_opt_results = run_suite(NHANES_OPTIMIZATION, SEEDS, "nh_opt")
    all_results.update(nh_opt_results)
    print_section("NHANES OPTIMIZATION", nh_opt_results)

    # Save all
    aggregated = {k: aggregate(v) for k, v in all_results.items() if v}
    out = PROJECT_ROOT / "runs" / "ablation_and_optimization_results.json"
    with open(out, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAll results saved to {out}")

    # Print combined summary
    print("\n" + "="*100)
    print("COMBINED SUMMARY")
    print("="*100)
    print_section("All Experiments", all_results)


if __name__ == "__main__":
    main()
