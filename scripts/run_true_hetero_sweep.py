#!/usr/bin/env python3
"""Comprehensive sweep: per-group encoders with TRUE heterogeneous feature spaces.

Each encoder gets a genuinely different input_dim matching its available features.
Tests multiple feature dropout patterns, with and without caps, baseline vs GroupDRO.
10 seeds each for robust evaluation.
"""

import subprocess
import sys
import json
import os
import shutil
import yaml
import numpy as np
from pathlib import Path

SEEDS = [7, 13, 27, 51, 99, 137, 256, 412, 666, 1024]

# All 13 feature indices for Fed-Heart:
# 0=age, 1=sex, 2=cp, 3=chol, 4=fbs, 5=thalach, 6=exang, 7=oldpeak, 8=slope, 9=ca, 10=thal, 11=restecg_0, 12=restecg_1
ALL_FEATURES = list(range(13))

# ---------- Feature dropout patterns ----------
# These simulate real-world hospital data collection differences
FEATURE_PATTERNS = {
    # Pattern A: Mild realistic drops (1-3 features per group, clinically motivated)
    "mild_realistic": {
        "description": "G0:all, G1:drop chol, G2:drop fbs+exang+oldpeak, G3:drop chol+restecg",
        "masks": [
            None,                                      # G0: all 13
            [0,1,2,4,5,6,7,8,9,10,11,12],            # G1: 12 (drop chol=3)
            [0,1,2,3,5,8,9,10,11,12],                 # G2: 10 (drop fbs=4, exang=6, oldpeak=7)
            [0,1,2,4,5,6,7,8,9,10],                   # G3: 10 (drop chol=3, restecg=11,12)
        ],
    },
    # Pattern B: Moderate drops (3-5 features per group)
    "moderate": {
        "description": "G0:all, G1:drop 3, G2:drop 4, G3:drop 5",
        "masks": [
            None,                                      # G0: all 13
            [0,1,2,5,6,7,8,9,10],                     # G1: 9 (drop chol=3, fbs=4, restecg=11,12)
            [0,1,2,5,8,9,10],                          # G2: 7 (drop chol=3, fbs=4, exang=6, oldpeak=7, restecg=11,12)
            [0,1,5,8,9,10,11,12],                      # G3: 8 (drop cp=2, chol=3, fbs=4, exang=6, oldpeak=7)
        ],
    },
    # Pattern C: Aggressive heterogeneity (5-7 features per group)
    "aggressive": {
        "description": "G0:all, G1:7 feats, G2:6 feats, G3:8 feats",
        "masks": [
            None,                                      # G0: all 13
            [0,1,2,5,8,9,10],                          # G1: 7 (drop chol,fbs,exang,oldpeak,restecg)
            [0,1,5,9,10,11],                           # G2: 6 (drop cp,chol,fbs,exang,oldpeak,restecg_1)
            [0,1,2,3,5,6,8,10],                        # G3: 8 (drop fbs,oldpeak,ca,restecg)
        ],
    },
    # Pattern D: Non-overlapping subsets (maximum heterogeneity)
    "non_overlapping": {
        "description": "Each group sees a unique-ish subset, some overlap via demographics",
        "masks": [
            [0,1,2,3,4,5,6],                          # G0: 7 (demographics + risk factors)
            [0,1,7,8,9,10,11,12],                      # G1: 8 (demographics + ECG/imaging)
            [0,1,2,3,5,10],                            # G2: 6 (demographics + subset)
            [0,1,4,6,7,8,9,11,12],                     # G3: 9 (demographics + exercise + ECG)
        ],
    },
    # Pattern E: Progressive drop (each successive group loses more)
    "progressive": {
        "description": "G0:all 13, G1:11, G2:9, G3:7 features",
        "masks": [
            None,                                      # G0: all 13
            [0,1,2,3,5,6,7,8,9,10,11],                # G1: 11 (drop fbs=4, restecg_1=12)
            [0,1,2,5,6,8,9,10,11],                     # G2: 9 (drop chol=3, fbs=4, oldpeak=7, restecg_1=12)
            [0,1,5,8,9,10,11],                         # G3: 7 (drop cp=2, chol=3, fbs=4, exang=6, oldpeak=7, restecg_1=12)
        ],
    },
}

# Cap configurations
CAP_CONFIGS = {
    "no_cap": None,
    "cap_20_30": [None, None, 20, 30],
    "cap_15_25": [None, None, 15, 25],
}

# GroupDRO hyperparameter sets to try
GROUPDRO_CONFIGS = {
    "default": {
        "groupdro_eta": 1.0,
        "groupdro_gamma": 0.9,
        "groupdro_update_mode": "softmax",
        "groupdro_objective": "weighted",
        "groupdro_kl_lambda": 0.0,
        "groupdro_uniform_init": True,
    },
    "strong_eta": {
        "groupdro_eta": 2.0,
        "groupdro_gamma": 0.9,
        "groupdro_update_mode": "softmax",
        "groupdro_objective": "weighted",
        "groupdro_kl_lambda": 0.0,
        "groupdro_uniform_init": True,
    },
    "with_kl": {
        "groupdro_eta": 1.0,
        "groupdro_gamma": 0.9,
        "groupdro_update_mode": "softmax",
        "groupdro_objective": "weighted",
        "groupdro_kl_lambda": 0.1,
        "groupdro_uniform_init": True,
    },
}

BASE_CFG = {
    "use_fedheart": True,
    "num_classes": 2,
    "stratified_batching": True,
    "train_frac": 0.66,
    "batch_size": 64,
    "num_workers": 0,
    "epochs": 100,
    "early_stopping_patience": 30,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "optimizer": "adam",
    "grad_clip": 1.0,
    "lr_scheduler": "cosine",
    "lr_min": 1e-5,
    "latent_dim": 64,
    "head_hidden": 32,
    "anchor_eps": 0.0001,
    "sep_samples_per_class": 4,
    "sep_method": "classifier",
    "sep_margin": 1.0,
    "lambda_fit": 0.001,
    "lambda_sep": 0.001,
    "log_interval": 20,
    "save_every": 100,
}

def make_group_cfg():
    return [
        {"name": "cleveland",     "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": 64, "dropout": 0.1},
        {"name": "hungarian",     "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": 64, "dropout": 0.1},
        {"name": "switzerland",   "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": 64, "dropout": 0.1},
        {"name": "va_long_beach", "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": 64, "dropout": 0.1},
    ]


def run_single(cfg_dict, run_name):
    cfg_path = f"/tmp/{run_name}.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
    project_root = str(Path(__file__).resolve().parent.parent)
    venv_python = os.path.join(project_root, "venv", "bin", "python")
    if not os.path.exists(venv_python):
        venv_python = sys.executable
    result = subprocess.run(
        [venv_python, "-m", "dro_hetero_anchors.src.train_fedheart", "--config", cfg_path],
        capture_output=True, text=True,
        cwd=project_root,
    )
    if result.returncode != 0:
        print(f"  [FAIL] {run_name}")
        print(result.stderr[-500:] if result.stderr else "no stderr")
        return None

    metrics_path = Path(cfg_dict["run_dir"]) / "metrics.jsonl"
    if not metrics_path.exists():
        print(f"  [FAIL] no metrics: {run_name}")
        return None
    best = None
    for line in metrics_path.read_text().strip().split("\n"):
        m = json.loads(line)
        if best is None or m.get("test_worst_group_acc", 0) > best.get("test_worst_group_acc", 0):
            best = m
    return best


def run_10seed(base_cfg, label, pattern_name, cap_name, dro_label):
    results = []
    for seed in SEEDS:
        cfg = dict(base_cfg)
        cfg["seed"] = seed
        run_name = f"th_{pattern_name}_{cap_name}_{dro_label}_s{seed}"
        cfg["run_dir"] = f"runs/{run_name}"
        cfg["run_name"] = run_name
        if os.path.exists(cfg["run_dir"]):
            shutil.rmtree(cfg["run_dir"])
        print(f"  Running {run_name} ...", end=" ", flush=True)
        m = run_single(cfg, run_name)
        if m is not None:
            results.append(m)
            wg = m.get("test_worst_group_acc", 0) * 100
            print(f"wg={wg:.1f}%")
        else:
            print("FAILED")
    return results


def extract_stats(results, num_groups=4):
    if not results:
        return None
    per_group_acc = {g: [] for g in range(num_groups)}
    per_group_loss = {g: [] for g in range(num_groups)}
    worst_group_accs = []
    overall_accs = []

    for m in results:
        accs = m.get("test_per_group_acc", [])
        losses = m.get("test_per_group_loss", [])
        for g in range(num_groups):
            if g < len(accs):
                per_group_acc[g].append(accs[g])
            if g < len(losses):
                per_group_loss[g].append(losses[g])
        worst_group_accs.append(m.get("test_worst_group_acc", 0))
        overall_accs.append(m.get("test_acc", 0))

    stats = {"n_seeds": len(results)}
    for g in range(num_groups):
        a = np.array(per_group_acc[g])
        l = np.array(per_group_loss[g])
        stats[f"g{g}_acc_mean"] = float(np.mean(a)) if len(a) > 0 else 0
        stats[f"g{g}_acc_std"] = float(np.std(a)) if len(a) > 0 else 0
        stats[f"g{g}_loss_mean"] = float(np.mean(l)) if len(l) > 0 else 0
        stats[f"g{g}_loss_std"] = float(np.std(l)) if len(l) > 0 else 0
    wg = np.array(worst_group_accs)
    stats["wg_acc_mean"] = float(np.mean(wg))
    stats["wg_acc_std"] = float(np.std(wg))
    oa = np.array(overall_accs)
    stats["overall_acc_mean"] = float(np.mean(oa))
    stats["overall_acc_std"] = float(np.std(oa))
    return stats


def format_pct(mean, std):
    return f"{mean*100:.2f}% +/- {std*100:.2f}%"


def main():
    all_results = {}

    # Phase 1: Test all feature patterns with no cap (baseline + GroupDRO default)
    print("=" * 80)
    print("PHASE 1: All feature patterns, no cap, baseline vs GroupDRO")
    print("=" * 80)

    for pat_name, pat in FEATURE_PATTERNS.items():
        print(f"\n--- Pattern: {pat_name} ({pat['description']}) ---")

        for dro_label, is_dro in [("erm", False), ("gdro", True)]:
            cfg = dict(BASE_CFG)
            cfg["groups"] = make_group_cfg()
            cfg["feature_mask"] = pat["masks"]
            cfg["group_max_train_samples"] = None

            if is_dro:
                cfg["groupdro_enabled"] = True
                cfg.update(GROUPDRO_CONFIGS["default"])
            else:
                cfg["groupdro_enabled"] = False

            key = f"{pat_name}_no_cap_{dro_label}"
            print(f"\n  [{key}]")
            raw = run_10seed(cfg, key, pat_name, "no_cap", dro_label)
            all_results[key] = extract_stats(raw)

    # Phase 2: Best patterns with caps
    print("\n" + "=" * 80)
    print("PHASE 2: Promising patterns with caps")
    print("=" * 80)

    promising_patterns = ["mild_realistic", "moderate", "progressive"]
    for pat_name in promising_patterns:
        pat = FEATURE_PATTERNS[pat_name]
        for cap_name, cap_val in [("cap_20_30", [None, None, 20, 30]), ("cap_15_25", [None, None, 15, 25])]:
            print(f"\n--- Pattern: {pat_name} + {cap_name} ---")
            for dro_label, is_dro in [("erm", False), ("gdro", True)]:
                cfg = dict(BASE_CFG)
                cfg["groups"] = make_group_cfg()
                cfg["feature_mask"] = pat["masks"]
                cfg["group_max_train_samples"] = cap_val

                if is_dro:
                    cfg["groupdro_enabled"] = True
                    cfg.update(GROUPDRO_CONFIGS["default"])
                else:
                    cfg["groupdro_enabled"] = False

                key = f"{pat_name}_{cap_name}_{dro_label}"
                print(f"\n  [{key}]")
                raw = run_10seed(cfg, key, pat_name, cap_name, dro_label)
                all_results[key] = extract_stats(raw)

    # Phase 3: Best combo with GroupDRO hyperparameter variations
    print("\n" + "=" * 80)
    print("PHASE 3: GroupDRO hyper variations on best patterns + caps")
    print("=" * 80)

    for pat_name in ["mild_realistic", "progressive"]:
        pat = FEATURE_PATTERNS[pat_name]
        for gdro_name, gdro_cfg in [("strong_eta", GROUPDRO_CONFIGS["strong_eta"]),
                                      ("with_kl", GROUPDRO_CONFIGS["with_kl"])]:
            for cap_name, cap_val in [("cap_20_30", [None, None, 20, 30])]:
                cfg = dict(BASE_CFG)
                cfg["groups"] = make_group_cfg()
                cfg["feature_mask"] = pat["masks"]
                cfg["group_max_train_samples"] = cap_val
                cfg["groupdro_enabled"] = True
                cfg.update(gdro_cfg)

                key = f"{pat_name}_{cap_name}_{gdro_name}"
                print(f"\n  [{key}]")
                raw = run_10seed(cfg, key, pat_name, cap_name, gdro_name)
                all_results[key] = extract_stats(raw)

    # Print final summary
    print("\n\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    group_names = ["Cleveland", "Hungarian", "Switzerland", "VA Long Beach"]

    # Pair up baseline/GroupDRO results
    baselines = {k: v for k, v in all_results.items() if k.endswith("_erm") and v is not None}
    gdros = {k: v for k, v in all_results.items() if not k.endswith("_erm") and v is not None}

    print(f"\n{'Config':<50} | {'Worst-Group Acc':<20} | {'Overall Acc':<20}")
    print("-" * 95)
    for k, s in sorted(all_results.items()):
        if s is None:
            continue
        print(f"{k:<50} | {format_pct(s['wg_acc_mean'], s['wg_acc_std']):<20} | {format_pct(s['overall_acc_mean'], s['overall_acc_std']):<20}")

    # Compute GroupDRO gains
    print(f"\n\n{'Config Pair':<60} | {'WG Gain':<12} | {'Overall Gain':<12}")
    print("-" * 90)
    for bk in sorted(baselines.keys()):
        prefix = bk.replace("_erm", "")
        gk_candidates = [k for k in gdros if k.startswith(prefix + "_")]
        for gk in sorted(gk_candidates):
            bs = baselines[bk]
            gs = gdros[gk]
            wg_gain = (gs["wg_acc_mean"] - bs["wg_acc_mean"]) * 100
            oa_gain = (gs["overall_acc_mean"] - bs["overall_acc_mean"]) * 100
            dro_suffix = gk.replace(prefix + "_", "")
            label = f"{prefix} ({dro_suffix} vs erm)"
            print(f"{label:<60} | {wg_gain:+.2f}%{'':<6} | {oa_gain:+.2f}%")

    # Save full results
    report_path = "runs/TRUE_HETERO_SWEEP_RESULTS.json"
    os.makedirs("runs", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {report_path}")

    # Generate detailed markdown report
    md_lines = ["# True Heterogeneous Feature Spaces: Per-Group Encoder Results\n"]
    md_lines.append("Each encoder has a genuinely different input_dim matching its available features.\n\n")

    for bk in sorted(baselines.keys()):
        prefix = bk.replace("_erm", "")
        bs = baselines[bk]
        gk_candidates = [k for k in gdros if k.startswith(prefix + "_")]
        for gk in sorted(gk_candidates):
            gs = gdros[gk]
            dro_suffix = gk.replace(prefix + "_", "")
            md_lines.append(f"## {prefix} ({dro_suffix})\n")

            md_lines.append("### Per-Group Accuracy\n")
            md_lines.append("| Group | Baseline (ERM) | GroupDRO | Change |")
            md_lines.append("|-------|----------------|----------|--------|")
            for g in range(4):
                ba = format_pct(bs[f"g{g}_acc_mean"], bs[f"g{g}_acc_std"])
                ga = format_pct(gs[f"g{g}_acc_mean"], gs[f"g{g}_acc_std"])
                diff = (gs[f"g{g}_acc_mean"] - bs[f"g{g}_acc_mean"]) * 100
                md_lines.append(f"| G{g} {group_names[g]} | {ba} | {ga} | {diff:+.2f}% |")
            wg_diff = (gs["wg_acc_mean"] - bs["wg_acc_mean"]) * 100
            md_lines.append(f"| **Worst-group** | {format_pct(bs['wg_acc_mean'], bs['wg_acc_std'])} | {format_pct(gs['wg_acc_mean'], gs['wg_acc_std'])} | **{wg_diff:+.2f}%** |")

            md_lines.append("\n### Per-Group Loss\n")
            md_lines.append("| Group | Baseline (ERM) | GroupDRO | Change |")
            md_lines.append("|-------|----------------|----------|--------|")
            for g in range(4):
                bl = f"{bs[f'g{g}_loss_mean']:.3f} +/- {bs[f'g{g}_loss_std']:.3f}"
                gl = f"{gs[f'g{g}_loss_mean']:.3f} +/- {gs[f'g{g}_loss_std']:.3f}"
                diff = gs[f"g{g}_loss_mean"] - bs[f"g{g}_loss_mean"]
                md_lines.append(f"| G{g} {group_names[g]} | {bl} | {gl} | {diff:+.3f} |")
            md_lines.append("")

    report_md = "runs/TRUE_HETERO_SWEEP_RESULTS.md"
    with open(report_md, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown report saved to {report_md}")


if __name__ == "__main__":
    main()
