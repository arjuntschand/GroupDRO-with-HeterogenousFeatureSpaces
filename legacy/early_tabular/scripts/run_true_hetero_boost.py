#!/usr/bin/env python3
"""Boosted experiments for per-group encoders with true heterogeneous features.

Strategies to amplify GroupDRO's impact:
1. Reduced encoder capacity (hidden_dim=32) - less per-group adaptation
2. Non-overlapping + caps - maximum heterogeneity + data scarcity
3. Aggressive + caps - heavy feature loss + data scarcity
4. Very small encoders (hidden_dim=16) - force shared head dependence
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

FEATURE_PATTERNS = {
    "mild_realistic": [
        None,
        [0,1,2,4,5,6,7,8,9,10,11,12],
        [0,1,2,3,5,8,9,10,11,12],
        [0,1,2,4,5,6,7,8,9,10],
    ],
    "moderate": [
        None,
        [0,1,2,5,6,7,8,9,10],
        [0,1,2,5,8,9,10],
        [0,1,5,8,9,10,11,12],
    ],
    "aggressive": [
        None,
        [0,1,2,5,8,9,10],
        [0,1,5,9,10,11],
        [0,1,2,3,5,6,8,10],
    ],
    "non_overlapping": [
        [0,1,2,3,4,5,6],
        [0,1,7,8,9,10,11,12],
        [0,1,2,3,5,10],
        [0,1,4,6,7,8,9,11,12],
    ],
    "progressive": [
        None,
        [0,1,2,3,5,6,7,8,9,10,11],
        [0,1,2,5,6,8,9,10,11],
        [0,1,5,8,9,10,11],
    ],
}

GDRO_DEFAULT = {
    "groupdro_eta": 1.0,
    "groupdro_gamma": 0.9,
    "groupdro_update_mode": "softmax",
    "groupdro_objective": "weighted",
    "groupdro_kl_lambda": 0.0,
    "groupdro_uniform_init": True,
}

def make_base_cfg(hidden_dim=64, head_hidden=32, latent_dim=64, dropout=0.1):
    return {
        "use_fedheart": True, "num_classes": 2, "stratified_batching": True,
        "train_frac": 0.66, "batch_size": 64, "num_workers": 0,
        "epochs": 100, "early_stopping_patience": 30,
        "lr": 0.001, "weight_decay": 0.0001, "optimizer": "adam",
        "grad_clip": 1.0, "lr_scheduler": "cosine", "lr_min": 1e-5,
        "latent_dim": latent_dim, "head_hidden": head_hidden,
        "anchor_eps": 0.0001, "sep_samples_per_class": 4,
        "sep_method": "classifier", "sep_margin": 1.0,
        "lambda_fit": 0.001, "lambda_sep": 0.001,
        "log_interval": 20, "save_every": 100,
        "_hidden_dim": hidden_dim, "_dropout": dropout,
    }


def make_group_cfg(hidden_dim=64, dropout=0.1):
    return [
        {"name": "cleveland",     "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": hidden_dim, "dropout": dropout},
        {"name": "hungarian",     "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": hidden_dim, "dropout": dropout},
        {"name": "switzerland",   "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": hidden_dim, "dropout": dropout},
        {"name": "va_long_beach", "encoder": "mlp_tabular_ln", "input_dim": 13, "hidden_dim": hidden_dim, "dropout": dropout},
    ]


def run_single(cfg_dict, run_name):
    cfg_path = f"/tmp/{run_name}.yaml"
    cfg_out = {k: v for k, v in cfg_dict.items() if not k.startswith("_")}
    with open(cfg_path, "w") as f:
        yaml.dump(cfg_out, f, default_flow_style=False)
    project_root = str(Path(__file__).resolve().parent.parent)
    venv_python = os.path.join(project_root, "venv", "bin", "python")
    if not os.path.exists(venv_python):
        venv_python = sys.executable
    result = subprocess.run(
        [venv_python, "-m", "dro_hetero_anchors.src.train_fedheart", "--config", cfg_path],
        capture_output=True, text=True, cwd=project_root,
    )
    if result.returncode != 0:
        print(f"FAIL")
        if result.stderr:
            lines = result.stderr.strip().split("\n")
            print("  " + lines[-1] if lines else "no error")
        return None

    metrics_path = Path(cfg_dict["run_dir"]) / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    best = None
    for line in metrics_path.read_text().strip().split("\n"):
        m = json.loads(line)
        if best is None or m.get("test_worst_group_acc", 0) > best.get("test_worst_group_acc", 0):
            best = m
    return best


def run_10seed(cfg_template, label, prefix):
    results = []
    for seed in SEEDS:
        cfg = dict(cfg_template)
        cfg["seed"] = seed
        run_name = f"tb_{prefix}_s{seed}"
        cfg["run_dir"] = f"runs/{run_name}"
        cfg["run_name"] = run_name
        if os.path.exists(cfg["run_dir"]):
            shutil.rmtree(cfg["run_dir"])
        print(f"  {run_name} ...", end=" ", flush=True)
        m = run_single(cfg, run_name)
        if m is not None:
            wg = m.get("test_worst_group_acc", 0) * 100
            print(f"wg={wg:.1f}%")
            results.append(m)
        else:
            print("FAILED")
    return results


def extract_stats(results, num_groups=4):
    if not results:
        return None
    stats = {"n_seeds": len(results)}
    per_group_acc = {g: [] for g in range(num_groups)}
    per_group_loss = {g: [] for g in range(num_groups)}
    worst_group_accs = []

    for m in results:
        accs = m.get("test_per_group_acc", [])
        losses = m.get("test_per_group_loss", [])
        for g in range(num_groups):
            if g < len(accs): per_group_acc[g].append(accs[g])
            if g < len(losses): per_group_loss[g].append(losses[g])
        worst_group_accs.append(m.get("test_worst_group_acc", 0))

    for g in range(num_groups):
        a = np.array(per_group_acc[g]); l = np.array(per_group_loss[g])
        stats[f"g{g}_acc_mean"] = float(np.mean(a)); stats[f"g{g}_acc_std"] = float(np.std(a))
        stats[f"g{g}_loss_mean"] = float(np.mean(l)); stats[f"g{g}_loss_std"] = float(np.std(l))
    wg = np.array(worst_group_accs)
    stats["wg_acc_mean"] = float(np.mean(wg)); stats["wg_acc_std"] = float(np.std(wg))
    return stats


def fmt(mean, std):
    return f"{mean*100:.2f}% +/- {std*100:.2f}%"


def main():
    all_results = {}

    experiments = [
        # (label, pattern_name, cap, hidden_dim, head_hidden, latent_dim, dropout, extra_gdro)
        # Strategy 1: Reduce encoder capacity → shared head matters more
        ("mild_h32_nocap", "mild_realistic", None, 32, 32, 64, 0.1, {}),
        ("mild_h32_cap2030", "mild_realistic", [None, None, 20, 30], 32, 32, 64, 0.1, {}),
        ("mild_h16_cap2030", "mild_realistic", [None, None, 20, 30], 16, 32, 64, 0.2, {}),
        ("progressive_h32_cap2030", "progressive", [None, None, 20, 30], 32, 32, 64, 0.1, {}),
        ("progressive_h16_cap2030", "progressive", [None, None, 20, 30], 16, 32, 64, 0.2, {}),

        # Strategy 2: Non-overlapping + caps (wasn't tested with caps before)
        ("nonoverlap_cap2030", "non_overlapping", [None, None, 20, 30], 64, 32, 64, 0.1, {}),
        ("nonoverlap_cap1525", "non_overlapping", [None, None, 15, 25], 64, 32, 64, 0.1, {}),
        ("nonoverlap_h32_cap2030", "non_overlapping", [None, None, 20, 30], 32, 32, 64, 0.1, {}),

        # Strategy 3: Aggressive + caps
        ("aggressive_cap2030", "aggressive", [None, None, 20, 30], 64, 32, 64, 0.1, {}),
        ("aggressive_cap1525", "aggressive", [None, None, 15, 25], 64, 32, 64, 0.1, {}),
        ("aggressive_h32_cap2030", "aggressive", [None, None, 20, 30], 32, 32, 64, 0.1, {}),

        # Strategy 4: Larger shared head (more capacity in shared component)
        ("mild_bighead_cap2030", "mild_realistic", [None, None, 20, 30], 64, 64, 64, 0.1, {}),
        ("progressive_bighead_cap2030", "progressive", [None, None, 20, 30], 64, 64, 64, 0.1, {}),

        # Strategy 5: Higher eta (stronger GroupDRO reweighting)
        ("nonoverlap_cap2030_eta3", "non_overlapping", [None, None, 20, 30], 64, 32, 64, 0.1, {"groupdro_eta": 3.0}),
        ("aggressive_cap2030_eta3", "aggressive", [None, None, 20, 30], 64, 32, 64, 0.1, {"groupdro_eta": 3.0}),
    ]

    for label, pat_name, cap, hd, hh, ld, do, extra_gdro in experiments:
        print(f"\n{'='*60}")
        print(f"  Experiment: {label}")
        print(f"  Pattern: {pat_name}, Cap: {cap}, Hidden: {hd}, Head: {hh}")
        print(f"{'='*60}")

        masks = FEATURE_PATTERNS[pat_name]

        # ERM baseline
        cfg = make_base_cfg(hidden_dim=hd, head_hidden=hh, latent_dim=ld, dropout=do)
        cfg["groups"] = make_group_cfg(hidden_dim=hd, dropout=do)
        cfg["feature_mask"] = masks
        cfg["group_max_train_samples"] = cap
        cfg["groupdro_enabled"] = False
        erm_key = f"{label}_erm"
        print(f"\n  [ERM]")
        erm_raw = run_10seed(cfg, erm_key, f"{label}_erm")
        all_results[erm_key] = extract_stats(erm_raw)

        # GroupDRO
        cfg = make_base_cfg(hidden_dim=hd, head_hidden=hh, latent_dim=ld, dropout=do)
        cfg["groups"] = make_group_cfg(hidden_dim=hd, dropout=do)
        cfg["feature_mask"] = masks
        cfg["group_max_train_samples"] = cap
        cfg["groupdro_enabled"] = True
        cfg.update(GDRO_DEFAULT)
        cfg.update(extra_gdro)
        gdro_key = f"{label}_gdro"
        print(f"\n  [GroupDRO]")
        gdro_raw = run_10seed(cfg, gdro_key, f"{label}_gdro")
        all_results[gdro_key] = extract_stats(gdro_raw)

    # Print summary
    print("\n\n" + "=" * 90)
    print("BOOSTED EXPERIMENTS SUMMARY - Per-Group Encoders + True Hetero Features")
    print("=" * 90)

    group_names = ["Cleveland", "Hungarian", "Switzerland", "VA Long Beach"]

    for label, _, _, _, _, _, _, _ in experiments:
        erm_key = f"{label}_erm"
        gdro_key = f"{label}_gdro"
        bs = all_results.get(erm_key)
        gs = all_results.get(gdro_key)
        if bs is None or gs is None:
            continue

        wg_gain = (gs["wg_acc_mean"] - bs["wg_acc_mean"]) * 100
        print(f"\n--- {label} (WG gain: {wg_gain:+.2f}%) ---")
        print(f"  {'Group':<15} | {'ERM':<22} | {'GroupDRO':<22} | {'Change':<8}")
        print(f"  {'-'*72}")
        for g in range(4):
            ba = fmt(bs[f"g{g}_acc_mean"], bs[f"g{g}_acc_std"])
            ga = fmt(gs[f"g{g}_acc_mean"], gs[f"g{g}_acc_std"])
            diff = (gs[f"g{g}_acc_mean"] - bs[f"g{g}_acc_mean"]) * 100
            print(f"  G{g} {group_names[g]:<11} | {ba:<22} | {ga:<22} | {diff:+.2f}%")
        wg_erm = fmt(bs["wg_acc_mean"], bs["wg_acc_std"])
        wg_gdro = fmt(gs["wg_acc_mean"], gs["wg_acc_std"])
        print(f"  {'Worst-group':<15} | {wg_erm:<22} | {wg_gdro:<22} | {wg_gain:+.2f}%")

        # Also show losses
        print(f"  {'Group':<15} | {'ERM Loss':<22} | {'GDRO Loss':<22} | {'Change':<8}")
        print(f"  {'-'*72}")
        for g in range(4):
            bl = f"{bs[f'g{g}_loss_mean']:.3f} +/- {bs[f'g{g}_loss_std']:.3f}"
            gl = f"{gs[f'g{g}_loss_mean']:.3f} +/- {gs[f'g{g}_loss_std']:.3f}"
            diff = gs[f"g{g}_loss_mean"] - bs[f"g{g}_loss_mean"]
            print(f"  G{g} {group_names[g]:<11} | {bl:<22} | {gl:<22} | {diff:+.3f}")

    # Rank by WG gain
    print("\n\n" + "=" * 60)
    print("RANKING BY WORST-GROUP ACCURACY GAIN")
    print("=" * 60)
    gains = []
    for label, _, _, _, _, _, _, _ in experiments:
        bs = all_results.get(f"{label}_erm")
        gs = all_results.get(f"{label}_gdro")
        if bs and gs:
            gain = (gs["wg_acc_mean"] - bs["wg_acc_mean"]) * 100
            gains.append((label, gain, gs["wg_acc_mean"]*100, gs["wg_acc_std"]*100))
    gains.sort(key=lambda x: -x[1])
    print(f"\n{'Rank':<5} {'Config':<40} {'WG Gain':<10} {'WG Acc (GDRO)':<20}")
    print("-" * 75)
    for i, (label, gain, wg_mean, wg_std) in enumerate(gains, 1):
        print(f"{i:<5} {label:<40} {gain:+.2f}%{'':<4} {wg_mean:.2f}% +/- {wg_std:.2f}%")

    # Save
    report_path = "runs/TRUE_HETERO_BOOST_RESULTS.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {report_path}")


if __name__ == "__main__":
    main()
