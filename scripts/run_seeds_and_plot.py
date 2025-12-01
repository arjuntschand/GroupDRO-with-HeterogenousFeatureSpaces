#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import json
from pathlib import Path
from collections import defaultdict
import csv
import re
import numpy as np
import matplotlib.pyplot as plt

WORKSPACE = Path(__file__).resolve().parents[1]
PYTHON = str(WORKSPACE / ".venv/bin/python")
TRAIN_MODULE = "dro_hetero_anchors.src.train"

CONFIGS = {
    "mnist20k_usps4k_nodro": "experiments/mnist_usps_mnist20k_usps4k_noskew.yaml",
    "mnist20k_usps4k_dro": "experiments/mnist_usps_mnist20k_usps4k_noskew_groupdro.yaml",
    "mnist500_usps5k_nodro": "experiments/mnist_usps_mnist500_usps5k_noskew.yaml",
    "mnist500_usps5k_dro": "experiments/mnist_usps_mnist500_usps5k_noskew_groupdro.yaml",
}

DEFAULT_SEED_BASE = 1337


def run_config(config_path: str, seed: int):
    # Override seed by writing a temp copy of the YAML
    cfg_file = WORKSPACE / config_path
    with open(cfg_file, "r") as f:
        content = f.read()
    content = re.sub(r"^seed:\s*\d+", f"seed: {seed}", content, flags=re.MULTILINE)
    tmp_path = WORKSPACE / "runs" / f"tmp_seed_{seed}_{Path(config_path).name}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w") as f:
        f.write(content)
    cmd = [PYTHON, "-m", TRAIN_MODULE, "--config", str(tmp_path.relative_to(WORKSPACE))]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=WORKSPACE)
    if res.returncode != 0:
        print(f"Run failed for {config_path} seed {seed}", file=sys.stderr)
        sys.exit(res.returncode)


def collect_runs(run_dir: Path):
    # Each training logs runs/index.csv and runs/<run_dir>/... via ResultsLogger
    # We'll parse the per-epoch entries from the run dir's CSVs if present.
    idx = WORKSPACE / "runs" / "index.csv"
    entries = []
    if idx.exists():
        with open(idx, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # filter by run_dir prefix match
                if str(run_dir) in row.get("run_dir", ""):
                    entries.append(row)
    # Also try reading per-run CSV inside the run_dir
    run_csv = run_dir / "results.csv"
    if run_csv.exists():
        with open(run_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)
    return entries


def aggregate(entries, mode: str = "final"):
    """Aggregate per-group accuracies per seed.

    mode options:
      final        -> use last epoch available per seed
      best_worst   -> use epoch with highest worst-group accuracy per seed
    """
    by_seed_rows = defaultdict(list)
    for row in entries:
        try:
            seed = int(row.get("seed", 0))
        except Exception:
            continue
        by_seed_rows[seed].append(row)

    per_seed_vecs = []
    for seed, rows in by_seed_rows.items():
        selected_row = None
        if mode == "final":
            max_epoch = -1
            for r in rows:
                try:
                    ep = int(r.get("epoch", -1))
                except Exception:
                    ep = -1
                if ep > max_epoch:
                    max_epoch = ep
                    selected_row = r
        elif mode == "best_worst":
            best_worst = -1.0
            for r in rows:
                # Prefer explicit worst_group_acc; fallback to min(per_group_acc)
                try:
                    if "worst_group_acc" in r and r["worst_group_acc"]:
                        worst = float(r["worst_group_acc"])
                    else:
                        pg = r.get("per_group_acc")
                        if isinstance(pg, str):
                            pg = json.loads(pg)
                        worst = float(min(pg)) if pg else -1.0
                except Exception:
                    worst = -1.0
                if worst > best_worst:
                    best_worst = worst
                    selected_row = r
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        if selected_row is None:
            continue
        per_group = selected_row.get("per_group_acc")
        if isinstance(per_group, str):
            try:
                per_group = json.loads(per_group)
            except Exception:
                continue
        vec = np.array(per_group, dtype=float)
        per_seed_vecs.append(vec)

    if not per_seed_vecs:
        return None
    arr = np.stack(per_seed_vecs)
    return arr.mean(axis=0), arr.std(axis=0)


def plot_bar(mean_std_map, title, outpath):
    # mean_std_map: {label: (mean_vec, std_vec)}
    labels = list(mean_std_map.keys())
    G = len(next(iter(mean_std_map.values()))[0])
    x = np.arange(G)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, label in enumerate(labels):
        mean, std = mean_std_map[label]
        ax.bar(x + i*width, mean, width, yerr=std, label=label, capsize=4)

    ax.set_xticks(x + width/2)
    ax.set_xticklabels([f"Group {i}" for i in range(G)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    print("Saved", outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "plot", "run_and_plot"]) 
    ap.add_argument("--seeds", type=int, default=1, help="Number of seeds to run (default: 1)")
    ap.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE, help="Starting seed value")
    ap.add_argument("--selection", choices=["final", "best_worst"], default="final", help="Epoch selection strategy for aggregation")
    args = ap.parse_args()

    # Map configs to run_dir names
    run_dirs = {
        "mnist20k_usps4k_nodro": WORKSPACE / "runs" / "mnist_usps_mnist20k_usps4k_noskew",
        "mnist20k_usps4k_dro": WORKSPACE / "runs" / "mnist_usps_mnist20k_usps4k_noskew_groupdro",
        "mnist500_usps5k_nodro": WORKSPACE / "runs" / "mnist_usps_mnist500_usps5k_noskew",
        "mnist500_usps5k_dro": WORKSPACE / "runs" / "mnist_usps_mnist500_usps5k_noskew_groupdro",
    }

    if args.action in ("run", "run_and_plot"):
        seeds_list = [args.seed_base + i for i in range(args.seeds)]
        for key, cfg in CONFIGS.items():
            for seed in seeds_list:
                run_config(cfg, seed)

    if args.action in ("plot", "run_and_plot"):
        # For each scenario, aggregate and plot bars
        # Scenario 1: 20k/4k no-DRO vs DRO
        s1_nodro_entries = collect_runs(run_dirs["mnist20k_usps4k_nodro"]) 
        s1_dro_entries = collect_runs(run_dirs["mnist20k_usps4k_dro"]) 
        s1_nodro = aggregate(s1_nodro_entries, mode=args.selection)
        s1_dro = aggregate(s1_dro_entries, mode=args.selection)
        if s1_nodro and s1_dro:
            plot_bar({"No DRO": s1_nodro, "GroupDRO": s1_dro},
                     f"MNIST20k vs USPS4k: Group accuracies ({args.selection} epoch)",
                     WORKSPACE / "runs" / "fig_mnist20k_usps4k.png")

        # Scenario 2: 500/5k no-DRO vs DRO
        s2_nodro_entries = collect_runs(run_dirs["mnist500_usps5k_nodro"]) 
        s2_dro_entries = collect_runs(run_dirs["mnist500_usps5k_dro"]) 
        s2_nodro = aggregate(s2_nodro_entries, mode=args.selection)
        s2_dro = aggregate(s2_dro_entries, mode=args.selection)
        if s2_nodro and s2_dro:
            plot_bar({"No DRO": s2_nodro, "GroupDRO": s2_dro},
                     f"MNIST500 vs USPS5k: Group accuracies ({args.selection} epoch)",
                     WORKSPACE / "runs" / "fig_mnist500_usps5k.png")

if __name__ == "__main__":
    main()
