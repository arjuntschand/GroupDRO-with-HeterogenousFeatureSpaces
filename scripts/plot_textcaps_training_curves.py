#!/usr/bin/env python3
"""
Plot training loss and per-group (test) accuracy vs epoch for the 6 TextCaps runs.
Reads metrics.jsonl from each run dir. Saves figures to scripts/plots/ (or --outdir).

Usage:
  python scripts/plot_textcaps_training_curves.py
  python scripts/plot_textcaps_training_curves.py --runs runs/textcaps_4class_baseline_v2 runs/...
"""

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = [
    ("runs/textcaps_4class_baseline_v2", "4-class 2-group baseline"),
    ("runs/textcaps_4class_groupdro_v2", "4-class 2-group GroupDRO"),
    ("runs/textcaps_10class_baseline_v3", "10-class 2-group baseline"),
    ("runs/textcaps_10class_groupdro_v3", "10-class 2-group GroupDRO"),
    ("runs/textcaps_4class_3group_baseline", "4-class 3-group baseline"),
    ("runs/textcaps_4class_3group_groupdro", "4-class 3-group GroupDRO"),
]


def load_metrics(run_dir: Path):
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None, help="Override run dirs (alternating with names not supported)")
    ap.add_argument("--outdir", type=str, default="scripts/plots", help="Output directory for figures")
    args = ap.parse_args()

    if args.runs:
        run_paths = [Path(p) for p in args.runs]
        run_names = [p.name for p in run_paths]
    else:
        run_paths = [Path(r[0]) for r in RUNS]
        run_names = [r[1] for r in RUNS]

    root = Path(__file__).resolve().parent.parent
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Load all metrics
    all_metrics = []
    for rdir in run_paths:
        full = root / rdir if not rdir.is_absolute() else rdir
        m = load_metrics(full)
        if m is None:
            print(f"Warning: no metrics at {full}")
            all_metrics.append([])
        else:
            all_metrics.append(m)

    # Filter to one record per epoch (take last occurrence if duplicates)
    def by_epoch(records):
        d = {}
        for r in records:
            d[r["epoch"]] = r
        epochs = sorted(d.keys())
        return epochs, [d[e] for e in epochs]

    # ----- Figure 1: Training loss vs epoch (all 6 runs) -----
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (records, name) in enumerate(zip(all_metrics, run_names)):
        if not records:
            continue
        epochs, recs = by_epoch(records)
        losses = [r["train_loss"] for r in recs]
        ax1.plot(epochs, losses, label=name, color=colors[i % 10], lw=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax1.set_title("Training loss vs epoch (all 6 runs)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(outdir / "train_loss_all_runs.png", dpi=150)
    plt.close(fig1)
    print(f"Saved {outdir / 'train_loss_all_runs.png'}")

    # ----- Figure 2: Per-group (test) accuracy vs epoch for each run -----
    # 2x3 subplots, one per run; each subplot has one line per group (visual, text, [combined])
    fig2, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    for i, (records, name) in enumerate(zip(all_metrics, run_names)):
        ax = axes[i]
        if not records:
            ax.set_title(name + " (no data)")
            continue
        epochs, recs = by_epoch(records)
        group_names = recs[0].get("group_names", ["visual", "text"])
        per_group = [r["per_group_acc"] for r in recs]
        # per_group is list of [acc_g0, acc_g1, ...] per epoch
        for g in range(len(group_names)):
            accs = [row[g] if g < len(row) else np.nan for row in per_group]
            ax.plot(epochs, accs, label=group_names[g], lw=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy")
        ax.set_title(name)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    fig2.tight_layout()
    fig2.savefig(outdir / "per_group_test_acc_vs_epoch.png", dpi=150)
    plt.close(fig2)
    print(f"Saved {outdir / 'per_group_test_acc_vs_epoch.png'}")

    # ----- Figure 2b: Per-group test ERROR (1 - acc) vs epoch — proxy for "per group loss" -----
    # (True per-group train loss was not logged in these runs; this is the best we have from metrics.)
    fig2b, axes2b = plt.subplots(2, 3, figsize=(14, 9))
    axes2b = axes2b.flatten()
    for i, (records, name) in enumerate(zip(all_metrics, run_names)):
        ax = axes2b[i]
        if not records:
            ax.set_title(name + " (no data)")
            continue
        epochs, recs = by_epoch(records)
        group_names = recs[0].get("group_names", ["visual", "text"])
        per_group = [r["per_group_acc"] for r in recs]
        for g in range(len(group_names)):
            accs = [row[g] if g < len(row) else np.nan for row in per_group]
            errors = [1.0 - a if not np.isnan(a) else np.nan for a in accs]
            ax.plot(epochs, errors, label=group_names[g], lw=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test error (1 − acc)")
        ax.set_title(name)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    fig2b.tight_layout()
    fig2b.savefig(outdir / "per_group_test_error_vs_epoch.png", dpi=150)
    plt.close(fig2b)
    print(f"Saved {outdir / 'per_group_test_error_vs_epoch.png'} (proxy for per-group 'loss'; true train loss per group was not logged in these runs)")

    # ----- Figure 3: Per-group TRAIN loss vs epoch (if present in metrics) -----
    has_per_group_loss = any(
        recs and "train_loss_per_group" in recs[0]
        for recs in all_metrics
    )
    if has_per_group_loss:
        fig3, axes3 = plt.subplots(2, 3, figsize=(14, 9))
        axes3 = axes3.flatten()
        for i, (records, name) in enumerate(zip(all_metrics, run_names)):
            ax = axes3[i]
            if not records:
                ax.set_title(name + " (no data)")
                continue
            epochs, recs = by_epoch(records)
            if "train_loss_per_group" not in recs[0]:
                ax.set_title(name + " (no per-group loss)")
                continue
            group_names = recs[0].get("group_names", ["visual", "text"])
            for g in range(len(group_names)):
                losses = [r["train_loss_per_group"][g] for r in recs if g < len(r.get("train_loss_per_group", []))]
                if len(losses) == len(epochs):
                    ax.plot(epochs, losses, label=group_names[g], lw=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Train loss (per group)")
            ax.set_title(name)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(outdir / "train_loss_per_group_vs_epoch.png", dpi=150)
        plt.close(fig3)
        print(f"Saved {outdir / 'train_loss_per_group_vs_epoch.png'}")
    else:
        print("Note: train_loss_per_group not in metrics; run with updated trainer to get per-group train loss plots.")

    print("Done.")


if __name__ == "__main__":
    main()
