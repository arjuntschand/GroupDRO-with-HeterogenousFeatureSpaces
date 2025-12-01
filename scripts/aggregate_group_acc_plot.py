import argparse
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Usage: python scripts/aggregate_group_acc_plot.py --runs runs --prefix mnist_usps_balanced_8k_5k_20e_baseline_lrdecay_lr0p1 --epoch 20 --out figures/mnist_usps_balanced_8k_5k_20e_baseline_lrdecay_lr0p1_agg.png

def load_final_group_acc(run_dir: Path, epoch: int):
    csv_path = run_dir / 'metrics.csv'
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        last_row = None
        for row in reader:
            if int(row.get('epoch', -1)) == epoch:
                last_row = row
        if last_row is None:
            return None
        # per_group_acc stored as JSON string
        pg_json = last_row.get('per_group_acc')
        if pg_json is None:
            return None
        try:
            per_group = json.loads(pg_json)
        except json.JSONDecodeError:
            return None
        return per_group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=str, default='runs')
    ap.add_argument('--prefix', type=str, required=True, help='Run name prefix to match')
    ap.add_argument('--epoch', type=int, default=20)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--groups', type=int, default=2)
    ap.add_argument('--colors', type=str, default='', help='Comma-separated colors for bars, e.g., "#ff7f0e,#1f77b4"')
    ap.add_argument('--labels', type=str, default='Group 0,Group 1', help='Comma-separated labels for groups')
    ap.add_argument('--title', type=str, default='Per-group test accuracy')
    ap.add_argument('--stat', type=str, default='se', choices=['se','std'], help='Error bar type: standard error (se) or std dev (std)')
    ap.add_argument('--only-seeded', action='store_true', help='If set, include only runs with names containing "_seed"')
    ap.add_argument('--include', type=str, default='', help='Comma-separated exact run directory names to include; if provided, only these runs are aggregated')
    args = ap.parse_args()

    runs_root = Path(args.runs)
    matched = [d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith(args.prefix)]
    if args.only_seeded:
        matched = [d for d in matched if '_seed' in d.name]
    if args.include:
        allow = set([s.strip() for s in args.include.split(',') if s.strip()])
        matched = [d for d in matched if d.name in allow]
    if not matched:
        print(f'No runs found with prefix {args.prefix}')
        return

    per_group_matrix = []
    run_names = []
    for rd in matched:
        accs = load_final_group_acc(rd, args.epoch)
        if accs is None:
            print(f'Skip {rd}: no epoch {args.epoch} record.')
            continue
        if len(accs) != args.groups:
            print(f'Skip {rd}: expected {args.groups} groups, got {len(accs)}')
            continue
        per_group_matrix.append(accs)
        run_names.append(rd.name)

    if not per_group_matrix:
        print('No valid runs to aggregate.')
        return

    arr = np.array(per_group_matrix)  # shape [num_runs, num_groups]
    means = arr.mean(axis=0)
    stds = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(means)
    # standard error
    stderrs = stds / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(means)

    labels = [s.strip() for s in args.labels.split(',')]
    if len(labels) != arr.shape[1]:
        labels = [f'Group {i}' for i in range(arr.shape[1])]

    fig, ax = plt.subplots(figsize=(7,5))
    bar_positions = np.arange(len(means))
    if args.stat == 'std' and arr.shape[0] > 1:
        yerr = stds
        err_label = '± SD'
    else:
        yerr = stderrs
        err_label = '± SE'
    colors = None
    if args.colors:
        cs = [c.strip() for c in args.colors.split(',')]
        if len(cs) == len(means):
            colors = cs
    ax.bar(bar_positions, means, yerr=yerr, capsize=6, color=colors)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f'Accuracy (mean {err_label})')
    # Title with extra padding so it doesn't overlap with top annotations
    ax.set_title(f'{args.title} (n={arr.shape[0]} seeds)', pad=14)
    # Place mean labels slightly above the bar; adapt offset if near the top
    offset = 0.015
    for i, m in enumerate(means):
        ax.text(i, m + offset, f'{m:.3f}', ha='center', va='bottom', fontsize=10)
    # Compute a headroom-aware upper bound to avoid clipping labels/error bars
    upper = (means + (yerr if isinstance(yerr, np.ndarray) else 0)).max() + 0.06
    upper = max(upper, 0.9)  # ensure reasonable scale
    upper = min(upper, 1.1)  # cap upper bound
    ax.set_ylim(0, upper)
    # Add a bit more space at the top of the layout
    fig.subplots_adjust(top=0.88)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f'Saved aggregated figure to {out_path}')
    # Also print numeric summary for convenience
    print('Aggregated stats:')
    for i, (label, mean) in enumerate(zip(labels, means)):
        err = yerr[i] if isinstance(yerr, np.ndarray) else yerr
        print(f"  {label}: mean={mean:.4f}, {err_label}={err:.4f}")
    # Per-run details
    print('Per-run final accuracies at epoch', args.epoch)
    for name, accs in zip(run_names, per_group_matrix):
        parts = ', '.join([f"{labels[i]}={accs[i]:.4f}" for i in range(len(accs))])
        print(f"  {name}: {parts}")

if __name__ == '__main__':
    main()
