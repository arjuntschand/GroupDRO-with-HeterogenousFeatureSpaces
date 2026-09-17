import argparse
import csv
import json
from pathlib import Path

"""
Analyze USPS class 0/1 issues by inspecting per-class accuracies and test set composition
from existing run artifacts (metrics.csv and embedded dataset summaries).

Usage:
  python scripts/analyze_usps_class_issues.py \
    --runs runs \
    --prefix mnist_usps_skewsubset_8000_5000_flip_20e_groupdro_eta0p30_gamma0p85_lrdecay_lr0p1 \
    --epoch 20

Outputs:
  - For each matched run: USPS per-class accuracies at the target epoch
  - Test dataset per-class counts and per-group-per-class counts (to see if USPS classes 0/1 are underrepresented)
  - Final group weights snapshot when available
"""


def read_metrics(run_dir: Path, epoch: int):
    csv_path = run_dir / 'metrics.csv'
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        target = None
        for row in reader:
            try:
                e = int(row.get('epoch', -1))
            except Exception:
                e = -1
            if e == epoch:
                target = row
        return target


def parse_json_field(row, key):
    val = row.get(key)
    if not val:
        return None
    try:
        return json.loads(val)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=str, default='runs')
    ap.add_argument('--prefix', type=str, required=True)
    ap.add_argument('--epoch', type=int, default=20)
    ap.add_argument('--only-seeded', action='store_true', help='Only include runs with names containing _seed')
    args = ap.parse_args()

    runs_root = Path(args.runs)
    matched = [d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith(args.prefix)]
    if args.only_seeded:
        matched = [d for d in matched if '_seed' in d.name]
    if not matched:
        print(f'No runs matched prefix {args.prefix}')
        return

    print(f'Inspecting {len(matched)} runs for epoch {args.epoch}…')
    for rd in sorted(matched, key=lambda p: p.name):
        row = read_metrics(rd, args.epoch)
        if row is None:
            print(f'- {rd.name}: missing epoch {args.epoch} in metrics.csv')
            continue
        per_class_acc = parse_json_field(row, 'per_class_acc') or []
        per_group_acc = parse_json_field(row, 'per_group_acc') or []
        per_group_by_class_acc = parse_json_field(row, 'per_group_by_class_acc') or []
        train_group_weights = parse_json_field(row, 'train_group_weights')

        # Dataset summaries are stored once per run; fetch from the most recent row
        test_summary = parse_json_field(row, 'test_dataset_summary')
        if test_summary is None:
            # fall back to reading the last row for summary
            test_summary = row.get('test_dataset_summary')
            try:
                test_summary = json.loads(test_summary) if test_summary else None
            except Exception:
                test_summary = None

        print(f'\nRun: {rd.name}')
        # Group 1 assumed USPS by config ordering
        if per_group_by_class_acc and len(per_group_by_class_acc) >= 2:
            usps_class_acc = per_group_by_class_acc[1]
            print('  USPS per-class accuracies:')
            for cid, acc in enumerate(usps_class_acc):
                print(f'    class {cid}: {acc:.4f}')
            if usps_class_acc and len(usps_class_acc) > 1:
                print(f'  USPS class 0/1: {usps_class_acc[0]:.4f}, {usps_class_acc[1]:.4f}')
        else:
            print('  Missing per-group-by-class accuracies.')

        if test_summary:
            # Show composition
            cc = test_summary.get('class_counts')
            gc = test_summary.get('group_counts')
            gxc = test_summary.get('group_class_counts')
            if cc:
                print('  Test set class counts (all groups):')
                print('   ', cc)
            if gc:
                print('  Test set group counts:')
                print('   ', gc)
            if gxc and len(gxc) > 1:
                print('  USPS (group 1) per-class counts:')
                print('   ', gxc[1])

        if train_group_weights:
            print('  Final group weights q:', train_group_weights)


if __name__ == '__main__':
    main()
