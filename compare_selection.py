"""How much did selecting the reported epoch on TEST inflate the tabular numbers?

Both tabular runners used to pick the epoch with the best test worst-group accuracy, out of up
to 100. That is model selection on the test set. The runs have now been repeated with a real
validation split, and the old directories were kept, so the bias can be measured rather than
guessed at.

The number that matters is not whether ours went down. Every method loses the same best-of-N
advantage, so what we need to know is whether the RANKING moved. If our lead over a baseline
only existed because we got more bites at the test set, that lead was never real.

  python compare_selection.py
"""
from __future__ import annotations
import csv, os
from collections import defaultdict

import numpy as np

PAIRS = [
    ("NHANES", "runs/matrix_nhanes_nested_TESTSEL/metrics_long.csv",
               "runs/matrix_nhanes_nested/metrics_long.csv"),
    ("Fed-Heart", "runs/fedheart_cv_TESTSEL/metrics_long.csv",
                  "runs/fedheart_cv/metrics_long.csv"),
]


def worst_group_by_seed(path):
    """method -> {seed: worst-group accuracy}, in percent."""
    if not os.path.exists(path):
        return None
    per = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        try:
            per[r["method"]][r["seed"]].append(float(r["accuracy"]))
        except (ValueError, KeyError):
            continue
    return {m: {s: min(v) * 100 for s, v in d.items() if v} for m, d in per.items()}


def main():
    for name, old_p, new_p in PAIRS:
        old, new = worst_group_by_seed(old_p), worst_group_by_seed(new_p)
        if not old or not new:
            missing = old_p if not old else new_p
            print(f"\n{name}: skipped, {missing} not written yet")
            continue

        print(f"\n{'='*72}\n{name}: epoch selected on TEST vs on VALIDATION\n{'='*72}")
        print(f"{'method':>22} | {'test-sel':>8} | {'val-sel':>8} | {'drop':>7} | seeds")
        rows = []
        for m in new:
            if m not in old:
                continue
            # Only seeds present in both, so the delta is not a change in the seed set.
            shared = sorted(set(old[m]) & set(new[m]))
            if not shared:
                continue
            o = np.mean([old[m][s] for s in shared])
            n = np.mean([new[m][s] for s in shared])
            rows.append((m, o, n, n - o, len(shared)))
        for m, o, n, d, k in sorted(rows, key=lambda r: -r[2]):
            print(f"{m:>22} | {o:8.2f} | {n:8.2f} | {d:+7.2f} | {k}")

        if rows:
            drops = [r[3] for r in rows]
            print(f"\n  average drop across methods: {np.mean(drops):+.2f} points")
            r_old = [m for m, *_ in sorted(rows, key=lambda r: -r[1])]
            r_new = [m for m, *_ in sorted(rows, key=lambda r: -r[2])]
            if r_old == r_new:
                print("  ranking unchanged: the bias was uniform, so every comparison survives")
            else:
                print("  RANKING MOVED. Order under test selection, then under val selection:")
                print(f"    test: {' > '.join(r_old[:5])}")
                print(f"    val:  {' > '.join(r_new[:5])}")
                moved = [m for m in r_new if r_new.index(m) != r_old.index(m)]
                print(f"    methods that changed place: {', '.join(moved)}")


if __name__ == "__main__":
    main()
