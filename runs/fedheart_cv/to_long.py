"""Convert the 5-fold CV results.json into the metrics_long.csv schema the site reads.

The CV protocol is the trustworthy one for Fed-Heart: median imputation keeps every patient
and rotating folds test each one exactly once, so all 925 are evaluated (Switzerland 125
rather than the 10 the single-split protocol left it with). results.json only records
per-group accuracy and counts, so loss and macro-F1 come out blank; the site renders those
as unavailable rather than inventing them.
"""
import csv, json, os

src = os.path.join(os.path.dirname(__file__), "results.json")
dst = os.path.join(os.path.dirname(__file__), "metrics_long.csv")
r = json.load(open(src))

with open(dst, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["method", "seed", "group", "n", "accuracy", "macro_f1", "loss",
                "R_star", "excess_loss"])
    n = 0
    for method, seeds in r.items():
        for seed, v in seeds.items():
            accs, counts = v.get("per_group_acc") or [], v.get("per_group_n") or []
            for gi, acc in enumerate(accs):
                cnt = counts[gi] if gi < len(counts) else ""
                w.writerow([method, seed, f"g{gi}", cnt, acc, "", "", "", ""])
                n += 1
print(f"wrote {dst} ({n} rows, {len(r)} methods)")
