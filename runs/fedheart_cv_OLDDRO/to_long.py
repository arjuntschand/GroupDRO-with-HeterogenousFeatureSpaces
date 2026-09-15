"""Convert the 5-fold CV results.json into the metrics_long.csv schema the site reads.

The CV protocol is the trustworthy one for Fed-Heart: median imputation keeps every patient
and rotating folds test each one exactly once, so all 925 are evaluated (Switzerland 125
rather than the 10 the single-split protocol left it with). The fold CSVs always carried per-group loss and macro-F1; the aggregator was dropping them,
so they are pooled here the same way accuracy is, weighted by fold test counts. R*_g comes
from runs/rstar_fedheart.json, the same constants the regret arms train against, and excess
loss is loss minus R* clipped at zero.
"""
import csv, json, os

src = os.path.join(os.path.dirname(__file__), "results.json")
dst = os.path.join(os.path.dirname(__file__), "metrics_long.csv")
r = json.load(open(src))

# R*_g is a property of the data, shared by every method, and lives in its own file
rstar_path = os.path.join(os.path.dirname(__file__), "..", "rstar_fedheart.json")
rstar = []
if os.path.exists(rstar_path):
    raw = json.load(open(rstar_path)).get("rstar", {})
    rstar = [raw[str(i)] if str(i) in raw else raw.get(i, 0.0) for i in range(len(raw))]

with open(dst, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["method", "seed", "group", "n", "accuracy", "macro_f1", "loss",
                "R_star", "excess_loss"])
    n = 0
    for method, seeds in r.items():
        for seed, v in seeds.items():
            accs = v.get("per_group_acc") or []
            counts = v.get("per_group_n") or []
            losses = v.get("per_group_loss") or []
            f1s = v.get("per_group_f1") or []
            for gi, acc in enumerate(accs):
                cnt = counts[gi] if gi < len(counts) else ""
                ls = losses[gi] if gi < len(losses) else ""
                f1 = f1s[gi] if gi < len(f1s) else ""
                rs = rstar[gi] if gi < len(rstar) else ""
                # Signed, NOT clamped. Xenia asked for the negative values to be visible:
                # a group below its own reference loss is doing better than a model trained on
                # that group alone, which is the whole point of pooling, and clamping to zero
                # throws that information away. The lambda update keeps max(0, .) per her
                # Step 4; this is the reporting column only.
                ex = (ls - rs) if (ls != "" and rs != "") else ""
                w.writerow([method, seed, f"g{gi}", cnt, acc, f1, ls, rs, ex])
                n += 1
print(f"wrote {dst} ({n} rows, {len(r)} methods)")
