"""Write metrics_long.csv for a run_fedheart_cv.py family from its results.json.

run_fedheart_cv.py pools the five folds per seed into results.json; the site, final_report.py and
preflight_check.py read the long-format CSV that run_method_matrix.py emits for the other
datasets. This produces the same columns from the pooled per-seed numbers.

  python fedheart_results_to_long.py --family runs/gamma_sweep_fh10/g0.1_s1 --rstar runs/rstar_fedheart_eq13.json
"""
import argparse, csv, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--family", required=True)
ap.add_argument("--rstar", required=True)
ap.add_argument("--params-from", default="runs/fedheart_uncapped/metrics_long.csv",
                help="CSV whose per-method n_params to reuse (architecture unchanged)")
args = ap.parse_args()

R = json.load(open(f"{args.family}/results.json"))
rs = json.load(open(args.rstar))["rstar"]
rstar = [rs[str(i)] for i in range(len(rs))]
n_params = {}
if os.path.exists(args.params_from):
    for row in csv.DictReader(open(args.params_from)):
        n_params.setdefault(row["method"], row["n_params"])

rows = []
for method, seeds in R.items():
    for seed, m in seeds.items():
        for g, (acc, f1, loss, n) in enumerate(zip(m["per_group_acc"], m["per_group_f1"],
                                                    m["per_group_loss"], m["per_group_n"])):
            rows.append(dict(method=method, seed=seed, group=f"g{g}", n=int(n),
                             n_params=n_params.get(method, ""), accuracy=round(acc, 5),
                             macro_f1=round(f1, 5), loss=round(loss, 5), R_star=round(rstar[g], 5),
                             excess_loss=round(loss - rstar[g], 5)))
out = f"{args.family}/metrics_long.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"wrote {out} ({len(rows)} rows, {len(R)} methods)")
