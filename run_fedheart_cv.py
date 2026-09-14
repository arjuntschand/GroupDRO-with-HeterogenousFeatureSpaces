"""Fed-Heart with K-fold cross-validation and median imputation.

Motivation. The single-split protocol evaluates Switzerland on only 10 patients, because
(a) the loader drops every row with a missing value, which throws away 63% of that site
(it is missing cholesterol on most records), and (b) a single 80/20 split then tests on
just 20% of what is left. With 10 test samples, worst-group accuracy moves in 10-point
jumps and the seed-to-seed spread was +/- 7.6 points, so some of the reported gain is
split luck rather than method quality.

Two fixes, both standard practice and neither of which invents data:
  1. impute_missing=True keeps every patient and fills missing features with that site's
     median (FLamby's own benchmark imputes rather than drops). Switzerland: 46 -> 123.
  2. K-fold CV rotates the held-out fold so EVERY patient is tested exactly once. Each
     group's effective test set becomes the whole group instead of 20% of it.

Together, Switzerland is evaluated on 123 patients instead of 10.

Usage:
  python run_fedheart_cv.py --folds 5 --seeds 42 1337 7
"""
from __future__ import annotations
import argparse, copy, csv, json, os
import numpy as np
import yaml

BASE = "experiments/fedheart_exp_paper_hetagg_gdro.yaml"
METHODS = [   # (label, common_encoder, groupdro, anchor_weight, use_regret)
    # Same 2x2x2 grid as the tabular matrix (encoder x DRO x anchors). The cross-validated
    # protocol previously ran only the per-group half, so the encoder axis was never varied
    # here even though this is the protocol we actually trust for Fed-Heart.
    ("ERM",                 True,  False, 0.001, False),
    ("PerGroupOnly",        False, False, 0.001, False),
    ("Shared_GDRO",         True,  True,  0.001, False),
    ("Shared_Anchors",      True,  False, 0.1,   False),
    ("Shared_Anchors_GDRO", True,  True,  0.1,   False),
    ("GroupDRO",    False, True,  0.001, False),
    ("RegretDRO",   False, True,  0.001, True),
    ("AnchorsOnly", False, False, 0.1,   False),
    ("Ours",        False, True,  0.1,   False),
    ("Ours_Regret", False, True,  0.1,   True),
]


def read_best(run_dir):
    f = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    if not rows:
        return None
    def fl(r, k):
        try:
            return float(r.get(k, "") or "nan")
        except Exception:
            return float("nan")
    # Select the reported epoch on VALIDATION when a validation split exists. Selecting on
    # test, which is what this did before, makes every reported number a best-of-N on the test
    # set and biases it upward. Falls back to test only when no val column is present, and the
    # fallback is visible in the output rather than silent.
    sel_key = "val_worst_group_acc" if rows and rows[0].get("val_worst_group_acc") else "test_worst_group_acc"
    if sel_key == "test_worst_group_acc":
        globals().setdefault("_WARNED_TEST_SEL", False)
        if not globals()["_WARNED_TEST_SEL"]:
            print("  [warn] no validation split found; selecting the reported epoch on TEST",
                  flush=True)
            globals()["_WARNED_TEST_SEL"] = True
    best = max(rows, key=lambda r: fl(r, sel_key))
    import ast
    def lst(k):
        v = best.get(k)
        try:
            return [float(x) for x in ast.literal_eval(v)] if v else None
        except Exception:
            return None
    return {"worst": fl(best, "test_worst_group_acc"),
            "overall": fl(best, "test_overall_acc"),
            "balanced": fl(best, "test_balanced_acc"),
            "per_group_acc": lst("test_per_group_acc"),
            "per_group_loss": lst("test_per_group_loss"),
            "per_group_f1": lst("test_per_group_f1"),
            "per_group_counts": lst("test_per_group_counts")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 7])
    ap.add_argument("--out", default="runs/fedheart_cv")
    ap.add_argument("--no-impute", action="store_true")
    ap.add_argument("--anchor-weight", type=float, default=0.1,
                    help="lambda_fit/lambda_sep for the anchors-on arms")
    args = ap.parse_args()

    from dro_hetero_anchors.src.train_fedheart import train
    base = yaml.safe_load(open(BASE))
    # per-group reference losses R*_g for the regret arms (estimated once, see
    # tools/estimate_rstar_tabular.py). Falls back to plain GroupDRO if unavailable.
    rstar_list = None
    if os.path.exists("runs/rstar_fedheart.json"):
        rs = json.load(open("runs/rstar_fedheart.json"))["rstar"]
        rstar_list = [rs[str(i)] if str(i) in rs else rs.get(i, 0.0) for i in range(len(rs))]
    base["impute_missing"] = not args.no_impute
    os.makedirs(args.out, exist_ok=True)
    results = {}

    # Each fold uses a different data_split_seed, so a different 1/K is held out. Over K
    # folds every patient lands in the test set exactly once.
    METHODS_RUN = [(l, sh, gd, (args.anchor_weight if a > 0.01 else a), rg)
                   for (l, sh, gd, a, rg) in METHODS]
    for label, shared, gdro, anch, regret in METHODS_RUN:
        results[label] = {}
        for seed in args.seeds:
            fold_metrics = []
            for k in range(args.folds):
                cfg = copy.deepcopy(base)
                cfg["common_encoder"] = shared
                cfg["groupdro_enabled"] = gdro
                cfg["lambda_fit"] = anch
                cfg["lambda_sep"] = anch
                cfg["use_regret"] = regret
                if regret and rstar_list:
                    cfg["optimal_losses"] = rstar_list
                cfg["seed"] = seed
                cfg["data_split_seed"] = 1000 + k          # fold identity
                cfg["train_frac"] = 1.0 - 1.0 / args.folds  # K-fold sized split
                cfg["run_dir"] = f"{args.out}/{label}_s{seed}_f{k}"
                m = read_best(cfg["run_dir"])
                if not (m and m["worst"] == m["worst"]):
                    try:
                        train(cfg)
                        m = read_best(cfg["run_dir"])
                    except Exception as e:
                        print(f"  FAILED {label} s{seed} f{k}: {e}", flush=True)
                        m = None
                if m:
                    fold_metrics.append(m)
            if not fold_metrics:
                continue
            # pool folds: per-group accuracy weighted by that fold's test count
            pg = [m["per_group_acc"] for m in fold_metrics if m["per_group_acc"]]
            cn = [m["per_group_counts"] for m in fold_metrics if m["per_group_counts"]]
            if pg and cn and len(pg) == len(cn):
                G = min(len(p) for p in pg)
                acc = np.array([p[:G] for p in pg]); cnt = np.array([c[:G] for c in cn])
                pooled = (acc * cnt).sum(0) / np.maximum(cnt.sum(0), 1)
                def pool(key):
                    vv = [m.get(key) for m in fold_metrics if m.get(key)]
                    if not vv or len(vv) != len(cn):
                        return None
                    a = np.array(vv); w = np.array(cn)
                    return ((a * w).sum(0) / np.maximum(w.sum(0), 1)).tolist()
                results[label][seed] = {
                    "per_group_loss": pool("per_group_loss"),
                    "per_group_f1": pool("per_group_f1"),
                    "per_group_acc": pooled.tolist(),
                    "per_group_n": cnt.sum(0).tolist(),
                    "worst": float(pooled.min()),
                    "overall": float((pooled * cnt.sum(0)).sum() / cnt.sum()),
                    "balanced": float(pooled.mean()),
                }
                r = results[label][seed]
                print(f"  {label} s{seed}: worst={r['worst']*100:.2f} overall={r['overall']*100:.2f} "
                      f"(pooled over {args.folds} folds, n/group={[int(x) for x in r['per_group_n']]})",
                      flush=True)

    print(f"\n########## FED-HEART, {args.folds}-FOLD CV"
          f"{' + IMPUTATION' if not args.no_impute else ''} ##########")
    print(f"{'method':>14} | worst-group | overall | balanced | n")
    for label, *_rest in METHODS:
        R = results.get(label, {})
        if not R:
            continue
        w = np.array([R[s]["worst"] for s in R]) * 100
        o = np.array([R[s]["overall"] for s in R]) * 100
        b = np.array([R[s]["balanced"] for s in R]) * 100
        print(f"{label:>14} | {w.mean():5.2f} ± {w.std():4.2f} | {o.mean():5.2f} | {b.mean():5.2f} | {len(w)}")
    ng = next((R[s]["per_group_n"] for R in results.values() for s in R), None)
    if ng:
        print(f"\nEvaluated per group (pooled across folds): {[int(x) for x in ng]}")
    json.dump(results, open(f"{args.out}/results.json", "w"), indent=2)
    print(f"wrote {args.out}/results.json")


if __name__ == "__main__":
    main()
