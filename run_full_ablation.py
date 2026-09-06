"""Production ablation: full 2x2x2 (encoder {shared,per-group} x GroupDRO {off,on}
x anchors {off,on}) for a tabular dataset, 10 seeds, extracting ALL metrics
(worst-group / overall / balanced acc, per-group acc + loss, and for NHANES also
AUROC / sensitivity / specificity / F1) from each run's metrics.csv.

Anchors "on" uses lambda_fit=lambda_sep=0.1 (the validated weight); "off" uses 0.0.

Usage:
  python run_full_ablation.py --dataset fedheart
  python run_full_ablation.py --dataset nhanes --base experiments/nhanes_disjoint_pergroup_gdro.yaml --tag disjoint
"""
import argparse, copy, csv, json, os
import numpy as np
import yaml

MODULES = {"fedheart": "dro_hetero_anchors.src.train_fedheart",
           "nhanes": "dro_hetero_anchors.src.train_nhanes"}
DEFAULT_BASE = {"fedheart": "experiments/fedheart_ablation_pergroup_noanchor.yaml",
                "nhanes": "experiments/nhanes_disjoint_pergroup_gdro.yaml"}
SEEDS = [42, 1337, 7, 2024, 31337, 11, 22, 33, 44, 55]
ANCHOR_ON = 0.1

# 8 cells: (label, common_encoder, groupdro, anchor_weight)
CELLS = [
    ("shared_erm",          True,  False, 0.0),
    ("shared_erm_anchor",   True,  False, ANCHOR_ON),
    ("shared_gdro",         True,  True,  0.0),
    ("shared_gdro_anchor",  True,  True,  ANCHOR_ON),
    ("pergroup_erm",        False, False, 0.0),
    ("pergroup_erm_anchor", False, False, ANCHOR_ON),
    ("pergroup_gdro",       False, True,  0.0),
    ("pergroup_gdro_anchor",False, True,  ANCHOR_ON),   # full method
]


def extract_metrics(run_dir):
    f = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    if not rows:
        return None
    def fl(r, k):
        try: return float(r.get(k, "") or "nan")
        except: return float("nan")
    best = max(rows, key=lambda r: fl(r, "test_worst_group_acc"))
    out = {}
    for k in best:
        if k.startswith("test_") and k not in ("test_per_group_acc", "test_per_group_loss",
                                               "test_per_group_sensitivity", "test_per_group_specificity",
                                               "test_per_group_f1", "test_per_group_auroc",
                                               "test_per_class_acc", "test_per_group_per_class_acc",
                                               "test_per_group_bal_acc"):
            v = fl(best, k)
            if v == v:
                out[k] = v
    # keep the JSON-list per-group columns as strings
    for k in ("test_per_group_acc", "test_per_group_loss", "test_per_group_auroc"):
        if k in best:
            out[k] = best[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(MODULES))
    ap.add_argument("--base", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import importlib
    train = importlib.import_module(MODULES[args.dataset]).train
    base = yaml.safe_load(open(args.base or DEFAULT_BASE[args.dataset]))
    base.setdefault("data_split_seed", 43 if args.dataset == "fedheart" else 100)
    tag = args.tag or args.dataset
    out = args.out or f"runs/ablation_{args.dataset}_{tag}"
    os.makedirs(out, exist_ok=True)
    results = {}

    for label, shared, gdro, anch in CELLS:
        results[label] = {}
        for seed in args.seeds:
            cfg = copy.deepcopy(base)
            cfg["common_encoder"] = shared
            cfg["groupdro_enabled"] = gdro
            cfg["lambda_fit"] = anch
            cfg["lambda_sep"] = anch
            cfg["seed"] = seed
            cfg["run_dir"] = f"{out}/{label}_s{seed}"
            print(f"\n=== [{tag}] {label} seed={seed} ===", flush=True)
            try:
                train(cfg)
                m = extract_metrics(cfg["run_dir"]) or {}
                results[label][seed] = m
                print(f"  worst={m.get('test_worst_group_acc',float('nan')):.4f} "
                      f"overall={m.get('test_overall_acc',float('nan')):.4f} "
                      f"bal={m.get('test_balanced_acc',float('nan')):.4f}", flush=True)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                results[label][seed] = {}

    # aggregate a few headline metrics
    print(f"\n\n########## FULL ABLATION: {tag} ##########")
    hdr = ["worst", "overall", "balanced"]
    print(f"{'cell':>22} | worst-grp | overall | balanced | n")
    summary = {}
    for label, *_ in CELLS:
        def col(metric):
            vs = [results[label][s].get(metric, float("nan")) for s in args.seeds]
            return [v for v in vs if v == v]
        w = col("test_worst_group_acc"); o = col("test_overall_acc"); b = col("test_balanced_acc")
        if w:
            print(f"{label:>22} | {np.mean(w)*100:5.2f}±{np.std(w)*100:4.2f} | "
                  f"{np.mean(o)*100:5.2f} | {np.mean(b)*100:5.2f} | {len(w)}")
            summary[label] = {m: {"mean": float(np.mean(col(f"test_{m2}_acc"))),
                                  "std": float(np.std(col(f"test_{m2}_acc")))}
                              for m, m2 in [("worst", "worst_group"), ("overall", "overall"),
                                            ("balanced", "balanced")] if col(f"test_{m2}_acc")}
    json.dump({"summary": summary, "raw": results}, open(f"{out}/results.json", "w"), indent=2)
    print(f"\nwrote {out}/results.json")


if __name__ == "__main__":
    main()
