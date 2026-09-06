"""THE CRUX EXPERIMENT: does adding class-conditional Gaussian anchors at a REAL
weight improve worst-group accuracy over per-group GroupDRO (anchors off)?

The repo's headline configs set lambda_fit=lambda_sep=0.001 — essentially off — which
is why anchors never appeared to help. Here we sweep the anchor weight properly.

For each (anchor_weight lam in {0, 0.01, 0.1, 0.3, 1.0}) x (seed), hold everything
else fixed (per-group encoders + GroupDRO + fixed data_split_seed), set
lambda_fit=lambda_sep=lam, and record test worst-group accuracy.

Usage:
  python run_anchor_sweep.py --dataset fedheart
  python run_anchor_sweep.py --dataset nhanes_disjoint
"""
import argparse, copy, json, os, sys
import numpy as np
import yaml

BASES = {
    "fedheart": ("experiments/fedheart_ablation_pergroup_noanchor.yaml",
                 "dro_hetero_anchors.src.train_fedheart"),
    "nhanes_disjoint": ("experiments/nhanes_disjoint_pergroup_gdro.yaml",
                        "dro_hetero_anchors.src.train_nhanes"),
}
LAMS = [0.0, 0.01, 0.1, 0.3, 1.0]
SEEDS = [42, 1337, 7, 2024, 31337]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(BASES))
    ap.add_argument("--lams", nargs="+", type=float, default=LAMS,
                    help="diagonal sweep: sets lambda_fit=lambda_sep=lam")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="explicit 'fit,sep' pairs, e.g. 0.1,0 0,0.1 0.1,0.1 "
                         "(for decomposing which anchor loss drives the gain)")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # build list of (label, lambda_fit, lambda_sep)
    if args.arms:
        arms = []
        for a in args.arms:
            f, s = (float(x) for x in a.split(","))
            arms.append((f"fit{f}_sep{s}", f, s))
    else:
        arms = [(str(l), l, l) for l in args.lams]

    base_path, mod = BASES[args.dataset]
    import importlib
    train = importlib.import_module(mod).train
    base = yaml.safe_load(open(base_path))
    # ensure a FIXED split so seeds only vary model init/subsampling
    base.setdefault("data_split_seed", 43 if args.dataset == "fedheart" else 100)

    out = args.out or f"runs/anchor_sweep_{args.dataset}"
    os.makedirs(out, exist_ok=True)
    results = {}   # label -> {seed -> metrics}

    for label, lfit, lsep in arms:
        results[label] = {}
        for seed in args.seeds:
            cfg = copy.deepcopy(base)
            cfg["lambda_fit"] = lfit
            cfg["lambda_sep"] = lsep
            cfg["seed"] = seed
            cfg["run_dir"] = f"{out}/{label}_s{seed}"
            cfg["groupdro_enabled"] = True         # per-group GDRO always on
            print(f"\n=== [{args.dataset}] {label} (fit={lfit} sep={lsep}) seed={seed} ===", flush=True)
            try:
                r = train(cfg)
                results[label][seed] = {
                    "worst": float(r.get("best_worst_group_acc", float("nan"))),
                    "balanced": float(r.get("best_balanced_acc", float("nan"))),
                }
                print(f"  -> worst={results[label][seed]['worst']:.4f} "
                      f"balanced={results[label][seed]['balanced']:.4f}", flush=True)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                results[label][seed] = {"worst": float("nan"), "balanced": float("nan")}

    # aggregate
    print(f"\n\n########## ANCHOR SWEEP: {args.dataset} ##########")
    print(f"{'arm':>14} | worst-group acc (mean±std) | balanced (mean±std) | n")
    summary = {}
    for label, _, _ in arms:
        w = [results[label][s]["worst"] for s in args.seeds if not np.isnan(results[label][s]["worst"])]
        b = [results[label][s]["balanced"] for s in args.seeds if not np.isnan(results[label][s]["balanced"])]
        if w:
            print(f"{label:>14} | {np.mean(w)*100:5.2f} ± {np.std(w)*100:4.2f}          | "
                  f"{np.mean(b)*100:5.2f} ± {np.std(b)*100:4.2f}     | {len(w)}")
            summary[label] = {"worst_mean": np.mean(w), "worst_std": np.std(w),
                              "bal_mean": np.mean(b), "bal_std": np.std(b), "n": len(w),
                              "worst_raw": w}
    json.dump({"summary": summary, "raw": results, "arms": [list(a) for a in arms]},
              open(f"{out}/results.json", "w"), indent=2)
    print(f"\nwrote {out}/results.json")


if __name__ == "__main__":
    main()
