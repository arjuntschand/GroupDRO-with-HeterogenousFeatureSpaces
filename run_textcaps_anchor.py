"""TextCaps anchor sweep (multimodal: visual + text/OCR groups) — the second
genuine-heterogeneity dataset to test whether anchors help worst-group.

Sweeps anchor weight lambda_fit=lambda_sep in {0.001 (~off), 0.1, 0.3} x seeds, holding
per-group modality encoders + GroupDRO fixed. Reads best worst-group acc + per-group acc
from each run's metrics.csv.

Run on the GPU box (image encoding). Usage:
  python run_textcaps_anchor.py --lams 0.001 0.1 0.3 --seeds 1337 42 7
"""
import argparse, copy, csv, json, os
import numpy as np
import yaml

BASE = "experiments/textcaps_multimodal_groupdro.yaml"


def read_best(run_dir):
    """Best (max) worst-group row from metrics.csv; return worst/overall/balanced/per_group."""
    f = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    if not rows:
        return None
    def col(keys):
        for k in rows[0]:
            if any(t in k.lower() for t in keys):
                return k
        return None
    wc = col(["worst_group_acc", "worst"])
    oc = col(["overall_acc", "test_acc"])
    bc = col(["balanced_acc", "balanced"])
    pc = col(["per_group_acc"])
    def fl(r, k):
        try: return float(r.get(k, "") or "nan")
        except: return float("nan")
    best = max(rows, key=lambda r: fl(r, wc) if wc else -1)
    return {"worst": fl(best, wc) if wc else float("nan"),
            "overall": fl(best, oc) if oc else float("nan"),
            "balanced": fl(best, bc) if bc else float("nan"),
            "per_group": best.get(pc, "") if pc else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", nargs="+", type=float, default=[0.001, 0.1, 0.3])
    ap.add_argument("--arms", nargs="+", default=None,
                    help="explicit 'fit,sep' pairs, e.g. 0.001,0.001 0.1,0 (fit-only)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 42, 7])
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=8, help="dataloader workers (GPU: 8)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--use-hf", action="store_true", help="use full HuggingFace dataset")
    ap.add_argument("--out", default="runs/textcaps_anchor")
    args = ap.parse_args()

    from dro_hetero_anchors.src.train_textcaps import train_textcaps
    base = yaml.safe_load(open(args.base))
    base["epochs"] = args.epochs
    base["textcaps_num_classes"] = args.num_classes
    base["textcaps_use_huggingface"] = bool(args.use_hf)
    base["num_workers"] = args.num_workers
    base["batch_size"] = args.batch_size
    os.makedirs(args.out, exist_ok=True)
    if args.arms:
        arms = []
        for a in args.arms:
            f, s = (float(x) for x in a.split(","))
            arms.append((f"fit{f}_sep{s}", f, s))
    else:
        arms = [(str(l), l, l) for l in args.lams]
    results = {}

    for label, lfit, lsep in arms:
        results[label] = {}
        for seed in args.seeds:
            cfg = copy.deepcopy(base)
            cfg["lambda_fit"] = lfit
            cfg["lambda_sep"] = lsep
            cfg["seed"] = seed
            cfg["run_dir"] = f"{args.out}/{label}_s{seed}"
            cfg["groupdro_enabled"] = True
            print(f"\n=== [textcaps] {label} (fit={lfit} sep={lsep}) seed={seed} ===", flush=True)
            try:
                train_textcaps(cfg)
                m = read_best(cfg["run_dir"]) or {}
                results[label][seed] = m
                print(f"  -> worst={m.get('worst', float('nan')):.4f} "
                      f"overall={m.get('overall', float('nan')):.4f} per_group={m.get('per_group','')[:40]}",
                      flush=True)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                results[label][seed] = {"worst": float("nan")}

    print("\n\n########## TEXTCAPS ANCHOR SWEEP ##########")
    print(f"{'arm':>16} | worst-group (mean±std) | overall | n")
    for label, _, _ in arms:
        w = [results[label][s].get("worst", float("nan")) for s in args.seeds]
        w = [x for x in w if x == x]
        o = [results[label][s].get("overall", float("nan")) for s in args.seeds]
        o = [x for x in o if x == x]
        if w:
            print(f"{label:>16} | {np.mean(w)*100:5.2f} ± {np.std(w)*100:4.2f}        | "
                  f"{np.mean(o)*100 if o else 0:5.2f} | {len(w)}")
    json.dump(results, open(f"{args.out}/results.json", "w"),
              default=lambda o: None, indent=2)
    print(f"\nwrote {args.out}/results.json")


if __name__ == "__main__":
    main()
