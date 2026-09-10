"""Re-run the anchor mechanism control with the class-proportion confound removed.

The first version drew pseudo-labels uniformly with randint. On NHANES (roughly 90/10) that
silently changed the class proportions to 50/50, so it confounded two things: the class
structure being destroyed, and the per-class moments being estimated from different subset
sizes. Permuting the batch's real labels breaks the sample-to-class correspondence while
keeping every class count exactly as it was, which isolates the structure.

Runs real anchors against both control variants so the confound's size is visible.
"""
from __future__ import annotations
import argparse, copy, json
import numpy as np, yaml
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="experiments/nhanes_pergroup_gdro.yaml")
    ap.add_argument("--dataset", default="nhanes")
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[42, 1337, 7, 2024, 31337, 11, 22, 33, 44, 55])
    ap.add_argument("--out", default="runs/anchor_control_v2.json")
    args = ap.parse_args()

    import importlib
    mod = importlib.import_module(
        "dro_hetero_anchors.src.train_nhanes" if args.dataset == "nhanes"
        else "dro_hetero_anchors.src.train_fedheart")
    base = yaml.safe_load(open(args.base))
    base.setdefault("data_split_seed", 100 if args.dataset == "nhanes" else 43)

    ARMS = [("real",     dict()),
            ("permuted", dict(random_anchor_targets="permute")),
            ("randint",  dict(random_anchor_targets="randint"))]

    res = {}
    for name, over in ARMS:
        vals = {}
        for seed in args.seeds:
            cfg = copy.deepcopy(base)
            cfg.update(lambda_fit=0.1, lambda_sep=0.1, common_encoder=False,
                       groupdro_enabled=True, seed=seed,
                       run_dir=f"runs/anchorctl2/{name}_s{seed}")
            cfg.update(over)
            try:
                r = mod.train(cfg)
            except Exception as e:
                print(f"  {name} s{seed} FAILED: {e}", flush=True); continue
            vals[seed] = r.get("best_worst_group_acc", float("nan")) * 100
            print(f"  {name} s{seed}: worst={vals[seed]:.2f}", flush=True)
        res[name] = vals

    print(f"\n{'arm':>10} | worst-group")
    for k, v in res.items():
        a = np.array(list(v.values()))
        print(f"{k:>10} | {a.mean():.2f} ± {a.std(ddof=1):.2f}  (n={len(a)})")

    print("\nreal vs each control (paired over seeds):")
    for ctl in ["permuted", "randint"]:
        sd = sorted(set(res["real"]) & set(res.get(ctl, {})))
        if len(sd) < 3:
            continue
        x = np.array([res["real"][s] for s in sd]); y = np.array([res[ctl][s] for s in sd])
        t, p = stats.ttest_rel(x, y)
        verdict = "class structure MATTERS" if (p < 0.05 and x.mean() > y.mean()) \
                  else "cannot distinguish real from control"
        print(f"  real - {ctl:<9} = {x.mean()-y.mean():+.2f}  p={p:.4f}  "
              f"wins {int((x>y).sum())}/{len(sd)}  -> {verdict}")
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
