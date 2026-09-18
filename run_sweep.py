"""Equal-budget hyperparameter sweep for our method and the baselines.

Two rules make this usable in the paper rather than just flattering.

Equal budget. Every method gets the same number of configurations. Sweeping our own method
hard while leaving the baselines at one setting produces a gap that measures search effort, not
method quality, and a reviewer comparing our grid against REMIND's Table 16 would see it
immediately.

Selection off test. Configurations are chosen by mean validation worst-group accuracy across
seeds, never by test. Picking the best of N configurations on test, on top of picking the best
of ~100 epochs on test, compounds into a number that will not reproduce. Sweep runs therefore
set val_frac and live under runs/sweep_*, so the canonical test-selected results are untouched
and nothing here silently redefines the headline tables.

Reported per method: the winning configuration, its validation score, and the test score at that
configuration. The gap between those two is itself worth reporting, since it says how much of a
sweep's apparent gain is selection noise.

  python run_sweep.py --dataset nhanes
  python run_sweep.py --dataset fedheart --folds 5
"""
from __future__ import annotations
import argparse, copy, csv, itertools, json, os, shutil, traceback

import numpy as np
import yaml

SEEDS = [42, 1337, 7]
N_CONFIGS = 12                      # identical for every method, ours included


def our_grid_frozen():
    """Frozen-protocol grid (2026-09-18): the step size is fixed at the selected gamma 0.1 per
    step (its own sweep is reported separately), so the equal budget goes to the anchor weight
    and the latent width."""
    g = list(itertools.product([0.01, 0.05, 0.1, 0.3], [32, 64, 128]))
    return [{"lambda_fit": a, "lambda_sep": a, "latent_dim": d} for a, d in g][:N_CONFIGS]


def our_grid():
    """Ours: the anchor weight, the DRO step size, and the latent width.

    lambda_fit and lambda_sep move together. They are separate knobs in the loss, but sweeping
    them independently would quadruple the grid and hand our method a budget advantage over the
    baselines, which is the thing this script exists to avoid.
    """
    g = list(itertools.product([0.01, 0.05, 0.1, 0.3], [0.5, 1.0, 2.0], [64]))
    return [{"lambda_fit": a, "lambda_sep": a, "groupdro_eta": e, "latent_dim": d}
            for a, e, d in g][:N_CONFIGS]


def baseline_grid(method):
    """Same count for each baseline. REMIND's own Table 16 sweeps the expert count over
    {32, 64, 128}, so that axis is theirs rather than something we invented for them."""
    if method == "Reweigh":
        combos = list(itertools.product([1e-3, 5e-4, 2e-3, 1e-4], [32, 64, 128]))
        return [{"lr": lr, "latent_dim": d} for lr, d in combos][:N_CONFIGS]
    combos = list(itertools.product([4, 8, 16, 32], [1e-3, 5e-4, 2e-3]))
    return [{"n_experts": e, "lr": lr} for e, lr in combos][:N_CONFIGS]


def read_scores(run_dir):
    """(best val worst-group acc, test worst-group acc at that same epoch)."""
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
    scored = [r for r in rows if fl(r, "val_worst_group_acc") == fl(r, "val_worst_group_acc")]
    if not scored:
        return None
    best = max(scored, key=lambda r: fl(r, "val_worst_group_acc"))
    return fl(best, "val_worst_group_acc"), fl(best, "test_worst_group_acc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nhanes", "fedheart"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--folds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="one config, one seed, few epochs")
    ap.add_argument("--base", default=None, help="config to sweep from (default: the paper base config)")
    ap.add_argument("--rstar", default=None, help="optimal-loss file (default: the paper R* file)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frozen", action="store_true",
                    help="frozen-protocol grid: anchor weight x latent width at the selected step size")
    args = ap.parse_args()

    import importlib
    mod = ("dro_hetero_anchors.src.train_nhanes" if args.dataset == "nhanes"
           else "dro_hetero_anchors.src.train_fedheart")
    train = importlib.import_module(mod).train
    base_path = args.base or ("experiments/nhanes_pergroup_gdro.yaml" if args.dataset == "nhanes"
                              else "experiments/fedheart_exp_paper_hetagg_gdro.yaml")
    base = yaml.safe_load(open(base_path))
    base["val_frac"] = base.get("val_frac") or 0.2   # keep the config's split when it has one
    if args.epochs:
        base["epochs"] = args.epochs
    if args.dataset == "nhanes":
        base.setdefault("data_split_seed", 100)

    out = args.out or f"runs/sweep_{args.dataset}"
    os.makedirs(out, exist_ok=True)
    seeds = args.seeds[:1] if args.smoke else args.seeds

    # (method label, config list, config->cfg patch)
    def ours_patch(cfg, hp):
        cfg.update(common_encoder=False, groupdro_enabled=True, use_regret=True, **hp)
        rp = args.rstar or (f"runs/rstar_{args.dataset}.json" if args.dataset == "fedheart"
                            else "runs/rstar_nhanes_nested.json")
        rs = json.load(open(rp))["rstar"]
        cfg["optimal_losses"] = [rs[str(i)] for i in range(len(rs))]
        return cfg

    _grid = our_grid_frozen() if args.frozen else our_grid()
    jobs = [("Ours_Regret", _grid[:1] if args.smoke else _grid, ours_patch)]

    results = {}
    for label, grid, patch in jobs:
        rows = []
        for ci, hp in enumerate(grid):
            vals, tests = [], []
            for seed in seeds:
                for fold in range(args.folds):
                    cfg = copy.deepcopy(base)
                    cfg = patch(cfg, hp)
                    cfg["seed"] = seed
                    if args.folds > 1:
                        cfg["data_split_seed"] = 1000 + fold
                        cfg["train_frac"] = 1.0 - 1.0 / args.folds
                    rd = f"{out}/{label}_c{ci}_s{seed}" + (f"_f{fold}" if args.folds > 1 else "")
                    cfg["run_dir"] = rd
                    sc = read_scores(rd)
                    if sc is None:
                        shutil.rmtree(rd, ignore_errors=True)   # never append to a stale header
                        try:
                            train(cfg)
                        except Exception:
                            traceback.print_exc()
                        sc = read_scores(rd)
                    if sc:
                        vals.append(sc[0]); tests.append(sc[1])
            if vals:
                rows.append({"config": hp, "val": float(np.mean(vals)),
                             "test": float(np.mean(tests)), "n": len(vals)})
                print(f"  [{label}] c{ci} {hp} val={np.mean(vals)*100:.2f} "
                      f"test={np.mean(tests)*100:.2f} (n={len(vals)})", flush=True)
        if rows:
            best = max(rows, key=lambda r: r["val"])
            results[label] = {"all": rows, "best": best}
            print(f"\n[{label}] chosen on VAL: {best['config']}  "
                  f"val={best['val']*100:.2f}  test={best['test']*100:.2f}\n", flush=True)

    json.dump(results, open(f"{out}/sweep.json", "w"), indent=2)
    print(f"wrote {out}/sweep.json")


if __name__ == "__main__":
    main()
