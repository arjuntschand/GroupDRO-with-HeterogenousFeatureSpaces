"""EMBED ablation runner — benchmark vs REMIND.

Runs the factorial ablation over our three method components:
    (A) encoder:   shared  vs  per-group (per-view-type)   [per_view_encoders]
    (B) anchors:   off     vs  on                          [lambda_fit/lambda_sep]
    (C) GroupDRO:  off     vs  on                          [groupdro_enabled]

plus the plain ERM control. Reports head / tail / overall accuracy (REMIND's metric)
per cell, averaged over seeds. Designed to run on the AWS GPU box after images are
downloaded. Start with --quick (subset + fewer epochs) to validate, then full.

Usage:
    python run_embed_ablation.py --cells core --seeds 42 1337 --epochs 20
    python run_embed_ablation.py --cells all --quick        # smoke on local subset
"""
import argparse, copy, json, os, time
import yaml

from dro_hetero_anchors.src.train_embed import train

BASE = "experiments/embed_base.yaml"

# name -> config overrides. anchors are "on" unless lambda_*=0.
CELLS = {
    # --- controls / single-component ---
    "erm_shared":        dict(per_view_encoders=False, groupdro_enabled=False, lambda_fit=0.0, lambda_sep=0.0),
    "gdro_shared":       dict(per_view_encoders=False, groupdro_enabled=True,  lambda_fit=0.0, lambda_sep=0.0),
    "anchors_shared":    dict(per_view_encoders=False, groupdro_enabled=False, lambda_fit=1e-3, lambda_sep=1e-3),
    # --- the anchor x GroupDRO synergy cell (our headline finding on tabular data) ---
    "anchors_gdro_shared": dict(per_view_encoders=False, groupdro_enabled=True, lambda_fit=1e-3, lambda_sep=1e-3),
    # --- per-group encoder arm ---
    "erm_pergroup":      dict(per_view_encoders=True,  groupdro_enabled=False, lambda_fit=0.0, lambda_sep=0.0),
    "gdro_pergroup":     dict(per_view_encoders=True,  groupdro_enabled=True,  lambda_fit=0.0, lambda_sep=0.0),
    # --- FULL method: per-group encoders + anchors + GroupDRO ---
    "full":              dict(per_view_encoders=True,  groupdro_enabled=True,  lambda_fit=1e-3, lambda_sep=1e-3),
}
CELL_SETS = {
    "core": ["erm_shared", "gdro_shared", "anchors_gdro_shared", "full"],
    "all": list(CELLS.keys()),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="core", choices=list(CELL_SETS) + ["*"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="local subset (require_local_images) + 2 epochs, for smoke testing")
    ap.add_argument("--out", default="runs/embed_ablation_results.json")
    args = ap.parse_args()

    with open(BASE) as f:
        base = yaml.safe_load(f)
    names = CELL_SETS.get(args.cells, list(CELLS.keys()))

    results = []
    for name in names:
        for seed in args.seeds:
            cfg = copy.deepcopy(base)
            cfg.update(CELLS[name])
            cfg["seed"] = seed
            cfg["run_name"] = f"embed_{name}_s{seed}"
            cfg["run_dir"] = f"runs/embed_ablation/{name}_s{seed}"
            if args.epochs is not None:
                cfg["epochs"] = args.epochs
            if args.quick:
                cfg["require_local_images"] = True
                cfg["epochs"] = 2
                cfg["stratified_batching"] = False
                cfg["min_group_size"] = 0   # keep the tiny local subset intact
            print(f"\n{'='*70}\n[cell={name} seed={seed}] "
                  f"per_view={cfg['per_view_encoders']} gdro={cfg['groupdro_enabled']} "
                  f"anchors={cfg['lambda_fit']>0}\n{'='*70}")
            t0 = time.time()
            out = train(cfg)
            results.append({"cell": name, "seed": seed, "minutes": round((time.time()-t0)/60, 1),
                            **out})
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} runs -> {args.out}")


if __name__ == "__main__":
    main()
