"""Look at the latent space directly, instead of inferring the mechanism from accuracy.

The random-target control matched the real class anchors, which would mean class-conditional
alignment is not what makes the anchors work. Before reframing the contribution around that,
it is worth checking what the latent space actually does, because there are three very
different explanations and accuracy alone cannot separate them:

  (a) the anchors never separate, so "class-conditional" was never really happening;
  (b) the anchors separate but the encoders ignore them, so the fit loss acts only as a
      scale or variance constraint;
  (c) the anchors separate AND classes align to them, and the random control still works
      because with few classes a random partition is not a strong enough scramble.

Measured for each configuration:
  anchor_sep      mean pairwise distance between class anchors (are they distinct at all)
  class_sep       between-class over within-class latent scatter (Fisher-style ratio)
  group_align     how close the per-group latent means are to each other (lower = better
                  aligned, which is what the shared head needs)
  latent_scale    RMS latent norm (shows whether the loss is mainly acting as a scale cap)

Usage:
  python run_latent_diagnostic.py --dataset nhanes --base experiments/nhanes_disjoint_pergroup_gdro.yaml
"""
from __future__ import annotations
import argparse, copy, json
import numpy as np
import torch
import yaml


def latent_stats(z, y, g, anchors_m):
    """z (N,D) latents, y labels, g group ids, anchors_m (C,D) anchor means."""
    z = z.detach().cpu().numpy(); y = y.cpu().numpy(); g = g.cpu().numpy()
    out = {}
    # anchor separation
    A = anchors_m.detach().cpu().numpy()
    d = [np.linalg.norm(A[i] - A[j]) for i in range(len(A)) for j in range(i + 1, len(A))]
    out["anchor_sep"] = float(np.mean(d)) if d else float("nan")
    # class separation: between-class scatter / within-class scatter
    mu = z.mean(0)
    between, within, n = 0.0, 0.0, 0
    for c in np.unique(y):
        zc = z[y == c]
        if len(zc) < 2:
            continue
        between += len(zc) * np.sum((zc.mean(0) - mu) ** 2)
        within += np.sum((zc - zc.mean(0)) ** 2)
        n += len(zc)
    out["class_sep"] = float(between / max(within, 1e-9))
    # group alignment: mean pairwise distance between per-group latent means (lower better)
    gm = [z[g == k].mean(0) for k in np.unique(g) if (g == k).sum() > 1]
    dd = [np.linalg.norm(gm[i] - gm[j]) for i in range(len(gm)) for j in range(i + 1, len(gm))]
    out["group_align"] = float(np.mean(dd)) if dd else float("nan")
    out["latent_scale"] = float(np.sqrt((z ** 2).sum(1).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nhanes")
    ap.add_argument("--base", default="experiments/nhanes_disjoint_pergroup_gdro.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 7])
    ap.add_argument("--out", default="runs/latent_diagnostic.json")
    args = ap.parse_args()

    import importlib
    mod = importlib.import_module(
        "dro_hetero_anchors.src.train_nhanes" if args.dataset == "nhanes"
        else "dro_hetero_anchors.src.train_fedheart")
    base = yaml.safe_load(open(args.base))
    base.setdefault("data_split_seed", 100)

    ARMS = [("no anchors",        dict(lambda_fit=0.001, lambda_sep=0.001)),
            ("real anchors",      dict(lambda_fit=0.1,   lambda_sep=0.1)),
            ("random anchors",    dict(lambda_fit=0.1,   lambda_sep=0.1, random_anchor_targets=True)),
            ("fit only (real)",   dict(lambda_fit=0.1,   lambda_sep=0.0)),
            ("sep only",          dict(lambda_fit=0.0,   lambda_sep=0.1))]

    results = {}
    for name, over in ARMS:
        per_seed = []
        for seed in args.seeds:
            cfg = copy.deepcopy(base); cfg.update(over)
            cfg["seed"] = seed
            cfg["common_encoder"] = False
            cfg["groupdro_enabled"] = True
            cfg["run_dir"] = f"runs/latdiag/{name.replace(' ','_')}_s{seed}"
            cfg["return_latents"] = True          # honoured by the patched trainers
            try:
                r = mod.train(cfg)
            except Exception as e:
                print(f"  {name} s{seed} FAILED: {e}", flush=True); continue
            lat = r.get("final_latents")
            if lat is None:
                print(f"  {name} s{seed}: trainer returned no latents"); continue
            st = latent_stats(lat["z"], lat["y"], lat["g"], lat["anchor_m"])
            st["worst"] = r.get("best_worst_group_acc", float("nan")) * 100
            per_seed.append(st)
            print(f"  {name} s{seed}: " + " ".join(f"{k}={v:.3f}" for k, v in st.items()), flush=True)
        if per_seed:
            results[name] = {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]}

    print(f"\n{'configuration':>18} | anchor_sep | class_sep | group_align | latent_scale | worst")
    for name, r in results.items():
        print(f"{name:>18} | {r['anchor_sep']:10.3f} | {r['class_sep']:9.3f} | "
              f"{r['group_align']:11.3f} | {r['latent_scale']:12.3f} | {r['worst']:.2f}")
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
