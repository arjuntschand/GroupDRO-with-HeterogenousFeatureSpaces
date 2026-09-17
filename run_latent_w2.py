"""Item 7 of Xenia's list: a Wasserstein distance between the learnt class distributions in latent
space, on the paper's NHANES configuration.

runs/latent_diagnostic.json already reports "group_align", but that is a raw Euclidean distance
between per-group centroids, and the anchors shrink the whole latent space by ~46x, so most of the
apparent alignment there is scale collapse. This reports the 2-Wasserstein distance of Appendix G
between per-group, per-class Gaussians fitted to the test latents (diagonal, matching eq. 16 and the
alignment loss itself), and divides by the latent scale so the number is comparable across arms.

Two quantities, per arm, averaged over seeds:
  W2(nu_g^c, nu_h^c)   pairwise between groups within a class: are groups aligned to each other?
  W2(nu_g^c, mu_c)     each group-class cloud to its anchor: did the fit term do its job?
plus the class-separation ratio W2(within class, across groups) / W2(across classes) so we can see
whether the anchors separate classes or merely shrink everything.

  python run_latent_w2.py --seeds 42 1337 7
"""
from __future__ import annotations
import argparse, copy, importlib, json, os
import numpy as np, torch, yaml


def diag_w2(m1, s1, m2, s2):
    """eq. (16): W2^2 between diagonal Gaussians."""
    return float(((m1 - m2) ** 2).sum() + ((np.sqrt(s1) - np.sqrt(s2)) ** 2).sum())


def fit(z):
    return z.mean(0), z.var(0) + 1e-6


def stats(z, y, g, anchor_m, anchor_var=None):
    z, y, g, A = (np.asarray(v) for v in (z, y, g, anchor_m))
    scale2 = float((z ** 2).sum(1).mean())            # normaliser: mean squared latent norm
    C, G = int(y.max()) + 1, int(g.max()) + 1
    within, to_anchor, across = [], [], []
    fits = {}
    for c in range(C):
        for k in range(G):
            zz = z[(y == c) & (g == k)]
            if len(zz) >= 5:
                fits[(c, k)] = fit(zz)
    for c in range(C):
        ks = [k for k in range(G) if (c, k) in fits]
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                within.append(diag_w2(*fits[(c, ks[i])], *fits[(c, ks[j])]))
            m, s = fits[(c, ks[i])]
            av = anchor_var[c] if anchor_var is not None else s
            to_anchor.append(diag_w2(m, s, A[c], av))
    for k in range(G):
        cs = [c for c in range(C) if (c, k) in fits]
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                across.append(diag_w2(*fits[(cs[i], k)], *fits[(cs[j], k)]))
    out = {"latent_scale": float(np.sqrt(scale2)),
           "w2_between_groups_same_class": float(np.mean(within)) if within else float("nan"),
           "w2_group_to_anchor": float(np.mean(to_anchor)) if to_anchor else float("nan"),
           "w2_between_classes_same_group": float(np.mean(across)) if across else float("nan")}
    for k in list(out):
        if k.startswith("w2_"):
            out[k + "_normalised"] = out[k] / scale2
    out["class_over_group_ratio"] = (out["w2_between_classes_same_group"] /
                                     max(out["w2_between_groups_same_class"], 1e-12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="experiments/nhanes_pergroup_gdro.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 7])
    ap.add_argument("--out", default="runs/latent_w2_nhanes.json")
    args = ap.parse_args()
    mod = importlib.import_module("dro_hetero_anchors.src.train_nhanes")
    base = yaml.safe_load(open(args.base))
    # ANCHOR_ON / ANCHOR_OFF exactly as run_method_matrix.py uses for the paper arms, so
    # "real anchors" here IS the Ours_GDRO cell and "no anchors" IS the GroupDRO cell.
    ARMS = [("no anchors", dict(lambda_fit=0.001, lambda_sep=0.001)),
            ("real anchors", dict(lambda_fit=0.1, lambda_sep=0.1)),
            ("random anchors", dict(lambda_fit=0.1, lambda_sep=0.1, random_anchor_targets=True))]
    results = {}
    for name, over in ARMS:
        per = []
        for seed in args.seeds:
            cfg = copy.deepcopy(base); cfg.update(over)
            cfg["seed"] = seed; cfg["return_latents"] = True
            cfg["run_dir"] = f"runs/latent_w2/{name.replace(' ', '_')}_s{seed}"
            cfg["run_name"] = f"latent_w2_{name.replace(' ', '_')}_s{seed}"
            r = mod.train(cfg)
            lat = (r or {}).get("final_latents")
            if not lat:
                print(f"  {name} s{seed}: no latents"); continue
            st = stats(lat["z"], lat["y"], lat["g"], lat["anchor_m"])
            st["worst"] = float(r.get("test_worst_group_acc", r.get("worst", float("nan"))))
            per.append(st)
            print(f"  {name} s{seed}: " + " ".join(f"{k}={v:.4g}" for k, v in st.items()), flush=True)
        if per:
            results[name] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n{'arm':>16} | scale | W2 grp-grp (norm) | W2 grp->anchor (norm) | W2 cls-cls (norm) | cls/grp")
    for n, r in results.items():
        print(f"{n:>16} | {r['latent_scale']:5.2f} | {r['w2_between_groups_same_class_normalised']:17.4f} | "
              f"{r['w2_group_to_anchor_normalised']:21.4f} | {r['w2_between_classes_same_group_normalised']:17.4f} | "
              f"{r['class_over_group_ratio']:6.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
