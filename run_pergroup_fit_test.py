"""Decisive test of the paper's central claim.

The write-up gives the anchor-fit loss twice, and the two are not equivalent:

  eq. 11-13 (Section 3.2, centralized): class moments POOLED over every group.
      Only the global class-c cloud is pulled to anchor c. No group is constrained
      individually, so cross-group alignment is at best a by-product.

  eq. 18 (Section 6.1, the GroupDRO setting we actually run): moments per (group, class).
      Each group's class-c cloud is pulled to anchor c separately. THIS is the operation
      that makes g1's class c land on g2's class c, which is the claim in Section 2.2.

Everything tabular was implemented as eq. 13. That matters because it changes what the
random-target control means. Under eq. 13, scrambling labels still leaves a loss that
collapses the latent space, so alignment survives and the control cannot discriminate.
Under eq. 18, scrambling should genuinely break it: group g1's "random class c" samples
and group g2's "random class c" samples have no reason to correspond to one another.

So: run real vs random anchors under BOTH forms. The claim predicts
    pooled:   real ~= random   (what we already observed)
    pergroup: real >> random   (the class-conditional structure is load-bearing)
If pergroup also shows real ~= random, the claim is genuinely unsupported and we say so.
"""
from __future__ import annotations
import argparse, copy, json
import numpy as np, yaml
from scipy import stats

from run_latent_diagnostic import latent_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="experiments/nhanes_disjoint_pergroup_gdro.yaml")
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[42, 1337, 7, 2024, 31337, 5, 99, 123, 777, 2718])
    ap.add_argument("--out", default="runs/pergroup_fit_test.json")
    args = ap.parse_args()

    import importlib
    mod = importlib.import_module("dro_hetero_anchors.src.train_nhanes")
    base = yaml.safe_load(open(args.base))
    base.setdefault("data_split_seed", 100)

    ARMS = []
    for fitform in ["pooled", "pergroup"]:
        for tgt in ["real", "random"]:
            ARMS.append((f"{fitform}/{tgt}",
                         dict(lambda_fit=0.1, lambda_sep=0.1,
                              per_group_fit=(fitform == "pergroup"),
                              random_anchor_targets=(tgt == "random"))))
    # reference point: anchors effectively off
    ARMS.append(("no anchors", dict(lambda_fit=0.001, lambda_sep=0.001)))

    res = {}
    for name, over in ARMS:
        worst, stats_acc = [], []
        for seed in args.seeds:
            cfg = copy.deepcopy(base); cfg.update(over)
            cfg.update(seed=seed, common_encoder=False, groupdro_enabled=True,
                       return_latents=True,
                       run_dir=f"runs/pgfit/{name.replace('/','_').replace(' ','_')}_s{seed}")
            try:
                r = mod.train(cfg)
            except Exception as e:
                print(f"  {name} s{seed} FAILED: {e}", flush=True); continue
            w = r.get("best_worst_group_acc", float("nan")) * 100
            worst.append(w)
            lat = r.get("final_latents")
            if lat is not None:
                stats_acc.append(latent_stats(lat["z"], lat["y"], lat["g"], lat["anchor_m"]))
            print(f"  {name} s{seed}: worst={w:.2f}", flush=True)
        if worst:
            entry = {"worst_mean": float(np.mean(worst)), "worst_std": float(np.std(worst)),
                     "worst_all": worst}
            if stats_acc:
                for k in stats_acc[0]:
                    entry[k] = float(np.mean([s[k] for s in stats_acc]))
            res[name] = entry

    print(f"\n{'arm':>18} | worst-group    | class_sep | group_align | latent_scale")
    for name, r in res.items():
        print(f"{name:>18} | {r['worst_mean']:6.2f}±{r['worst_std']:4.2f} | "
              f"{r.get('class_sep',float('nan')):9.3f} | {r.get('group_align',float('nan')):11.3f} | "
              f"{r.get('latent_scale',float('nan')):12.3f}")

    print("\nreal vs random (paired t-test over seeds):")
    for form in ["pooled", "pergroup"]:
        a, b = res.get(f"{form}/real"), res.get(f"{form}/random")
        if not (a and b):
            continue
        n = min(len(a["worst_all"]), len(b["worst_all"]))
        d = np.array(a["worst_all"][:n]) - np.array(b["worst_all"][:n])
        t, p = stats.ttest_rel(a["worst_all"][:n], b["worst_all"][:n])
        verdict = ("class structure MATTERS" if (p < 0.05 and d.mean() > 0)
                   else "cannot distinguish real from random")
        print(f"  {form:>9}: real-random = {d.mean():+.2f}  p={p:.4f}  -> {verdict}")
        res[f"{form}/_delta"] = {"mean": float(d.mean()), "p": float(p)}

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
