"""Every table for Xenia's 2026-09-17 request list, regenerated from runs/. Nothing typed by hand.

  1  sample size / features vs the estimated floor      (R* files across protocols)
  2  lambda init p_g vs 1/G                              (runs/rstar_tight vs runs/uniform_init)
  3  lambda update cadence                               (runs/dro_cadence)
  4  constant-predictor certificate, Appendix F Table 1  (eq13 R* files)
  5  NHANES sample-size swap, regimes A/B                (runs/matrix_nhanes_cap{A,B} + baselines)
  6  NHANES zero-overlap partition, REMIND vs ERM        (runs/matrix_nhanes_partition + baselines)
  7  latent 2-Wasserstein geometry                       (runs/latent_w2_nhanes.json)
  +  Appendix F Table 2, ordering certificate on nested pairs

  python xenia_checks.py
"""
from __future__ import annotations
import csv, json, os
from collections import defaultdict
import numpy as np

SEEDS3 = {"42", "1337", "7"}


def load(p):
    d = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(p):
        return d
    for r in csv.DictReader(open(p)):
        try:
            d[r["method"]][r["seed"]][r["group"]] = (float(r.get("accuracy") or r.get("acc")), float(r["loss"]))
        except (ValueError, TypeError, KeyError):
            pass
    return d


def nseeds(d, m):
    return len([1 for g in d.get(m, {}).values() if g])


def worst(d, m, seeds=None, i=0):
    v = [(min if i == 0 else max)(x[i] for x in g.values()) for s, g in d.get(m, {}).items()
         if g and (seeds is None or s in seeds)]
    return (np.mean(v) * 100 if i == 0 else np.mean(v)) if v else float("nan")


def rs(p, key="rstar"):
    if not os.path.exists(p):
        return None
    d = json.load(open(p)); d = d.get(key, d)
    return [round(float(v), 3) for _, v in sorted(d.items(), key=lambda kv: str(kv[0]))]


def sec(t):
    print(f"\n{'=' * 96}\n{t}\n{'=' * 96}")


def main():
    sec("1  Sample size and features vs the estimated floor")
    print("  Fed-Heart, same groups, only the protocol changed (Prop 2: the true floor is fixed)")
    for lab, p in [("capped 20/25, old estimator", "runs/rstar_fedheart_OLD.json"),
                   ("uncapped, per-group + joint", "runs/rstar_fedheart_v2.json"),
                   ("uncapped, pooled/tight", "runs/rstar_fedheart_tight.json"),
                   ("eq13 before margin", "runs/rstar_fedheart_eq13.json")]:
        v = rs(p, "rstar_before_margin" if "eq13" in p else "rstar")
        print(f"    {lab:34} {v}")
    print("  NHANES nested, features 10 -> 13 -> 20:", rs("runs/rstar_nhanes_eq13.json", "rstar_before_margin"))

    sec("2  lambda init: p_g vs 1/G   (Fed-Heart uncapped, 3 seeds x 5 folds, same tight R*)")
    A = json.load(open("runs/rstar_tight/results.json")) if os.path.exists("runs/rstar_tight/results.json") else {}
    B = json.load(open("runs/uniform_init/results.json")) if os.path.exists("runs/uniform_init/results.json") else {}
    m = lambda d, a: np.mean([s["worst"] * 100 for s in d.get(a, {}).values() if "worst" in s]) if d.get(a) else float("nan")
    print(f"  {'arm':14} {'p_g':>8} {'1/G':>8}")
    for a in ["PerGroupOnly", "GroupDRO", "RegretDRO", "Ours", "Ours_Regret"]:
        print(f"  {a:14} {m(A, a):8.2f} {m(B, a):8.2f}")

    sec("3  lambda update cadence  (Fed-Heart uncapped, 3 seeds x 5 folds)")
    C = {t: (json.load(open(f"runs/dro_cadence/cv_{t}/results.json")) if os.path.exists(f"runs/dro_cadence/cv_{t}/results.json") else {})
         for t in ["perepoch", "every4", "every1"]}
    print(f"  {'arm':14} {'per-epoch':>10} {'every 4':>10} {'every step':>11}")
    for a in ["GroupDRO", "RegretDRO", "Ours", "Ours_Regret"]:
        print(f"  {a:14} " + " ".join(f"{m(C[t], a):10.2f}" for t in C))

    sec("4  Appendix F Table 1: constant-predictor certificate (i)")
    for ds, p, names in [("Fed-Heart", "runs/rstar_fedheart_eq13.json", ["Cleveland", "Hungarian", "Switzerland", "VA"]),
                         ("NHANES", "runs/rstar_nhanes_eq13.json", ["G0 survey", "G1 +exam", "G2 +labs"])]:
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        fit = d.get("rstar_before_margin", d["rstar"]); const = d.get("constant_predictor_bound", {})
        chosen = d.get("chosen_estimator", {})
        print(f"  {ds:10} {'group':13} {'R^ (fit)':>9} {'R^const':>9} {'verdict':>28}   chosen")
        for k in sorted(fit):
            f_, c_ = float(fit[k]), float(const.get(k, float("nan")))
            # the fit-only estimate is what the certificate tests; if the constant won, the fit was worse
            fail = chosen.get(k, "").startswith("constant")
            print(f"  {'':10} {names[int(k)]:13} {f_:9.3f} {c_:9.3f} {'FAIL (fit worse than base rate)' if fail else 'pass':>28}   {chosen.get(k, '')}")

    sec("+  Appendix F Table 2: ordering certificate (ii), nested pairs")
    E = json.load(open("runs/embed_fix_final/rstar.json")) if os.path.exists("runs/embed_fix_final/rstar.json") else {}
    N = rs("runs/rstar_nhanes_eq13.json", "rstar_before_margin") or [float("nan")] * 3
    pairs = [("NHANES", "G0 in G1", N[0], N[1]), ("NHANES", "G1 in G2", N[1], N[2])]
    if E:
        pairs += [("EMBED", "g1 in g2", E["g1"], E["g2"]), ("EMBED", "g1 in g4", E["g1"], E["g4"]), ("EMBED", "g3 in g4", E["g3"], E["g4"]),
                  ("EMBED", "g2 in g5", E["g2"], E["g5"]), ("EMBED", "g4 in g6", E["g4"], E["g6"]), ("EMBED", "g5 in g6", E["g5"], E["g6"])]
    for ds, pr, a, b in pairs:
        print(f"  {ds:8} {pr:12} R^_g {a:.3f}  R^_h {b:.3f}   {'FAIL  spread(eps) >= %.3f' % (b - a) if b > a + 1e-6 else 'pass'}")

    sec("5  NHANES sample-size swap: A = G2 (richest) capped 500, B = G0 (poorest) capped 500.  worst-group acc, 3 seeds")
    F, FB = load("runs/matrix_nhanes_nested/metrics_long.csv"), load("runs/baselines_nhanes/metrics_long.csv")
    A_, AB = load("runs/matrix_nhanes_capA/metrics_long.csv"), load("runs/baselines_nhanes_capA/metrics_long.csv")
    B_, BB = load("runs/matrix_nhanes_capB/metrics_long.csv"), load("runs/baselines_nhanes_capB/metrics_long.csv")
    print(f"  {'arm':14} {'full':>8} {'A':>8} {'B':>8}     loss: {'full':>6} {'A':>6} {'B':>6}   n seeds full/A/B")
    for a, ext in [("ERM", 0), ("PerGroupOnly", 0), ("GroupDRO", 0), ("RegretDRO", 0), ("Ours_GDRO", 0), ("Ours_Regret", 0),
                   ("Reweigh", 1), ("FlexMoE", 1), ("REMIND", 1)]:
        f, x, y = (F, A_, B_) if ext == 0 else (FB, AB, BB)
        print(f"  {a:14} {worst(f, a, None):8.2f} {worst(x, a, None):8.2f} {worst(y, a, None):8.2f}           "
              f"{worst(f, a, None, 1):6.3f} {worst(x, a, None, 1):6.3f} {worst(y, a, None, 1):6.3f}   {nseeds(f, a)}/{nseeds(x, a)}/{nseeds(y, a)}")
    for lab, p in [("full", "runs/rstar_nhanes_eq13.json"), ("A", "runs/rstar_nhanes_capA_infoScarce.json"), ("B", "runs/rstar_nhanes_capB_poorScarce.json")]:
        print(f"  R^ before margin {lab:5} {rs(p, 'rstar_before_margin')}   c_g {rs(p, 'margin_c_g')}   R~ {rs(p)}")

    sec("6  NHANES zero-overlap partition (G0 survey / G1 body+2 labs / G2 BP+3 labs; no shared column).  worst-group acc, 3 seeds")
    P, PB = load("runs/matrix_nhanes_partition/metrics_long.csv"), load("runs/baselines_nhanes_partition/metrics_long.csv")
    print(f"  R^ before margin {rs('runs/rstar_nhanes_partition.json', 'rstar_before_margin')}   const bound {rs('runs/rstar_nhanes_partition.json', 'constant_predictor_bound')}   R~ {rs('runs/rstar_nhanes_partition.json')}")
    print(f"  {'arm':14} {'nested':>8} {'partition':>10}     loss: {'nested':>7} {'partition':>10}")
    for a, ext in [("ERM", 0), ("PerGroupOnly", 0), ("GroupDRO", 0), ("RegretDRO", 0), ("Ours_GDRO", 0), ("Ours_Regret", 0),
                   ("Reweigh", 1), ("FlexMoE", 1), ("REMIND", 1)]:
        f, x = (F, P) if ext == 0 else (FB, PB)
        print(f"  {a:14} {worst(f, a, None):8.2f} {worst(x, a, None):10.2f}           {worst(f, a, None, 1):7.3f} {worst(x, a, None, 1):10.3f}")
    # Collapse guard. NHANES is 89.5% negative, so a constant "no CVD" classifier scores ~89
    # worst-group accuracy. Report AUROC and class-balanced worst-group accuracy beside it and
    # flag any arm at AUROC ~0.5: that row is the base rate, not a result.
    def arms_level(d):
        while isinstance(d, dict) and "ERM" not in d:
            d = next(iter(d.values()))
        return d
    for lab, rp in [("nested", "runs/matrix_nhanes_nested/results.json"),
                    ("partition", "runs/matrix_nhanes_partition/results.json")]:
        if not os.path.exists(rp):
            continue
        Aj = arms_level(json.load(open(rp)))
        print(f"  {lab}: {'arm':13} {'AUROC':>6} {'per-group AUROC':>22} {'cls-bal worst':>14} {'worst acc':>10}")
        for a in ["ERM", "PerGroupOnly", "GroupDRO", "RegretDRO", "Ours_GDRO", "Ours_Regret"]:
            rows = [v for s_, v in Aj.get(a, {}).items() if str(s_) in SEEDS3]
            if not rows:
                continue
            au = np.mean([r.get("overall_auroc", np.nan) for r in rows])
            pga = np.round(np.mean([r.get("per_group_auroc", [np.nan] * 3) for r in rows], 0), 2)
            wb = np.mean([r.get("worst_group_bal_acc", np.nan) for r in rows]) * 100
            wa = np.mean([r.get("worst_group_acc", np.nan) for r in rows]) * 100
            flag = "   <-- AUROC 0.5: majority-class collapse, accuracy is the base rate" if au < 0.55 else ""
            print(f"  {'':10} {a:13} {au:6.3f} {str(pga):>22} {wb:14.1f} {wa:10.1f}{flag}")
    # Baseline CSVs carry no AUROC, so use the base rate itself as the collapse test: NHANES is
    # 89.5% negative, so any group whose accuracy sits within a point of 89.5 is being served the
    # majority label. Flag baselines with such groups on the partition.
    BASE_RATE = 89.5
    def per_group(d, m):
        arr = [[v[0] * 100 for _, v in sorted(g.items())] for s_, g in d.get(m, {}).items() if g]
        return np.mean(arr, 0) if arr else None
    for b in ["Reweigh", "FlexMoE", "REMIND"]:
        pg = per_group(PB, b)
        if pg is None:
            continue
        near = [i for i, v in enumerate(pg) if abs(v - BASE_RATE) < 1.5]
        flag = f"   <-- groups {near} at the 89.5% base rate: majority-class on those groups" if near else ""
        print(f"  partition {b:9} per-group acc {np.round(pg, 1)}{flag}")
    if PB and "REMIND" in PB and "ERM" in P:
        print(f"  REMIND - ERM on partition (worst-group acc): {worst(PB, 'REMIND', None) - worst(P, 'ERM', None):+.2f}"
              f"   -- read against the collapse flag above; compare loss and AUROC, not accuracy")

    sec("7  Latent 2-Wasserstein geometry (NHANES nested, eq16 W2 / mean squared latent norm, 3 seeds)")
    if os.path.exists("runs/latent_w2_nhanes.json"):
        R = json.load(open("runs/latent_w2_nhanes.json"))
        print(f"  {'arm':15} {'scale':>7} {'grp-grp':>9} {'cls-cls':>9} {'->anchor':>9} {'cls/grp':>8} {'worst%':>7}")
        for a in ["no anchors", "real anchors", "random anchors"]:
            r = R[a]
            print(f"  {a:15} {r['latent_scale']:7.3f} {r['w2_between_groups_same_class_normalised']:9.3f} "
                  f"{r['w2_between_classes_same_group_normalised']:9.3f} {r['w2_group_to_anchor_normalised']:9.3f} "
                  f"{r['class_over_group_ratio']:8.2f} {r.get('worst_val_selected', float('nan')):7.2f}")

    sec("Algorithm-1-conformant Fed-Heart (1/G init + eq13 R~), 3 seeds x 5 folds")
    G = json.load(open("runs/fedheart_alg1/results.json")) if os.path.exists("runs/fedheart_alg1/results.json") else {}
    print(f"  {'arm':14} {'p_g + old R*':>13} {'1/G + eq13 R~':>14}")
    for a in ["PerGroupOnly", "GroupDRO", "RegretDRO", "Ours", "Ours_Regret"]:
        print(f"  {a:14} {m(A, a):13.2f} {m(G, a):14.2f}")


if __name__ == "__main__":
    main()
