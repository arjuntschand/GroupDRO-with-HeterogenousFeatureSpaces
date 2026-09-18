"""Verify every arm on a dataset was measured the same way, before any table is built.

Three times now a comparison has silently mixed a method difference with a protocol difference:
Fed-Heart baselines on a single 185-patient split against our 925-patient CV; ten "seeds" that
were one initialisation repeated; and baselines training on 522 patients against our 449 after a
validation split was added on our side only. Each was found by eye, after the numbers were
reported. This checks the invariants instead.
"""
from __future__ import annotations
import csv, json, os, sys
from collections import defaultdict

DATASETS = {
    "fedheart": ("runs/final_fedheart/metrics_long.csv",
                 ["runs/baselines_fedheart_uncapped/metrics_long.csv",
                  "runs/baselines_fedheart_uncapped_matched/metrics_long.csv"],
                 "runs/rstar_fedheart_eq13.json"),
    "nhanes":   ("runs/final_nhanes/metrics_long.csv",
                 ["runs/baselines_nhanes/metrics_long.csv",
                  "runs/baselines_nhanes_matched/metrics_long.csv"],
                 "runs/rstar_nhanes_eq13.json"),
    "embed":    ("runs/final_embed/metrics_long.csv",
                 ["runs/baselines_embed/metrics_long.csv",
                  "runs/baselines_embed_remind128/metrics_long.csv",
                  "runs/baselines_embed_matched/metrics_long.csv"],
                 "runs/embed_xenia_production/rstar.json"),
}


def load(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    out = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        ak = "accuracy" if "accuracy" in r else "acc"
        try:
            out[r["method"]][r["seed"]][r["group"]] = (float(r[ak]), r.get("n"),
                                                       r.get("excess_loss"))
        except (ValueError, KeyError):
            continue
    return out


def check(name):
    ours_p, base_ps, rstar_p = DATASETS[name]
    fails = []
    O = load(ours_p)
    if O is None:
        return [f"{name}: {ours_p} missing"]
    srcs = [("ours", O)] + [(os.path.basename(os.path.dirname(p)), load(p)) for p in base_ps]
    srcs = [(n, d) for n, d in srcs if d]

    # 1. every arm evaluated on the same number of rows per group
    sizes = {}
    for sname, d in srcs:
        for m, seeds in d.items():
            for s, gs in seeds.items():
                for g, (_, n, _) in gs.items():
                    if n:
                        sizes.setdefault(g, {}).setdefault(str(n), []).append(f"{sname}/{m}")
    for g, byn in sorted(sizes.items()):
        if len(byn) > 1:
            detail = "; ".join(f"n={k}: {', '.join(sorted(set(v))[:3])}" for k, v in byn.items())
            fails.append(f"{name}/{g}: evaluation set differs -> {detail}")

    # 2. same seed set everywhere
    seedsets = {}
    for sname, d in srcs:
        for m, seeds in d.items():
            seedsets[f"{sname}/{m}"] = frozenset(seeds)
    uniq = set(seedsets.values())
    if len(uniq) > 1:
        big = max(uniq, key=len)
        odd = [k for k, v in seedsets.items() if v != big]
        fails.append(f"{name}: seed sets differ; {len(big)} seeds for most, odd ones: {odd[:4]}")

    # 3. excess must be signed everywhere, i.e. at least one source shows a negative
    #    (a source with zero negatives AND a zero minimum is the clamp signature)
    for sname, d in srcs:
        vals = [float(e) for m in d for s in d[m] for (_, _, e) in d[m][s].values()
                if e not in (None, "")]
        if vals and min(vals) == 0.0:
            fails.append(f"{name}/{sname}: excess_loss floors at exactly 0.0, looks clamped")

    # 4. R* identical across sources
    if os.path.exists(rstar_p):
        rs = json.load(open(rstar_p)); rs = rs.get("rstar", rs)
        want = {str(k).lstrip("g"): round(float(v), 4) for k, v in rs.items()}
        for p in base_ps:
            if not os.path.exists(p):
                continue
            got = {}
            for r in csv.DictReader(open(p)):
                if r.get("R_star"):
                    got[r["group"].lstrip("g")] = round(float(r["R_star"]), 4)
            bad = {k: (want.get(k), v) for k, v in got.items() if want.get(k) != v}
            if bad:
                fails.append(f"{name}/{os.path.basename(os.path.dirname(p))}: R* mismatch {bad}")
    return fails


def main():
    names = sys.argv[1:] or list(DATASETS)
    allf = []
    for n in names:
        f = check(n)
        allf += f
        print(f"{n:10} {'OK' if not f else str(len(f)) + ' PROBLEM(S)'}")
        for x in f:
            print(f"    - {x}")
    print("\n" + ("ALL CHECKS PASSED" if not allf else f"{len(allf)} problem(s) to fix before reporting"))
    return 1 if allf else 0


if __name__ == "__main__":
    sys.exit(main())
