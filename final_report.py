"""The whole result set on one protocol, printed once.

Reads only metrics_long.csv files, so nothing here can drift from what the site shows. Every
comparison is against arms trained on the same rows and evaluated on the same rows; preflight
checks that before this runs.

Fed-Heart appears twice on purpose. The capped configuration holds Switzerland at 20 training
patients and the VA at 25, which is a scarcity manipulation we imposed; the uncapped one is
FLamby's standard protocol. Both are reported with their own matched baselines so the choice of
which is primary is a decision rather than an artefact of whichever happened to be run.

  python final_report.py
"""
from __future__ import annotations
import csv, os
from collections import defaultdict

import numpy as np
from scipy import stats

BASE = [
    # Primary families are the frozen protocol of 2026-09-18 (runs/final_*/PROTOCOL.md). The
    # capped Fed-Heart study and the old frozen-weight families stay as appendix material.
    ("Fed-Heart (uncapped, primary)", "runs/final_fedheart/metrics_long.csv",
     "runs/baselines_fedheart_uncapped/metrics_long.csv",
     {"erm": "ERM", "dro": "GroupDRO", "regret": "RegretDRO",
      "anc_dro": "Ours", "full": "Ours_Regret"}),
    ("NHANES (primary)", "runs/final_nhanes/metrics_long.csv",
     "runs/baselines_nhanes/metrics_long.csv",
     {"erm": "ERM", "dro": "GroupDRO", "regret": "RegretDRO",
      "anc_dro": "Ours_GDRO", "full": "Ours_Regret"}),
    ("EMBED (primary)", "runs/final_embed/metrics_long.csv",
     "runs/baselines_embed/metrics_long.csv",
     {"erm": "erm", "dro": "groupdro", "regret": "regret_only",
      "anc_dro": "align_only", "full": "ours"}),
    ("Fed-Heart (capped scarcity study, old weight schedule)", "runs/fedheart_cv/metrics_long.csv",
     "runs/baselines_fedheart/metrics_long.csv",
     {"erm": "ERM", "dro": "GroupDRO", "regret": "RegretDRO",
      "anc_dro": "Ours", "full": "Ours_Regret"}),
    ("Fed-Heart (uncapped, old weight schedule)", "runs/fedheart_uncapped/metrics_long.csv",
     "runs/baselines_fedheart_uncapped/metrics_long.csv",
     {"erm": "ERM", "dro": "GroupDRO", "regret": "RegretDRO",
      "anc_dro": "Ours", "full": "Ours_Regret"}),
    ("NHANES (old weight schedule)", "runs/matrix_nhanes_nested/metrics_long.csv",
     "runs/baselines_nhanes/metrics_long.csv",
     {"erm": "ERM", "dro": "GroupDRO", "regret": "RegretDRO",
      "anc_dro": "Ours_GDRO", "full": "Ours_Regret"}),
    ("EMBED (old weight schedule)", "runs/embed_fix_final/metrics_long.csv",
     "runs/baselines_embed/metrics_long.csv",
     {"erm": "erm", "dro": "groupdro", "regret": "regret_only",
      "anc_dro": "align_only", "full": "ours"}),
]
BASELINES = ["Reweigh", "FlexMoE", "REMIND"]


def read(p):
    d = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(p):
        return {}
    for r in csv.DictReader(open(p)):
        acc = r.get("accuracy") or r.get("acc")
        try:
            d[r["method"]][r["seed"]][r["group"]] = (
                float(acc), float(r["loss"]), float(r.get("excess_loss") or "nan"),
                int(float(r["n_params"])) if r.get("n_params") else 0)
        except (ValueError, KeyError, TypeError):
            continue
    return d


def series(d, m, i):
    if m not in d:
        return []
    if i == 0:
        return [min(v[0] for v in g.values()) * 100 for g in d[m].values() if g]
    return [max(v[i] for v in g.values()) for g in d[m].values() if g]


def par(d, m):
    for g in d.get(m, {}).values():
        for v in g.values():
            if v[3]:
                return v[3]
    return 0


def verdict(a, b, higher_better):
    if not a or not b:
        return "  n/a"
    n = min(len(a), len(b))
    _, p = stats.ttest_ind(a[:n], b[:n])
    better = (np.mean(a) > np.mean(b)) if higher_better else (np.mean(a) < np.mean(b))
    if p >= 0.05:
        return "  tie"
    return "  WIN" if better else " LOSS"


def main():
    tally = {"acc": [0, 0, 0], "loss": [0, 0, 0]}
    for name, op, bp, m in BASE:
        O, B = read(op), read(bp)
        if not O:
            print(f"\n{name}: not available yet"); continue
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        print(f"{'method':34} {'worst-grp':>10} {'worst loss':>11} {'params':>10}")
        rows = [("ERM, common features", m["erm"], O),
                ("Per-group + GroupDRO", m["dro"], O),
                ("Per-group + Regret-DRO", m["regret"], O),
                ("Per-group + anchors + GroupDRO", m["anc_dro"], O),
                ("Ours: + anchors + Regret", m["full"], O)]
        rows += [(f"{b} (baseline)", b, B) for b in BASELINES]
        for lab, key, src in rows:
            a = series(src, key, 0)
            if not a:
                continue
            print(f"{lab:34} {np.mean(a):6.2f}±{np.std(a):<3.1f} "
                  f"{np.mean(series(src, key, 1)):11.3f} {par(src, key):10,}")

        # the ablation ladder, paired because the arms share seeds
        print("\n  ablation (paired over seeds):")
        for a_key, b_key, lab in [(m["erm"], m["dro"], "per-group + GroupDRO vs ERM"),
                                  (m["dro"], m["regret"], "regret vs GroupDRO"),
                                  (m["dro"], m["anc_dro"], "anchors vs GroupDRO")]:
            if a_key not in O or b_key not in O:
                continue
            ks = sorted(set(O[a_key]) & set(O[b_key]))
            if len(ks) < 3:
                continue
            x = np.array([min(v[0] for v in O[a_key][s].values()) * 100 for s in ks])
            y = np.array([min(v[0] for v in O[b_key][s].values()) * 100 for s in ks])
            _, p = stats.ttest_rel(y, x)
            print(f"    {lab:36} {np.mean(y - x):+6.2f}  p={p:.4f}"
                  f"{'  significant' if p < 0.05 else ''}")

        # our full method against each baseline, both metrics
        print("\n  vs baselines (our full method):")
        for b in BASELINES:
            if b not in B:
                continue
            for i, (lab, hi, key) in enumerate([("worst-group acc", True, "acc"),
                                                ("worst-group loss", False, "loss")]):
                a = series(O, m["full"], 0 if i == 0 else 1)
                y = series(B, b, 0 if i == 0 else 1)
                v = verdict(a, y, hi)
                tally[key][0 if "WIN" in v else (2 if "LOSS" in v else 1)] += 1
                print(f"    {b:9} {lab:18} {np.mean(a):7.3f} vs {np.mean(y):7.3f} {v}")
    print(f"\n{'=' * 78}\nTALLY of our full method against the three baselines")
    for k, (w, t, l) in tally.items():
        print(f"  {k:5} win {w}  tie {t}  loss {l}")


if __name__ == "__main__":
    main()
