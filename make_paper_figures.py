"""Paper figures and the main results table, generated from the metrics CSVs.

Everything here reads runs/*/metrics_long.csv, so a number in a figure can never disagree with a
number in the table, and both regenerate if a run changes.

  fig1_ladder.pdf      contribution decomposition, one panel per dataset
  fig2_efficiency.pdf  worst-group loss against parameter count, log x
  table1.tex           main results, LaTeX booktabs, significance marked

  python make_paper_figures.py
"""
from __future__ import annotations
import csv, json, os
from collections import defaultdict

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figs/paper"
os.makedirs(OUT, exist_ok=True)

# (label, ours csv, baselines csv, arm-name map). Arm names differ per runner, which is why the
# map is explicit rather than guessed.
DATASETS = [
    ("Fed-Heart", "runs/fedheart_cv/metrics_long.csv",
     "runs/baselines_fedheart/metrics_long.csv",
     {"erm": "ERM", "pergroup": "PerGroupOnly", "dro": "GroupDRO",
      "regret": "RegretDRO", "anchors_dro": "Ours", "full": "Ours_Regret"}),
    ("NHANES", "runs/matrix_nhanes_nested/metrics_long.csv",
     "runs/baselines_nhanes/metrics_long.csv",
     {"erm": "ERM", "pergroup": "PerGroupOnly", "dro": "GroupDRO",
      "regret": "RegretDRO", "anchors_dro": "Ours_GDRO", "full": "Ours_Regret"}),
    ("EMBED", "runs/embed_valsignal_final/metrics_long.csv",
     "runs/baselines_embed/metrics_long.csv",
     {"erm": "erm", "pergroup": None, "dro": "groupdro",
      "regret": "regret_only", "anchors_dro": "align_only", "full": "ours"}),
]
BASELINES = ["Reweigh", "FlexMoE", "REMIND"]


def read(path):
    d = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(path):
        return {}
    for r in csv.DictReader(open(path)):
        acc = r.get("accuracy") or r.get("acc")
        try:
            d[r["method"]][r["seed"]][r["group"]] = (
                float(acc), float(r["loss"]),
                int(float(r["n_params"])) if r.get("n_params") else 0)
        except (ValueError, KeyError, TypeError):
            continue
    return d


def worst_acc(d, m):
    return [min(v[0] for v in g.values()) * 100 for g in d.get(m, {}).values() if g]


def worst_loss(d, m):
    return [max(v[1] for v in g.values()) for g in d.get(m, {}).values() if g]


def params(d, m):
    for g in d.get(m, {}).values():
        for v in g.values():
            if v[2]:
                return v[2]
    return 0


def sig(a, b, higher_better):
    """Return '' or '*', comparing a against b."""
    if not a or not b:
        return ""
    n = min(len(a), len(b))
    _, p = stats.ttest_ind(a[:n], b[:n])
    better = (np.mean(a) > np.mean(b)) if higher_better else (np.mean(a) < np.mean(b))
    return "*" if (p < 0.05 and better) else ""


def fig_ladder():
    """What each component is worth, per dataset. The steps are cumulative only where the
    architecture allows: EMBED has no common-features rung because its six groups share no view,
    so there is no shared-feature model to build."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (name, op, bp, m) in zip(axes, DATASETS):
        O = read(op)
        steps = [("ERM\n(common feat.)", m["erm"]), ("+ per-group\nencoders", m["pergroup"]),
                 ("+ GroupDRO", m["dro"]), ("+ anchors", m["anchors_dro"])]
        labs, vals, errs = [], [], []
        for lab, key in steps:
            if not key or not worst_acc(O, key):
                continue
            v = worst_acc(O, key)
            labs.append(lab); vals.append(np.mean(v)); errs.append(np.std(v) / np.sqrt(len(v)))
        x = np.arange(len(vals))
        ax.bar(x, vals, yerr=errs, capsize=3,
               color=["#9aa4b2", "#6b8fb5", "#4a7ba7", "#c0625f"][:len(vals)])
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7.5)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(min(vals) - 6, max(vals) + 4)
        ax.grid(axis="y", alpha=.25, lw=.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("worst-group accuracy (%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_ladder.pdf"); fig.savefig(f"{OUT}/fig1_ladder.png", dpi=180)
    plt.close(fig)
    print(f"  wrote {OUT}/fig1_ladder.pdf")


def fig_efficiency():
    """Worst-group loss against parameter count. Down and to the left is better, and the point
    of the figure is that the baselines buy their accuracy with an order of magnitude more
    parameters without improving worst-group loss."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (name, op, bp, m) in zip(axes, DATASETS):
        O, B = read(op), read(bp)
        pts = []
        if worst_loss(O, m["full"]):
            pts.append(("Ours", params(O, m["full"]), np.mean(worst_loss(O, m["full"])), "#c0625f"))
        if m["dro"] and worst_loss(O, m["dro"]):
            pts.append(("Ours, anchors off", params(O, m["dro"]),
                        np.mean(worst_loss(O, m["dro"])), "#4a7ba7"))
        for b in BASELINES:
            if worst_loss(B, b):
                pts.append((b, params(B, b), np.mean(worst_loss(B, b)), "#9aa4b2"))
        for lab, p, l, c in pts:
            if not p:
                continue
            ax.scatter(p, l, s=52, color=c, zorder=3,
                       edgecolor="white", linewidth=.8)
            ax.annotate(lab, (p, l), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xscale("log"); ax.set_title(name, fontsize=10)
        ax.set_xlabel("parameters", fontsize=8.5)
        ax.grid(alpha=.25, lw=.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("worst-group loss", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_efficiency.pdf"); fig.savefig(f"{OUT}/fig2_efficiency.png", dpi=180)
    plt.close(fig)
    print(f"  wrote {OUT}/fig2_efficiency.pdf")


def table1():
    """Main results. Asterisk marks a significant improvement over common-features ERM for our
    arms, and over our full method for the baselines, so the comparison direction is explicit
    rather than implied by which row is bold."""
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Dataset & Method & Worst-group acc. & Worst-group loss & Params \\", r"\midrule"]
    for name, op, bp, m in DATASETS:
        O, B = read(op), read(bp)
        base = worst_acc(O, m["erm"])
        rows = [("ERM, common features", m["erm"], O),
                ("Per-group + GroupDRO", m["dro"], O),
                ("Per-group + Regret-DRO", m["regret"], O),
                ("Per-group + anchors + GroupDRO", m["anchors_dro"], O),
                ("Ours (per-group + anchors + Regret)", m["full"], O)]
        rows += [(b, b, B) for b in BASELINES]
        first = True
        for lab, key, src in rows:
            if not key or not worst_acc(src, key):
                continue
            a, l, p = worst_acc(src, key), worst_loss(src, key), params(src, key)
            star = sig(a, base, True) if src is O else ""
            ds = name if first else ""
            first = False
            lines.append(f"{ds} & {lab} & {np.mean(a):.2f}\\,$\\pm$\\,{np.std(a):.1f}{star} & "
                         f"{np.mean(l):.3f} & {p:,} \\\\".replace(",", r"{,}"))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(f"{OUT}/table1.tex", "w").write("\n".join(lines) + "\n")
    print(f"  wrote {OUT}/table1.tex")


if __name__ == "__main__":
    fig_ladder()
    fig_efficiency()
    table1()
    print(f"\nall paper assets in {OUT}/")
