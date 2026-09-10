"""Regenerate the per-dataset figures from the current metrics CSVs.

The committed figures under documentation/figures/ are stale: they show Fed-Heart's
Switzerland group near 98%, which came from the 10-test-patient split we replaced with
5-fold cross validation, and they include NHANES-disjoint, which has been cut. Rather than
put wrong numbers back on the site, these are drawn fresh from the same files the tables
read, so a figure can never disagree with the table above it.

Two per dataset:
  ladder     what each step of the pipeline is worth, common features -> per-group -> DRO
             -> anchors, with 95% CI error bars
  pergroup   per-group accuracy, common-features baseline against the full method
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_site import load, DATASETS, worst_by_seed

OUT = "site/figs"
INK, BASE, OURS = "#16181d", "#9aa1ac", "#b03a3a"

LADDER = [("common features\nERM",        ["ERM"]),
          ("+ per-group\nencoders",       ["PerGroupOnly", "erm"]),
          ("+ GroupDRO",                  ["GroupDRO", "groupdro"]),
          ("+ anchors\n(full method)",    ["Ours_GDRO", "Ours", "align_only"])]


def pick(data, aliases):
    return next((data[a] for a in aliases if a in data), None)


def ci95(v):
    v = np.asarray(v)
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def fig_ladder(data, label, path):
    xs, mus, errs = [], [], []
    for name, al in LADDER:
        b = pick(data, al)
        if not b:
            continue
        v = list(worst_by_seed(b).values())
        xs.append(name); mus.append(np.mean(v)); errs.append(ci95(v))
    if len(xs) < 2:
        return False
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    cols = [BASE] * (len(xs) - 1) + [OURS]
    ax.bar(range(len(xs)), mus, yerr=errs, capsize=4, color=cols,
           edgecolor="none", width=.62)
    for i, (m, e) in enumerate(zip(mus, errs)):
        ax.text(i, m + e + .5, f"{m:.1f}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=9)
    ax.set_ylabel("worst-group accuracy (%)", fontsize=10)
    ax.set_title(f"{label}: what each step is worth", fontsize=11, color=INK)
    lo = min(mus) - max(6, max(errs) * 2); ax.set_ylim(max(0, lo), max(mus) + max(errs) + 4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return True


def fig_pergroup(data, label, glegend, path):
    base, ours = pick(data, ["ERM"]), pick(data, ["Ours_GDRO", "Ours", "align_only"])
    if not (base and ours):
        return False
    gs = sorted({g for gr in base.values() for g in gr})
    def per(b, g):
        v = [gr[g]["acc"] for gr in b.values() if g in gr]
        return (np.mean(v), ci95(v)) if v else (np.nan, 0)
    bm = [per(base, g) for g in gs]; om = [per(ours, g) for g in gs]
    x = np.arange(len(gs)); w = .38
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(x - w/2, [m for m, _ in bm], w, yerr=[e for _, e in bm], capsize=3,
           label="common-features ERM", color=BASE, edgecolor="none")
    ax.bar(x + w/2, [m for m, _ in om], w, yerr=[e for _, e in om], capsize=3,
           label="full method", color=OURS, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n{glegend.get(g,'').split(',')[0][:16]}" for g in gs], fontsize=8.5)
    ax.set_ylabel("accuracy (%)", fontsize=10)
    ax.set_title(f"{label}: per-group accuracy", fontsize=11, color=INK)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.18)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return True


def main():
    os.makedirs(OUT, exist_ok=True)
    made = []
    for d in DATASETS:
        data = load(d["path"])
        if not data:
            continue
        for fn, f in [(f"{d['key']}_ladder.png", lambda p: fig_ladder(data, d["label"], p)),
                      (f"{d['key']}_pergroup.png",
                       lambda p: fig_pergroup(data, d["label"], d["groups"], p))]:
            if f(os.path.join(OUT, fn)):
                made.append(fn)
    print("wrote:", ", ".join(made) if made else "nothing")


if __name__ == "__main__":
    main()
