"""The two figures Xenia's spec asks for, built from the run CSVs.

fig_lambda_vs_rstar
    Her "One plot" section: R*_g on x, final lambda_g on y, one series per method. The claim
    being tested is that GroupDRO's weight tracks R*_g, because it reweights on raw loss and a
    group whose features are poor has a high loss it can never reduce, while regret subtracts
    that floor and so should not track it. A fitted slope and correlation make the comparison
    quantitative rather than a matter of eyeballing the scatter.

fig_loss_curves
    The per-group loss curves, fixed. The single-seed version oscillated by +/-0.1 epoch to
    epoch on NHANES, which is seed noise rather than anything about the method, and showed every
    group including the two that behave normally. This averages over seeds with a +/-1 SE band
    and keeps only the groups that carry the point.

  python plot_mechanism.py
"""
from __future__ import annotations
import ast, csv, glob, json, os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figs/paper"
os.makedirs(OUT, exist_ok=True)

DS = [
    ("Fed-Heart", "runs/fedheart_cv/{a}_s*_f0/metrics.csv", "runs/rstar_fedheart.json",
     ["Cleveland", "Hungarian", "Switzerland", "VA"], ["Switzerland", "VA"]),
    ("NHANES", "runs/matrix_nhanes_nested/{a}_s*/metrics.csv", "runs/rstar_nhanes_nested.json",
     ["survey only", "+ exam", "+ labs"], ["survey only", "+ labs"]),
]
ARMS = [("GroupDRO", "GroupDRO", "#4a7ba7", "o"), ("RegretDRO", "Regret-DRO", "#c0625f", "s")]


def rstar(path):
    d = json.load(open(path)); d = d.get("rstar", d)
    return [float(d[str(i)]) for i in range(len(d))]


def final_lambdas(pat, arm):
    """One final-lambda vector per seed."""
    out = []
    for f in sorted(glob.glob(pat.format(a=arm))):
        Q = [ast.literal_eval(r["groupdro_weights"]) for r in csv.DictReader(open(f))
             if r.get("groupdro_weights") not in (None, "None")]
        if Q:
            out.append(Q[-1])
    return np.array(out) if out else None


def curves(pat, arm, key):
    """Per-epoch per-group series, stacked over seeds and truncated to the shortest run."""
    series = []
    for f in sorted(glob.glob(pat.format(a=arm))):
        rows = list(csv.DictReader(open(f)))
        vals = [ast.literal_eval(r[key]) for r in rows if r.get(key) not in (None, "None")]
        if vals:
            series.append(np.array(vals))
    if not series:
        return None
    n = min(len(s) for s in series)
    return np.stack([s[:n] for s in series])           # (seeds, epochs, groups)


def fig_lambda_vs_rstar():
    fig, axes = plt.subplots(1, len(DS), figsize=(8.2, 3.5))
    for ax, (name, pat, rp, gnames, _) in zip(np.atleast_1d(axes), DS):
        R = rstar(rp)
        for arm, lab, colour, mk in ARMS:
            L = final_lambdas(pat, arm)
            if L is None:
                continue
            mu, se = L.mean(0), L.std(0) / np.sqrt(len(L))
            ax.errorbar(R, mu, yerr=se, fmt=mk, color=colour, ms=7, capsize=3,
                        lw=0, elinewidth=1.1, label=lab, zorder=3)
            # least squares fit; the slope is the quantity the claim is about
            if len(R) > 2:
                b, a = np.polyfit(R, mu, 1)
                xs = np.linspace(min(R), max(R), 20)
                ax.plot(xs, a + b * xs, color=colour, lw=1.2, alpha=.55, zorder=2)
                r = np.corrcoef(R, mu)[0, 1]
                ax.annotate(f"{lab}: slope {b:+.2f}, r {r:+.2f}",
                            xy=(.03, .93 if arm == "GroupDRO" else .84),
                            xycoords="axes fraction", fontsize=7.5, color=colour)
        for x, g in zip(R, gnames):
            ax.annotate(g, (x, 0), xytext=(0, -26), textcoords="offset points",
                        fontsize=6.5, ha="center", color="#6c7480", rotation=20)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(r"$R^*_g$  (group's achievable floor)", fontsize=8.5)
        ax.grid(alpha=.25, lw=.6); ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel(r"final group weight $\lambda_g$", fontsize=9)
    np.atleast_1d(axes)[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_lambda_vs_rstar.pdf")
    fig.savefig(f"{OUT}/fig3_lambda_vs_rstar.png", dpi=180)
    plt.close(fig)
    print(f"  wrote {OUT}/fig3_lambda_vs_rstar.pdf")


def fig_loss_curves():
    rows = []
    for name, pat, rp, gnames, keep in DS:
        for g in keep:
            rows.append((name, pat, rp, gnames, g))
    fig, axes = plt.subplots(1, len(rows), figsize=(3.0 * len(rows), 3.1), sharey=False)
    for ax, (name, pat, rp, gnames, gname) in zip(np.atleast_1d(axes), rows):
        gi = gnames.index(gname)
        R = rstar(rp)[gi]
        for key, lab, colour in [("test_per_group_loss", "test", "#c0625f"),
                                 ("train_per_group_loss", "train", "#d9a441")]:
            C = curves(pat, "RegretDRO", key)
            if C is None:
                continue
            y = C[:, :, gi]
            mu, se = y.mean(0), y.std(0) / np.sqrt(len(y))
            x = np.arange(len(mu))
            ax.plot(x, mu, color=colour, lw=1.5, label=f"{lab} loss", zorder=3)
            ax.fill_between(x, mu - se, mu + se, color=colour, alpha=.22, lw=0, zorder=2)
        ax.axhline(R, ls="--", lw=1.1, color="#6c7480", zorder=1,
                   label=rf"$R^*$ = {R:.2f}")
        ax.set_title(f"{name}: {gname}", fontsize=9)
        ax.set_xlabel("epoch", fontsize=8.5)
        ax.grid(alpha=.25, lw=.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("loss", fontsize=9)
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Per-group loss, mean over 10 seeds with $\\pm$1 SE", fontsize=9.5, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_loss_curves.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/fig4_loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/fig4_loss_curves.pdf")


if __name__ == "__main__":
    fig_lambda_vs_rstar()
    fig_loss_curves()
