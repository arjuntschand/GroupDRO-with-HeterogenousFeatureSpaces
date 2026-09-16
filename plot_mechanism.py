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
    """Groups ordered left to right by how hard they intrinsically are, with each method's final
    weight as a bar.

    The scatter version of this was hard to read: R* on a numeric axis put Cleveland and
    Hungarian almost on top of each other, the fitted lines implied a trend on Fed-Heart where
    there is none, and the group names had to be rotated under the axis. Ordering the groups by
    R* and drawing weights as bars says the same thing directly. If GroupDRO chases groups it
    cannot help, its bars grow to the right. If regret does not, its bars stay level.
    """
    fig, axes = plt.subplots(1, len(DS), figsize=(9.5, 3.6))
    for ax, (name, pat, rp, gnames, _) in zip(np.atleast_1d(axes), DS):
        R = np.array(rstar(rp))
        order = np.argsort(R)                       # easiest group first
        x = np.arange(len(order)); w = 0.38
        for k, (arm, lab, colour, _mk) in enumerate(ARMS):
            L = final_lambdas(pat, arm)
            if L is None:
                continue
            mu, se = L.mean(0)[order], (L.std(0) / np.sqrt(len(L)))[order]
            ax.bar(x + (k - .5) * w, mu, w, yerr=se, capsize=2.5, label=lab,
                   color=colour, edgecolor="white", linewidth=.6, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{gnames[i]}\n$R^*$={R[i]:.2f}" for i in order], fontsize=7.5)
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", alpha=.25, lw=.6); ax.set_axisbelow(True)
        ax.set_ylim(0, None)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.annotate("harder for any model $\\rightarrow$", xy=(.5, -.28),
                    xycoords="axes fraction", ha="center", fontsize=7.5, color="#6c7480")
    np.atleast_1d(axes)[0].set_ylabel(r"final group weight $\lambda_g$", fontsize=9)
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_lambda_vs_rstar.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/fig3_lambda_vs_rstar.png", dpi=180, bbox_inches="tight")
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


def fig_full_dynamics():
    """Complete replacement for the single-seed per-run panels.

    Those showed every group and both arms but from one fold at seed 42, so the NHANES traces
    oscillated by +/-0.1 epoch to epoch and it was impossible to tell signal from seed noise.
    The smoothed figure that replaced them covered only two groups and only the regret arm, so it
    was a subset rather than a substitute. This is the full thing: every group, both arms, loss
    on the top row against that group's R*, and the group's DRO weight underneath, all averaged
    over 10 seeds with a one standard error band.
    """
    for name, pat, rp, gnames, _keep in DS:
        R = rstar(rp)
        ng = len(gnames)
        fig, axes = plt.subplots(2, ng, figsize=(2.75 * ng, 5.2), squeeze=False)
        for gi in range(ng):
            ax_l, ax_w = axes[0][gi], axes[1][gi]
            for arm, lab, colour, _mk in ARMS:
                # solid test, dashed train. The gap between them is the memorisation evidence:
                # on a group capped at 20 training samples the two separate immediately, and the
                # DRO weight underneath is driven by whichever one the max player reads.
                for key, ls, alpha, suffix in [("test_per_group_loss", "-", .20, " test"),
                                               ("train_per_group_loss", "--", .12, " train")]:
                    C = curves(pat, arm, key)
                    if C is None:
                        continue
                    y = C[:, :, gi]; mu = y.mean(0); se = y.std(0) / np.sqrt(len(y))
                    x = np.arange(len(mu))
                    wide = 2.6 if arm == "GroupDRO" else 1.2
                    ax_l.plot(x, mu, color=colour, lw=wide, ls=ls,
                              zorder=3 if arm == "GroupDRO" else 4,
                              alpha=.75 if arm == "GroupDRO" else 1.0,
                              label=lab + suffix)
                    ax_l.fill_between(x, mu - se, mu + se, color=colour, alpha=alpha, lw=0)
                W = curves(pat, arm, "groupdro_weights")
                if W is not None:
                    y = W[:, :, gi]; mu = y.mean(0); se = y.std(0) / np.sqrt(len(y))
                    x = np.arange(len(mu))
                    ax_w.plot(x, mu, color=colour,
                              lw=2.6 if arm == "GroupDRO" else 1.2,
                              zorder=3 if arm == "GroupDRO" else 4,
                              alpha=.75 if arm == "GroupDRO" else 1.0, label=lab)
                    ax_w.fill_between(x, mu - se, mu + se, color=colour, alpha=.20, lw=0)
            ax_l.axhline(R[gi], ls="--", lw=1.0, color="#6c7480", zorder=1)
            ax_l.set_title(f"{gnames[gi]}   $R^*$={R[gi]:.2f}", fontsize=8.5)
            ax_w.set_xlabel("epoch", fontsize=8)
            ax_w.set_ylim(0, 1)
            for ax in (ax_l, ax_w):
                ax.grid(alpha=.25, lw=.6); ax.set_axisbelow(True)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
            if gi:
                ax_l.set_yticklabels([]); ax_w.set_yticklabels([])
        axes[0][0].set_ylabel("loss", fontsize=9)
        axes[1][0].set_ylabel(r"group weight $\lambda_g$", fontsize=9)
        axes[0][0].legend(fontsize=6.5, frameon=False, ncol=2)
        # a shared loss axis makes the groups comparable, which is the point of the R* line
        lo = min(a.get_ylim()[0] for a in axes[0]); hi = max(a.get_ylim()[1] for a in axes[0])
        for a in axes[0]:
            a.set_ylim(lo, hi)
        fig.suptitle(f"{name}: per-group loss and DRO weight, mean of 10 seeds "
                     r"with $\pm$1 SE", fontsize=10, y=.98)
        fig.tight_layout(rect=(0, 0, 1, .96))
        stem = f"{OUT}/fig5_dynamics_{name.lower().replace('-','')}"
        fig.savefig(stem + ".pdf", bbox_inches="tight")
        fig.savefig(stem + ".png", dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {stem}.pdf")
