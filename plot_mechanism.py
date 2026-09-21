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
import sys
# --v3: draw from the corrected-pipeline families (review of 2026-09-20) and write *_v3 files, so
# the figures of the earlier pipeline are left as they are.
V3 = "--v3" in sys.argv
SUF = "_v3" if V3 else ""
os.makedirs(OUT, exist_ok=True)

# Fed-Heart appears as the UNCAPPED run: that is FLamby's standard protocol, our best arm
# beats all three baselines there, and the capped variant caps Switzerland and the VA at 20 and
# 25 patients, which is a scarcity manipulation we imposed. The capped dynamics are kept below
# as the scarcity study rather than as the headline. The uncapped and NHANES sources are the
# frozen-protocol families (1/G init, eq. 13 floors, held-out signal, gamma 0.1 per step).
DS = [
    ("Fed-Heart", "runs/gamma_sweep_fh10/g0.1_s1/{a}_s*_f0/metrics.csv", "runs/rstar_fedheart_eq13.json",
     ["Cleveland", "Hungarian", "Switzerland", "VA"], ["Switzerland", "VA"]),
    ("Fed-Heart capped", "runs/fedheart_cv/{a}_s*_f0/metrics.csv", "runs/rstar_fedheart.json",
     ["Cleveland", "Hungarian", "Switzerland", "VA"], ["Switzerland", "VA"]),
    ("NHANES", "runs/gamma_sweep_nh_val10/g0.1_s1/{a}_s*/metrics.csv", "runs/rstar_nhanes_eq13.json",
     ["survey only", "+ exam", "+ labs"], ["survey only", "+ labs"]),
]
if V3:
    DS = [("Fed-Heart", "runs/fedheart_v3/{a}_s*_f0/metrics.csv", "runs/rstar_v3/fedheart/fold0.json",
           ["Cleveland", "Hungarian", "Switzerland", "VA"], ["Switzerland", "VA"]),
          ("NHANES", "runs/nhanes_v3/{a}_s*/metrics.csv", "runs/rstar_v3/nhanes/nested.json",
           ["survey only", "+ exam", "+ labs"], ["survey only", "+ labs"])]
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
    fig.savefig(f"{OUT}/fig3_lambda_vs_rstar{SUF}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/fig3_lambda_vs_rstar{SUF}.png", dpi=180, bbox_inches="tight")
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
    fig.savefig(f"{OUT}/fig4_loss_curves{SUF}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/fig4_loss_curves{SUF}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/fig4_loss_curves.pdf")


def fig_full_dynamics():
    """Complete replacement for the single-seed per-run panels.

    Those showed every group and both arms but from one fold at seed 42, so the NHANES traces
    oscillated by +/-0.1 epoch to epoch and it was impossible to tell signal from seed noise.

    ONE FIGURE PER METHOD. The previous version stacked GroupDRO and Regret-DRO weights on the
    same axis, which put four bands and six lines in every panel and made the figure unreadable
    at page width. Each method now gets its own figure, so a panel carries one method's three
    loss curves against that group's R* and, underneath, that same method's weight. Comparing
    the two is then a matter of putting the figures side by side, which is what the eye is
    actually good at.
    """
    for name, pat, rp, gnames, _keep in DS:
        R = rstar(rp)
        ng = len(gnames)
        for arm, arm_lab, colour, _mk in ARMS:
            # skip an arm with no runs rather than emitting an empty figure
            if curves(pat, arm, "test_per_group_loss") is None:
                continue
            fig, axes = plt.subplots(2, ng, figsize=(2.75 * ng, 5.2), squeeze=False)
            for gi in range(ng):
                ax_l, ax_w = axes[0][gi], axes[1][gi]
                # train is memorised, test is what we report, val is what drives the lambda
                # update and selects the epoch, so all three carry distinct information.
                for key, ls, alpha, lab in [("train_per_group_loss", "--", .12, "train"),
                                            ("val_per_group_loss", ":", .12, "val"),
                                            ("test_per_group_loss", "-", .18, "test")]:
                    C = curves(pat, arm, key)
                    if C is None:
                        continue
                    y = C[:, :, gi]; mu = y.mean(0); se = y.std(0) / np.sqrt(len(y))
                    x = np.arange(len(mu))
                    ax_l.plot(x, mu, color=colour, lw=1.5, ls=ls, zorder=3, label=lab)
                    ax_l.fill_between(x, mu - se, mu + se, color=colour, alpha=alpha, lw=0)

                W = curves(pat, arm, "groupdro_weights")
                if W is not None:
                    y = W[:, :, gi]; mu = y.mean(0); se = y.std(0) / np.sqrt(len(y))
                    x = np.arange(len(mu))
                    ax_w.plot(x, mu, color=colour, lw=2.0, zorder=3)
                    ax_w.fill_between(x, mu - se, mu + se, color=colour, alpha=.22, lw=0)
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
            axes[0][0].legend(fontsize=6.5, frameon=False, ncol=3)
            # a shared loss axis makes the groups comparable, which is the point of the R* line
            lo = min(a.get_ylim()[0] for a in axes[0]); hi = max(a.get_ylim()[1] for a in axes[0])
            for a in axes[0]:
                a.set_ylim(lo, hi)
            fig.suptitle(f"{name} — {arm_lab}: per-group loss and group weight, "
                         r"mean of 10 seeds with $\pm$1 SE", fontsize=9.5, y=.98)
            fig.tight_layout(rect=(0, 0, 1, .96))
            slug = arm.lower().replace("dro", "dro")
            ds_slug = name.lower().replace('-', '').replace(' ', '_')
            stem = f"{OUT}/fig5_dynamics_{ds_slug}_{slug}{SUF}"
            fig.savefig(stem + ".pdf", bbox_inches="tight")
            fig.savefig(stem + ".png", dpi=170, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {stem}.pdf")


if __name__ == "__main__":
    # This block used to sit above fig_full_dynamics, so fig5 was defined but never
    # called and the script silently produced only figs 3 and 4.
    fig_lambda_vs_rstar()
    fig_loss_curves()
    fig_full_dynamics()
