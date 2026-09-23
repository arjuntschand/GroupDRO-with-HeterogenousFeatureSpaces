"""Paper figures from the protocol-v4 runs (runs/v4b tabular, runs/v4 EMBED), mean over 10 seeds.

  fig_dynamics_v4        main text: worst-group test loss per epoch, every method, one panel per dataset,
                         with the full method's group weights beneath; epoch 10 (the reported budget) marked
  fig_dynamics_v4_appx   appendix: per-group validation and test loss per epoch for the full method,
                         per-group GroupDRO and REMIND, all three datasets

Every curve is the mean over seeds of the run at the step size the fixed-10 view reports (validated on the
validation split); nothing here is chosen on test.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "figure.dpi": 200})
GROUPS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, DIM, RED, GREEN = "#141413", "#6b6b6b", "#b03a3a", "#1a7f4b"
OURS, ABL = "#1a7f4b", "#4a7ba7"
DS = [("nhanes", "runs/v4b", "NHANES", "Ours_Regret", {"Ours_Regret": ("Ours", OURS, "-", 1.8), "GroupDRO": ("Per-group GroupDRO", ABL, "-", 1.2),
                                                       "PerGroupOnly": ("Per-group ERM", "#9a9a94", "-", 1.0), "REMIND": ("REMIND", "#c0625f", "--", 1.3),
                                                       "Reweigh": ("Reweigh", "#e8a33d", "--", 1.1), "FlexMoE": ("Flex-MoE", "#8c6bb1", "--", 1.1)},
        ["G0 survey", "G1 + exam", "G2 + labs"]),
      ("fedheart", "runs/v4b", "Fed-Heart", "Ours_Regret", None, ["Cleveland", "Hungarian", "Switzerland", "VA"]),
      ("embed", "runs/v4", "EMBED", "ours", {"ours": ("Ours", OURS, "-", 1.8), "groupdro": ("Per-group GroupDRO", ABL, "-", 1.2),
                                              "erm": ("Per-group ERM", "#9a9a94", "-", 1.0), "REMIND": ("REMIND", "#c0625f", "--", 1.3),
                                              "Reweigh": ("Reweigh", "#e8a33d", "--", 1.1), "FlexMoE": ("Flex-MoE", "#8c6bb1", "--", 1.1)},
        ["g1", "g2", "g3", "g4", "g5", "g6"])]
DS[1] = (DS[1][0], DS[1][1], DS[1][2], DS[1][3], DS[0][4], DS[1][5])
BUDGET = {"nhanes": 10, "fedheart": 10, "embed": 10}


def load(root):
    return json.load(open(f"{root}/curves.json")), json.load(open(f"{root}/summary.json"))


def series(curves, summ, fam, method, metric, view="fixed10"):
    """Per-group series at the step size the view reports for that method: {group: [(epoch, value)]}."""
    st = summ[fam][view][method]["step"]
    key = next(k for k in curves[fam] if k.split("|")[0] == method and abs(float(k.split("|")[1]) - st) < 1e-9)
    out = {}
    for r in curves[fam][key]:
        if r[metric] == r[metric]:
            out.setdefault(r["group"], []).append((r["epoch"], r[metric]))
    return {g: sorted(v) for g, v in out.items()}


def worst(curves, summ, fam, method, metric):
    per = series(curves, summ, fam, method, metric)
    eps = sorted({e for v in per.values() for e, _ in v})
    return eps, [max(dict(v).get(e, np.nan) for v in per.values()) for e in eps]


def main_figure():
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), gridspec_kw={"height_ratios": [1.35, 1]})
    for j, (fam, root, title, full, arms, glabels) in enumerate(DS):
        curves, summ = load(root)
        ax = axes[0, j]
        for m, (lab, col, ls, lw) in arms.items():
            if not any(k.split("|")[0] == m for k in curves[fam]):
                continue
            eps, w = worst(curves, summ, fam, m, "test_loss")
            eps, w = zip(*[(e, v) for e, v in zip(eps, w) if e >= 1])
            ax.plot(eps, w, ls, color=col, lw=lw, label=lab, zorder=3 if m == full else 2)
        ax.axvline(BUDGET[fam], color=DIM, lw=0.7, ls=":", zorder=1)
        ax.text(BUDGET[fam] + 0.3, ax.get_ylim()[1] * 0.98, "reported\nbudget", fontsize=6, color=DIM, va="top")
        ax.set_title(title, fontweight="bold", loc="left")
        ax.set_ylabel("worst-group test loss" if j == 0 else "")
        ax.grid(alpha=0.25, lw=0.5)
        if fam == "embed":
            ax.set_ylim(0.8, 3.2)
        if j == 0:
            ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.0), handlelength=1.6)
        # weights
        ax2 = axes[1, j]
        per = series(curves, summ, fam, full, "weight")
        for gi, (g, v) in enumerate(sorted(per.items())):
            e, w = zip(*v)
            ax2.plot(e, w, color=GROUPS[gi % 6], lw=1.3, label=glabels[gi] if gi < len(glabels) else g)
        ax2.axvline(BUDGET[fam], color=DIM, lw=0.7, ls=":")
        ax2.set_ylim(0, 1); ax2.set_xlabel("epoch"); ax2.grid(alpha=0.25, lw=0.5)
        ax2.set_ylabel("group weight $\\lambda_g$ (ours)" if j == 0 else "")
        ax2.legend(frameon=False, ncol=2 if len(per) > 4 else 1, loc="upper left", handlelength=1.2, fontsize=6)
    fig.tight_layout(h_pad=0.8, w_pad=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(f"figs/paper/fig_dynamics_v4.{ext}", bbox_inches="tight")
    print("wrote figs/paper/fig_dynamics_v4.{png,pdf}")


def appendix_figure():
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.6))
    ARMS = [("full", "Ours"), ("gdro", "Per-group GroupDRO"), ("REMIND", "REMIND")]
    for i, (fam, root, title, full, arms, glabels) in enumerate(DS):
        curves, summ = load(root)
        keys = {"full": full, "gdro": "groupdro" if fam == "embed" else "GroupDRO", "REMIND": "REMIND"}
        for j, (k, lab) in enumerate(ARMS):
            ax = axes[i, j]
            va = series(curves, summ, fam, keys[k], "val_loss"); te = series(curves, summ, fam, keys[k], "test_loss")
            for gi, g in enumerate(sorted(va)):
                e, v = zip(*[(e_, v_) for e_, v_ in va[g] if e_ >= 1]); ax.plot(e, v, ":", color=GROUPS[gi % 6], lw=1.0)
                e, v = zip(*[(e_, v_) for e_, v_ in te[g] if e_ >= 1]); ax.plot(e, v, "-", color=GROUPS[gi % 6], lw=1.2, label=glabels[gi] if gi < len(glabels) else g)
            ax.axvline(BUDGET[fam], color=DIM, lw=0.7, ls=":")
            ax.set_title(f"{title}: {lab}", loc="left", fontsize=8)
            if fam == "embed":
                ax.set_ylim(0.3, 3.5)
            ax.grid(alpha=0.25, lw=0.5)
            if j == 0:
                ax.set_ylabel("per-group loss (solid: test, dotted: validation)")
            if i == 2:
                ax.set_xlabel("epoch")
            if j == 0:
                ax.legend(frameon=False, fontsize=6, ncol=2 if len(va) > 4 else 1, handlelength=1.2)
    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(f"figs/paper/fig_dynamics_v4_appx.{ext}", bbox_inches="tight")
    print("wrote figs/paper/fig_dynamics_v4_appx.{png,pdf}")


if __name__ == "__main__":
    main_figure(); appendix_figure()
