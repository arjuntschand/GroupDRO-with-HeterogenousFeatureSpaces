"""Appendix E of the draft: log the lambda trajectory for every run; a trace that never leaves its
initialisation invalidates the run. This is that figure for the gamma x cadence sweep.

One column per step size gamma, GroupDRO on the top row and Regret-DRO underneath, per-step
refresh (the draft's cadence). Each panel: the four groups' weights against epoch, mean of 3 seeds
at fold 0, with the epoch that gets reported marked. The two objectives should pick DIFFERENT
groups once lambda moves (Section 3.3): raw loss goes to the highest-floor group, regret to the
group furthest above its own floor. Colours are the validated 4-slot categorical palette.

  python plot_gamma_sweep.py
"""
import csv, ast, glob, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--dataset", choices=["fedheart", "nhanes"], default="fedheart")
_args = _ap.parse_args()
G = [0.02, 0.1, 0.5, 2.0]
if _args.dataset == "fedheart":
    GROUPS = ["Cleveland", "Hungarian", "Switzerland", "VA"]
    PAT = "runs/gamma_sweep_fh/g{g}_s{stride}/{arm}_s*_f0/metrics.csv"
    TITLE = ("Fed-Heart: group weight trajectories by step size γ, per-step refresh, 1/G init, eq. 13 floors "
             "(mean of 3 seeds, fold 0). Flat at 0.25 = the max player never engaged.")
    INIT = 0.25; OUT = "fig8_gamma_lambda"
else:
    GROUPS = ["G0 survey", "G1 +exam", "G2 +labs"]
    PAT = "runs/gamma_sweep_nh/g{g}/{arm}_s*/metrics.csv"
    TITLE = ("NHANES: group weight trajectories by step size γ, train-batch signal every step, 1/G init, eq. 13 floors "
             "(mean of 3 seeds). Flat at 0.33 = the max player never engaged.")
    INIT = 1 / 3; OUT = "fig9_gamma_lambda_nhanes"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # validated slots 1-4
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
ARMS = [("GroupDRO", "GroupDRO (raw loss)"), ("RegretDRO", "Regret-DRO (excess over R~)")]

def traj(g, arm, stride=1):
    W, sel = [], []
    for f in sorted(glob.glob(PAT.format(g=g, stride=stride, arm=arm))):
        rs = list(csv.DictReader(open(f)))
        w = [ast.literal_eval(r["groupdro_weights"]) for r in rs if r.get("groupdro_weights") not in (None, "None", "")]
        va = []
        for r in rs:
            try: va.append(float(r["val_worst_group_acc"]))
            except (ValueError, KeyError, TypeError): va.append(-1.0)
        if w:
            W.append(np.array(w)); sel.append(int(np.argmax(va)) if va else len(w) - 1)
    if not W:
        return None, None
    n = min(len(x) for x in W)
    return np.stack([x[:n] for x in W]).mean(0), int(round(np.mean(sel)))

fig, axes = plt.subplots(2, len(G), figsize=(3.1 * len(G), 5.4), facecolor=SURF, sharey=True)
for j, g in enumerate(G):
    for i, (arm, lab) in enumerate(ARMS):
        ax = axes[i][j]; ax.set_facecolor(SURF)
        W, sel = traj(g, arm)
        if W is None:
            ax.text(.5, .5, "no runs", ha="center", transform=ax.transAxes); continue
        x = np.arange(len(W))
        for k, name in enumerate(GROUPS):
            ax.plot(x, W[:, k], color=COL[k], lw=2, label=name)
            ax.annotate(name, (x[-1], W[-1, k]), xytext=(3, 0), textcoords="offset points",
                        fontsize=6.5, color=INK2, va="center")
        ax.axvline(sel, color=INK2, ls=":", lw=1)
        ax.axhline(INIT, color=GRID, lw=1)
        ax.set_ylim(0, 1); ax.set_xlim(0, len(W) + 6)
        ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
        ax.tick_params(labelsize=7.5, colors=INK2, length=0)
        if i == 0: ax.set_title(f"γ = {g}", fontsize=10, color=INK)
        if j == 0: ax.set_ylabel(lab + "\nλ_g", fontsize=8.5, color=INK)
        if i == 1: ax.set_xlabel("epoch  (dotted = reported epoch)", fontsize=8, color=INK2)
axes[0][0].legend(fontsize=7, frameon=False, loc="upper left")
fig.suptitle(TITLE, fontsize=9.5, color=INK, x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .95))
os.makedirs("figs/paper", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figs/paper/{OUT}.{ext}", dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"  wrote figs/paper/{OUT}.png/.pdf")
