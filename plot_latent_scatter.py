"""The latent space as a picture: every test patient as a point, with and without the anchors.

Columns: no anchors | class anchors | randomly assigned anchors. Each column is that model's own
64-d latent space projected onto its first two principal components.
  top row     coloured by GROUP  -- are patients from different feature spaces mapped to the same
              region? Ringed markers are the group centroids.
  bottom row  the same points coloured by CLASS -- is the outcome still separable? Stars are the
              learnt anchor means (drawn only where the anchors are switched on).
Groups are subsampled to at most 600 points each so the 2,338-patient group does not bury the
two 530-patient groups; the subsample is seeded and the centroids use every point.

Needs the latents written by:  python run_latent_w2.py --seeds 42 --save-latents runs/latent_points
  python plot_latent_scatter.py
"""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

ARMS = [("no_anchors", "No anchors", False), ("real_anchors", "Class anchors (ours)", True),
        ("random_anchors", "Randomly assigned anchors (control)", True)]
GROUPS = ["G0  survey only", "G1  + exam", "G2  + labs"]
GCOL = ["#2a78d6", "#eb6834", "#1baf7a"]                  # validated categorical slots 1-3
NEG, POS = "#9a9a94", "#e87ba4"                           # outcome: muted context vs highlighted cases
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SEED, CAP = 42, 600
W2 = json.load(open("runs/latent_w2_nhanes.json")) if os.path.exists("runs/latent_w2_nhanes.json") else {}

def pca2(z):
    mu = z.mean(0); u, s, vt = np.linalg.svd(z - mu, full_matrices=False)
    return mu, vt[:2], (s[:2] ** 2) / (s ** 2).sum()

fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6), facecolor=SURF)
rng = np.random.default_rng(0)
for j, (key, title, show_anchor) in enumerate(ARMS):
    d = np.load(f"runs/latent_points/{key}_s{SEED}.npz")
    z, y, g, A = d["z"], d["y"], d["g"], d["anchor_m"]
    mu, P, var = pca2(z)
    xy = (z - mu) @ P.T; axy = (A - mu) @ P.T
    keep = np.concatenate([rng.choice(np.where(g == k)[0], size=min(CAP, (g == k).sum()), replace=False) for k in range(3)])
    lx = np.percentile(np.abs(xy[:, 0]), 99.5) * 1.10; ly = np.percentile(np.abs(xy[:, 1]), 99.5) * 1.35
    if show_anchor:                       # never clip an anchor out of its own panel
        lx = max(lx, np.abs(axy[:, 0]).max() * 1.25); ly = max(ly, np.abs(axy[:, 1]).max() * 1.25)
    for i in (0, 1):
        ax = axes[i][j]; ax.set_facecolor(SURF)
        if i == 0:
            for k in (2, 0, 1):                                   # largest group underneath
                m = keep[g[keep] == k]
                ax.scatter(xy[m, 0], xy[m, 1], s=7, color=GCOL[k], alpha=.38, linewidths=0, zorder=2)
            for k in range(3):
                c = xy[g == k].mean(0)
                ax.scatter(*c, s=150, color=GCOL[k], edgecolor=SURF, linewidth=2.2, zorder=5)
                ax.scatter(*c, s=150, facecolor="none", edgecolor=INK, linewidth=.9, zorder=6)
        else:
            m0, m1 = keep[y[keep] == 0], keep[y[keep] == 1]
            ax.scatter(xy[m0, 0], xy[m0, 1], s=7, color=NEG, alpha=.30, linewidths=0, zorder=2)
            ax.scatter(xy[m1, 0], xy[m1, 1], s=11, color=POS, alpha=.75, linewidths=0, zorder=3)
            if show_anchor:
                for c_, lab in enumerate(["anchor: no CVD", "anchor: CVD"]):
                    ax.scatter(*axy[c_], marker="*", s=330, color=[NEG, POS][c_], edgecolor=INK, linewidth=1.1, zorder=6)
                    ax.annotate(lab, axy[c_], xytext=(9, 9), textcoords="offset points", fontsize=8.5, color=INK, zorder=7,
                                path_effects=[pe.withStroke(linewidth=3, foreground=SURF)])
        ax.set_xlim(-lx, lx); ax.set_ylim(-ly, ly)
        ax.grid(color=GRID, lw=.7, zorder=0); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.tick_params(labelsize=7, colors=INK2, length=0)
        ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% of variance)", fontsize=8, color=INK2)
        if j == 0: ax.set_ylabel(("coloured by group\n" if i == 0 else "coloured by outcome\n") + f"PC2 ({var[1]*100:.0f}%)", fontsize=8.5, color=INK)
        else: ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)", fontsize=8, color=INK2)
    w = W2.get(key.replace("_", " "), {})
    note = (f"W₂ between groups {w['w2_between_groups_same_class_normalised']:.2f}  ·  between classes "
            f"{w['w2_between_classes_same_group_normalised']:.2f}") if w else ""
    axes[0][j].set_title(f"{title}\n", fontsize=11, color=INK, loc="left")
    axes[0][j].text(0, 1.02, note, transform=axes[0][j].transAxes, fontsize=8, color=INK2, va="bottom")
h1 = [Line2D([], [], marker="o", ls="", ms=6, color=GCOL[k], alpha=.8, label=GROUPS[k]) for k in range(3)]
h1 += [Line2D([], [], marker="o", ls="", ms=10, markerfacecolor="#ffffff", markeredgecolor=INK, label="group centroid")]
axes[0][0].legend(handles=h1, fontsize=8, frameon=False, loc="lower left")
h2 = [Line2D([], [], marker="o", ls="", ms=6, color=NEG, label="no CVD"), Line2D([], [], marker="o", ls="", ms=6, color=POS, label="CVD")]
axes[1][0].legend(handles=h2, fontsize=8, frameon=False, loc="lower left")
fig.suptitle("NHANES: where each test patient lands in the shared latent space. Three groups with 10, 13 and 20 features, each through "
             "its own encoder.\nW₂ values are scale-normalised means over 10 seeds; the points are one seed. Axes are scaled per panel "
             "(the anchors shrink the space about 40×).", fontsize=10.5, color=INK, x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .93), w_pad=2.0, h_pad=1.6)
os.makedirs("figs/paper", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figs/paper/fig11_latent_scatter.{ext}", dpi=170, bbox_inches="tight", facecolor=SURF)
print("  wrote figs/paper/fig11_latent_scatter.png/.pdf")
