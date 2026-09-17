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

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--dataset", choices=["nhanes", "fedheart", "embed"], default="nhanes")
_ap.add_argument("--columns", type=int, default=3, help="2 drops the random-anchor control (main-text version)")
ARGS = _ap.parse_args()
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]      # categorical slots 1-6, fixed order
RAMP4 = ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"]                            # ordinal outcome: one hue, light -> dark
NEG, POS = "#9a9a94", "#e87ba4"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
DS = {
  "nhanes":   dict(dir="runs/latent_points", seed=42, cap=600, w2="runs/latent_w2_nhanes.json",
                   arms=[("no_anchors", "No anchors"), ("real_anchors", "Class anchors (ours)"), ("random_anchors", "Randomly assigned anchors (control)")],
                   groups=["G0  survey only", "G1  + exam", "G2  + labs"], classes=["no CVD", "CVD"], ordinal=False,
                   title="NHANES: where each test patient lands in the shared latent space. Three groups with 10, 13 and 20 features, each through its own encoder."),
  "fedheart": dict(dir="runs/latent_points_fedheart", seed=42, cap=10**9, w2=None,
                   arms=[("no_anchors", "No anchors"), ("real_anchors", "Class anchors (ours)"), ("random_anchors", "Randomly assigned anchors (control)")],
                   groups=["Cleveland", "Hungarian", "Switzerland", "VA"], classes=["no disease", "heart disease"], ordinal=False,
                   title="Fed-Heart: the 150 test patients of one split in the shared latent space. Four hospitals, each through its own encoder."),
  "embed":    dict(dir="runs/latent_points_embed", seed=0, cap=500, w2=None,
                   arms=[("groupdro", "No anchors"), ("align_only", "Class anchors (ours)"), ("rand_anchor", "Randomly assigned anchors (control)")],
                   groups=["g1 {FFDM CC}", "g2 {C-View CC, FFDM CC}", "g3 {FFDM MLO}", "g4 {FFDM CC, MLO}", "g5 {C-View CC, FFDM CC+MLO}", "g6 all four views"],
                   classes=["density A", "density B", "density C", "density D"], ordinal=True,
                   title="EMBED: test exams in the shared latent space. Six view-availability groups, each through its own encoder over frozen ViT features."),
}
D = DS[ARGS.dataset]
ARMS = [(k, t, k not in ("no_anchors", "groupdro")) for k, t in D["arms"]][:ARGS.columns]
GROUPS, GCOL = D["groups"], SLOTS[:len(D["groups"])]
CCOL = RAMP4 if D["ordinal"] else [NEG, POS]
SEED, CAP = D["seed"], D["cap"]
W2 = json.load(open(D["w2"])) if D["w2"] and os.path.exists(D["w2"]) else {}

def w2_stats(z, y, g):
    """Scale-normalised eq.-16 W2 between groups (same class) and between classes (same group), from these points."""
    s2 = float((z ** 2).sum(1).mean()); fits = {}
    for c in np.unique(y):
        for k in np.unique(g):
            zz = z[(y == c) & (g == k)]
            if len(zz) >= 5: fits[(c, k)] = (zz.mean(0), zz.var(0) + 1e-6)
    d = lambda a, b: float(((a[0] - b[0]) ** 2).sum() + ((np.sqrt(a[1]) - np.sqrt(b[1])) ** 2).sum())
    gg = [d(fits[a], fits[b]) for a in fits for b in fits if a[0] == b[0] and a[1] < b[1]]
    cc = [d(fits[a], fits[b]) for a in fits for b in fits if a[1] == b[1] and a[0] < b[0]]
    return (np.mean(gg) / s2 if gg else np.nan, np.mean(cc) / s2 if cc else np.nan)

def pca2(z):
    mu = z.mean(0); u, s, vt = np.linalg.svd(z - mu, full_matrices=False)
    return mu, vt[:2], (s[:2] ** 2) / (s ** 2).sum()

NC = len(ARMS)
fig, axes = plt.subplots(2, NC, figsize=(4.85 * NC, 7.6), facecolor=SURF, squeeze=False)
rng = np.random.default_rng(0)
for j, (key, title, show_anchor) in enumerate(ARMS):
    d = np.load(f"{D['dir']}/{key}_s{SEED}.npz")
    z, y, g, A = d["z"], d["y"], d["g"], d["anchor_m"]
    mu, P, var = pca2(z)
    xy = (z - mu) @ P.T; axy = (A - mu) @ P.T
    NG = len(GROUPS)
    keep = np.concatenate([rng.choice(np.where(g == k)[0], size=min(CAP, (g == k).sum()), replace=False) for k in range(NG) if (g == k).any()])
    big = int(np.argmax(np.bincount(g, minlength=NG))); order = [big] + [k for k in range(NG) if k != big]
    ms = 22 if len(z) < 400 else 7
    lx = np.percentile(np.abs(xy[:, 0]), 99.5) * 1.10; ly = np.percentile(np.abs(xy[:, 1]), 99.5) * 1.35
    if show_anchor:                       # never clip an anchor out of its own panel
        lx = max(lx, np.abs(axy[:, 0]).max() * 1.25); ly = max(ly, np.abs(axy[:, 1]).max() * 1.25)
    for i in (0, 1):
        ax = axes[i][j]; ax.set_facecolor(SURF)
        if i == 0:
            for k in order:                                        # largest group underneath
                m = keep[g[keep] == k]
                ax.scatter(xy[m, 0], xy[m, 1], s=ms, color=GCOL[k], alpha=.45 if ms > 7 else .38, linewidths=0, zorder=2)
            for k in range(NG):
                if not (g == k).any():
                    continue
                c = xy[g == k].mean(0)
                ax.scatter(*c, s=150, color=GCOL[k], edgecolor=SURF, linewidth=2.2, zorder=5)
                ax.scatter(*c, s=150, facecolor="none", edgecolor=INK, linewidth=.9, zorder=6)
        else:
            ncls = len(D["classes"])
            for c_ in ([0] + list(range(1, ncls)) if not D["ordinal"] else range(ncls)):
                mc = keep[y[keep] == c_]
                hi = (not D["ordinal"]) and c_ == 1
                ax.scatter(xy[mc, 0], xy[mc, 1], s=(ms * 1.6 if hi else ms), color=CCOL[c_], alpha=(.75 if hi else (.55 if D["ordinal"] else .30)), linewidths=0, zorder=(3 if hi else 2))
            if show_anchor:
                spread_a = max(np.ptp(axy[:len(D["classes"]), 0]), np.ptp(axy[:len(D["classes"]), 1]))
                coincident = spread_a < 0.08 * 2 * lx
                for c_, lab in enumerate(D["classes"]):
                    ax.scatter(*axy[c_], marker="*", s=330, color=CCOL[c_], edgecolor=INK, linewidth=1.1, zorder=6)
                    if not coincident:
                        ax.annotate("anchor: " + lab, axy[c_], xytext=(9, 9), textcoords="offset points", fontsize=8.5, color=INK, zorder=7,
                                    path_effects=[pe.withStroke(linewidth=3, foreground=SURF)])
                if coincident:
                    ax.annotate(f"all {len(D['classes'])} class anchors (nearly coincident at this scale)", axy[:len(D["classes"])].mean(0),
                                xytext=(14, 16), textcoords="offset points", fontsize=8.5, color=INK, zorder=7,
                                path_effects=[pe.withStroke(linewidth=3, foreground=SURF)])
        ax.set_xlim(-lx, lx); ax.set_ylim(-ly, ly)
        ax.grid(color=GRID, lw=.7, zorder=0); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.tick_params(labelsize=7, colors=INK2, length=0)
        ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% of variance)", fontsize=8, color=INK2)
        if j == 0: ax.set_ylabel(("coloured by group\n" if i == 0 else "coloured by outcome\n") + f"PC2 ({var[1]*100:.0f}%)", fontsize=8.5, color=INK)
        else: ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)", fontsize=8, color=INK2)
    w = W2.get(key.replace("_", " "), {})
    if w:
        bg, bc, src = w["w2_between_groups_same_class_normalised"], w["w2_between_classes_same_group_normalised"], "10 seeds"
    else:
        bg, bc = w2_stats(z, y, g); src = "this seed"
    note = f"W₂ between groups {bg:.2f}  ·  between classes {bc:.2f}  ({src})"
    axes[0][j].set_title(f"{title}\n", fontsize=11, color=INK, loc="left")
    axes[0][j].text(0, 1.02, note, transform=axes[0][j].transAxes, fontsize=8, color=INK2, va="bottom")
h1 = [Line2D([], [], marker="o", ls="", ms=6, color=GCOL[k], alpha=.8, label=GROUPS[k]) for k in range(len(GROUPS))]
h1 += [Line2D([], [], marker="o", ls="", ms=10, markerfacecolor="#ffffff", markeredgecolor=INK, label="group centroid")]
axes[0][0].legend(handles=h1, fontsize=7.5 if len(GROUPS) > 4 else 8, frameon=False, loc="lower left")
h2 = [Line2D([], [], marker="o", ls="", ms=6, color=CCOL[c_], label=D["classes"][c_]) for c_ in range(len(D["classes"]))]
axes[1][0].legend(handles=h2, fontsize=8, frameon=False, loc="lower left")
fig.suptitle(D["title"] + "\nW₂ values are scale-normalised. Axes are scaled per panel." + ("" if ARGS.dataset == "embed" else " The anchors shrink the latent space by an order of magnitude or more.") + "",
             fontsize=10.5, color=INK, x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .93), w_pad=2.0, h_pad=1.6)
os.makedirs("figs/paper", exist_ok=True)
STEM = "fig11_latent_scatter" + ("" if ARGS.dataset == "nhanes" else "_" + ARGS.dataset) + ("" if ARGS.columns == 3 else "_main")
for ext in ("png", "pdf"):
    fig.savefig(f"figs/paper/{STEM}.{ext}", dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"  wrote figs/paper/{STEM}.png/.pdf")
