"""Figure for Xenia's item 7: what the anchors do to the latent geometry, as 2-Wasserstein distances.

Three small multiples, one per question, each with the same three arms in the same order so a
reader compares bars within a panel and never across scales (one axis per panel, no dual axes):
  groups apart?   W2 between groups within a class      -- alignment wants this SMALL
  classes apart?  W2 between classes within a group     -- discrimination wants this LARGE
  on target?      W2 from each (group, class) cloud to its anchor
Every W2 is divided by the mean squared latent norm, so shrinking the space cannot masquerade as
alignment (the anchors shrink it ~40x). Colours are the validated categorical slots 1-3 in fixed
order; bars carry direct labels so identity never rests on colour alone.

  python plot_latent_w2.py
"""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("runs/latent_w2_nhanes.json"))
ARMS = [("no anchors", "#2a78d6"), ("real anchors", "#eb6834"), ("random anchors", "#1baf7a")]
PANELS = [("w2_between_groups_same_class_normalised", "groups apart?\nW2 between groups, same class", "smaller = aligned"),
          ("w2_between_classes_same_group_normalised", "classes apart?\nW2 between classes, same group", "larger = separable"),
          ("w2_group_to_anchor_normalised", "on target?\nW2 from each (group, class) cloud to its anchor", "smaller = fitted")]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), facecolor=SURF)
for ax, (key, title, hint) in zip(axes, PANELS):
    ax.set_facecolor(SURF)
    vals = [R[a][key] for a, _ in ARMS]
    x = np.arange(len(ARMS))
    bars = ax.bar(x, vals, width=0.56, color=[c for _, c in ARMS], edgecolor=SURF, linewidth=2, zorder=3)
    for b, v in zip(bars, vals):                       # direct labels in text ink, not series colour
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([a for a, _ in ARMS], fontsize=8.5, color=INK)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.text(0.99, 0.97, hint, transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8, colors=INK2, length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_ylim(0, max(vals) * 1.22)
axes[0].set_ylabel("W2 / mean squared latent norm", fontsize=8.5, color=INK2)
fig.suptitle("NHANES: 2-Wasserstein geometry of the learnt latent space (test latents, 3 seeds, GroupDRO on)",
             fontsize=10, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
os.makedirs("figs/paper", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figs/paper/fig7_latent_w2.{ext}", dpi=180, bbox_inches="tight", facecolor=SURF)
print("  wrote figs/paper/fig7_latent_w2.png/.pdf")
