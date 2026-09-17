"""Does the max player get enough time? Per-group held-out loss and group weight against epoch, the
paper's lambda setting beside the recommended one, for the full method on both tabular datasets.

Columns: Fed-Heart at gamma 0.02 per-epoch | Fed-Heart at gamma 0.5 per-step | NHANES at gamma 0.02
per-epoch | NHANES at gamma 0.5 per-step. Everything else is identical inside a dataset (1/G init,
eq. 13 floors, held-out signal). Top row: per-group validation loss with each group's floor dashed.
Bottom row: lambda_g. Dotted vertical line: the epoch that gets reported (validation-selected). If
lambda is still at 1/G when that line is reached, the DRO arm was scored as plain averaging.

  python plot_recommended_dynamics.py [--arm Ours_Regret]
"""
import argparse, ast, csv, glob, json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="Ours_Regret"); args = ap.parse_args()
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
FH_G = ["Cleveland", "Hungarian", "Switzerland", "VA"]; NH_G = ["G0 survey", "G1 +exam", "G2 +labs"]
def floors(p):
    d = json.load(open(p))["rstar"]; return [float(d[str(i)]) for i in range(len(d))]
PANELS = [("Fed-Heart · γ 0.02, per-epoch (paper's λ setting)", f"runs/gamma_sweep_fh/g0.02_s0/{args.arm}_s*_f0/metrics.csv", FH_G, floors("runs/rstar_fedheart_eq13.json")),
          ("Fed-Heart · γ 0.5, per-step", f"runs/gamma_sweep_fh/g0.5_s1/{args.arm}_s*_f0/metrics.csv", FH_G, floors("runs/rstar_fedheart_eq13.json")),
          ("NHANES · γ 0.02, per-epoch (paper's λ setting)", f"runs/gamma_sweep_nh_val/g0.02_s0/{args.arm}_s*/metrics.csv", NH_G, floors("runs/rstar_nhanes_eq13.json")),
          ("NHANES · γ 0.5, per-step", f"runs/gamma_sweep_nh_val/g0.5_s1/{args.arm}_s*/metrics.csv", NH_G, floors("runs/rstar_nhanes_eq13.json"))]

def load(pat):
    L, W, sel = [], [], []
    for f in sorted(glob.glob(pat)):
        rs = list(csv.DictReader(open(f)))
        l = [ast.literal_eval(r["val_per_group_loss"]) for r in rs if r.get("val_per_group_loss") not in (None, "None", "")]
        w = [ast.literal_eval(r["groupdro_weights"]) for r in rs if r.get("groupdro_weights") not in (None, "None", "")]
        va = []
        for r in rs:
            try: va.append(float(r["val_worst_group_acc"]))
            except (ValueError, KeyError, TypeError): va.append(-1.0)
        if l and w:
            L.append(np.array(l)); W.append(np.array(w)); sel.append(int(np.argmax(va)))
    n = min(min(len(x) for x in L), min(len(x) for x in W))
    return np.stack([x[:n] for x in L]).mean(0), np.stack([x[:n] for x in W]).mean(0), int(round(np.mean(sel))), len(L)

fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.2), facecolor=SURF)
for j, (title, pat, names, R) in enumerate(PANELS):
    L, W, sel, nseed = load(pat)
    x = np.arange(len(L))
    for i, (ax, Y, ylab) in enumerate([(axes[0][j], L, "held-out loss per group"), (axes[1][j], W, "group weight λ_g")]):
        ax.set_facecolor(SURF)
        for k, nm in enumerate(names):
            ax.plot(x, Y[:, k], color=COL[k], lw=2, label=nm)
            if i == 0:
                ax.axhline(R[k], color=COL[k], lw=1, ls="--", alpha=.55)
        ax.axvline(sel, color=INK2, ls=":", lw=1.2)
        ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
        ax.tick_params(labelsize=7.5, colors=INK2, length=0)
        if j in (0, 2): ax.set_ylabel(ylab, fontsize=8.5, color=INK)
        if i == 1:
            ax.set_ylim(0, 1); ax.axhline(1 / len(names), color=GRID, lw=1); ax.set_xlabel("epoch", fontsize=8, color=INK2)
            moved = float(np.abs(W[min(sel, len(W) - 1)] - W[0]).sum())
            ax.text(.98, .95, f"λ moved {moved:.2f} (L1) by the reported epoch", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
        else:
            ax.set_title(title, fontsize=9.5, color=INK, loc="left")
            ax.annotate(f"reported\nepoch {sel}", (sel, ax.get_ylim()[1]), xytext=(4, -4), textcoords="offset points", fontsize=7, color=INK2, va="top")
    axes[0][j].legend(fontsize=7, frameon=False, loc="upper right", ncol=2)
fig.suptitle(f"{args.arm}: held-out loss per group (dashed = that group's floor R̃) and group weight, mean of 3 seeds. "
             "Same model and floors in each pair; only the λ step size and refresh cadence differ.", fontsize=10, color=INK, x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .94))
os.makedirs("figs/paper", exist_ok=True)
stem = f"figs/paper/fig10_recommended_dynamics_{args.arm.lower()}"
for ext in ("png", "pdf"):
    fig.savefig(f"{stem}.{ext}", dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"  wrote {stem}.png/.pdf")
