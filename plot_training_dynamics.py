"""Per-group training-dynamics plots: loss vs epoch with R*, and lambda vs epoch.

Requested by Xenia: for each configuration (anchors + per-group + GroupDRO, and
anchors + per-group + regret-DRO), one plot per group showing

  (a) that group's loss over training, with its reference loss R*_g drawn as a
      horizontal line, so you can see whether the group is above or below its floor and
      when it crosses
  (b) that group's DRO weight lambda_g over training, so you can see how the max player
      redistributes attention

Everything needed is already written to metrics.csv by the trainers: `epoch`,
`test_per_group_loss` and `groupdro_weights`. Nothing has to be retrained to draw these.

  python plot_training_dynamics.py --run runs/matrix_nhanes_nested/Ours_GDRO_s42 \
      --rstar runs/rstar_nhanes_nested.json --tag nhanes_gdro
"""
from __future__ import annotations
import argparse, ast, csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, LOSS, LAM, REF = "#16181d", "#b03a3a", "#1d6f8b", "#6b7280"


def read(run):
    rows = list(csv.DictReader(open(os.path.join(run, "metrics.csv"))))
    ep, loss, lam = [], [], []
    for r in rows:
        try:
            ep.append(int(float(r["epoch"])))
        except Exception:
            continue
        loss.append(ast.literal_eval(r.get("test_per_group_loss") or "[]"))
        # Prefer the epoch MEAN. `groupdro_weights` is q after the final batch, and in
        # softmax mode q is overwritten from that batch alone; a partial last batch holding
        # one group produces an exact one-hot, which made GroupDRO look permanently collapsed
        # when the weights actually sit near uniform all through training.
        w = r.get("groupdro_weights_mean") or r.get("groupdro_weights") or ""
        lam.append(ast.literal_eval(w) if w and w != "None" else None)
    return ep, loss, lam


def load_rstar(path, n):
    if not path or not os.path.exists(path):
        return [None] * n
    raw = json.load(open(path))
    raw = raw.get("rstar", raw)
    return [float(raw[str(i)]) if str(i) in raw else float(raw.get(i, "nan"))
            for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rstar", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--names", nargs="*", default=None, help="group display names")
    ap.add_argument("--out", default="site/figs/dynamics")
    args = ap.parse_args()

    ep, loss, lam = read(args.run)
    if not ep:
        print(f"  no epoch rows in {args.run}"); return
    G = max(len(l) for l in loss if l)
    rstar = load_rstar(args.rstar, G)
    names = args.names or [f"g{i}" for i in range(G)]
    os.makedirs(args.out, exist_ok=True)

    # one row of loss plots, one row of lambda plots, one column per group
    have_lam = any(l is not None for l in lam)
    nrow = 2 if have_lam else 1
    fig, axes = plt.subplots(nrow, G, figsize=(3.5 * G, 3.2 * nrow), squeeze=False)

    for gi in range(G):
        ax = axes[0][gi]
        ys = [l[gi] if gi < len(l) else None for l in loss]
        xs = [e for e, y in zip(ep, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        ax.plot(xs, ys, color=LOSS, lw=1.6, label="group loss")
        if rstar[gi] == rstar[gi] and rstar[gi] is not None:
            ax.axhline(rstar[gi], color=REF, ls="--", lw=1.4,
                       label=f"R* = {rstar[gi]:.3f}")
        ax.set_title(f"{names[gi]}", fontsize=11, color=INK)
        ax.set_xlabel("epoch", fontsize=9); ax.set_ylabel("loss", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.18)

        if have_lam:
            ax2 = axes[1][gi]
            yl = [(l[gi] if l and gi < len(l) else None) for l in lam]
            x2 = [e for e, y in zip(ep, yl) if y is not None]
            y2 = [y for y in yl if y is not None]
            ax2.plot(x2, y2, color=LAM, lw=1.6)
            ax2.set_ylim(-0.04, 1.04)
            ax2.set_xlabel("epoch", fontsize=9)
            ax2.set_ylabel("group weight  lambda", fontsize=9)
            ax2.spines[["top", "right"]].set_visible(False); ax2.grid(alpha=.18)
            if y2:
                ax2.annotate(f"final {y2[-1]:.3f}", (x2[-1], y2[-1]), fontsize=8.5,
                             color=INK, xytext=(-52, 6), textcoords="offset points")

    fig.suptitle(args.tag.replace("_", " "), fontsize=12, color=INK, y=1.0)
    fig.tight_layout()
    p = os.path.join(args.out, f"{args.tag}.png")
    fig.savefig(p, dpi=135, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}   ({G} groups, {len(ep)} epochs, lambda={'yes' if have_lam else 'no'})")


if __name__ == "__main__":
    main()
