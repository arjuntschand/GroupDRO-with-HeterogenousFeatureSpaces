"""EMBED per-group validation loss and group weight against epoch, from runs/embed_dynamics
(seed 0, frozen protocol: uniform init, train-signal schedule, gamma 2.0). One figure per arm.

  python plot_embed_dynamics.py
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "runs/embed_dynamics"; OUT = "figs/paper"
RSTAR = json.load(open(f"{SRC}/rstar.json"))
GROUPS = ["g1", "g2", "g3", "g4", "g5", "g6"]
LABEL = {"g1": "g1 FFDM CC", "g2": "g2 C-View CC + FFDM CC", "g3": "g3 FFDM MLO",
         "g4": "g4 FFDM CC + MLO", "g5": "g5 C-View CC + FFDM CC + MLO", "g6": "g6 all four views"}
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

for arm, title in [("groupdro", "GroupDRO"), ("regret_only", "Regret-DRO"), ("ours", "ours: anchors + Regret-DRO")]:
    p = f"{SRC}/curve_{arm}_s0.json"
    if not os.path.exists(p):
        continue
    c = json.load(open(p))["curve"]; ep = [e["epoch"] for e in c]
    fig, ax = plt.subplots(2, 6, figsize=(16, 6.2), sharex=True)
    for j, g in enumerate(GROUPS):
        a = ax[0, j]; a.plot(ep, [e["per_group_loss"][g] for e in c], color=COL[j], lw=2)
        a.axhline(RSTAR[g], color="#555", ls="--", lw=1); a.set_title(f"{LABEL[g]}\nR*={RSTAR[g]:.2f}", fontsize=9)
        a.grid(alpha=.3); a.set_ylim(0, max(2.6, max(e["per_group_loss"][g] for e in c) * 1.05))
        b = ax[1, j]; b.plot(ep, [e["lambda"][j] for e in c], color=COL[j], lw=2); b.set_ylim(0, 1.02); b.grid(alpha=.3); b.set_xlabel("epoch")
    ax[0, 0].set_ylabel("validation loss"); ax[1, 0].set_ylabel("group weight λ_g")
    fig.suptitle(f"EMBED — {title}: per-group validation loss and group weight, seed 0 (uniform init, γ 2.0, train-signal schedule)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig5_dynamics_embed_{arm}.{ext}", dpi=150)
    plt.close(fig); print(f"  wrote {OUT}/fig5_dynamics_embed_{arm}.png/.pdf")
