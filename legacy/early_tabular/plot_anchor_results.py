"""Generate the anchor-result figures from the sweep JSONs:
  1. Specificity: Δ worst-group (anchors on−off) across datasets/feature-modes.
  2. NHANES-disjoint anchor-weight sweep (worst-group vs λ).
  3. fit vs sep decomposition.
Reads whatever result JSONs are present; skips missing ones.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = "documentation/figures"
os.makedirs(OUT, exist_ok=True)


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def arm_worst(res, label):
    """mean±std worst-group (%) and raw list for an arm label in results.json['raw']."""
    raw = res["raw"].get(label)
    if raw is None:
        return None
    w = [raw[s]["worst"] for s in raw if raw[s]["worst"] == raw[s]["worst"]]
    return np.array(w) * 100 if w else None


def paired_delta(res, off_label, on_label):
    raw = res["raw"]
    if off_label not in raw or on_label not in raw:
        return None
    seeds = [s for s in raw[off_label] if s in raw[on_label]]
    off = np.array([raw[off_label][s]["worst"] for s in seeds]) * 100
    on = np.array([raw[on_label][s]["worst"] for s in seeds]) * 100
    m = np.isfinite(off) & np.isfinite(on)
    off, on = off[m], on[m]
    if len(off) < 2:
        return None
    d = on - off
    try:
        _, p = stats.ttest_rel(on, off)
    except Exception:
        p = np.nan
    return dict(delta=d.mean(), std=d.std(ddof=1), n=len(d), p=p,
               wins=int((d > 0).sum()), off=off.mean(), on=on.mean())


# ---------- Figure 1: specificity ----------
specs = [
    ("Fed-Heart\n(overlapping)", "runs/anchor_sweep_fedheart/results.json", "0.0", "0.1"),
    ("NHANES nested\n(nested)", "runs/anchor_spec_nested/results.json", "0.001", "0.1"),
    ("NHANES expanded\n(nested+)", "runs/anchor_spec_expanded/results.json", "0.001", "0.1"),
    ("NHANES disjoint\n(disjoint)", "runs/anchor_val_nhanes_disjoint/results.json", "0.001", "0.1"),
]
labels, deltas, errs, ps = [], [], [], []
for name, path, off, on in specs:
    res = load(path)
    if not res:
        continue
    r = paired_delta(res, off, on)
    if not r:
        continue
    labels.append(name); deltas.append(r["delta"]); errs.append(r["std"] / np.sqrt(r["n"]))
    ps.append(r["p"])
    print(f"{name.splitlines()[0]:20s} Δ={r['delta']:+.2f} (off {r['off']:.1f}→on {r['on']:.1f}) "
          f"{r['wins']}/{r['n']} p={r['p']:.4f}")
if labels:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#bbb" if d < 1 else "#c0392b" for d in deltas]
    bars = ax.bar(labels, deltas, yerr=errs, color=colors, capsize=4)
    for b, p in zip(bars, ps):
        if p < 0.01: star = "**"
        elif p < 0.05: star = "*"
        else: star = "ns"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, star,
                ha="center", fontsize=11)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Δ worst-group acc, anchors on − off (pts)")
    ax.set_title("Anchors fire on genuinely disjoint feature spaces\n(** p<0.01, * p<0.05, ns = not significant)")
    plt.tight_layout(); plt.savefig(f"{OUT}/anchor_specificity.png", dpi=140)
    print(f"wrote {OUT}/anchor_specificity.png")

# ---------- Figure 2: NHANES-disjoint weight sweep ----------
res = load("runs/anchor_val_nhanes_disjoint/results.json")
if res:
    order = ["0.001", "0.1", "0.3"]
    xs, means, sds = [], [], []
    for l in order:
        w = arm_worst(res, l)
        if w is not None:
            xs.append(l); means.append(w.mean()); sds.append(w.std())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(range(len(xs)), means, yerr=sds, marker="o", color="#c0392b", capsize=4)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"λ={x}" for x in xs])
    ax.set_ylabel("worst-group acc (%)"); ax.set_xlabel("anchor weight (λ_fit = λ_sep)")
    ax.set_title("NHANES-disjoint: worst-group vs anchor weight (10 seeds)")
    plt.tight_layout(); plt.savefig(f"{OUT}/anchor_weight_sweep.png", dpi=140)
    print(f"wrote {OUT}/anchor_weight_sweep.png")

# ---------- Figure 3: fit vs sep decomposition ----------
res = load("runs/anchor_decomp_nhanes/results.json")
base = load("runs/anchor_val_nhanes_disjoint/results.json")
if res:
    arms = [("off", base, "0.001") if base else None,
            ("fit only\n(alignment)", res, "fit0.1_sep0.0"),
            ("sep only", res, "fit0.0_sep0.1"),
            ("both", res, "fit0.1_sep0.1")]
    names, means, sds = [], [], []
    for a in arms:
        if not a: continue
        nm, r, lab = a
        w = arm_worst(r, lab)
        if w is not None:
            names.append(nm); means.append(w.mean()); sds.append(w.std())
    if names:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(names, means, yerr=sds, color=["#bbb", "#e67e22", "#3498db", "#c0392b"][:len(names)], capsize=4)
        ax.set_ylabel("worst-group acc (%)")
        ax.set_ylim(min(means) - 2, max(means) + 2)
        ax.set_title("NHANES-disjoint: which anchor loss drives the gain?")
        plt.tight_layout(); plt.savefig(f"{OUT}/anchor_fit_vs_sep.png", dpi=140)
        print(f"wrote {OUT}/anchor_fit_vs_sep.png")
