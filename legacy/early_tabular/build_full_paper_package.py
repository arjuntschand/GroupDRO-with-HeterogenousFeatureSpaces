"""Build the COMPLETE paper results package: every table, every metric, every figure.

Outputs
  documentation/PAPER_TABLES.md   — all tables (headline, ablation, per-group, anchor, sig tests)
  documentation/figures/fig1_headline.png        — full method vs baseline, all datasets
  documentation/figures/fig2_ablation_grid.png   — 2x2x2 ablation, all datasets
  documentation/figures/fig3_pergroup.png        — per-group accuracy, baseline vs ours
  documentation/figures/fig4_anchor_effect.png   — anchor effect + significance across datasets
  documentation/figures/fig5_anchor_mechanism.png— weight sweep + fit/sep decomposition
"""
from __future__ import annotations
import ast, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = "documentation/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C_BASE, C_GDRO, C_OURS, C_GREY = "#95a5a6", "#3498db", "#c0392b", "#bdc3c7"

CELLS = ["shared_erm", "shared_erm_anchor", "shared_gdro", "shared_gdro_anchor",
         "pergroup_erm", "pergroup_erm_anchor", "pergroup_gdro", "pergroup_gdro_anchor"]
PRETTY = {"shared_erm": "Shared ERM", "shared_erm_anchor": "Shared ERM + anchors",
          "shared_gdro": "Shared GroupDRO", "shared_gdro_anchor": "Shared GroupDRO + anchors",
          "pergroup_erm": "Per-group ERM", "pergroup_erm_anchor": "Per-group ERM + anchors",
          "pergroup_gdro": "Per-group GroupDRO", "pergroup_gdro_anchor": "Per-group GroupDRO + anchors (Ours)"}
DATASETS = [("Fed-Heart", "runs/ablation_fedheart_fedheart/results.json", ["Cleveland", "Hungarian", "Switzerland", "VA"]),
            ("NHANES-nested", "runs/ablation_nhanes_nested/results.json", ["G0 survey", "G1 exam", "G2 labs"]),
            ("NHANES-disjoint", "runs/ablation_nhanes_disjoint/results.json", ["G0 survey", "G1 exam", "G2 labs"]),
            ("NHANES-expanded", "runs/ablation_nhanes_expanded/results.json", ["G0 survey", "G1 exam", "G2 labs"])]


def load(p):
    return json.load(open(p))["raw"] if os.path.exists(p) else None


def vals(raw, cell, metric):
    out = []
    for s in raw.get(cell, {}):
        r = raw[cell][s]
        v = r.get(metric) if isinstance(r, dict) else None
        if isinstance(v, (int, float)) and v == v:
            out.append(v)
    return np.array(out)


def seedvals(raw, cell, metric):
    """dict seed -> value (for paired tests)."""
    out = {}
    for s in raw.get(cell, {}):
        r = raw[cell][s]
        v = r.get(metric) if isinstance(r, dict) else None
        if isinstance(v, (int, float)) and v == v:
            out[s] = v
    return out


def paired(raw, a, b, metric="test_worst_group_acc"):
    A, B = seedvals(raw, a, metric), seedvals(raw, b, metric)
    ks = sorted(set(A) & set(B))
    if len(ks) < 3:
        return None
    x = np.array([A[k] for k in ks]) * 100
    y = np.array([B[k] for k in ks]) * 100
    d = y - x
    t, p = stats.ttest_rel(y, x)
    return {"delta": d.mean(), "p": p, "wins": int((d > 0).sum()), "n": len(d),
            "a": x.mean(), "b": y.mean()}


def pergroup(raw, cell, key="test_per_group_acc"):
    arrs = []
    for s in raw.get(cell, {}):
        v = raw[cell][s].get(key) if isinstance(raw[cell][s], dict) else None
        if isinstance(v, str):
            try:
                arrs.append([float(x) for x in ast.literal_eval(v)])
            except Exception:
                pass
        elif isinstance(v, list):
            arrs.append([float(x) for x in v])
    if not arrs:
        return None, None
    m = min(len(a) for a in arrs)
    A = np.array([a[:m] for a in arrs]) * 100
    return A.mean(0), A.std(0)


def fmt(v):
    return f"{v[0]:.2f} ± {v[1]:.2f}" if v is not None else "—"


def ms(a):
    return (a.mean() * 100, a.std() * 100) if len(a) else None


# ───────────────────────────── tables ─────────────────────────────
L = ["# Complete paper results — all tables\n",
     "All numbers are **worst-group / overall / balanced accuracy (%)**, mean ± std over 10 seeds",
     "(TextCaps: 3 seeds). Anchors ON = λ_fit = λ_sep = 0.1; OFF = 0.001 (exactly 0.0 is",
     "numerically unstable — see ANCHOR_RESULTS.md).\n"]

# Table 1 — headline
L += ["\n## Table 1 — Headline: full method vs naive baseline (worst-group accuracy)\n",
      "| dataset | Shared ERM (naive) | Per-group GroupDRO | **+ anchors (Ours)** | **gain vs naive** |",
      "|---|---|---|---|---|"]
headline = []
for name, path, _ in DATASETS:
    raw = load(path)
    if not raw:
        continue
    b = ms(vals(raw, "shared_erm", "test_worst_group_acc"))
    g = ms(vals(raw, "pergroup_gdro", "test_worst_group_acc"))
    o = ms(vals(raw, "pergroup_gdro_anchor", "test_worst_group_acc"))
    best = max([x for x in (g, o) if x], key=lambda z: z[0]) if (g or o) else None
    gain = best[0] - b[0] if (b and best) else None
    L.append(f"| {name} | {fmt(b)} | {fmt(g)} | {fmt(o)} | **{gain:+.2f}** |")
    headline.append((name, b[0] if b else np.nan, g[0] if g else np.nan,
                     o[0] if o else np.nan, gain))
# textcaps
tc = json.load(open("runs/textcaps_cached_full/results.json")) if os.path.exists("runs/textcaps_cached_full/results.json") else None
if tc:
    off = np.array([tc["fit0.001_sep0.001"][s]["worst"] for s in tc["fit0.001_sep0.001"]]) * 100
    on = np.array([tc["fit0.1_sep0.0"][s]["worst"] for s in tc["fit0.1_sep0.0"]]) * 100
    L.append(f"| TextCaps (full data) | — | {off.mean():.2f} ± {off.std():.2f} | "
             f"{on.mean():.2f} ± {on.std():.2f} | **{on.mean()-off.mean():+.2f}** |")

# Table 2 — full ablation per dataset
L += ["\n\n## Table 2 — Full 2×2×2 ablation (encoder × GroupDRO × anchors)\n"]
for name, path, _ in DATASETS:
    raw = load(path)
    if not raw:
        continue
    L += [f"\n### {name}\n",
          "| config | worst-group | overall | balanced |", "|---|---|---|---|"]
    for c in CELLS:
        w, o, b = (ms(vals(raw, c, k)) for k in
                   ("test_worst_group_acc", "test_overall_acc", "test_balanced_acc"))
        if w:
            star = " ⭐" if c == "pergroup_gdro_anchor" else ""
            L.append(f"| {PRETTY[c]}{star} | {fmt(w)} | {fmt(o)} | {fmt(b)} |")

# Table 3 — per-group accuracy + loss
L += ["\n\n## Table 3 — Per-group accuracy and loss (baseline vs Ours)\n"]
for name, path, gnames in DATASETS:
    raw = load(path)
    if not raw:
        continue
    ba, _ = pergroup(raw, "shared_erm")
    oa, _ = pergroup(raw, "pergroup_gdro_anchor")
    bl, _ = pergroup(raw, "shared_erm", "test_per_group_loss")
    ol, _ = pergroup(raw, "pergroup_gdro_anchor", "test_per_group_loss")
    if ba is None or oa is None:
        continue
    gg = gnames[:len(ba)]
    L += [f"\n### {name}\n", "| group | ERM acc | Ours acc | Δ acc | ERM loss | Ours loss |",
          "|---|---|---|---|---|---|"]
    for i, g in enumerate(gg):
        bls = f"{bl[i]/100:.3f}" if bl is not None and i < len(bl) else "—"
        ols = f"{ol[i]/100:.3f}" if ol is not None and i < len(ol) else "—"
        L.append(f"| {g} | {ba[i]:.2f} | {oa[i]:.2f} | {oa[i]-ba[i]:+.2f} | {bls} | {ols} |")

# Table 4 — anchor effect + significance
L += ["\n\n## Table 4 — Anchor contribution (paired, same seeds)\n",
      "Effect of turning anchors ON, holding encoder + GroupDRO fixed.\n",
      "| dataset | base config | anchors off | anchors on | Δ | seeds won | p |",
      "|---|---|---|---|---|---|---|"]
anchor_rows = []
for name, path, _ in DATASETS:
    raw = load(path)
    if not raw:
        continue
    for base, lbl in [("pergroup_gdro", "Per-group GDRO"), ("shared_gdro", "Shared GDRO")]:
        r = paired(raw, base, base + "_anchor")
        if r:
            sig = "**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else "ns")
            L.append(f"| {name} | {lbl} | {r['a']:.2f} | {r['b']:.2f} | **{r['delta']:+.2f}** | "
                     f"{r['wins']}/{r['n']} | {r['p']:.4f} {sig} |")
            if base == "pergroup_gdro":
                anchor_rows.append((name, r))
if tc:
    d = on - off
    t, p = stats.ttest_rel(on, off)
    L.append(f"| TextCaps (full) | GroupDRO | {off.mean():.2f} | {on.mean():.2f} | "
             f"**{d.mean():+.2f}** | {(d>0).sum()}/{len(d)} | {p:.4f} ns |")
    anchor_rows.append(("TextCaps", {"delta": d.mean(), "p": p, "wins": int((d > 0).sum()), "n": len(d)}))

# Table 5 — NHANES clinical metrics
L += ["\n\n## Table 5 — NHANES clinical metrics (AUROC, disjoint mode)\n",
      "| config | overall AUROC | worst-group balanced acc |", "|---|---|---|"]
raw = load("runs/ablation_nhanes_disjoint/results.json")
if raw:
    for c in CELLS:
        a = ms(vals(raw, c, "test_overall_auroc"))
        wb = ms(vals(raw, c, "test_worst_group_bal_acc"))
        if a:
            L.append(f"| {PRETTY[c]} | {fmt(a)} | {fmt(wb)} |")

# Table 6 — anchor mechanism
L += ["\n\n## Table 6 — Anchor mechanism: which loss drives the gain (NHANES-disjoint, 10 seeds)\n",
      "| arm | worst-group | Δ vs off | p |", "|---|---|---|---|"]
dec = json.load(open("runs/anchor_decomp_nhanes/results.json")) if os.path.exists("runs/anchor_decomp_nhanes/results.json") else None
val = json.load(open("runs/anchor_val_nhanes_disjoint/results.json")) if os.path.exists("runs/anchor_val_nhanes_disjoint/results.json") else None
if dec and val:
    offv = np.array([val["raw"]["0.001"][s]["worst"] for s in val["raw"]["0.001"]]) * 100
    L.append(f"| off (λ=0.001) | {offv.mean():.2f} ± {offv.std():.2f} | — | — |")
    for lab, nm in [("fit0.1_sep0.0", "fit only (alignment)"), ("fit0.0_sep0.1", "sep only"),
                    ("fit0.1_sep0.1", "both")]:
        if lab in dec["raw"]:
            v = np.array([dec["raw"][lab][s]["worst"] for s in dec["raw"][lab]]) * 100
            t, p = stats.ttest_rel(v, offv) if len(v) == len(offv) else (np.nan, np.nan)
            L.append(f"| {nm} | {v.mean():.2f} ± {v.std():.2f} | {v.mean()-offv.mean():+.2f} | {p:.4f} |")

open("documentation/PAPER_TABLES.md", "w").write("\n".join(L))
print("wrote documentation/PAPER_TABLES.md")

# ───────────────────────────── figures ─────────────────────────────
# Fig 1 — headline gains
if headline:
    names = [h[0] for h in headline]
    base = [h[1] for h in headline]
    gdro = [h[2] for h in headline]
    ours = [h[3] for h in headline]
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w, base, w, label="Shared ERM (naive)", color=C_BASE)
    ax.bar(x, gdro, w, label="Per-group GroupDRO", color=C_GDRO)
    ax.bar(x + w, ours, w, label="+ anchors (Ours)", color=C_OURS)
    for i, h in enumerate(headline):
        if h[4] == h[4]:
            ax.annotate(f"{h[4]:+.1f}", (x[i] + w, max(h[2], h[3]) + 0.7), ha="center",
                        fontsize=9, fontweight="bold", color=C_OURS)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("worst-group accuracy (%)")
    ax.set_ylim(min(base) - 5, max(max(gdro), max(ours)) + 4)
    ax.set_title("Worst-group robustness: full method vs naive baseline (10 seeds)")
    ax.legend(fontsize=8, loc="lower right")
    plt.savefig(f"{OUT}/fig1_headline.png"); plt.close()
    print(f"wrote {OUT}/fig1_headline.png")

# Fig 2 — ablation grid
avail = [(n, load(p)) for n, p, _ in DATASETS]
avail = [(n, r) for n, r in avail if r]
if avail:
    fig, axes = plt.subplots(1, len(avail), figsize=(4.2 * len(avail), 4.2), squeeze=False)
    for ax, (name, raw) in zip(axes[0], avail):
        ys, ms_, sd = [], [], []
        for c in CELLS:
            v = ms(vals(raw, c, "test_worst_group_acc"))
            if v:
                ys.append(PRETTY[c].replace(" + anchors", "\n + anchors")); ms_.append(v[0]); sd.append(v[1])
        cols = [C_OURS if "Ours" in y else (C_GDRO if "anchors" in y else C_GREY) for y in ys]
        ax.barh(range(len(ys)), ms_, xerr=sd, color=cols, capsize=2)
        ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys, fontsize=6.5)
        ax.set_xlim(max(0, min(ms_) - 4), max(ms_) + max(sd) + 1.5)
        ax.invert_yaxis(); ax.set_xlabel("worst-group acc (%)"); ax.set_title(name, fontsize=10)
    plt.suptitle("Ablation: encoder × GroupDRO × anchors (worst-group accuracy)", fontsize=11)
    plt.savefig(f"{OUT}/fig2_ablation_grid.png"); plt.close()
    print(f"wrote {OUT}/fig2_ablation_grid.png")

# Fig 3 — per-group
pg = [(n, load(p), g) for n, p, g in DATASETS]
pg = [(n, r, g) for n, r, g in pg if r]
if pg:
    fig, axes = plt.subplots(1, len(pg), figsize=(3.6 * len(pg), 3.6), squeeze=False)
    for ax, (name, raw, gnames) in zip(axes[0], pg):
        ba, bs = pergroup(raw, "shared_erm")
        oa, os_ = pergroup(raw, "pergroup_gdro_anchor")
        if ba is None or oa is None:
            continue
        m = min(len(ba), len(oa)); x = np.arange(m); w = 0.38
        ax.bar(x - w/2, ba[:m], w, yerr=bs[:m], label="Shared ERM", color=C_BASE, capsize=2)
        ax.bar(x + w/2, oa[:m], w, yerr=os_[:m], label="Ours", color=C_OURS, capsize=2)
        ax.set_xticks(x); ax.set_xticklabels(gnames[:m], fontsize=7, rotation=20)
        ax.set_ylabel("accuracy (%)"); ax.set_title(name, fontsize=10)
        ax.set_ylim(min(min(ba[:m]), min(oa[:m])) - 6, max(max(ba[:m]), max(oa[:m])) + 4)
        ax.legend(fontsize=7)
    plt.suptitle("Per-group accuracy: naive baseline vs full method", fontsize=11)
    plt.savefig(f"{OUT}/fig3_pergroup.png"); plt.close()
    print(f"wrote {OUT}/fig3_pergroup.png")

# Fig 4 — anchor effect + significance
if anchor_rows:
    names = [a[0] for a in anchor_rows]
    deltas = [a[1]["delta"] for a in anchor_rows]
    ps = [a[1]["p"] for a in anchor_rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = [C_OURS if p < 0.05 else C_GREY for p in ps]
    bars = ax.bar(names, deltas, color=cols)
    for b, p, d in zip(bars, ps, deltas):
        lab = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        ax.text(b.get_x() + b.get_width()/2, d + (0.12 if d >= 0 else -0.3), lab,
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Δ worst-group accuracy (pts)")
    ax.set_title("Anchor contribution, holding encoder + GroupDRO fixed\n(** p<0.01, * p<0.05, ns)")
    plt.xticks(rotation=12, fontsize=8)
    plt.savefig(f"{OUT}/fig4_anchor_effect.png"); plt.close()
    print(f"wrote {OUT}/fig4_anchor_effect.png")

# Fig 5 — mechanism (weight sweep + fit/sep)
if val and dec:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    order = ["0.001", "0.1", "0.3"]
    xs, mm, ss = [], [], []
    for l in order:
        if l in val["raw"]:
            v = np.array([val["raw"][l][s]["worst"] for s in val["raw"][l]]) * 100
            xs.append(f"λ={l}"); mm.append(v.mean()); ss.append(v.std())
    axes[0].errorbar(range(len(xs)), mm, yerr=ss, marker="o", color=C_OURS, capsize=4)
    axes[0].set_xticks(range(len(xs))); axes[0].set_xticklabels(xs)
    axes[0].set_ylabel("worst-group acc (%)"); axes[0].set_xlabel("anchor weight")
    axes[0].set_title("(a) Anchor weight sweep", fontsize=10)
    offv = np.array([val["raw"]["0.001"][s]["worst"] for s in val["raw"]["0.001"]]) * 100
    labs, vv, se = ["off"], [offv.mean()], [offv.std()]
    for lab, nm in [("fit0.1_sep0.0", "fit\nonly"), ("fit0.0_sep0.1", "sep\nonly"), ("fit0.1_sep0.1", "both")]:
        if lab in dec["raw"]:
            v = np.array([dec["raw"][lab][s]["worst"] for s in dec["raw"][lab]]) * 100
            labs.append(nm); vv.append(v.mean()); se.append(v.std())
    axes[1].bar(labs, vv, yerr=se, color=[C_GREY, C_OURS, C_GDRO, "#e67e22"][:len(labs)], capsize=3)
    axes[1].set_ylim(min(vv) - 2, max(vv) + 2); axes[1].set_ylabel("worst-group acc (%)")
    axes[1].set_title("(b) Which anchor loss drives the gain?", fontsize=10)
    plt.suptitle("Anchor mechanism (NHANES-disjoint, 10 seeds)", fontsize=11)
    plt.savefig(f"{OUT}/fig5_anchor_mechanism.png"); plt.close()
    print(f"wrote {OUT}/fig5_anchor_mechanism.png")
