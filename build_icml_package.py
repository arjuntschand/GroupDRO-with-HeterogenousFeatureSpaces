"""ICML-grade results package for the tabular datasets.

Consumes runs/matrix_<tag>/results.json (+ metrics_long.csv) and emits every table and
figure the paper needs, following Xenia's Step 6 reporting spec:
  per group  -> accuracy, macro-F1, loss, R*_g, excess loss
  overall    -> accuracy, macro-F1
  worst-group raw loss (and which group);  max excess loss (and which group)

Outputs
  documentation/ICML_RESULTS.md
  documentation/figures/icml_fig{1..6}_*.png
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = "documentation/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

METHODS = ["ERM", "GroupDRO", "RegretDRO", "AnchorsOnly", "Ours_GDRO", "Ours_Regret"]
PRETTY = {"ERM": "ERM", "GroupDRO": "GroupDRO", "RegretDRO": "Regret-DRO",
          "AnchorsOnly": "Anchors only", "Ours_GDRO": "Ours (anchors+GroupDRO)",
          "Ours_Regret": "Ours (anchors+regret)"}
COL = {"ERM": "#95a5a6", "GroupDRO": "#3498db", "RegretDRO": "#9b59b6",
       "AnchorsOnly": "#f39c12", "Ours_GDRO": "#e74c3c", "Ours_Regret": "#c0392b"}
TAGS = [("Fed-Heart", "fedheart", ["Cleveland", "Hungarian", "Switzerland", "VA"]),
        ("NHANES-nested", "nhanes_nested", ["G0 survey", "G1 exam", "G2 labs"]),
        ("NHANES-disjoint", "nhanes_disjoint", ["G0 survey", "G1 exam", "G2 labs"]),
        ("NHANES-expanded", "nhanes_expanded", ["G0 survey", "G1 exam", "G2 labs"])]


def load(tag):
    p = f"runs/matrix_{tag}/results.json"
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p))
    return d["results"], d["rstar"]


def scalar(res, m, key):
    """Mean±std over seeds (x100). Derived keys:
       mean_group_f1 / worst_group_f1 fall back to per_group_f1 when the trainer did not
       log an overall macro-F1 (NHANES logs per-group F1 only)."""
    v = []
    for s in res.get(m, {}):
        r = res[m][s]
        if not isinstance(r, dict):
            continue
        x = r.get(key)
        if not (isinstance(x, (int, float)) and x == x):
            pg = r.get("per_group_f1")
            if isinstance(pg, list) and pg:
                if key in ("mean_group_f1", "overall_macro_f1"):
                    x = float(np.mean(pg))
                elif key == "worst_group_f1":
                    x = float(np.min(pg))
        if isinstance(x, (int, float)) and x == x:
            v.append(x)
    return (np.mean(v) * 100, np.std(v) * 100, len(v)) if v else None


def perseed(res, m, key):
    out = {}
    for s in res.get(m, {}):
        v = res[m][s].get(key) if isinstance(res[m][s], dict) else None
        if isinstance(v, (int, float)) and v == v:
            out[s] = v * 100
    return out


def pergroup(res, m, key):
    arrs = [res[m][s].get(key) for s in res.get(m, {}) if isinstance(res[m][s], dict)]
    arrs = [a for a in arrs if isinstance(a, list) and a]
    if not arrs:
        return None, None
    k = min(len(a) for a in arrs)
    A = np.array([a[:k] for a in arrs], dtype=float)
    return A.mean(0), A.std(0)


def worst_loss_and_group(res, m):
    """mean over seeds of (max_g L_g), plus the modal argmax group."""
    vals, args_ = [], []
    for s in res.get(m, {}):
        pgl = res[m][s].get("per_group_loss") if isinstance(res[m][s], dict) else None
        if pgl:
            vals.append(max(pgl)); args_.append(int(np.argmax(pgl)))
    if not vals:
        return None
    return float(np.mean(vals)), int(stats.mode(args_, keepdims=False).mode) if args_ else None


def max_excess_and_group(res, m, rstar):
    vals, args_ = [], []
    for s in res.get(m, {}):
        pgl = res[m][s].get("per_group_loss") if isinstance(res[m][s], dict) else None
        if pgl:
            ex = [max(0.0, l - rstar[i]) for i, l in enumerate(pgl) if i < len(rstar)]
            if ex:
                vals.append(max(ex)); args_.append(int(np.argmax(ex)))
    if not vals:
        return None
    return float(np.mean(vals)), int(stats.mode(args_, keepdims=False).mode) if args_ else None


def paired_p(res, a, b, key="worst_group_acc"):
    A, B = perseed(res, a, key), perseed(res, b, key)
    ks = sorted(set(A) & set(B))
    if len(ks) < 3:
        return None
    x = np.array([A[k] for k in ks]); y = np.array([B[k] for k in ks])
    t, p = stats.ttest_rel(y, x)
    return {"delta": (y - x).mean(), "p": p, "wins": int((y > x).sum()), "n": len(ks)}


def f(v, pct=True):
    if v is None:
        return "—"
    return f"{v[0]:.2f} ± {v[1]:.2f}"


L = ["# ICML results package — NHANES & Fed-Heart\n",
     "Protocol follows Xenia's Step 5/6 spec (EMBED_Experiments_Description.docx) extended",
     "with regret optimization. **6 methods × 10 seeds.** All methods share identical model",
     "capacity and differ only in the objective. Anchors ON = λ_fit = λ_sep = 0.1; OFF = 0.001.",
     "R*_g estimated by 5-fold out-of-fold dedicated per-group models (nested-CV early stopping).\n",
     "Metrics: per-group accuracy / macro-F1 / loss / R*_g / excess loss; overall accuracy and",
     "macro-F1; worst-group raw loss (max_g L_g) and max excess loss (max_g L_g − R*_g), each",
     "with the group it points at. Raw rows: `runs/matrix_<tag>/metrics_long.csv`.\n"]

loaded = []
for name, tag, gnames in TAGS:
    res, rstar = load(tag)
    if res:
        loaded.append((name, tag, gnames, res, rstar))

# ── Table 1: method comparison (headline) ──
L += ["\n## Table 1 — Method comparison (worst-group accuracy, mean ± std over 10 seeds)\n",
      "| method | " + " | ".join(n for n, *_ in loaded) + " |",
      "|" + "---|" * (len(loaded) + 1)]
for m in METHODS:
    row = [f(scalar(res, m, "worst_group_acc")) for *_, res, _ in loaded]
    star = " ⭐" if m == "Ours_Regret" else ""
    L.append(f"| {PRETTY[m]}{star} | " + " | ".join(row) + " |")

# ── Table 2: all overall metrics per dataset ──
L += ["\n\n## Table 2 — Overall metrics per dataset\n"]
for name, tag, gnames, res, rstar in loaded:
    L += [f"\n### {name}\n",
          "| method | worst-group acc | overall acc | balanced acc | mean F1 | worst-group F1 | worst-group loss (group) | max excess loss (group) |",
          "|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        wl = worst_loss_and_group(res, m)
        me = max_excess_and_group(res, m, rstar)
        gl = f"{wl[0]:.3f} ({gnames[wl[1]] if wl[1] is not None and wl[1] < len(gnames) else '—'})" if wl else "—"
        ml = f"{me[0]:.3f} ({gnames[me[1]] if me[1] is not None and me[1] < len(gnames) else '—'})" if me else "—"
        L.append(f"| {PRETTY[m]} | {f(scalar(res,m,'worst_group_acc'))} | {f(scalar(res,m,'overall_acc'))} | "
                 f"{f(scalar(res,m,'balanced_acc'))} | {f(scalar(res,m,'mean_group_f1'))} | "
                 f"{f(scalar(res,m,'worst_group_f1'))} | {gl} | {ml} |")

# ── Table 3: per-group full breakdown (Xenia Step 6) ──
L += ["\n\n## Table 3 — Per-group breakdown: accuracy, macro-F1, loss, R*_g, excess loss\n"]
for name, tag, gnames, res, rstar in loaded:
    L += [f"\n### {name}\n"]
    for m in ["ERM", "GroupDRO", "RegretDRO", "Ours_Regret"]:
        acc, _ = pergroup(res, m, "per_group_acc")
        f1, _ = pergroup(res, m, "per_group_f1")
        ls, _ = pergroup(res, m, "per_group_loss")
        if acc is None:
            continue
        L += [f"\n**{PRETTY[m]}**\n",
              "| group | n | accuracy | macro-F1 | loss | R*_g | excess loss |",
              "|---|---|---|---|---|---|---|"]
        cnt, _ = pergroup(res, m, "per_group_counts")
        for i in range(len(acc)):
            R = rstar[i] if i < len(rstar) else float("nan")
            Lg = ls[i] if ls is not None and i < len(ls) else float("nan")
            ex = max(0.0, Lg - R) if (Lg == Lg and R == R) else float("nan")
            L.append(f"| {gnames[i] if i < len(gnames) else f'g{i}'} | "
                     f"{int(cnt[i]) if cnt is not None and i < len(cnt) else '—'} | "
                     f"{acc[i]*100:.2f} | {f1[i]*100:.2f} | {Lg:.3f} | {R:.3f} | {ex:.3f} |"
                     if f1 is not None and i < len(f1) else
                     f"| {gnames[i]} | — | {acc[i]*100:.2f} | — | {Lg:.3f} | {R:.3f} | {ex:.3f} |")

# ── Table 4: 2x2 anchors x regret + significance ──
L += ["\n\n## Table 4 — 2×2 interaction (anchors × regret) and paired significance\n",
      "Worst-group accuracy. GroupDRO (row 2) vs Ours-regret is the headline comparison.\n",
      "| dataset | GroupDRO | Regret-DRO | Ours (anc+GDRO) | Ours (anc+regret) | anchor effect | regret effect |",
      "|---|---|---|---|---|---|---|"]
for name, tag, gnames, res, rstar in loaded:
    anc = paired_p(res, "GroupDRO", "Ours_GDRO")
    reg = paired_p(res, "GroupDRO", "RegretDRO")
    def sig(r):
        if not r:
            return "—"
        s = "**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else "ns")
        return f"{r['delta']:+.2f} ({r['wins']}/{r['n']}, p={r['p']:.3f}) {s}"
    L.append(f"| {name} | {f(scalar(res,'GroupDRO','worst_group_acc'))} | {f(scalar(res,'RegretDRO','worst_group_acc'))} | "
             f"{f(scalar(res,'Ours_GDRO','worst_group_acc'))} | {f(scalar(res,'Ours_Regret','worst_group_acc'))} | "
             f"{sig(anc)} | {sig(reg)} |")

# ── Table 5: parameter testing (group definitions) ──
nh = [(n, t, g, r, rs) for n, t, g, r, rs in loaded if t.startswith("nhanes")]
if nh:
    L += ["\n\n## Table 5 — Parameter testing: group definition (NHANES feature modes)\n",
          "Same method, different *group structure* — nested (G0⊂G1⊂G2), expanded (nested, more",
          "features), disjoint (each group has unique features). Worst-group accuracy.\n",
          "| method | " + " | ".join(n for n, *_ in nh) + " |",
          "|" + "---|" * (len(nh) + 1)]
    for m in METHODS:
        L.append(f"| {PRETTY[m]} | " + " | ".join(f(scalar(r, m, "worst_group_acc")) for *_, r, _ in nh) + " |")

open("documentation/ICML_RESULTS.md", "w").write("\n".join(L))
print("wrote documentation/ICML_RESULTS.md")

# ───────────────────────── figures ─────────────────────────
if loaded:
    # Fig 1: method comparison grouped bars
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(loaded)); w = 0.13
    for j, m in enumerate(METHODS):
        vals = [scalar(res, m, "worst_group_acc") for *_, res, _ in loaded]
        mu = [v[0] if v else np.nan for v in vals]; sd = [v[1] if v else 0 for v in vals]
        ax.bar(x + (j - 2.5) * w, mu, w, yerr=sd, label=PRETTY[m], color=COL[m], capsize=2)
    ax.set_xticks(x); ax.set_xticklabels([n for n, *_ in loaded])
    ax.set_ylabel("worst-group accuracy (%)")
    allv = [scalar(res, m, "worst_group_acc") for *_, res, _ in loaded for m in METHODS]
    allv = [v[0] for v in allv if v]
    ax.set_ylim(min(allv) - 4, max(allv) + 3)
    ax.set_title("Method comparison across datasets (10 seeds)")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    plt.savefig(f"{OUT}/icml_fig1_methods.png"); plt.close()
    print(f"wrote {OUT}/icml_fig1_methods.png")

    # Fig 2: per-group accuracy, ERM vs Ours
    fig, axes = plt.subplots(1, len(loaded), figsize=(3.5 * len(loaded), 3.6), squeeze=False)
    for ax, (name, tag, gnames, res, rstar) in zip(axes[0], loaded):
        a1, s1 = pergroup(res, "ERM", "per_group_acc")
        a2, s2 = pergroup(res, "Ours_Regret", "per_group_acc")
        if a1 is None or a2 is None:
            continue
        k = min(len(a1), len(a2)); xx = np.arange(k); w = 0.38
        ax.bar(xx - w/2, a1[:k]*100, w, yerr=s1[:k]*100, label="ERM", color=COL["ERM"], capsize=2)
        ax.bar(xx + w/2, a2[:k]*100, w, yerr=s2[:k]*100, label="Ours", color=COL["Ours_Regret"], capsize=2)
        ax.set_xticks(xx); ax.set_xticklabels(gnames[:k], fontsize=7, rotation=20)
        ax.set_title(name, fontsize=10); ax.set_ylabel("accuracy (%)"); ax.legend(fontsize=7)
    plt.suptitle("Per-group accuracy: ERM vs full method", fontsize=11)
    plt.savefig(f"{OUT}/icml_fig2_pergroup.png"); plt.close()
    print(f"wrote {OUT}/icml_fig2_pergroup.png")

    # Fig 3: R*_g vs achieved loss (the regret picture)
    fig, axes = plt.subplots(1, len(loaded), figsize=(3.5 * len(loaded), 3.5), squeeze=False)
    for ax, (name, tag, gnames, res, rstar) in zip(axes[0], loaded):
        for m, mk in [("GroupDRO", "o"), ("Ours_Regret", "s")]:
            ls, _ = pergroup(res, m, "per_group_loss")
            if ls is None:
                continue
            k = min(len(ls), len(rstar))
            ax.scatter(rstar[:k], ls[:k], label=PRETTY[m], color=COL[m], marker=mk, s=45)
        lim = [0, max(max(rstar), 1.0) * 1.15]
        ax.plot(lim, lim, "k--", lw=0.8, label="L_g = R*_g")
        ax.set_xlabel("R*_g (achievable floor)"); ax.set_ylabel("achieved loss L_g")
        ax.set_title(name, fontsize=10); ax.legend(fontsize=6.5)
    plt.suptitle("Achieved loss vs per-group reference loss R*_g (points below the line beat the dedicated model)", fontsize=10)
    plt.savefig(f"{OUT}/icml_fig3_regret.png"); plt.close()
    print(f"wrote {OUT}/icml_fig3_regret.png")

    # Fig 4: 2x2 anchors x regret interaction
    fig, axes = plt.subplots(1, len(loaded), figsize=(3.2 * len(loaded), 3.4), squeeze=False)
    for ax, (name, tag, gnames, res, rstar) in zip(axes[0], loaded):
        cells = [("GroupDRO", "no anchors\nno regret"), ("Ours_GDRO", "anchors\nno regret"),
                 ("RegretDRO", "no anchors\nregret"), ("Ours_Regret", "anchors\nregret")]
        mu = [scalar(res, m, "worst_group_acc") for m, _ in cells]
        vals = [v[0] if v else np.nan for v in mu]; sds = [v[1] if v else 0 for v in mu]
        ax.bar(range(4), vals, yerr=sds, capsize=3,
               color=[COL["GroupDRO"], COL["Ours_GDRO"], COL["RegretDRO"], COL["Ours_Regret"]])
        ax.set_xticks(range(4)); ax.set_xticklabels([c[1] for c in cells], fontsize=6.5)
        ax.set_ylim(min(v for v in vals if v == v) - 3, max(v for v in vals if v == v) + 2)
        ax.set_ylabel("worst-group acc (%)"); ax.set_title(name, fontsize=10)
    plt.suptitle("2×2: anchors × regret", fontsize=11)
    plt.savefig(f"{OUT}/icml_fig4_2x2.png"); plt.close()
    print(f"wrote {OUT}/icml_fig4_2x2.png")

    # Fig 5: parameter testing across group definitions
    if nh:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(nh)); w = 0.13
        for j, m in enumerate(METHODS):
            vals = [scalar(r, m, "worst_group_acc") for *_, r, _ in nh]
            mu = [v[0] if v else np.nan for v in vals]
            ax.bar(x + (j - 2.5) * w, mu, w, label=PRETTY[m], color=COL[m])
        ax.set_xticks(x); ax.set_xticklabels([n for n, *_ in nh])
        ax.set_ylabel("worst-group accuracy (%)")
        vv = [v[0] for *_, r, _ in nh for v in [scalar(r, m, "worst_group_acc") for m in METHODS] if v]
        ax.set_ylim(min(vv) - 3, max(vv) + 3)
        ax.set_title("Parameter test: group definition (NHANES feature modes)")
        ax.legend(fontsize=7, ncol=3)
        plt.savefig(f"{OUT}/icml_fig5_groupdef.png"); plt.close()
        print(f"wrote {OUT}/icml_fig5_groupdef.png")

    # Fig 6: per-group loss vs excess
    fig, axes = plt.subplots(1, len(loaded), figsize=(3.5 * len(loaded), 3.4), squeeze=False)
    for ax, (name, tag, gnames, res, rstar) in zip(axes[0], loaded):
        ls_e, _ = pergroup(res, "ERM", "per_group_loss")
        ls_o, _ = pergroup(res, "Ours_Regret", "per_group_loss")
        if ls_e is None or ls_o is None:
            continue
        k = min(len(ls_e), len(ls_o), len(rstar)); xx = np.arange(k); w = 0.28
        ax.bar(xx - w, ls_e[:k], w, label="ERM loss", color=COL["ERM"])
        ax.bar(xx, ls_o[:k], w, label="Ours loss", color=COL["Ours_Regret"])
        ax.bar(xx + w, np.array(rstar[:k]), w, label="R*_g", color="#2ecc71")
        ax.set_xticks(xx); ax.set_xticklabels(gnames[:k], fontsize=7, rotation=20)
        ax.set_ylabel("cross-entropy"); ax.set_title(name, fontsize=10); ax.legend(fontsize=6.5)
    plt.suptitle("Per-group loss vs achievable floor R*_g", fontsize=11)
    plt.savefig(f"{OUT}/icml_fig6_losses.png"); plt.close()
    print(f"wrote {OUT}/icml_fig6_losses.png")
