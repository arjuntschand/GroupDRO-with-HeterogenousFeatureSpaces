"""Assemble the paper-grade results package from the ablation run JSONs:
  - per-dataset 2x2x2 ablation table (worst / overall / balanced acc, mean±std)
  - per-group accuracy + loss table (full method vs shared-ERM baseline)
  - figures: ablation bar (worst-group by cell), per-group accuracy bars
  - consolidated documentation/PAPER_RESULTS.md

Reads whatever runs/ablation_*/results.json are present; skips missing.
"""
import ast, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "documentation/figures"
os.makedirs(OUT, exist_ok=True)
CELL_ORDER = ["shared_erm", "shared_erm_anchor", "shared_gdro", "shared_gdro_anchor",
              "pergroup_erm", "pergroup_erm_anchor", "pergroup_gdro", "pergroup_gdro_anchor"]
PRETTY = {"shared_erm": "Shared ERM", "shared_erm_anchor": "Shared ERM +anc",
          "shared_gdro": "Shared GDRO", "shared_gdro_anchor": "Shared GDRO +anc",
          "pergroup_erm": "Per-group ERM", "pergroup_erm_anchor": "Per-group ERM +anc",
          "pergroup_gdro": "Per-group GDRO", "pergroup_gdro_anchor": "Per-group GDRO +anc (Ours)"}


def parse_list(s):
    try:
        return [float(x) for x in ast.literal_eval(s)]
    except Exception:
        return None


def agg_cell(raw, cell, metric):
    if cell not in raw:
        return None
    vs = [raw[cell][s].get(metric) for s in raw[cell] if isinstance(raw[cell][s], dict)]
    vs = [v for v in vs if isinstance(v, (int, float)) and v == v]
    return (np.mean(vs) * 100, np.std(vs) * 100, len(vs)) if vs else None


def per_group(raw, cell, key="test_per_group_acc"):
    """Mean per-group values across seeds for a cell."""
    if cell not in raw:
        return None
    arrs = []
    for s in raw[cell]:
        v = raw[cell][s].get(key) if isinstance(raw[cell][s], dict) else None
        pl = parse_list(v) if isinstance(v, str) else None
        if pl:
            arrs.append(pl)
    if not arrs:
        return None
    m = min(len(a) for a in arrs)
    arrs = np.array([a[:m] for a in arrs])
    return arrs.mean(0)


def main():
    datasets = []
    for path in sorted(glob.glob("runs/ablation_*/results.json")):
        tag = path.split("/")[1].replace("ablation_", "")
        d = json.load(open(path))
        datasets.append((tag, d["raw"]))

    lines = ["# Paper results — full ablation package\n",
             "2×2×2 ablation: encoder {shared, per-group} × GroupDRO {off, on} × anchors {off, on}.",
             "Anchors on = λ_fit=λ_sep=0.1 (validated). 10 seeds, mean±std.\n"]
    for tag, raw in datasets:
        lines.append(f"\n## {tag}\n")
        lines.append("| config | worst-group | overall | balanced |")
        lines.append("|---|---|---|---|")
        for c in CELL_ORDER:
            w = agg_cell(raw, c, "test_worst_group_acc")
            o = agg_cell(raw, c, "test_overall_acc")
            b = agg_cell(raw, c, "test_balanced_acc")
            if w:
                lines.append(f"| {PRETTY.get(c,c)} | {w[0]:.2f} ± {w[1]:.2f} | "
                             f"{o[0]:.2f} ± {o[1]:.2f} | {b[0]:.2f} ± {b[1]:.2f} |")
        # anchor effect on the full method
        pg = agg_cell(raw, "pergroup_gdro", "test_worst_group_acc")
        pga = agg_cell(raw, "pergroup_gdro_anchor", "test_worst_group_acc")
        if pg and pga:
            lines.append(f"\n- **anchors on the full method (per-group GDRO): "
                         f"{pg[0]:.2f} → {pga[0]:.2f} = {pga[0]-pg[0]:+.2f} pts worst-group**")

        # per-group accuracy figure: shared-ERM vs Ours
        base = per_group(raw, "shared_erm"); ours = per_group(raw, "pergroup_gdro_anchor")
        if base is not None and ours is not None and len(base) == len(ours):
            g = np.arange(len(base)); wdt = 0.38
            plt.figure(figsize=(6, 4))
            plt.bar(g - wdt/2, base*100, wdt, label="Shared ERM", color="#bbb")
            plt.bar(g + wdt/2, ours*100, wdt, label="Ours (per-group GDRO +anchors)", color="#c0392b")
            plt.xticks(g, [f"G{i}" for i in g]); plt.ylabel("accuracy (%)")
            plt.title(f"{tag}: per-group accuracy"); plt.legend(fontsize=8)
            plt.tight_layout(); plt.savefig(f"{OUT}/pergroup_{tag}.png", dpi=140); plt.close()
            lines.append(f"- per-group figure: `figures/pergroup_{tag}.png`")

    # ablation bar: worst-group by cell, one dataset per subplot
    if datasets:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4), squeeze=False)
        for ax, (tag, raw) in zip(axes[0], datasets):
            ws = [(PRETTY.get(c, c), agg_cell(raw, c, "test_worst_group_acc")) for c in CELL_ORDER]
            ws = [(n, v) for n, v in ws if v]
            names = [n for n, _ in ws]; means = [v[0] for _, v in ws]; sds = [v[1] for _, v in ws]
            colors = ["#c0392b" if "Ours" in n else "#888" for n in names]
            ax.barh(range(len(names)), means, xerr=sds, color=colors)
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
            lo = max(0, min(means) - 4); ax.set_xlim(lo, max(means) + max(sds) + 2)
            ax.set_xlabel("worst-group acc (%)"); ax.set_title(tag); ax.invert_yaxis()
        plt.tight_layout(); plt.savefig(f"{OUT}/ablation_worst_group.png", dpi=140); plt.close()
        lines.append("\n## Figures\n- `figures/ablation_worst_group.png` (worst-group by cell, all datasets)")

    open("documentation/PAPER_RESULTS.md", "w").write("\n".join(lines))
    print("\n".join(lines))
    print("\nwrote documentation/PAPER_RESULTS.md")


if __name__ == "__main__":
    main()
