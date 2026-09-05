"""Aggregate metrics_long.csv (+ curve_*.json) into Xenia's report artifacts:
  - per-group tables (method x group) for accuracy / macro-F1 / loss, mean+-std over seeds
  - overall / tail-mean / worst-group summary
  - the headline R*_g (x) vs final lambda_g (y) plot: GroupDRO line vs Ours line

Usage: python -m dro_hetero_anchors.src.report_embed_xenia --run runs/embed_xenia
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import pandas as pd

GROUPS = ["g1", "g2", "g3", "g4", "g5", "g6"]
HEAD = {"g4", "g6"}
METHOD_ORDER = ["erm", "groupdro", "align_only", "regret_only", "ours", "group_only"]
PRETTY = {"erm": "ERM", "groupdro": "GroupDRO (R*=0)", "align_only": "Ablation: align-only",
          "regret_only": "Ablation: regret-only", "ours": "Ours (anchors+regret)",
          "group_only": "Group-only (dedicated)"}


def agg_table(df, metric):
    """Return method x group DataFrame of 'mean±std' strings, + summary columns."""
    rows = {}
    for method in [m for m in METHOD_ORDER if m in df.method.unique()]:
        sub = df[df.method == method]
        cell = {}
        # per-group mean over seeds
        for g in GROUPS:
            gg = sub[sub.group == g]
            if len(gg):
                cell[g] = f"{gg[metric].mean():.3f}"
        # summary per seed then mean±std
        by_seed = sub.groupby("seed")
        overall = by_seed.apply(lambda s: s[metric].mean(), include_groups=False)
        tail = by_seed.apply(lambda s: s[~s.group.isin(HEAD)][metric].mean(), include_groups=False)
        if metric == "acc":
            worst = by_seed.apply(lambda s: s.set_index("group")[metric].min(), include_groups=False)
        else:
            worst = by_seed.apply(lambda s: s.set_index("group")[metric].max(), include_groups=False)
        cell["overall"] = f"{overall.mean():.3f}±{overall.std():.3f}"
        cell["tail"] = f"{tail.mean():.3f}±{tail.std():.3f}"
        cell["worst"] = f"{worst.mean():.3f}±{worst.std():.3f}"
        rows[PRETTY[method]] = cell
    return pd.DataFrame(rows).T[GROUPS + ["overall", "tail", "worst"]]


def md_table(df):
    cols = list(df.columns)
    out = ["| method | " + " | ".join(cols) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for name, r in df.iterrows():
        out.append("| " + name + " | " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def rstar_lambda_plot(run, df, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # R*_g (method-independent) from any regret-aware rows, mean over seeds
    rstar = {g: df[df.group == g]["R_star"].replace(0, np.nan).mean() for g in GROUPS}
    # final lambda per method from curve json, mean over seeds
    def final_lambda(method):
        acc = {g: [] for g in GROUPS}
        for f in glob.glob(os.path.join(run, f"curve_{method}_s*.json")):
            info = json.load(open(f))
            for g, l in zip(info["groups"], info["lambda"]):
                acc[g].append(l)
        return {g: (np.mean(v) if v else np.nan) for g, v in acc.items()}
    lam_gdro = final_lambda("groupdro")
    lam_reg = final_lambda("regret_only")
    plt.figure(figsize=(6.5, 4.5))
    for lam, name, c, mk in [(lam_gdro, "GroupDRO (R*=0)", "#888", "o"),
                             (lam_reg, "Regret-only", "#c0392b", "s")]:
        order = sorted(GROUPS, key=lambda g: (rstar[g] if not np.isnan(rstar[g]) else 0))
        px = [rstar[g] for g in order]; py = [lam[g] for g in order]
        plt.plot(px, py, marker=mk, color=c, label=name)
    for g in GROUPS:
        if not np.isnan(rstar[g]):
            plt.annotate(g, (rstar[g], max(lam_gdro[g], lam_reg[g])), fontsize=8,
                         xytext=(3, 3), textcoords="offset points")
    plt.xlabel("group optimal loss  R*_g  (best achievable CE)")
    plt.ylabel("final group weight  λ_g")
    plt.title("Both DRO variants abandon high-R* tail groups (memorized → λ≈0);\n"
              "regret only reallocates between the two heads g4↔g6", fontsize=10)
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=130)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/embed_xenia")
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(args.run, "metrics_long.csv"))
    # per-group test size (constant across methods/seeds for a fixed split)
    ncounts = {g: int(df[df.group == g]["n"].max()) for g in GROUPS if g in df.group.values}
    lines = ["# EMBED (Xenia spec) — results\n",
             f"seeds: {sorted(df.seed.unique())}   groups: {sorted(df.group.unique())}",
             f"n_params (shared model): {int(df.n_params.iloc[0]):,}",
             "test-set size per group: " + ", ".join(f"{g}={ncounts.get(g,0)}" for g in GROUPS),
             "(tail groups g2/g5 are tiny → their per-group accuracy is high-variance)\n"]
    for metric, title in [("acc", "Accuracy"), ("macro_f1", "Macro-F1"), ("loss", "Cross-entropy loss")]:
        lines.append(f"\n## {title} (per group; overall/tail/worst = mean±std over seeds)\n")
        lines.append(md_table(agg_table(df, metric)))
    # headline
    lines.append("\n## Headline: GroupDRO vs Ours\n")
    lines.append("(overall-macro = mean over the 6 groups; overall-weighted = sample-weighted "
                 "'entire dataset', dominated by heads g4/g6; tail = mean over tail groups.)\n")
    for meth in [m for m in METHOD_ORDER if m in df.method.unique()]:
        sub = df[df.method == meth]
        def per_seed(fn):
            return sub.groupby("seed").apply(fn, include_groups=False)
        macro = per_seed(lambda s: s["acc"].mean())
        micro = per_seed(lambda s: np.average(s["acc"], weights=s["n"]))
        wg = per_seed(lambda s: s.set_index("group")["acc"].min())
        tl = per_seed(lambda s: s[~s.group.isin(HEAD)]["acc"].mean())
        lines.append(f"- **{PRETTY[meth]}**: overall-weighted {micro.mean():.3f}±{micro.std():.3f}, "
                     f"overall-macro {macro.mean():.3f}±{macro.std():.3f}, "
                     f"tail {tl.mean():.3f}±{tl.std():.3f}, worst-group {wg.mean():.3f}±{wg.std():.3f}")
    txt = "\n".join(lines)
    open(os.path.join(args.run, "REPORT.md"), "w").write(txt)
    print(txt)
    try:
        rstar_lambda_plot(args.run, df, os.path.join(args.run, "rstar_vs_lambda.png"))
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
