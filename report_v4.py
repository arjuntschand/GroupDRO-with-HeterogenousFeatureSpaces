"""Protocol v4 reporting: turn the per-epoch logs into the five selection views.

Reads every family under runs/v4/ (tabular: runs/v4/<family>/epochs/*.csv written by
train_v4_tabular.py; EMBED: curve_*.json written on the GPU box and synced to runs/v4/embed*/)
and writes, per family and view, a metrics_long.csv in the same schema the site already reads
(method, seed, group, n, n_params, accuracy, macro_f1, loss, R_star, excess_loss, auroc), plus
runs/v4/summary.json with the per-seed worst-group series behind every table, and
runs/v4/curves.json with mean per-epoch validation / test loss and weight trajectories.

Selection is per run (seed, and fold on Fed-Heart) on the VALIDATION split; the step size of each
DRO arm is then the grid value with the best mean validation value of the same rule over seeds.
Nothing reads a test number to decide anything.

  python report_v4.py
"""
import csv, glob, json, os, re
from collections import defaultdict
import numpy as np
import pandas as pd

VIEWS = ["fixed", "fixed5", "fixed10", "fixed15", "fixed20", "fixed25", "overall", "worst", "groupavg", "excess"]
import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "runs/v4"


def val_score(df_epoch, view):
    """Validation criterion for one (run, epoch) block of per-group rows. Lower is better."""
    n, l = df_epoch["n_val"].values.astype(float), df_epoch["val_loss"].values.astype(float)
    if view == "overall":
        return float((n * l).sum() / n.sum())
    if view == "worst":
        return float(l.max())
    if view == "groupavg":
        return float(l.mean())
    if view == "excess":
        return float((l - df_epoch["R_star"].values).max())
    raise ValueError(view)


def select_epoch(run, view, last):
    if view == "fixed":
        return last
    if view.startswith("fixed"):
        return min(int(view[5:]), last)          # fixed budget of N epochs (capped at the run's last epoch)
    by = run[run.epoch >= 1].groupby("epoch")
    scores = {e: val_score(d, view) for e, d in by}
    return min(scores, key=scores.get)


def tabular_family(fam):
    files = glob.glob(f"{OUT}/{fam}/epochs/*.csv")
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    last = int(df.epoch.max())
    out = {v: [] for v in VIEWS}
    curves = {}
    for (m, st), dm in df.groupby(["method", "step"]):
        # mean curves for the plots (validation-independent)
        cur = dm[dm.epoch >= 0].groupby(["epoch", "group"]).agg(val_loss=("val_loss", "mean"), test_loss=("test_loss", "mean"),
                                                                weight=("weight", "mean")).reset_index()
        curves[f"{m}|{st:g}"] = cur.to_dict("records")
        for v in VIEWS:
            for (sd, fold), run in dm.groupby(["seed", "fold"]):
                e = select_epoch(run, v, last)
                sel = run[run.epoch == e]
                vs = val_score(sel, "worst" if v.startswith("fixed") else v)
                for _, r in sel.iterrows():
                    out[v].append(dict(method=m, step=st, seed=sd, fold=fold, group=r.group, n=r.n_test, n_params=r.n_params,
                                       accuracy=r.test_acc, macro_f1=r.test_f1, loss=r.test_loss, R_star=r.R_star,
                                       excess_loss=r.test_loss - r.R_star, auroc=r.test_auroc, val_score=vs, epoch=e))
    for v in VIEWS:
        d = pd.DataFrame(out[v])
        if not d.empty:
            d.loc[(d.method == "REMIND") & (d.step == 0.02), "method"] = "REMIND_pub"
        out[v] = d
    return {v: out[v] for v in VIEWS}, curves


def embed_family(fam):
    # 'embed' must not pick up 'embed_disj_*' (it did until 2026-09-23, mixing the two settings)
    files = [f for f in glob.glob(f"{OUT}/{fam}_*/curve_*.json") if fam != "embed" or "embed_disj" not in f]
    if not files:
        return None
    rows = []
    for f in files:
        m = re.search(r"curve_(.+)_s(\d+)\.json$", os.path.basename(f))
        method, seed = m.group(1), int(m.group(2))
        d = os.path.basename(os.path.dirname(f))
        g = re.search(r"g([\d.]+)$", d)
        step = float(g.group(1)) if g else (0.02 if "REMIND" in os.path.basename(f) else 0.0)   # the plain *_bl dirs ran REMIND at its published 0.02
        c = json.load(open(f))
        n_params = c.get("n_params", float("nan"))
        for e in c["curve"]:
            if "val_full" not in e:
                continue
            for grp, v in e["val_full"].items():
                t = e["test_full"].get(grp, {})
                lam = e.get("lambda")
                gi = c["groups"].index(grp) if lam and grp in c["groups"] else None
                rows.append(dict(method=method, step=step, seed=seed, fold=0, epoch=e["epoch"] + 1, group=grp,
                                 n_val=v["n"], val_loss=v["loss"], val_acc=v["acc"], val_auroc=float("nan"),
                                 n_test=t.get("n"), test_loss=t.get("loss"), test_acc=t.get("acc"), test_f1=t.get("macro_f1"),
                                 test_auroc=float("nan"), weight=(lam[gi] if gi is not None else float("nan")),
                                 R_star=v.get("R_star", 0.0), n_params=n_params))
    df = pd.DataFrame(rows)
    # REMIND at its published gamma is a separate row from REMIND with a validated gamma
    df.loc[(df.method == "REMIND") & (df.step == 0.02), "method"] = "REMIND_pub"
    last = int(df.epoch.max())
    out = {v: [] for v in VIEWS}; curves = {}
    for (m, st), dm in df.groupby(["method", "step"]):
        cur = dm.groupby(["epoch", "group"]).agg(val_loss=("val_loss", "mean"), test_loss=("test_loss", "mean"),
                                                 weight=("weight", "mean")).reset_index()
        curves[f"{m}|{st:g}"] = cur.to_dict("records")
        for v in VIEWS:
            for sd, run in dm.groupby("seed"):
                e = select_epoch(run, v, last)
                sel = run[run.epoch == e]
                vs = val_score(sel, "worst" if v.startswith("fixed") else v)
                for _, r in sel.iterrows():
                    out[v].append(dict(method=m, step=st, seed=sd, fold=0, group=r.group, n=r.n_test, n_params=r.n_params,
                                       accuracy=r.test_acc, macro_f1=r.test_f1, loss=r.test_loss, R_star=r.R_star,
                                       excess_loss=r.test_loss - r.R_star, auroc=float("nan"), val_score=vs, epoch=e))
    return {v: pd.DataFrame(out[v]) for v in VIEWS}, curves


STEP_OPTS = [0.1, 0.5, 2.0, 10.0]      # second selector on the site: force one step size for every arm that was run at it


def pick_step(df, force=None):
    """One step size per method: the grid value with the best mean validation score (over seeds and
    folds), or `force` for every method that was run at that value (others keep their validated choice)."""
    keep = []
    for m, dm in df.groupby("method"):
        by = dm.groupby("step").val_score.mean()
        best = by.idxmin()
        if force is not None and any(abs(st - force) < 1e-9 for st in by.index):
            best = [st for st in by.index if abs(st - force) < 1e-9][0]
        keep.append(dm[dm.step == best].assign(chosen_step=best))
    return pd.concat(keep, ignore_index=True)


def pool_folds(df):
    """Fed-Heart: pool the folds per seed, weighting each fold's group metric by its test count,
    so every patient counts once (same as run_fedheart_cv)."""
    if df.fold.nunique() == 1:
        return df
    rows = []
    for (m, sd, g), d in df.groupby(["method", "seed", "group"]):
        w = d.n.values.astype(float); W = w.sum()
        r = d.iloc[0].to_dict()
        for k in ["accuracy", "macro_f1", "loss", "excess_loss", "auroc"]:
            vals = d[k].values.astype(float); ok = ~np.isnan(vals)
            r[k] = float((vals[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")
        r["n"] = int(W); r["fold"] = -1; rows.append(r)
    return pd.DataFrame(rows)


def main():
    summary = {}; all_curves = {}
    fams = [("nhanes", tabular_family), ("nhanes_nooverlap", tabular_family), ("fedheart", tabular_family),
            ("fedheart_nooverlap", tabular_family), ("embed", embed_family), ("embed_disj", embed_family)]
    if OUT != "runs/v4":       # EMBED ran once, under runs/v4; the tabular variants borrow it
        global_embed = "runs/v4"
        fams = [(f, fn) for f, fn in fams if not f.startswith("embed")]
    for fam, fn in fams:
        res = fn(fam)
        if res is None:
            print(f"{fam}: nothing yet"); continue
        views, curves = res
        all_curves[fam] = curves
        summary[fam] = {}
        for v, df0 in views.items():
          if df0.empty:
              continue
          for force in [None] + STEP_OPTS:
            v_key = v if force is None else f"{v}|s{force:g}"
            df = pool_folds(pick_step(df0, force))
            os.makedirs(f"{OUT}/{fam}/views/{v_key.replace('|', '_')}", exist_ok=True)
            cols = ["method", "seed", "group", "n", "n_params", "accuracy", "macro_f1", "loss", "R_star", "excess_loss", "auroc", "chosen_step", "epoch"]
            df[cols].to_csv(f"{OUT}/{fam}/views/{v_key.replace('|', '_')}/metrics_long.csv", index=False)
            tab = {}
            for m, dm in df.groupby("method"):
                s = dm.groupby("seed").agg(wacc=("accuracy", "min"), wloss=("loss", "max"), wex=("excess_loss", "max"), wauc=("auroc", "min"))
                ov = dm.groupby("seed").apply(lambda x: float(np.average(x.accuracy, weights=x.n)))
                tab[m] = dict(n_seeds=len(s), worst_acc=float(s.wacc.mean() * 100), worst_loss=float(s.wloss.mean()),
                              worst_excess=float(s.wex.mean()), worst_auroc=float(s.wauc.mean()), overall_acc=float(ov.mean() * 100),
                              step=float(dm.chosen_step.iloc[0]), mean_epoch=float(dm.epoch.mean()),
                              n_params=float(dm.n_params.iloc[0]),
                              per_seed={"worst_acc": s.wacc.to_dict(), "worst_loss": s.wloss.to_dict(), "worst_excess": s.wex.to_dict(), "worst_auroc": s.wauc.to_dict()})
            summary[fam][v_key] = tab
            if force is None: print(f"{fam:20s} {v:9s} " + "  ".join(f"{m}:{t['worst_acc']:.1f}|{t['worst_loss']:.3f}" for m, t in sorted(tab.items())))
    json.dump(summary, open(f"{OUT}/summary.json", "w"))
    json.dump(all_curves, open(f"{OUT}/curves.json", "w"))
    print("wrote", f"{OUT}/summary.json")


if __name__ == "__main__":
    main()
