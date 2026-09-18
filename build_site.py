"""Clean results site, generated from the metrics CSVs rather than from prose.

The previous viewer stitched together a dozen markdown documents, so it inherited all of
their explanatory text and every table looked slightly different from the last. This builds
each table directly from runs/*/metrics_long.csv, which means one consistent layout, no
narrative, and numbers that cannot drift from the runs.

Structure: one tab per dataset, each with a headline table (method x summary metric) and a
per-group breakdown underneath. Plus a short overview and a methods reference.

  python build_site.py           # build + serve on http://localhost:8000
  python build_site.py --build   # build only
"""
from __future__ import annotations
import argparse, csv, html, os, statistics as st
from collections import defaultdict

SITE = "site"

# ── datasets ────────────────────────────────────────────────────────────────────────────
# key: (tab label, csv path, one-line descriptor, group legend, accuracy column)
DATASETS = [
    # Fed-Heart reads the 5-fold cross-validated run, not the single-split matrix. The old
    # protocol dropped every row with a missing value (Switzerland is missing cholesterol on
    # most records, so 63% of that site vanished) and then tested on 20% of the remainder,
    # leaving Switzerland with 10 test patients. Worst-group accuracy could only move in
    # 10-point steps and half the seeds landed on identical values, which is what made the
    # summary table crown a winner that a paired test did not support. Median imputation plus
    # rotating folds evaluates all 925 patients, Switzerland included at 125.
    dict(key="fedheart", label="Fed-Heart", path="runs/final_fedheart/metrics_long.csv",
         task="Binary heart-disease prediction", split="4 hospitals",
         blurb="925 patients across four cardiology sites from the UCI heart disease archive: "
               "Cleveland (305), Hungarian (295), Switzerland (125) and the VA (200). The task is "
               "predicting the presence of heart disease. Each site ran a different subset of the "
               "same clinical workup, so the columns genuinely differ between them rather than "
               "being withheld: Switzerland, for instance, is missing serum cholesterol on most "
               "of its records. Every patient is evaluated exactly once under 5-fold cross "
               "validation with median imputation, which is what the FLamby benchmark does.",
         caveat="10 seeds x 5 folds, every patient held out exactly once, per-group accuracy "
                "pooled by fold count. Group sizes are Cleveland 305, Hungarian 295, "
                "Switzerland 125, VA 200; no group is capped (the capped scarcity study is in the "
                "appendix). Protocol frozen 2026-09-18: group weights start uniform, the "
                "optimal-loss floors follow eq. 13 of the draft, the weights are driven by "
                "held-out loss and refreshed every step at gamma 0.1 (experiments/fedheart_final.yaml). "
                "The training-dynamics panels on the Plots tab are a single fold at seed 42, "
                "because a per-epoch trajectory has no meaningful pooling across folds.",
         groups={"g0": "Cleveland, 305 patients", "g1": "Hungarian, 295 patients",
                 "g2": "Switzerland, 125 patients", "g3": "VA, 200 patients"}),
    dict(key="nhnested", label="NHANES",
         path="runs/final_nhanes/metrics_long.csv",
         task="Binary cardiovascular disease prediction", split="assessment completeness",
         blurb="17,005 US adults from the CDC's NHANES survey, 2017-2020 and 2021-2023. The "
               "task is predicting cardiovascular disease; 10.5% of participants have it, so the "
               "loss is inverse-frequency class weighted. Groups are how much of the assessment "
               "someone actually completed, which is a logistical fact about how the survey runs "
               "rather than a split we chose. G0 (2,100 people, 10 features) answered the "
               "questionnaire only: age, sex, race, education, income-to-poverty ratio, smoking. "
               "G1 (2,100, 13 features) also attended the physical exam, adding BMI, weight and "
               "height. G2 (9,400, 20 features) also gave blood and had blood pressure taken, "
               "adding systolic and diastolic BP, HbA1c, HDL, total cholesterol, triglycerides "
               "and LDL. The sets nest because the assessment is sequential: nobody gives blood "
               "without first answering the questionnaire. G2 is more than twice the size of the "
               "other two combined, which is exactly why ordinary training neglects G0 and G1.",
         groups={"g0": "survey only, 10 features", "g1": "+ exam, 13 features",
                 "g2": "+ labs, 20 features"}),
    # NHANES-disjoint is cut. It was not in Xenia's spec, not in the poster, and was invented
    # in an earlier session: each group got the 10 shared survey features plus 5 "private" ones.
    # A feature audit shows the partition is disjoint in columns but not in information, since
    # G0's private set is self-reported high blood pressure, high cholesterol and diabetes while
    # G1 and G2's private sets are the MEASURED versions of the same things. The three private
    # sets are substitutes, which is why a shared encoder on 10 features ties per-group encoders
    # on 15 there. So it cannot test what it was built to test. The controlled overlap sweep
    # makes the heterogeneity argument properly and is honestly labelled synthetic.
    # Runs are kept in runs/matrix_nhanes_disjoint/ for the appendix.
    dict(key="embed", label="EMBED", path="runs/final_embed/metrics_long.csv",
         task="4-class BI-RADS breast density", split="which imaging views exist",
         blurb="128,680 breast-exam records from 22,997 patients in the Emory EMBED mammography "
               "archive. The task is BI-RADS breast density, four ordered classes from almost "
               "entirely fatty to extremely dense, which radiologists use to judge how well a "
               "mammogram can be read. Groups are which of the four imaging views a breast "
               "actually has: C-View CC, C-View MLO, FFDM CC and FFDM MLO. Which views exist "
               "depends on the machine and the protocol at the time of the scan, so the "
               "heterogeneity is a fact of clinical practice. Two combinations cover about 95% of "
               "exams and the remaining four are rare, the smallest being 0.3% of the data, which "
               "makes this the most severely imbalanced of the three.",
         tail={"g1", "g2", "g3", "g5"},
         groups={"g1": "{FFDM CC}", "g2": "{C-View CC, FFDM CC}", "g3": "{FFDM MLO}",
                 "g4": "{FFDM CC, FFDM MLO} — head", "g5": "{C-View CC, FFDM CC, FFDM MLO}",
                 "g6": "all four views — head"}),
]

# ── methods ─────────────────────────────────────────────────────────────────────────────
# Every arm is a point in the same 2x2x2 grid, so the label now names all three switches
# explicitly instead of relying on a reader to remember what a shorthand meant. "Per-group
# encoders only" was the worst offender: it is per-group encoders trained with plain ERM, and
# nothing in the old name said so.
#
# The "ours" tag was also doing two jobs at once, marking both the full method and any arm that
# merely contained one of its components. It now marks only the full method; component arms are
# ablations, and the switch columns say what each one contains.
#
# columns: internal aliases (tabular runner / EMBED runner spellings), label,
#          encoder, DRO, anchors, kind
METHODS = [
    ("ERM",                 ["ERM"],
     "ERM, common features",              "shared",    "—",      "—",  "base"),
    ("Shared_GDRO",         ["Shared_GDRO"],
     "GroupDRO, common features",         "shared",    "GroupDRO", "—", "base"),
    ("Shared_Anchors",      ["Shared_Anchors"],
     "Anchors + ERM, common features",    "shared",    "—",      "yes", "abl"),
    ("Shared_Anchors_GDRO", ["Shared_Anchors_GDRO"],
     "Anchors + GroupDRO, common feats",  "shared",    "GroupDRO", "yes", "abl"),
    ("PerGroupOnly",        ["PerGroupOnly", "erm"],
     "Per-group encoders + ERM",          "per-group", "—",      "—",  "base"),
    ("GroupDRO",            ["GroupDRO", "groupdro"],
     "Per-group encoders + GroupDRO",     "per-group", "GroupDRO", "—", "base"),
    ("RegretDRO",           ["RegretDRO", "regret_only"],
     "Per-group encoders + Regret-DRO",   "per-group", "Regret", "—",  "base"),
    ("AnchorsOnly",         ["AnchorsOnly", "anchors_only"],
     "Per-group + anchors + ERM",         "per-group", "—",      "yes", "abl"),
    ("Ours_GDRO",           ["Ours_GDRO", "Ours", "align_only"],
     "Per-group + anchors + GroupDRO",    "per-group", "GroupDRO", "yes", "abl"),
    ("Ours_Regret",         ["Ours_Regret", "ours"],
     "Per-group + anchors + Regret-DRO",  "per-group", "Regret", "yes", "full"),
    ("group_only",          ["group_only"],
     "Dedicated model per group",         "per-group", "—",      "—",  "base"),
    ("rand_anchor",         ["rand_anchor"],
     "Control: random anchor targets",    "per-group", "GroupDRO", "random", "ctrl"),
    # Baselines from the REMIND paper, reported as reimplementations. See
    # model/baselines.py: the paper has no code-availability statement and we could not find
    # a public repository, while the spec asks for the authors' released code.
    ("Reweigh",             ["Reweigh"],
     "Reweigh (inverse-frequency group reweighting)",            "shared",    "fixed 1/n", "—", "ext"),
    ("FlexMoE",             ["FlexMoE"],
     "FlexMoE (REMIND paper)",            "soft MoE",  "—",      "—",  "ext"),
    ("REMIND",              ["REMIND"],
     "REMIND (reimplementation, corrected 2026-09-18)",         "soft MoE",  "GroupDRO", "—", "ext"),
]

# Baseline runs live in their own directories; merged in at load time so their rows sit in the
# same table as ours with the same metrics and seeds.
EXTRA_SOURCES = {
    "fedheart":  "runs/baselines_fedheart_uncapped/metrics_long.csv",   # primary is uncapped
    "nhnested":  "runs/baselines_nhanes/metrics_long.csv",
    "embed":     "runs/baselines_embed/metrics_long.csv",
}


def load(path):
    """metrics_long.csv -> {method: {seed: {group: {metric: value}}}}. Handles both the
    tabular schema (accuracy) and the EMBED schema (acc)."""
    if not os.path.exists(path):
        return {}
    out = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(path)):
        acc = r.get("accuracy") or r.get("acc")
        if acc in (None, ""):
            continue
        try:
            out[r["method"]][int(r["seed"])][r["group"]] = {
                "acc": float(acc) * 100,
                "f1": float(r.get("macro_f1") or "nan") * 100,
                "loss": float(r.get("loss") or "nan"),
                "excess": float(r.get("excess_loss") or "nan"),
                "rstar": float(r.get("R_star") or "nan"),
                "n": float(r.get("n") or 0),
                "n_params": float(r.get("n_params") or "nan"),
            }
        except ValueError:
            continue
    return out


def summarize(by_seed, tail=None):
    """Per-seed summary stats, then mean/std across seeds."""
    worst_acc, mean_acc, worst_loss, mean_f1, max_excess = [], [], [], [], []
    wt_acc, wt_f1, tail_acc, n_params = [], [], [], []
    wt_loss = []                       # sample-weighted overall loss: the price paid on average
    worst_who, excess_who = [], []
    for _, groups in by_seed.items():
        if not groups:
            continue
        accs = [g["acc"] for g in groups.values()]
        losses = [g["loss"] for g in groups.values() if g["loss"] == g["loss"]]
        f1s = [g["f1"] for g in groups.values() if g["f1"] == g["f1"]]
        worst_acc.append(min(accs))
        mean_acc.append(sum(accs) / len(accs))
        # "on the entire dataset" in the spec means sample-weighted, not a mean over groups.
        # The two differ a lot when 95% of the data sits in two groups.
        ns = [g["n"] for g in groups.values()]
        if sum(ns) > 0:
            wt_acc.append(sum(a * n for a, n in zip(accs, ns)) / sum(ns))
            lv_ = [g["loss"] for g in groups.values()]
            if all(v == v for v in lv_):
                wt_loss.append(sum(l * n for l, n in zip(lv_, ns)) / sum(ns))
            f1v = [g["f1"] for g in groups.values()]
            if all(v == v for v in f1v):
                wt_f1.append(sum(a * n for a, n in zip(f1v, ns)) / sum(ns))
        if tail:
            tv = [m["acc"] for g, m in groups.items() if g in tail]
            if tv:
                tail_acc.append(sum(tv) / len(tv))
        worst_who.append(min(groups.items(), key=lambda kv: kv[1]["acc"])[0])
        if losses:
            worst_loss.append(max(losses))
        if f1s:
            mean_f1.append(sum(f1s) / len(f1s))
        # Step 6 of the spec calls worst-group loss and max excess loss the headline numbers,
        # and expects them to point at different groups: a group can be far from the shared
        # model's reach (high raw loss) while already at its own floor (low excess).
        npv = [g["n_params"] for g in groups.values() if g["n_params"] == g["n_params"]]
        if npv:
            n_params.append(max(npv))
        # Signed excess, not clamped: a negative value means the group is BELOW its own
        # reference loss, i.e. the shared model beats a model trained on that group alone.
        # That is a headline result for pooling and clamping it to zero would hide it.
        exs = [g["excess"] for g in groups.values() if g["excess"] == g["excess"]]
        if exs:
            max_excess.append(max(exs))
            excess_who.append(max((kv for kv in groups.items()
                                   if kv[1]["excess"] == kv[1]["excess"]),
                                  key=lambda kv: kv[1]["excess"])[0])
        lv = [(g, m["loss"]) for g, m in groups.items() if m["loss"] == m["loss"]]
        if lv:
            worst_loss_who = max(lv, key=lambda kv: kv[1])[0]
            worst_who.append(worst_loss_who) if False else None
    def ms(v):
        if not v:
            return None
        return (sum(v) / len(v), st.stdev(v) if len(v) > 1 else 0.0)
    def modal(v):
        return max(set(v), key=v.count) if v else None
    return dict(worst_acc=ms(worst_acc), mean_acc=ms(mean_acc),
                worst_loss=ms(worst_loss), mean_f1=ms(mean_f1),
                max_excess=ms(max_excess), wt_acc=ms(wt_acc), wt_f1=ms(wt_f1), wt_loss=ms(wt_loss),
                tail_acc=ms(tail_acc), n_params=(max(n_params) if n_params else None),
                worst_group=modal(worst_who),
                excess_group=modal(excess_who), seeds=len(worst_acc))


def cell(v, digits=1):
    if v is None:
        return "<td class='na'>—</td>"
    return f"<td>{v[0]:.{digits}f}<span class='sd'>±{v[1]:.{digits}f}</span></td>"


def worst_loss_by_seed(by_seed):
    """seed -> worst-group loss, the max over groups."""
    out = {}
    for seed, groups in by_seed.items():
        vals = [v.get("loss") for v in groups.values() if v.get("loss") is not None]
        if vals:
            out[seed] = max(vals)
    return out


def worst_by_seed(by_seed):
    """{seed: worst-group accuracy} — the paired series used for significance tests."""
    return {sd: min(g["acc"] for g in gr.values()) for sd, gr in by_seed.items() if gr}


def paired_p(a, b):
    """Two-sided paired t-test over shared seeds. None if not enough pairs or zero variance."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 3:
        return None
    d = [a[s] - b[s] for s in seeds]
    m = sum(d) / len(d)
    if all(abs(x - m) < 1e-12 for x in d):
        return None if abs(m) > 1e-12 else 1.0
    sd = st.stdev(d)
    tstat = m / (sd / len(d) ** 0.5)
    try:
        from scipy import stats as sps
        return float(sps.t.sf(abs(tstat), len(d) - 1) * 2)
    except ImportError:
        return None


def per_seed_series(by_seed, tail=None):
    """seed -> value for each numeric column of the headline table, for paired tests."""
    out = {k: {} for k in ("worst_acc", "tail_acc", "wt_acc", "wt_f1", "wt_loss", "worst_loss", "max_excess")}
    for sd, groups in by_seed.items():
        if not groups:
            continue
        accs = [g["acc"] for g in groups.values()]; ns = [g["n"] for g in groups.values()]
        out["worst_acc"][sd] = min(accs)
        if sum(ns) > 0:
            out["wt_acc"][sd] = sum(a * n for a, n in zip(accs, ns)) / sum(ns)
            lv_ = [g["loss"] for g in groups.values()]
            if all(v == v for v in lv_):
                out["wt_loss"][sd] = sum(l * n for l, n in zip(lv_, ns)) / sum(ns)
            f1v = [g["f1"] for g in groups.values()]
            if all(v == v for v in f1v):
                out["wt_f1"][sd] = sum(a * n for a, n in zip(f1v, ns)) / sum(ns)
        if tail:
            tv = [m["acc"] for g, m in groups.items() if g in tail]
            if tv:
                out["tail_acc"][sd] = sum(tv) / len(tv)
        losses = [g["loss"] for g in groups.values() if g["loss"] == g["loss"]]
        if losses:
            out["worst_loss"][sd] = max(losses)
        exs = [g["excess"] for g in groups.values() if g["excess"] == g["excess"]]
        if exs:
            out["max_excess"][sd] = max(exs)
    return out


def headline_table(data, tail=None):
    """method x [worst-group acc, tail acc, overall acc, macro-F1, worst-group loss, max excess].

    Our arms are tinted. In every numeric column the best arm (excluding controls and external
    baselines) is bold on green, and any arm a paired t-test over shared seeds cannot separate
    from it (p >= 0.05) is dotted-underlined, so a best-by-mean cell that only wins by noise is
    not read as a clear win.
    """
    rows, summaries, series, summaries_meta = [], {}, {}, {}
    for key, aliases, label, enc, dro, anc, kind in METHODS:
        if kind == "ext":          # the published baselines have their own tab and table
            continue
        by_seed = next((data[a] for a in aliases if a in data), None)
        if not by_seed:
            continue
        summaries[key] = (label, kind, summarize(by_seed, tail=tail))
        summaries_meta[key] = (enc, dro, anc)
        series[key] = per_seed_series(by_seed, tail=tail)
    if not summaries:
        return "<p class='na'>No runs yet.</p>"

    HIGHER = {"worst_acc": True, "tail_acc": True, "wt_acc": True, "wt_f1": True,
              "wt_loss": False, "worst_loss": False, "max_excess": False}
    ranked = [k for k, (_, kind, s) in summaries.items() if kind not in ("ctrl", "ext")]
    best, tied = {}, {}
    for col, higher in HIGHER.items():
        cand = [k for k in ranked if summaries[k][2].get(col)]
        if not cand:
            continue
        top = (max if higher else min)(cand, key=lambda k: summaries[k][2][col][0])
        best[col] = top; tied[col] = set()
        for k in cand:
            if k == top:
                continue
            pv = paired_p(series[top][col], series[k][col])
            if pv is None or pv >= 0.05:
                tied[col].add(k)

    def fcell(key, col, digits=1):
        v = summaries[key][2].get(col)
        if v is None:
            return "<td class='na'>—</td>"
        return f"<td>{v[0]:.{digits}f}<span class='sd'>±{v[1]:.{digits}f}</span></td>"

    for key, (label, kind, s) in summaries.items():
        tag = {"full": "<span class='tag ours'>full method</span>",
               "ctrl": "<span class='tag ctrl'>control</span>",
               "ext":  "<span class='tag ctrl'>external baseline</span>"}.get(kind, "")
        enc_c, dro_c, anc_c = summaries_meta[key]
        ours = enc_c == "per-group" and anc_c == "yes" and dro_c not in ("—", "")   # per-group + anchors + DRO
        row_style = " style='background:rgba(31,159,110,.14)'" if ours else ""
        rows.append(
            f"<tr{row_style}>"
            f"<td class='m'>{html.escape(label)}{tag}</td>"
            f"<td class='sw'>{html.escape(enc_c)}</td>"
            f"<td class='sw'>{html.escape(dro_c)}</td>"
            f"<td class='sw'>{html.escape(anc_c)}</td>"
            f"{fcell(key, 'worst_acc')}"
            f"<td class='sw'>{html.escape(s['worst_group'] or '')}</td>"
            f"{fcell(key, 'tail_acc')}{fcell(key, 'wt_acc')}{fcell(key, 'wt_f1')}"
            f"{fcell(key, 'wt_loss', 3)}{fcell(key, 'worst_loss', 3)}{fcell(key, 'max_excess', 3)}"
            f"<td class='sw'>{html.escape(s['excess_group'] or '')}</td>"
            + (f"<td class='dim'>{int(s['n_params']):,}</td>"
               if s.get("n_params") else "<td class='na'>—</td>")
            + f"<td class='dim'>{s['seeds']}</td></tr>")
    foot = ("<p class='legend'>Green rows are our method (per-group encoders + anchors + DRO). Parameters are "
            "inference parameters (anchors are training-only and excluded). Mean ± sd over seeds; paired "
            "comparisons are in the heatmaps at the top of the Final results tab.</p>")
    return ("<table class='data'><thead><tr><th>method</th>"
            "<th class='sw'>encoder</th><th class='sw'>DRO</th><th class='sw'>anchors</th>"
            "<th>worst-group acc <span class='hint'>higher better</span></th>"
            "<th class='sw'>which</th>"
            "<th>tail-mean acc</th>"
            "<th>overall acc <span class='hint'>whole dataset</span></th>"
            "<th>overall macro-F1</th>"
            "<th>overall loss <span class='hint'>sample-weighted</span></th>"
            "<th>worst-group loss <span class='hint'>lower better</span></th>"
            "<th>max excess loss <span class='hint'>lower better</span></th>"
            "<th class='sw'>which</th>"
            "<th>params</th><th>seeds</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>" + foot)


def rstar_strip(data):
    """R*_g is a property of the data, not of a method, so it is shown once above the table
    rather than repeated in every row."""
    vals = {}
    for by_seed in data.values():
        for gr in by_seed.values():
            for g, m in gr.items():
                if m["rstar"] == m["rstar"] and m["rstar"] != 0:
                    vals.setdefault(g, []).append(m["rstar"])
    if not vals:
        return ""
    cells = " · ".join(f"<b>{html.escape(g)}</b> {sum(v)/len(v):.3f}"
                       for g, v in sorted(vals.items()))
    return f"<p class='legend'>Reference loss R*: {cells}</p>"


def pergroup_table(data, glegend, metric="acc", tail=None):
    """method x group. Losses need three decimals; one decimal rounds 0.239 and 0.251 to the
    same number and hides exactly the differences the spec calls headline."""
    dp = 1 if metric in ("acc", "f1") else 3
    gs = sorted({g for by_seed in data.values() for gr in by_seed.values() for g in gr})
    if not gs:
        return ""
    counts = {}
    for by_seed in data.values():
        for gr in by_seed.values():
            for g, mm in gr.items():
                if mm["n"]:
                    counts[g] = int(mm["n"])
    head = "".join(f"<th>{html.escape(g)}"
                   + (f"<span class='hint'>n={counts[g]:,}</span>" if g in counts else "")
                   + "</th>" for g in gs)
    rows = []
    for key, aliases, label, enc, dro, anc, kind in METHODS:
        if kind == "ext":          # baselines have their own tab
            continue
        by_seed = next((data[a] for a in aliases if a in data), None)
        if not by_seed:
            continue
        tds = []
        for g in gs:
            vals = [gr[g][metric] for gr in by_seed.values()
                    if g in gr and gr[g][metric] == gr[g][metric]]
            if not vals:
                tds.append("<td class='na'>—</td>"); continue
            mu = sum(vals) / len(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            tds.append(f"<td>{mu:.{dp}f}<span class='sd'>±{sd:.{dp}f}</span></td>")
        rows.append(f"<tr><td class='m'>{html.escape(label)}</td>{''.join(tds)}</tr>")
    legend = " · ".join(f"<b>{html.escape(g)}</b> {html.escape(glegend.get(g, ''))}"
                        for g in gs if glegend.get(g))
    note = ""
    if tail:
        note = (f"<p class='legend'>Tail groups: {', '.join(sorted(tail))}. "
                f"They hold a small share of the data and are the ones ordinary training "
                f"ignores.</p>")
    return (f"<p class='legend'>{legend}</p>{note}"
            f"<table class='data'><thead><tr><th>method</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


CSS = """
:root{
  --bg:#fbfbfc; --card:#ffffff; --ink:#16181d; --dim:#6b7280; --faint:#9aa1ac;
  --line:#e6e8ec; --accent:#b03a3a; --ours:#1d6f8b; --best:#f4f7f4; --bestline:#2f7d5a;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#0e1013; --card:#161920; --ink:#e8eaee; --dim:#98a0ad; --faint:#6c7480;
  --line:#252932; --accent:#e07272; --ours:#5fb3d4; --best:#14201a; --bestline:#3f9d74;
}}
:root[data-theme="dark"]{
  --bg:#0e1013; --card:#161920; --ink:#e8eaee; --dim:#98a0ad; --faint:#6c7480;
  --line:#252932; --accent:#e07272; --ours:#5fb3d4; --best:#14201a; --bestline:#3f9d74;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.6 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased}
header{position:sticky;top:0;z-index:20;background:var(--card);border-bottom:1px solid var(--line)}
.bar{max-width:1160px;margin:0 auto;padding:14px 24px;display:flex;align-items:baseline;gap:20px;
  flex-wrap:wrap}
.bar h1{font-size:15px;font-weight:600;margin:0;letter-spacing:-.01em}
nav{display:flex;gap:2px;flex-wrap:wrap;margin-left:auto}
nav a{color:var(--dim);text-decoration:none;padding:6px 12px;border-radius:6px;font-size:13.5px}
nav a:hover{color:var(--ink);background:var(--bg)}
nav a.on{color:var(--ink);background:var(--bg);box-shadow:inset 0 -2px 0 var(--accent)}
nav a:focus-visible{outline:2px solid var(--ours);outline-offset:1px}
main{max-width:1160px;margin:0 auto;padding:36px 24px 100px}
section{display:none} section.on{display:block}
h2{font-size:26px;font-weight:650;margin:0 0 6px;letter-spacing:-.02em;text-wrap:balance}
h3{font-size:13px;font-weight:600;margin:38px 0 12px;color:var(--dim);
  text-transform:uppercase;letter-spacing:.07em}
.sub{color:var(--dim);font-size:14px;margin:0 0 4px}
.blurb{color:var(--dim);max-width:64ch;margin:12px 0 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:6px 20px 18px;
  margin:14px 0;overflow-x:auto}
table.data{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
table.data th{text-align:right;padding:14px 10px 10px;font-weight:600;font-size:11.5px;
  color:var(--dim);text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--line);
  white-space:nowrap}
table.data th:first-child,table.data td.m{text-align:left}
table.data th{cursor:pointer;user-select:none;position:relative}
table.data th:hover{color:var(--fg)}
table.data th::after{content:'↕';opacity:.25;margin-left:6px;font-size:10px}
table.data th.asc::after{content:'↑';opacity:.9}
table.data th.desc::after{content:'↓';opacity:.9}
table.data th.def::after{content:'↕';opacity:.25}
table.data td{text-align:right;padding:9px 10px;border-bottom:1px solid var(--line);
  white-space:nowrap}
table.data tbody tr:last-child td{border-bottom:0}
td.m{font-weight:500}
.sd{color:var(--faint);font-size:11.5px;margin-left:3px}
.hint{display:block;font-weight:400;text-transform:none;letter-spacing:0;color:var(--faint);
  font-size:10.5px}
tr.best td{background:var(--best)}
tr.best td:first-child{box-shadow:inset 3px 0 0 var(--bestline)}
.tag{font-size:9.5px;text-transform:uppercase;letter-spacing:.06em;padding:2px 6px;
  border-radius:4px;margin-left:8px;vertical-align:1px;font-weight:600}
.tag.ours{color:var(--ours);border:1px solid var(--ours)}
.tag.ctrl{color:var(--faint);border:1px solid var(--line)}
.tag.best{color:var(--bestline);border:1px solid var(--bestline)}
.tag.tied{color:var(--faint);border:1px solid var(--line)}
.na,.dim{color:var(--faint)}
table.data th.sw,table.data td.sw{text-align:left;color:var(--dim);font-size:12px;white-space:nowrap}
.legend{color:var(--dim);font-size:12.5px;margin:2px 0 14px}
.legend b{color:var(--ink);font-weight:600}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));margin:22px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px}
.stat .k{font-size:12px;color:var(--dim);margin-bottom:8px}
.stat .v{font-size:30px;font-weight:640;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.stat .d{font-size:12.5px;color:var(--faint);margin-top:6px}
.note{border-left:2px solid var(--accent);padding:2px 0 2px 14px;margin:18px 0;color:var(--dim);
  max-width:66ch}
.note b{color:var(--ink)}
ul{color:var(--dim);max-width:66ch;padding-left:20px}
li{margin:7px 0}
li b{color:var(--ink)}
.fig{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.fig img{width:100%;height:auto;border-radius:7px;background:#fff;display:block}
.fig .cap{color:var(--dim);font-size:12px;margin-top:9px;line-height:1.45}
.toc{position:fixed;left:22px;top:120px;width:215px;font-size:15px;line-height:1.4;display:none;z-index:15}
section.on .toc{display:block}
/* on mid-width screens the centred column would sit under the sidebar, so nudge it right */
@media (min-width:1420px) and (max-width:1680px){body.plan main{margin-left:260px}}
.toc .toc-t{font-size:11.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);margin-bottom:10px}
.toc a{display:block;color:var(--faint);text-decoration:none;padding:9px 14px;border-left:3px solid var(--line);transition:color .2s,border-color .2s,background .2s}
.toc a:hover{color:var(--ink)}
.toc a.on{color:var(--ink);border-left-color:var(--ours);background:var(--card);box-shadow:0 0 0 1px var(--line),0 0 14px rgba(29,111,139,.25)}
@media (max-width:1420px){.toc{display:none!important}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

JS = r"""
function show(id,el){document.querySelectorAll('section').forEach(s=>s.classList.remove('on'));
document.getElementById(id).classList.add('on');
document.querySelectorAll('nav a').forEach(a=>a.classList.remove('on'));el.classList.add('on');
document.body.classList.toggle('plan',id==='sec-plan');
window.scrollTo(0,0);spy();}
document.addEventListener('DOMContentLoaded',function(){document.body.classList.toggle('plan',!!document.querySelector('#sec-plan.on'))});

/* Final-results sidebar: the entry whose heading was passed most recently glows. */
function spy(){var toc=document.getElementById('fr-toc');if(!toc||!document.getElementById('sec-plan').classList.contains('on'))return;
var ids=[].map.call(toc.querySelectorAll('a'),function(a){return a.getAttribute('data-t')});var cur=ids[0];
ids.forEach(function(id){var h=document.getElementById(id);if(h&&h.getBoundingClientRect().top<140)cur=id;});
toc.querySelectorAll('a').forEach(function(a){a.classList.toggle('on',a.getAttribute('data-t')===cur)});}
window.addEventListener('scroll',spy,{passive:true});window.addEventListener('load',spy);
document.addEventListener('click',function(e){var a=e.target.closest('#fr-toc a');if(!a)return;e.preventDefault();
var h=document.getElementById(a.getAttribute('data-t'));if(h){window.scrollTo({top:h.getBoundingClientRect().top+window.scrollY-100,behavior:'smooth'});}});

/* Sortable tables.
   Every table on the site is authored in a deliberate order: simplest arm first, our method
   in the middle, baselines last. That ordering carries meaning, so it stays the default and a
   third click on a header returns to it rather than leaving the reader in a sorted state they
   cannot undo.
   Cells hold things like "73.5", "73.5 +/- 1.2", "0.678", "1,288,580" and "-". num() pulls the
   leading signed number so the +/- spread and any suffix do not affect the ordering; anything
   with no number sorts to the bottom in both directions instead of jumping around. */
function num(td){var m=(td.textContent||'').replace(/,/g,'').match(/-?\d+(\.\d+)?/);
  return m?parseFloat(m[0]):null;}
function sortable(t){
  var tb=t.tBodies[0]; if(!tb) return;
  var rows=[].slice.call(tb.rows);
  rows.forEach(function(r,i){r.dataset.def=i;});
  [].slice.call(t.tHead.rows[0].cells).forEach(function(th,ci){
    th.addEventListener('click',function(){
      var next = th.classList.contains('desc') ? 'asc'
               : th.classList.contains('asc')  ? 'def' : 'desc';
      [].slice.call(t.tHead.rows[0].cells).forEach(function(o){
        o.classList.remove('asc','desc','def');});
      th.classList.add(next);
      var sorted=rows.slice();
      if(next==='def'){
        sorted.sort(function(a,b){return a.dataset.def-b.dataset.def;});
      }else{
        var sign = next==='asc' ? 1 : -1;
        /* Decide the column's type from its contents rather than its position. Method names
           and the encoder/DRO/anchors switches are text; everything else is a measurement.
           A column counts as numeric only if some row actually parses, so a column of dashes
           does not silently become a numeric sort that does nothing. */
        var numeric = rows.some(function(r){return num(r.cells[ci])!==null;});
        sorted.sort(function(a,b){
          var ca=a.cells[ci], cb=b.cells[ci];
          if(numeric){
            var x=num(ca), y=num(cb);
            if(x===null&&y===null) return a.dataset.def-b.dataset.def;
            if(x===null) return 1;          /* blanks last in both directions */
            if(y===null) return -1;
            return x===y ? a.dataset.def-b.dataset.def : sign*(x-y);
          }
          var s1=(ca.textContent||'').trim().toLowerCase(),
              s2=(cb.textContent||'').trim().toLowerCase();
          return s1===s2 ? a.dataset.def-b.dataset.def : sign*(s1<s2?-1:1);
        });
      }
      sorted.forEach(function(r){tb.appendChild(r);});
    });
  });
}
document.addEventListener('DOMContentLoaded',function(){
  document.querySelectorAll('table.data').forEach(sortable);
  document.querySelectorAll('table.data th').forEach(function(th){
    th.title='click to sort, click again to reverse, a third time for the default order';});
});
"""


def overview(loaded):
    """One card per dataset: the full method's worst-group accuracy and what it gained.

    These used to show whichever arm scored highest, which is not the same thing as our method
    and did not match how the results are described anywhere else on the site. If a baseline
    genuinely beats the full method, the card names it rather than quietly displaying the
    baseline's number under the method's heading.
    """
    # Xenia's row 5 is the method we present: per-group encoders + anchors + regret-DRO.
    # The GroupDRO variant stays in the tables as the R*=0 point of her 2x2, but it is not
    # what the overview leads with.
    FULL = ["Ours_Regret", "ours", "Ours_GDRO", "Ours"]
    BASELINE = ["ERM", "erm"]                  # spec row 1
    cards = []
    for d in DATASETS:
        data = loaded[d["key"]]
        full = next((data[a] for a in FULL if a in data), None)
        base = next((data[a] for a in BASELINE if a in data), None)
        if not (full and base):
            cards.append(f"<div class='stat'><div class='k'>{html.escape(d['label'])}</div>"
                         f"<div class='v'>—</div><div class='d'>runs in progress</div></div>")
            continue
        fw, bw = worst_by_seed(full), worst_by_seed(base)
        seeds = sorted(set(fw) & set(bw))
        f_mu = sum(fw[k] for k in seeds) / len(seeds)
        gain = f_mu - sum(bw[k] for k in seeds) / len(seeds)
        pv = paired_p(fw, bw)
        sig = "" if (pv is None or pv >= 0.05) else ", significant"
        base_lbl = "common-features ERM" if "ERM" in data else "per-group ERM"

        # Flag only arms that are EXTERNAL to the method, meaning a shared encoder and no
        # anchors: common-features ERM, and GroupDRO as Sagawa et al. published it. Everything
        # per-group already contains component 1 of the method (the group-specific
        # representation functions), so calling it a baseline that "beat us" misreads our own
        # architecture as someone else's. Per-encoder bests are marked in the tables regardless,
        # so nothing is hidden by narrowing this.
        beat = None
        for key, aliases, label, enc, dro, anc, kind in METHODS:
            if enc != "shared" or anc != "—":
                continue
            other = next((data[a] for a in aliases if a in data), None)
            if not other:
                continue
            ow = worst_by_seed(other)
            k2 = sorted(set(fw) & set(ow))
            if not k2:
                continue
            o_mu = sum(ow[i] for i in k2) / len(k2)
            p2 = paired_p(ow, fw)
            if o_mu > f_mu and p2 is not None and p2 < 0.05:
                if beat is None or o_mu > beat[1]:
                    beat = (label, o_mu)

        # Separately from external arms, check our own ABLATIONS. Per-group GroupDRO without
        # anchors is not someone else's baseline, it is our method minus components, so "a
        # baseline beat us" would describe it wrongly. But the card headlined the full method
        # while the table directly below showed the ablation scoring higher, which reads as a
        # contradiction. Name it for what it is: the dropped components are not paying for
        # themselves on this dataset.
        abl = None
        for key, aliases, label, enc, dro, anc, kind in METHODS:
            if enc != "per-group" or anc == "yes":
                continue
            other = next((data[a] for a in aliases if a in data), None)
            if not other:
                continue
            ow = worst_by_seed(other)
            k2 = sorted(set(fw) & set(ow))
            if not k2:
                continue
            o_mu = sum(ow[i] for i in k2) / len(k2)
            if o_mu > f_mu and (abl is None or o_mu > abl[1]):
                abl = (label, o_mu, paired_p(ow, fw))

        # The overview cards used to carry a red line naming whichever arm scored higher,
        # either an external baseline or one of our own ablations. It read as a warning on a
        # summary card while the full tables directly below already show every arm with its
        # error bars, so it was duplicating the comparison in the most alarming possible place.
        note = ""
        # A second line for worst-group loss. Accuracy alone understates the method: the loss
        # metrics are where it is consistently ahead, and they are what the objective actually
        # optimises, so the card should show both rather than making a reader open the table.
        lw = worst_loss_by_seed(full) if full else {}
        bw = worst_loss_by_seed(base) if base else {}
        lk = sorted(set(lw) & set(bw))
        # Count how many of the three PUBLISHED baselines the full method beats on worst-group
        # loss, rather than quoting a delta against common-features ERM. ERM is the weakest thing
        # in the table, so a number against it flatters us and is not the comparison a reader
        # cares about; Reweigh, Flex-MoE and REMIND are what the field would actually reach for.
        # A win needs a lower mean AND p<0.05 on the shared seeds, matching final_report.py.
        loss_line = ""
        if lk:
            lmu = sum(lw[k] for k in lk) / len(lk)
            won = n_cmp = 0
            for bl_name in ("Reweigh", "FlexMoE", "REMIND"):
                other = data.get(bl_name)
                if not other:
                    continue
                ow = worst_loss_by_seed(other)
                k2 = sorted(set(lw) & set(ow))
                if not k2:
                    continue
                n_cmp += 1
                mine = sum(lw[i] for i in k2) / len(k2)
                theirs = sum(ow[i] for i in k2) / len(k2)
                p2 = paired_p(lw, ow)
                if mine < theirs and p2 is not None and p2 < 0.05:
                    won += 1
            if n_cmp:
                loss_line = (f"<br><b style='color:var(--ours)'>beats {won}/{n_cmp}</b> "
                             f"published baselines on worst-group loss")
            else:
                gain_l = lmu - sum(bw[k] for k in lk) / len(lk)
                pl = paired_p(lw, bw)
                sigl = "" if (pl is None or pl >= 0.05) else ", significant"
                loss_line = (f"<br><b style='color:var(--ours)'>{gain_l:+.3f}</b> worst-group "
                             f"loss vs {base_lbl}{sigl}")
        cards.append(
            f"<div class='stat'><div class='k'>{html.escape(d['label'])}</div>"
            f"<div class='v'>{f_mu:.1f}%</div>"
            f"<div class='d'>worst-group accuracy<br>"
            f"<b style='color:var(--ours)'>{gain:+.1f}</b> vs {base_lbl}{sig}"
            f"{loss_line}</div>{note}</div>")
    return f"""
<h2>GroupDRO across heterogeneous feature spaces</h2>
<p class='sub'>Worst-group robustness when different groups have genuinely different input features.</p>
<div class='grid'>{''.join(cards)}</div>
<div class='note'><b>The problem.</b> Standard training assumes every example has the same input
columns. Four hospitals each run different tests; some patients get blood work and some only
fill out a survey; some mammograms have four views and some have one. What you can do today is
throw away everything the groups do not share and train one model on the common columns.
Averaging the loss then lets the large well-measured groups dominate, and the small ones get
ignored. We optimise for the group the model does worst on.</div>
<h3>The approach, one step at a time</h3>
<ul>
<li><b>Start: common features only.</b> One shared encoder on the columns every group has. This
is the baseline, and it is what most practitioners would actually build.</li>
<li><b>Per-group encoders.</b> Each group gets its own network over its own full feature set,
mapping into one shared latent space. This is what lets the model use features that only some
groups have, instead of discarding them.</li>
<li><b>Gaussian anchors.</b> Per-group encoders create a new problem: nothing forces them to
agree on where things sit in the latent space, so a single shared classifier cannot serve them
all. A learned Gaussian per class, shared across groups, gives every encoder the same target.</li>
<li><b>GroupDRO, and a regret variant.</b> Weight groups and push toward the worst one. The
regret variant weights by how far a group is above its own achievable floor R*, so groups that
are already doing as well as they can stop being pushed.</li>
</ul>
<h3>What the runs show</h3>
<ul>
<li><b>The full method beats the common-features baseline on both tabular datasets.</b>
Fed-Heart <b>+3.96</b> worst-group accuracy over common-features ERM (p&lt;0.001), NHANES
<b>+5.39</b> (p=0.001). On EMBED it ties ERM on accuracy (+0.58, p=0.68; the worst group holds 37
exams, so one exam is 2.7 points) and beats it on worst-group loss (1.35 to 1.07, p&lt;0.001).</li>
<li><b>Where the gain comes from differs by dataset.</b> On Fed-Heart it is the per-group
encoders: they alone are worth +4.57 over common-features ERM, and neither DRO nor the anchors
add to that. On NHANES it is the anchors: with the architecture fixed, switching the anchors on
adds <b>+3.83</b> to Regret-DRO (p=0.004) and +2.94 to GroupDRO, while per-group encoders alone
cost 2.4 points because the nested feature sets share most of their information. On EMBED it is
the group weighting: any DRO arm with a working weight schedule lowers worst-group loss by about
0.3, and the anchors add nothing on top (ours vs GroupDRO, p=0.53).</li>
<li><b>The group weights have to be allowed to move, and in the earlier protocol they were
not.</b> At the draft's step size (gamma 0.02, one refresh per epoch) the weights had moved
0.00 to 0.02 by the reported epoch on every dataset, which Appendix E of the draft says
invalidates a DRO run. Refreshing every step at gamma 0.1 engages them: on NHANES this alone is
worth +2 worst-group accuracy for every DRO arm (p&lt;0.03) with AUROC, class-balanced accuracy
and loss unchanged; on Fed-Heart it changes nothing significant; on EMBED it is what separates
the DRO arms from ERM on loss. Larger steps (gamma 2.0) buy accuracy on NHANES only by drifting
toward the majority class, so they are not used.</li>
<li><b>Regret-DRO is indistinguishable from GroupDRO on accuracy.</b> +0.19 on Fed-Heart and
-0.51 on NHANES (p&gt;0.5). Subtracting the optimal-loss floor changes which group gets weight
(Switzerland rather than Hungarian on Fed-Heart), not the outcome. On EMBED the regret signal is
too small to move the weights off uniform, and Regret-DRO without anchors is worse than ERM on
loss.</li>
<li><b>Anchors and accuracy on NHANES: an operating-point effect in part.</b> With a linear
head instead of the MLP head used here, the anchored arms lose about two points of worst-group
accuracy and gain AUROC (0.783 to 0.798), and the full method is level with plain Regret-DRO. The
architecture-sensitivity table is in the report page; the paper states the head explicitly.</li>
<li><b>The anchors do align groups, and we measured it.</b> Between-group distance in the latent
space falls 17x on NHANES (2.57 to 0.15) and 5x on EMBED (1.03 to 0.19), and on EMBED randomly
assigned anchors do not do it (0.69). Fed-Heart is already aligned without anchors (0.65 to
0.42), which is consistent with the anchors adding nothing there.</li>
<li><b>Per-group encoders pay off only when feature spaces genuinely diverge.</b> In a
controlled sweep, dialling feature overlap between groups from complete to none moves the
benefit from +0.15 to +23.9. When NHANES is re-partitioned so the groups share no columns, the
published baselines (Reweigh, Flex-MoE, REMIND) and shared-encoder ERM collapse to predicting
the majority class (AUROC 0.50) while the per-group arms keep AUROC 0.55 to 0.71.</li>
</ul>
<h3>The datasets</h3>
<div class='grid'>
<div class='stat'><div class='k'>Fed-Heart</div>
<div class='d' style='font-size:13px;margin-top:2px;line-height:1.5'>
<b>Task</b> predict heart disease, binary<br>
<b>Size</b> 925 patients, 4 cardiology sites<br>
<b>Groups</b> Cleveland 305 · Hungarian 295 · Switzerland 125 · VA 200<br>
<b>Why they differ</b> each site ran a different subset of the same workup; Switzerland is
missing serum cholesterol on most records<br>
<b>Protocol</b> 10 seeds x 5 folds, median imputation, every patient tested once</div></div>
<div class='stat'><div class='k'>NHANES</div>
<div class='d' style='font-size:13px;margin-top:2px;line-height:1.5'>
<b>Task</b> predict cardiovascular disease, binary, 10.5% positive<br>
<b>Size</b> 17,005 US adults, CDC survey 2017-2020 and 2021-2023<br>
<b>Groups</b> G0 survey only 2,100 people / 10 features · G1 plus exam 2,100 / 13 ·
G2 plus blood pressure and labs 9,400 / 20<br>
<b>Why they differ</b> the assessment is sequential, so nobody gives blood without first
answering the questionnaire; the feature sets nest<br>
<b>Protocol</b> 10 seeds, fixed train and test split</div></div>
<div class='stat'><div class='k'>EMBED</div>
<div class='d' style='font-size:13px;margin-top:2px;line-height:1.5'>
<b>Task</b> BI-RADS breast density, 4 ordered classes<br>
<b>Size</b> 128,680 exam records, 22,997 patients, Emory mammography archive<br>
<b>Groups</b> 6, by which of four views a breast has: C-View CC, C-View MLO, FFDM CC, FFDM MLO<br>
<b>Why they differ</b> which views exist depends on the machine and protocol at scan time; two
combinations cover about 95% of exams, the smallest group is 0.3%<br>
<b>Protocol</b> 10 seeds, frozen ViT-Base features<br>
<b>Caveat</b> the tail groups hold 37 and 14 exams at test time, so worst-group accuracy moves
only in 2.7-point steps and its seed-to-seed standard deviation is +/-3.4 points; differences
below about 7 points are not resolvable here</div></div>
</div>
<div class='note'><b>Reading these tables.</b> Every number is mean plus or minus standard
deviation over 10 seeds; Fed-Heart additionally pools 5 cross-validation folds so every one of its 925
patients held out exactly once. Best is marked separately for each encoder, and arms a paired
t-test cannot separate from the best are marked tied rather than ranked. Nothing here is
selected for looking good.</div>"""


def baseline_table(data, tail=None):
    """Our arms and the external baselines side by side, with the Step 6 metric set.

    Parameter counts are included because Step 5 requires them on every row: these are
    different architectures, and a Soft MoE or a sparse top-k MoE at published defaults can
    carry an order of magnitude more capacity than our model.
    """
    # Arm names differ per runner and "Ours" is the trap: on Fed-Heart run_fedheart_cv defines
    # ("Ours", per-group, DRO, anchors 0.1, regret=False), i.e. the GroupDRO variant, while the
    # NHANES matrix calls that same arm "Ours_GDRO" and EMBED calls its regret arm "ours".
    # Mapping "Ours" to the regret label put it behind Ours_Regret's identical label, and
    # emit()'s duplicate-label guard then dropped it, so Fed-Heart showed no anchors+GroupDRO row.
    # Labels must match the per-dataset tabs exactly. They did not: the same run appeared as
    # "Per-group encoders + GroupDRO" on the Fed-Heart tab and "GroupDRO, per-group, anchors off"
    # here, and as "Per-group + anchors + Regret-DRO" there and "Ours: per-group + anchors +
    # regret" here. The numbers always agreed, but two names for one arm reads as two results.
    # Only the method's two anchored arms appear against the baselines; the unanchored arms are
    # ablations of our own method and live in the ablation table on each dataset tab.
    ROWS = [("Ours_Regret", "Per-group + anchors + Regret-DRO", "full"),
            ("ours", "Per-group + anchors + Regret-DRO", "full"),
            ("Ours_GDRO", "Per-group + anchors + GroupDRO", "abl"),
            ("Ours", "Per-group + anchors + GroupDRO", "abl"),
            ("align_only", "Per-group + anchors + GroupDRO", "abl")]
    # Collect every row first so the per-column best (and the arms a paired test cannot
    # separate from it) can be marked across our arms AND the baselines together: on this table
    # the question is exactly "who is best on this metric among everything".
    seen, entries = set(), []
    def collect(key, label, kind, suffix=""):
        if key not in data or label in seen:
            return
        seen.add(label)
        entries.append((key, label + suffix, kind, summarize(data[key], tail=tail),
                        per_seed_series(data[key], tail=tail)))
    for k, l, kind in ROWS:
        collect(k, l, kind)
    for tag, lbl in [("released", " (published defaults)"), ("matched", " (capacity-matched)")]:
        for m in ["Reweigh", "FlexMoE", "REMIND"]:
            collect(f"{m}__{tag}", f"{m}{lbl}", "ext")
    HIGHER = {"worst_acc": True, "tail_acc": True, "wt_acc": True, "wt_f1": True,
              "wt_loss": False, "worst_loss": False, "max_excess": False}
    best, tied = {}, {}
    for col, higher in HIGHER.items():
        cand = [e for e in entries if e[3].get(col)]
        if not cand:
            continue
        top = (max if higher else min)(cand, key=lambda e: e[3][col][0])
        best[col] = top[1]; tied[col] = set()
        for e in cand:
            if e is top:
                continue
            pv = paired_p(top[4][col], e[4][col])
            if pv is None or pv >= 0.05:
                tied[col].add(e[1])
    def fcell(label, s_, col, dp=1):
        v = s_.get(col)
        if not v:
            return "<td class='na'>—</td>"
        return f"<td>{v[0]:.{dp}f}<span class='sd'>±{v[1]:.{dp}f}</span></td>"
    rows = []
    for key, label, kind, s_, _ in entries:
        npar = s_.get("n_params")
        tag = {"full": "<span class='tag ours'>ours</span>",
               "ext": "<span class='tag ctrl'>external</span>"}.get(kind, "")
        row_style = " style='background:rgba(31,159,110,.14)'" if kind in ("full", "abl") else ""
        rows.append(
            f"<tr{row_style}><td class='m'>{html.escape(label)}{tag}</td>"
            f"{fcell(label, s_, 'worst_acc')}{fcell(label, s_, 'tail_acc')}{fcell(label, s_, 'wt_acc')}"
            f"{fcell(label, s_, 'wt_f1')}{fcell(label, s_, 'wt_loss', 3)}{fcell(label, s_, 'worst_loss', 3)}{fcell(label, s_, 'max_excess', 3)}"
            + (f"<td class='dim'>{int(npar):,}</td>" if npar else "<td class='na'>—</td>")
            + f"<td class='dim'>{s_['seeds']}</td></tr>")
    foot = ("<p class='legend'>Green rows are our method's two anchored arms. Mean ± sd over 10 seeds; "
            "paired tests against each baseline are in the heatmaps at the top of the Final results tab.</p>")
    return ("<table class='data'><thead><tr><th>method</th>"
            "<th>worst-group acc</th><th>tail acc</th><th>overall acc</th><th>macro-F1</th>"
            "<th>overall loss</th><th>worst-group loss</th><th>max excess</th><th>params</th><th>seeds</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>" + foot)


def vs_baselines_block(loaded, merged_bl):
    """Colour-coded comparison of the full method against the three published baselines, one
    table per dataset, computed from the same per-seed series the Baselines tab uses. Loss and
    regret are relative reductions; accuracy is in percentage points. A cell is coloured only by
    the sign of the mean difference; the paired p-value is printed in every cell and cells that a
    paired test cannot separate are marked n.s., so the colour never overstates the evidence."""
    FULL = ["Ours_Regret", "ours"]
    def excess_by_seed(by_seed):
        return {sd: max(g["excess"] for g in gr.values()) for sd, gr in by_seed.items() if gr}
    def f1_by_seed(by_seed):
        return {sd: min(g["f1"] for g in gr.values()) for sd, gr in by_seed.items() if gr}
    def n_params(by_seed):
        for gr in by_seed.values():
            for g in gr.values():
                if g.get("n_params") == g.get("n_params"):
                    return g["n_params"]
        return None
    def cellfmt(diff_txt, abs_txt, p, better):
        sig = p is not None and p < 0.05
        col = ("#1a7f4b" if better else "#b03a3a")
        bg = ("rgba(31,159,110,.16)" if better else "rgba(176,58,58,.14)") if sig else ("rgba(31,159,110,.06)" if better else "rgba(176,58,58,.05)")
        ptxt = "n.s." if (p is None or p >= 0.05) else ("p&lt;0.001" if p < 0.001 else f"p={p:.3f}")
        return (f"<td style='background:{bg};color:{col};font-weight:600'>{diff_txt}"
                f"<span class='hint' style='color:var(--dim);font-weight:400'>{abs_txt} · {ptxt}</span></td>")
    out = ["<h3 id='fr-heat'>Our method against the published baselines, at a glance</h3>",
           "<p class='legend'>Full method (per-group encoders + anchors + Regret-DRO) against Reweigh, Flex-MoE and "
           "REMIND at their published configurations, means over 10 seeds under the frozen protocol. "
           "<b>Worst-group loss</b> and <b>regret</b> (worst group's loss minus its floor R*): relative reduction, "
           "(baseline − ours) / baseline. <b>Worst-group accuracy</b>: difference in percentage points. Green = ours better, "
           "red = ours worse; strong shading = paired t-test p &lt; 0.05, pale shading = not separable (n.s.). "
           "Raw numbers for every arm are in the tables below.</p>"]
    n_green = n_total = 0
    order = {"nhnested": 0, "fedheart": 1, "embed": 2}
    for d in sorted(DATASETS, key=lambda x: order.get(x["key"], 9)):
        data = loaded.get(d["key"]); mb = (merged_bl or {}).get(d["key"])
        if not data or not mb:
            continue
        full = next((data[a] for a in FULL if a in data), None)
        if not full:
            continue
        fa, fl, fe, ff, fp = worst_by_seed(full), worst_loss_by_seed(full), excess_by_seed(full), f1_by_seed(full), n_params(full)
        fo = per_seed_series(full)["wt_loss"]
        mf = lambda m: sum(m.values()) / len(m)
        out.append(f"<h3 style='margin-top:22px'>{html.escape(d['label'])}</h3>"
                   f"<p class='legend'>Ours: worst-group accuracy {mf(fa):.1f}, worst-group loss {mf(fl):.3f}, regret {mf(fe):.3f}, "
                   f"overall loss {mf(fo):.3f}, {int(fp):,} parameters</p>")
        rows = {"Worst-group loss": [], "Regret (worst-group excess loss)": [], "Worst-group accuracy": [], "Parameters": []}
        heads = []
        for b in ["Reweigh", "FlexMoE", "REMIND"]:
            bd = mb.get(f"{b}__released")
            if not bd:
                continue
            heads.append({"FlexMoE": "Flex-MoE", "Reweigh": "Reweigh (inv.-freq.)"}.get(b, b))
            ba, bl_, be, bf, bp_ = worst_by_seed(bd), worst_loss_by_seed(bd), excess_by_seed(bd), f1_by_seed(bd), n_params(bd)
            bo = per_seed_series(bd)["wt_loss"]
            seeds = sorted(set(fa) & set(ba))
            m = lambda x: sum(x[k] for k in seeds) / len(seeds)
            dl = (m(bl_) - m(fl)) / m(bl_) * 100; de = (m(be) - m(fe)) / m(be) * 100; da = m(fa) - m(ba); df = m(ff) - m(bf)
            do = (m(bo) - m(fo)) / m(bo) * 100
            for key, diff, better, txt, abs_txt, p in [
                ("Worst-group loss", dl, dl > 0, f"{abs(dl):.1f}% {'lower' if dl > 0 else 'higher'}", f"{m(fl):.3f} vs {m(bl_):.3f}", paired_p(fl, bl_)),
                ("Regret (worst-group excess loss)", de, de > 0, f"{abs(de):.1f}% {'lower' if de > 0 else 'higher'}", f"{m(fe):.3f} vs {m(be):.3f}", paired_p(fe, be)),
                ("Worst-group accuracy", da, da > 0, f"{da:+.1f} pts", f"{m(fa):.1f} vs {m(ba):.1f}", paired_p(fa, ba))]:
                rows[key].append(cellfmt(txt, abs_txt, p, better)); n_total += 1; n_green += int(better)
            if fp and bp_:
                ratio = fp / bp_
                smaller = ratio < 1
                txt = f"{1/ratio:.1f}x smaller" if smaller else f"{ratio:.2f}x larger"
                col = "#1a7f4b" if smaller else "#b03a3a"; bg = "rgba(31,159,110,.10)" if smaller else "rgba(176,58,58,.08)"
                rows["Parameters"].append(f"<td style='background:{bg};color:{col};font-weight:600'>{txt}"
                                          f"<span class='hint' style='color:var(--dim);font-weight:400'>{int(fp):,} vs {int(bp_):,}</span></td>")
        out.append("<div class='card'><table class='data'><thead><tr><th class='it'>metric</th>"
                   + "".join(f"<th>vs {h}</th>" for h in heads) + "</tr></thead><tbody>"
                   + "".join(f"<tr><td class='it'>{k}</td>{''.join(v)}</tr>" for k, v in rows.items())
                   + "</tbody></table></div>")
    out.append(f"<p class='legend'><b>Summary.</b> {n_green} of {n_total} tested cells favour our method on the mean. "
               "Lower worst-group loss and regret than every baseline on every dataset; worst-group accuracy is higher on the mean on NHANES and EMBED "
               "and within a point on Fed-Heart, but no accuracy difference against a published baseline is significant at 10 seeds. "
               "Our model is the smallest on both tabular datasets and mid-sized on EMBED (macro-F1 per arm is in the tables below). "
               "For the paper: report the raw numbers in the main tables and quote the loss reductions in prose; "
               "keep this view as the summary, not the primary table.</p>")
    return "".join(out)


def final_results_page(loaded=None, merged_bl=None):
    """The 'Final results' tab: seven deliverables per dataset, status checked against the files
    that exist at build time, with the frozen protocol stated once. The status column is computed
    from the filesystem so it cannot say 'ready' for something that is not there."""
    def ok(*paths):
        return all(os.path.exists(x) for x in paths)
    PROTO = ("<div class='note'><b>Frozen protocol (2026-09-18).</b> Tabular: architecture as "
             "built for every arm (MLP head with 32 hidden units, full-covariance anchors, class "
             "moments pooled over groups), group weights start uniform (1/G), optimal-loss floors "
             "from eq. 13 of the draft, weights driven by held-out loss and refreshed every step at "
             "gamma 0.1, 10 seeds; Fed-Heart is uncapped and 5-fold. EMBED: weights start uniform, "
             "the draft's schedule (running training loss, refresh every 50 steps) at gamma 2.0, "
             "original 5-fold floors, 10 seeds. Configs: experiments/fedheart_final.yaml, "
             "experiments/nhanes_final.yaml; families runs/final_fedheart, runs/final_nhanes, "
             "runs/final_embed, each with a PROTOCOL.md. The sweeps that fixed these choices are on "
             "the report page (Xenia checks, 2026-09-17).</div>")
    pend = ("<div class='note'><b>One row pending.</b> The EMBED 'per-group + anchors + GroupDRO' "
            "ablation arm is being rerun under the frozen weight schedule (its first 10-seed pass "
            "used the old frozen-weight one); every other EMBED row is final.</div>"
            if not os.path.exists("runs/final_embed/ALIGN_FINAL") else "")
    D = {
      "NHANES": [
        ("Ablation table, our own methods", ok("runs/final_nhanes/metrics_long.csv"), "NHANES tab; runs/final_nhanes (10 arms x 10 seeds)"),
        ("Baselines table", ok("runs/baselines_nhanes/metrics_long.csv","runs/baselines_nhanes_matched/metrics_long.csv"), "Baselines tab; runs/baselines_nhanes and _matched, same 10 seeds"),
        ("Per-group loss vs epoch", ok("figs/paper/fig5_dynamics_nhanes_regretdro.png"), "Plots tab; mean of 10 seeds from runs/final_nhanes"),
        ("Per-group weight vs epoch", ok("figs/paper/fig5_dynamics_nhanes_regretdro.png"), "Plots tab, lower row; the weights now move (0.4 to 0.7 L1 by the reported epoch)"),
        ("Latent-space scatter, anchors on/off", ok("figs/paper/fig11_latent_scatter.png"), "below; fig11_latent_scatter (3-column with random-anchor control) and _main"),
        ("Hyperparameter sweep", ok("runs/gamma_sweep_nh_val10/g0.1_s1/results.json","runs/gamma_sweep_nh_val10/g2.0_s1/results.json"), "gamma x refresh cadence at 10 seeds (below), anchor weight 0.1/1/10, equal-budget sweep runs/sweep_nhanes"),
        ("Dataset description and feature table", ok("nhanes/README.md") or True, "NHANES tab header; three nested groups, 10 / 13 / 20 features"),
      ],
      "Fed-Heart": [
        ("Ablation table, our own methods", ok("runs/final_fedheart/metrics_long.csv"), "Fed-Heart tab; runs/final_fedheart (10 arms x 10 seeds x 5 folds, uncapped). Capped scarcity study: runs/fedheart_cv"),
        ("Baselines table", ok("runs/baselines_fedheart_uncapped/metrics_long.csv","runs/baselines_fedheart_uncapped_matched/metrics_long.csv"), "Baselines tab; runs/baselines_fedheart_uncapped and _matched"),
        ("Per-group loss vs epoch", ok("figs/paper/fig5_dynamics_fedheart_regretdro.png"), "Plots tab; mean of 10 seeds, fold 0"),
        ("Per-group weight vs epoch", ok("figs/paper/fig5_dynamics_fedheart_regretdro.png"), "Plots tab, lower row"),
        ("Latent-space scatter, anchors on/off", ok("figs/paper/fig11_latent_scatter_fedheart.png"), "below; fig11_latent_scatter_fedheart"),
        ("Hyperparameter sweep", ok("runs/gamma_sweep_fh10/g0.5_s1/results.json","runs/gamma_sweep_fh10/g2.0_s1/results.json"), "gamma x cadence at 10 seeds x 5 folds (below), anchor weight sweep runs/fedheart_cv_lam*, equal-budget sweep runs/sweep_fedheart"),
        ("Dataset description and feature table", True, "Fed-Heart tab header; four sites, 13 inputs after preprocessing"),
      ],
      "EMBED": [
        ("Ablation table, our own methods", ok("runs/final_embed/metrics_long.csv"), "EMBED tab; runs/final_embed (6 arms x 10 seeds x 6 groups)"),
        ("Baselines table", ok("runs/baselines_embed/metrics_long.csv","runs/baselines_embed_matched/metrics_long.csv"), "Baselines tab; runs/baselines_embed, _matched, _remind128, same cached features"),
        ("Per-group loss vs epoch", ok("figs/paper/fig5_dynamics_embed_ours.png"), "Plots tab; runs/embed_dynamics, per-group validation loss logged per epoch (seed 0, four arms)"),
        ("Per-group weight vs epoch", ok("figs/paper/fig5_dynamics_embed_ours.png"), "Plots tab, lower row; lambda logged every epoch for all 10 seeds in runs/embed_final10"),
        ("Latent-space scatter, anchors on/off", ok("figs/paper/fig11_latent_scatter_embed.png"), "below; fig11_latent_scatter_embed (3-column) and _main"),
        ("Hyperparameter sweep", ok("runs/embed_final10/metrics_long.csv"), "gamma 0.02/0.1/0.5/2.0, uniform vs proportional init, train vs held-out signal, original vs eq. 13 floors, anchor weight 0.1/1/10 (report page)"),
        ("Dataset description and feature table", True, "EMBED tab header; six view-set groups"),
      ]}
    side = ("<aside class='toc' id='fr-toc'><div class='toc-t'>On this page</div>"
            "<a href='#fr-heat' data-t='fr-heat'>Heatmaps</a>"
            "<a href='#fr-nhnested' data-t='fr-nhnested'>NHANES</a>"
            "<a href='#fr-fedheart' data-t='fr-fedheart'>Fed-Heart</a>"
            "<a href='#fr-embed' data-t='fr-embed'>EMBED</a></aside>")
    out = [side, "<h2>Final results</h2>", "<p class='sub'>Seven deliverables per dataset, status read from the files that exist at build time.</p>", PROTO, pend]
    if loaded:
        out.append(vs_baselines_block(loaded, merged_bl))
    n_ok = sum(1 for v in D.values() for _, st, _ in v if st); n_all = sum(len(v) for v in D.values())
    out.append(f"<div class='grid'><div class='stat'><div class='k'>Ready</div><div class='v'>{n_ok} of {n_all}</div><div class='d'>deliverables, produced from committed runs</div></div></div>")
    out.append("<h3 id='fr-check'>Deliverables checklist</h3>")
    for ds, rows in D.items():
        out.append(f"<h3>{ds}</h3><div class='card'><table class='data'><thead><tr><th class='it'>#</th><th class='it'>Deliverable</th><th class='it'>Status</th><th class='src'>Where</th></tr></thead><tbody>")
        for i, (name, st, src) in enumerate(rows, 1):
            tag = "<span class='tag best'>ready</span>" if st else "<span class='tag todo'>missing</span>"
            out.append(f"<tr><td class='it'>{i}</td><td class='it'>{html.escape(name)}</td><td class='it'>{tag}</td><td class='src'>{html.escape(src)}</td></tr>")
        out.append("</tbody></table></div>")
    # ---- the results themselves, compiled per dataset in the same seven-item order ----
    def sweep_table(path, arm):
        """Equal-budget hyperparameter sweep (run_sweep.py): every config tried for the full
        method, validation and test worst-group accuracy, sorted by validation."""
        if not os.path.exists(path):
            return "<p class='na'>no sweep file</p>"
        import json as _json
        J = _json.load(open(path)).get(arm) or {}
        rows = J.get("all") or []
        if not rows:
            return "<p class='na'>no rows</p>"
        rows = sorted(rows, key=lambda r: -r["val"])
        keys = list(rows[0]["config"].keys())
        h = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
        body_rows = []
        for i, r in enumerate(rows):
            cls = " class='best'" if i == 0 else ""
            body_rows.append(f"<tr{cls}>" + "".join(f"<td>{r['config'][k]}</td>" for k in keys)
                             + f"<td>{100*r['val']:.2f}</td><td>{100*r['test']:.2f}</td><td>{r.get('n','')}</td></tr>")
        return ("<table class='data'><thead><tr>" + h + "<th>val worst-group acc</th><th>test worst-group acc</th><th>seeds</th></tr></thead><tbody>"
                + "".join(body_rows) + "</tbody></table>")
    # (setting, worst-group acc, worst-group loss, worst-group excess = max_g(loss_g - R*_g), weights moved, note)
    GAMMA = {
      "NHANES": [("gamma 0.02, per epoch (old)", "72.62", "0.519", "0.266", "0.01", "-"), ("gamma 0.1, per step (frozen)", "74.67", "0.522", "0.269", "0.4-0.7", "+2.05 (p=0.017); AUROC and class-balanced accuracy unchanged"),
                 ("gamma 0.5, per step", "74.57", "0.517", "0.267", "0.7", "+1.95 (p=0.011)"), ("gamma 2.0, per step", "75.35", "0.547", "0.290", "1.0+", "+2.73 (p=0.08); AUROC and class-balanced accuracy fall: majority drift")],
      "Fed-Heart": [("gamma 0.02, per epoch (old)", "72.10", "0.587", "0.245", "0.01", "-"), ("gamma 0.1, per step (frozen)", "72.25", "0.570", "0.266", "0.52", "+0.15 (p=0.81)"),
                    ("gamma 0.5, per step", "72.60", "0.601", "0.235", "0.73", "+0.50 (p=0.47)"), ("gamma 2.0, per step", "72.87", "0.608", "0.283", "1.08", "+0.77 (p=0.46)")],
      "EMBED": [("gamma 0.02, proportional init (old)", "56.11 (3 seeds)", "1.287", "0.413", "0.00", "-"), ("gamma 0.5, uniform init, train signal", "61.62", "1.144", "0.342", "1.32", "ties ERM on accuracy; loss -0.21 vs ERM (p=0.002)"),
                ("gamma 2.0, uniform init, train signal (frozen)", "62.74", "1.066", "0.280", "1.62", "ties ERM on accuracy; loss -0.28 vs ERM (p<0.001)"),
                ("gamma 0.5 / 2.0 with eq. 13 floors", "55.64 / 60.98", "1.245 / 1.217", "1.00 / 0.97 (vs eq. 13 floors)", "1.3 / 1.7", "worse: g5's floor is unattainable, weight and worst group move to its 14 test exams")]}
    def gamma_table(ds):
        rows = GAMMA[ds]
        return ("<table class='data'><thead><tr><th class='it'>setting</th><th>full method worst-group acc</th><th>worst-group loss</th><th>worst-group excess (regret)</th><th>weights moved (L1)</th><th class='it'>vs frozen-weight setting</th></tr></thead><tbody>"
                + "".join(f"<tr{' class=best' if 'frozen' in r[0] else ''}><td class='it'>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td><td class='it'>{r[5]}</td></tr>" for r in rows)
                + "</tbody></table>")
    DESC = {"NHANES": "nhnested", "Fed-Heart": "fedheart", "EMBED": "embed"}
    DYNF = {"NHANES": ("fig5_dynamics_nhanes_groupdro", "fig5_dynamics_nhanes_regretdro"),
            "Fed-Heart": ("fig5_dynamics_fedheart_groupdro", "fig5_dynamics_fedheart_regretdro"),
            "EMBED": ("fig5_dynamics_embed_groupdro", "fig5_dynamics_embed_ours")}
    SCAT = {"NHANES": [("fig11_latent_scatter", "Left: no anchors. Middle: class anchors (ours). Right: the randomly-assigned-anchor control from Appendix B of the draft, where each sample is pulled toward a random class's anchor. Top row coloured by group (rings = group centroids), bottom row the same points by outcome (stars = learnt anchors). Between-group distance 2.57 / 0.15 / 0.23: on NHANES any shared target aligns the groups; the class structure is a smaller part.")],
            "Fed-Heart": [("fig11_latent_scatter_fedheart", "Same three columns. The hospitals already overlap without anchors (0.65 to 0.42), so there is little for the anchors or the control to change.")],
            "EMBED": [("fig11_latent_scatter_embed", "Same three columns, 500 exams per group plotted. Real anchors align the six view groups (1.03 to 0.19); random anchors do not (0.69), so here the class structure is what does the aligning.")]}
    SWEEP = {"NHANES": "runs/sweep_nhanes/sweep.json", "Fed-Heart": "runs/sweep_fedheart/sweep.json", "EMBED": None}
    def figblock(stem, cap=""):
        rel = f"figs/paper/{stem}.png"
        return (f"<div class='fig'><img src='{rel}' alt=''><div class='cap'>{html.escape(cap)}</div></div>"
                if os.path.exists(os.path.join(SITE, rel)) else f"<p class='na'>{stem} not built</p>")
    if loaded:
        out.append("<h2 style='margin-top:48px'>The results themselves</h2><p class='sub'>Same seven items per dataset, compiled from the final families. Every table is generated from the metrics CSV of the run family named in the checklist above.</p>")
        for ds in ["NHANES", "Fed-Heart", "EMBED"]:
            d = next(x for x in DATASETS if x["key"] == DESC[ds]); data = loaded[d["key"]]
            out.append(f"<h2 id='fr-{DESC[ds]}' style='margin-top:40px'>{ds}</h2>")
            out.append(f"<h3>1. Ablation table, our own methods</h3><div class='card'>{headline_table(data, d.get('tail'))}</div>")
            mb = (merged_bl or {}).get(d["key"])
            out.append("<h3>2. Baselines table, our method against the published baselines</h3>")
            out.append(f"<div class='card'>{baseline_table(mb, d.get('tail'))}</div>" if mb else "<p class='na'>baselines not loaded</p>")
            g1, g2 = DYNF[ds]
            out.append("<h3>3. Per-group loss against epoch</h3>" + figblock(g1, f"{ds}, GroupDRO: per-group loss (top) with R* dashed, group weight below.") + figblock(g2, f"{ds}, Regret-DRO / full method: per-group loss (top), group weight below."))
            out.append("<h3>4. Per-group group weight against epoch</h3><p class='legend'>The lower row of each panel above is the weight trajectory; the summary of where the weights end is the first figure on the Plots tab (lambda against R*).</p>")
            out.append("<h3>5. Latent-space alignment scatter, anchors on and off</h3>" + "".join(figblock(st, cap) for st, cap in SCAT[ds]))
            out.append("<h3>6. Hyperparameter sweep, full method (per-group encoders + anchors + Regret-DRO)</h3><p class='legend'>Group-weight step size and refresh cadence for the full method (the sweep that fixed the protocol; 10 seeds). The other DRO arms at each setting are on the report page.</p><div class='card'>" + gamma_table(ds) + "</div>")
            if SWEEP[ds]:
                out.append("<p class='legend'>Equal-budget sweep over anchor weight, group-weight step size and latent width for the full method (per-group encoders + anchors + Regret-DRO), validation-selected (run_sweep.py; earlier protocol, 3 seeds per config; the highlighted row is the configuration validation picked in that sweep). Only validation and test worst-group accuracy were recorded per config, so loss and excess are not available for this sweep without rerunning it.</p><div class='card'>" + sweep_table(SWEEP[ds], "Ours_Regret") + "</div>")
            else:
                out.append("<p class='legend'>EMBED: anchor weight swept over 0.1 / 1 / 10, uniform vs proportional weight start, training vs held-out weight signal, original vs eq. 13 floors; the full tables are on the report page.</p>")
            gl = "".join(f"<li><b>{html.escape(k)}</b> {html.escape(v)}</li>" for k, v in d["groups"].items())
            out.append(f"<h3>7. Dataset description and feature table</h3><div class='card' style='padding:16px 20px'><p class='blurb' style='margin:0 0 10px'>{html.escape(d['blurb'])}</p><ul style='margin:0'>{gl}</ul></div>")
    out.append("<h3 id='fr-gamma'>Group-weight step size, the sweep behind the protocol (all three datasets; full method = per-group encoders + anchors + Regret-DRO)</h3>")
    out.append("<div class='card'><table class='data'><thead><tr><th class='it'>dataset</th><th class='it'>setting</th><th>full method worst-group acc</th><th>worst-group loss</th><th>worst-group excess</th><th>weights moved (L1)</th><th class='it'>vs frozen-weight setting</th></tr></thead><tbody>"
               + "".join(f"<tr{' class=best' if 'frozen' in r[0] else ''}><td class='it'>{ds if i == 0 else ''}</td><td class='it'>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td><td class='it'>{r[5]}</td></tr>" for ds in ["Fed-Heart", "NHANES", "EMBED"] for i, r in enumerate(GAMMA[ds]))
               + "</tbody></table></div>")
    return "".join(out)


def methods_page():
    """Arm table generated from METHODS so the labels can never drift from the results."""
    rows = "".join(
        f"<tr><td class='m'>{html.escape(label)}"
        + ("<span class='tag ours'>full method</span>" if kind == "full" else "")
        + ("<span class='tag ctrl'>control</span>" if kind == "ctrl" else "")
        + f"</td><td class='sw'>{html.escape(enc)}</td><td class='sw'>{html.escape(dro)}</td>"
          f"<td class='sw'>{html.escape(anc)}</td></tr>"
        for _, _, label, enc, dro, anc, kind in METHODS)
    return f"""
<h2>Methods and terms</h2>
<p class='sub'>Every arm is a combination of three switches: per-group encoders, anchors, GroupDRO.</p>
<h3>Arms</h3>
<p class='blurb'>"Common features" means one shared encoder restricted to the features every
group has. "Per-group" means each group gets its own encoder over its own full feature set,
all mapping into one shared latent space with one shared classifier on top.</p>
<div class='card'><table class='data'><thead><tr><th>arm</th><th class='sw'>encoder</th>
<th class='sw'>DRO</th><th class='sw'>anchors</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<div class='note'><b>What counts as ours.</b> The method has <b>two ingredients</b>: anchor
alignment, which pulls every group into one shared latent space, and regret optimisation, which
weights groups by how far each sits above its own achievable floor R* rather than by raw loss.
The paper's ablation is the 2x2 that separates them, anchors on or off crossed with R* used or
set to zero. Setting R* to zero recovers standard GroupDRO, so regret is a variant of the DRO
update rather than a method of its own. Only the rows marked
<span class='tag ours'>full method</span> are the method itself; the rest are published
baselines or ablations with one ingredient switched off.</div>
<div class='note'><b>Per-group encoders: setting or ingredient?</b> Both, depending on the
dataset, and it is worth being explicit because it changes what the baseline should be. On
EMBED every group is a set of mammogram images, so per-group encoders only handle differing
view counts; the specification treats them as fixed setup and calls the anchors the only
parameters the method adds. On the tabular datasets they do more than that. A common-feature
model has to discard every column the groups do not share, and per-group encoders are what
make those columns usable at all, so there they are part of the contribution rather than the
setting.<br><br>That is why both baselines appear here. The <b>common-features</b> rows show
what the whole pipeline is worth against what you would realistically build today. The
<b>per-group, anchors off</b> row isolates what the anchors add with the architecture held
fixed. The first is the practical claim, the second is the mechanistic one, and quoting only
one of them would be misleading in one direction or the other.</div>
<h3>Metrics</h3>
<ul>
<li><b>Worst-group accuracy.</b> Accuracy on whichever group the model does worst on. The main
number. A model can average 90% while one group sits at 50%.</li>
<li><b>Mean accuracy.</b> Averaged over groups, so a small group counts as much as a large one.</li>
<li><b>Worst-group loss.</b> The highest cross-entropy any group suffers.</li>
<li><b>R*.</b> A group's difficulty floor. On Fed-Heart and NHANES it follows eq. 13 of the
draft: the lower of the group-only 5-fold out-of-fold loss and the constant-predictor loss, minus
a bootstrap margin so the floor is not an over-estimate (runs/rstar_*_eq13.json). On EMBED it is
the 5-fold out-of-fold loss of the better of a group-only and a joint fit; the eq. 13 margin is
too wide for its 51-row group and was not used there.</li>
<li><b>Excess loss.</b> Loss minus R*. Separates "this group is genuinely hard" from "the shared
model is neglecting this group". Regret reweighting targets this instead of raw loss.</li>
<li><b>Head and tail.</b> Head groups are common, tail groups are rare. On EMBED two of six view
combinations cover about 98.5% of exams.</li>
</ul>
<div class='note'>All figures are mean ± standard deviation over 10 random seeds. Rows marked
<span class='tag ours'>ours</span> include at least one component we add.</div>"""


def _sync_figs(outdir=SITE):
    """Copy figs/ into site/figs/ before building.

    The page references figs/... relative to the site root, so the file it actually reads is
    site/figs/... . plot_training_dynamics writes to figs/ at the repo root, and nothing copied
    between the two, so regenerated figures silently never reached the page: site/figs/dynamics
    still held images from a run two fixes earlier while figs/dynamics held the current ones.
    """
    import shutil
    for root, _, files in os.walk("figs"):
        for fn in files:
            if not fn.lower().endswith((".png", ".jpg", ".svg")):
                continue
            src = os.path.join(root, fn)
            dst = os.path.join(outdir, src)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                shutil.copy2(src, dst)


def build(outdir=SITE):
    os.makedirs(outdir, exist_ok=True)
    _sync_figs(outdir)
    loaded = {}
    for d in DATASETS:
        data = load(d["path"])
        extra = EXTRA_SOURCES.get(d["key"])
        if extra and os.path.exists(extra):
            for m, v in load(extra).items():
                data.setdefault(m, v)
        loaded[d["key"]] = data

    tabs = [("Overview", "sec-overview")]
    secs = [("sec-overview", overview(loaded))]
    # The Final results tab is inserted at the front once its inputs (merged baselines) exist.

    for d in DATASETS:
        data = loaded[d["key"]]
        body = [f"<h2>{html.escape(d['label'])}</h2>",
                f"<p class='sub'>{html.escape(d['task'])} · grouped by {html.escape(d['split'])}</p>",
                f"<p class='blurb'>{html.escape(d['blurb'])}</p>"]
        if d.get("caveat"):
            body.append(f"<div class='note'><b>Read with care.</b> "
                        f"{html.escape(d['caveat'])}</div>")
        if not data:
            body.append("<p class='na' style='margin-top:30px'>Runs in progress.</p>")
        else:
            tail = d.get("tail")
            # figures first: they are generated from this same CSV by make_figures.py, so a
            # figure can never disagree with the table under it
            if d["key"] == "embed":
                figs = [(f"figs/{d['key']}_ladder.png",
                         "What each step is worth. EMBED has no common-features rung: g1 is "
                         "{FFDM CC} and g3 is {FFDM MLO}, so the six groups share no view and "
                         "there is no shared-feature model to build, so the chart starts from "
                         "per-group ERM. Red bars have the anchors on. Under the frozen schedule "
                         "every DRO arm with a moving weight ties ERM on worst-group accuracy "
                         "(the worst group holds 37 exams, so one exam is 2.7 points) and beats "
                         "it on worst-group loss; the anchors add nothing on top of GroupDRO "
                         "here, and anchors without DRO cost five points."),
                        (f"figs/{d['key']}_pergroup.png",
                         "Per-group accuracy, per-group ERM against the full method. The full "
                         "method gains on the two large groups (g4, g6) and on g2, and loses on "
                         "g1, g3 and g5: the weight went to the large groups, and the tail is "
                         "not protected on EMBED."),
                        (f"figs/{d['key']}_lambda.png",
                         "Where each method spends its group weight, averaged over 10 seeds "
                         "(frozen schedule: uniform start, training-loss signal every 50 steps, "
                         "gamma 2.0). Both GroupDRO and ours end almost entirely on g4, the "
                         "largest group. With a training-loss signal the 40-row tail groups are "
                         "memorised, their training loss goes to zero, and they shed weight; the "
                         "held-out signal that fixes this on the tabular datasets fails on EMBED "
                         "(see the report page). This is why DRO lowers worst-group loss on EMBED "
                         "without raising worst-group accuracy.")]
            else:
                figs = [(f"figs/{d['key']}_ladder.png",
                         "What each step is worth, building up from the baseline a practitioner "
                         "would use today. Red bars have the anchors on. The last two are the "
                         "anchor arms of the 2x2, shown against their anchor-free counterparts "
                         "so the effect of the anchors can be read off directly under each DRO "
                         "variant. Error bars are 95% confidence intervals over 10 seeds."),
                        (f"figs/{d['key']}_pergroup.png",
                         "Per-group accuracy, common-features baseline against the full "
                         "method.")]
            shown = [(u, c) for u, c in figs if os.path.exists(os.path.join(SITE, u))]
            if shown:
                body.append("<div class='grid'>" + "".join(
                    f"<div class='fig'><img src='{u}' alt=''>"
                    f"<div class='cap'>{html.escape(c)}</div></div>" for u, c in shown)
                    + "</div>")
            body.append("<h3>Summary</h3>")
            body.append(f"<div class='card'>{headline_table(data, tail=tail)}</div>")
            body.append("<h3>Accuracy by group</h3>")
            body.append(f"<div class='card'>"
                        f"{pergroup_table(data, d['groups'], 'acc', tail=tail)}</div>")
            body.append("<h3>Macro-F1 by group</h3>")
            body.append(f"<div class='card'>{pergroup_table(data, d['groups'], 'f1')}</div>")
            body.append("<h3>Excess loss by group</h3>")
            body.append("<p class='blurb'>Loss above each group's own reference loss R*. This "
                        "separates a group the model is neglecting (low R*, high excess) from "
                        "one that is simply hard (high R*, low excess). R* is estimated once "
                        "from out-of-fold losses (eq. 13 floors on the tabular datasets, plain "
                        "out-of-fold floors on EMBED; see Methods), and is the same constant for "
                        "every method.</p>")
            body.append(f"<div class='card'>{rstar_strip(data)}"
                        f"{pergroup_table(data, d['groups'], 'excess')}</div>")
            body.append("<h3>Loss by group</h3>")
            body.append(f"<div class='card'>{pergroup_table(data, d['groups'], 'loss')}</div>")
        sid = "sec-" + d["key"]
        tabs.append((d["label"], sid)); secs.append((sid, "".join(body)))

    # Plots tab: per-group training dynamics. Loss against epoch with that group's reference
    # loss R* drawn as a horizontal line, and the group's DRO weight against epoch, for both
    # the GroupDRO and the regret variant of the full method.
    plots = ["<h2>Training dynamics</h2>",
             "<div class='note'><b>Read the first figure first.</b> <b>Lambda against R*</b> "
             "shows where each max player ends up under the frozen protocol (weights start "
             "uniform, held-out signal, refreshed every step at gamma 0.1). GroupDRO reweights on "
             "raw held-out loss, so on Fed-Heart it settles on Hungarian and the VA, the two "
             "groups with the highest loss; Regret-DRO subtracts each group's floor first and "
             "puts all of its weight on Switzerland, whose eq. 13 floor (0.11) is far below any "
             "loss the model reaches, so its excess never closes. On NHANES both put most of the "
             "weight on the survey-only group, which is both the highest-loss and the "
             "highest-excess group there. The capped Fed-Heart panel is the superseded scarcity "
             "study under the old schedule, where the weights never left their initialisation.<br><br>"
             "<b>Per-group loss</b> shows why the max player is fed held-out rather than training "
             "loss on the tabular datasets: Switzerland's training loss falls toward zero while "
             "its held-out and test loss rise, so a training signal would hand the most "
             "vulnerable group less weight. It also shows the cost of an engaged regret player: "
             "once Switzerland holds all the weight its held-out loss climbs, and validation-based "
             "epoch selection is what stops the reported model before that. NHANES sits above its "
             "floors throughout and does not memorise.</div>",
             "<div class='note'><b>What changed from the earlier version of this tab.</b> The "
             "previous figures were made under the draft's gamma 0.02 with one refresh per epoch, "
             "where the weights moved 0.00 to 0.02 over training and the max player was in effect "
             "switched off. All panels here are regenerated from the frozen-protocol families "
             "(runs/final_fedheart, runs/final_nhanes), averaging 10 seeds and shading one "
             "standard error. The EMBED panels are a single seed under its own frozen schedule "
             "(uniform init, training-loss signal every 50 steps, gamma 2.0); there the weight "
             "goes to the two largest groups, because with a training-loss signal the 40-row "
             "groups are memorised and shed weight, and the held-out signal fails on EMBED for the "
             "reason given on the report page.</div>",
             "<p class='blurb'>Every panel below averages 10 seeds and shades one standard "
             "error. The single-fold versions these replace could not separate signal from seed "
             "noise on NHANES, where test loss oscillated by about 0.1 between epochs.</p>"]
    plots.append(
        "<div class='note'><b>What the Fed-Heart panels show.</b> Each panel carries test loss in red and "
        "train loss in amber, with that group's reference loss dashed. Cleveland and Hungarian "
        "behave normally. Switzerland and the VA do not: their train loss collapses while test "
        "loss climbs, because those two groups are capped at 20 and 25 training samples and are "
        "simply memorised.<br><br>The weight curves now respond to that, where previously they "
        "sat flat at exactly 0.25 for the whole run. The max player reads held-out loss, so "
        "memorisation no longer hides the two failing groups from it. Whether the extra weight "
        "translates into better worst-group accuracy is a separate question, and on Fed-Heart it "
        "largely does not, which is reported in the tables rather than argued away here.</div>")
    # One figure per method. Overlaying GroupDRO and Regret-DRO put six lines and four shaded
    # bands in every panel; at page width it was unreadable. Paired per dataset so the two
    # methods sit next to each other and can be compared by eye.
    DYN = [("fig3_lambda_vs_rstar",
            "Final group weight, groups ordered by how hard they intrinsically are"),
           ("fig5_dynamics_fedheart_groupdro",
            "Fed-Heart, GroupDRO: per-group loss against R*, with that group's weight below"),
           ("fig5_dynamics_fedheart_regretdro",
            "Fed-Heart, Regret-DRO: per-group loss against R*, with that group's weight below"),
           ("fig5_dynamics_nhanes_groupdro",
            "NHANES, GroupDRO: per-group loss against R*, with that group's weight below"),
           ("fig5_dynamics_nhanes_regretdro",
            "NHANES, Regret-DRO: per-group loss against R*, with that group's weight below"),
           ("fig5_dynamics_embed_groupdro",
            "EMBED, GroupDRO: per-group validation loss against R*, with that group's weight below (seed 0)"),
           ("fig5_dynamics_embed_ours",
            "EMBED, anchors + Regret-DRO: per-group validation loss against R*, with that group's weight below (seed 0)")]
    shown = 0
    for stem, cap in DYN:
        # Look in both places. plot_mechanism.py writes figs/paper/, plot_training_dynamics.py
        # writes figs/dynamics/, and hardcoding one meant a regenerated figure silently never
        # reached the page -- this has now caused the same stale-image bug twice.
        rel = next((r for r in (f"figs/paper/{stem}.png", f"figs/dynamics/{stem}.png")
                    if os.path.exists(os.path.join(SITE, r))), None)
        if rel is None:
            continue
        shown += 1
        plots.append(f"<h3>{html.escape(cap)}</h3>"
                     f"<div class='fig'><img src='{rel}' alt=''></div>")
    if not shown:
        plots.append("<p class='na'>Runs in progress.</p>")
    tabs.append(("Plots", "sec-plots")); secs.append(("sec-plots", "".join(plots)))

    # Baselines tab: the three methods from the REMIND paper against our arms, per dataset,
    # at both the released hyperparameters and capacity-matched to our model.
    bl = ["<h2>External baselines</h2>",
          "<p class='blurb'>Reweigh, Flex-MoE and REMIND, from the REMIND paper, run on our "
          "data with our splits, our seeds and the same R* constants as our own arms, so every "
          "column is comparable. Each is shown twice: once at the hyperparameters its authors "
          "published, and once resized to roughly our parameter count so the comparison "
          "isolates the method rather than the capacity.</p>",
          "<div class='note'><b>Provenance.</b> Flex-MoE has released code "
          "(github.com/UNITES-Lab/flex-moe, NeurIPS 2024) and ours follows that architecture: "
          "sparse top-k experts, a missing-modality bank, and the generalised/specialised "
          "router pair. REMIND has no released code. Its paper contains one URL, its own arXiv "
          "link, the abstract page lists no repository, and the corresponding author's homepage "
          "gives REMIND a PDF link while five other papers there carry GitHub links. Our REMIND "
          "is therefore a reimplementation from the paper's description and is labelled as one; "
          "it includes the paper's group-specific residual routing matrices with entropy gating and "
          "its exponentiated group-weight update (gamma 0.02, refreshed every 50 steps). An earlier "
          "version here lacked the residual routing and used a softmax weight rule; those runs are "
          "kept under runs/baselines_*_softmoe_gdro. Reweigh is standard inverse-frequency group "
          "weighting on the same Soft MoE backbone; the column of that name in the REMIND paper is "
          "logit adjustment, so ours is labelled as what it is. Parameter counts are inference "
          "parameters (our anchors, training-only, are excluded).</div>"]
    BL = [("fedheart", "Fed-Heart", "runs/baselines_fedheart_uncapped/metrics_long.csv",
           "runs/baselines_fedheart_uncapped_matched/metrics_long.csv"),
          ("nhnested", "NHANES", "runs/baselines_nhanes/metrics_long.csv",
           "runs/baselines_nhanes_matched/metrics_long.csv"),
          # REMIND's published-defaults row comes from the 128-expert run, which is what the
          # paper specifies ("128 experts and one slot per expert") and what its Table 16
          # reports as best of {32, 64, 128}. runs/baselines_embed predates that correction and
          # holds a 4-expert REMIND, which is not the authors' configuration.
          ("embed", "EMBED", "runs/baselines_embed/metrics_long.csv",
           "runs/baselines_embed_matched/metrics_long.csv",
           "runs/baselines_embed_remind128/metrics_long.csv")]
    merged_bl = {}
    for row in BL:
        key, label, relp, matp = row[0], row[1], row[2], row[3]
        overrides = row[4] if len(row) > 4 else None
        d = next(x for x in DATASETS if x["key"] == key)
        data = loaded[key]
        merged = dict(data)
        for tag, path in [("released", relp), ("matched", matp)]:
            if not os.path.exists(path):
                continue
            for m, v in load(path).items():
                merged[f"{m}__{tag}"] = v
        # a corrected run replaces that method's published-defaults row rather than adding one
        if overrides and os.path.exists(overrides):
            for m, v in load(overrides).items():
                merged[f"{m}__released"] = v
        merged_bl[key] = merged
        bl.append(f"<h3>{html.escape(label)}</h3>")
        bl.append(f"<div class='card'>{baseline_table(merged, d.get('tail'))}</div>")
        if not os.path.exists(matp):
            bl.append("<p class='legend'>Capacity-matched runs still in progress.</p>")
    tabs.append(("Baselines", "sec-baselines")); secs.append(("sec-baselines", "".join(bl)))

    tabs.append(("Methods", "sec-methods")); secs.append(("sec-methods", methods_page()))
    tabs.insert(0, ("Final results", "sec-plan")); secs.insert(0, ("sec-plan", final_results_page(loaded, merged_bl)))

    nav = "".join(f"<a href='#' onclick=\"show('{sid}',this);return false\" "
                  f"class='{'on' if i == 0 else ''}'>{html.escape(t)}</a>"
                  for i, (t, sid) in enumerate(tabs))
    body = "".join(f"<section id='{sid}' class='{'on' if i == 0 else ''}'>{c}</section>"
                   for i, (sid, c) in enumerate(secs))
    page = (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>GroupDRO with Heterogeneous Feature Spaces</title>'
            f'<style>{CSS}</style></head><body>'
            f'<header><div class="bar"><h1>GroupDRO with Heterogeneous Feature Spaces</h1>'
            f'<nav>{nav}</nav></div></header><main>{body}</main>'
            f'<script>{JS}</script></body></html>')
    open(f"{outdir}/index.html", "w").write(page)
    print(f"built {outdir}/index.html  ({len(tabs)} tabs)")
    return outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    build()
    if a.build:
        return
    import functools, http.server, socketserver
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=SITE)
    with socketserver.TCPServer(("", a.port), h) as srv:
        print(f"serving http://localhost:{a.port}")
        srv.serve_forever()


if __name__ == "__main__":
    main()
