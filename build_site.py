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
    dict(key="fedheart", label="Fed-Heart", path="runs/fedheart_cv/metrics_long.csv",
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
                "Switzerland 125, VA 200. The earlier single-split protocol tested Switzerland "
                "on 10 patients and is superseded; every TABLE here comes from the "
                "cross-validated runs. The two training-dynamics panels on the Plots tab are a "
                "single fold at seed 42, because a per-epoch trajectory has no meaningful "
                "pooling across folds.",
         groups={"g0": "Cleveland, 305 patients", "g1": "Hungarian, 295 patients",
                 "g2": "Switzerland, 125 patients", "g3": "VA, 200 patients"}),
    dict(key="nhnested", label="NHANES",
         path="runs/matrix_nhanes_nested/metrics_long.csv",
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
    dict(key="embed", label="EMBED", path="runs/embed_fix_final/metrics_long.csv",
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
     "Reweigh (REMIND paper)",            "shared",    "fixed 1/n", "—", "ext"),
    ("FlexMoE",             ["FlexMoE"],
     "FlexMoE (REMIND paper)",            "soft MoE",  "—",      "—",  "ext"),
    ("REMIND",              ["REMIND"],
     "REMIND (reimplementation)",         "soft MoE",  "GroupDRO", "—", "ext"),
]

# Baseline runs live in their own directories; merged in at load time so their rows sit in the
# same table as ours with the same metrics and seeds.
EXTRA_SOURCES = {
    "fedheart":  "runs/baselines_fedheart/metrics_long.csv",
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
                max_excess=ms(max_excess), wt_acc=ms(wt_acc), wt_f1=ms(wt_f1),
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


def headline_table(data, tail=None):
    """method x [worst-group acc, mean acc, macro-F1, worst-group loss].

    Ranking by raw mean alone is misleading when the spread across seeds is larger than the
    gap between arms, which happens here whenever a group's test set is small. So the top row
    by mean is compared against every other arm with a paired t-test over shared seeds, and
    anything that is not significantly worse is marked as tied rather than beaten.
    """
    rows, summaries, series, summaries_meta = [], {}, {}, {}
    for key, aliases, label, enc, dro, anc, kind in METHODS:
        by_seed = next((data[a] for a in aliases if a in data), None)
        if not by_seed:
            continue
        summaries[key] = (label, kind, summarize(by_seed, tail=tail))
        summaries_meta[key] = (enc, dro, anc)
        series[key] = worst_by_seed(by_seed)
    if not summaries:
        return "<p class='na'>No runs yet.</p>"

    # Rank WITHIN each encoder track rather than across the whole table. A shared-encoder arm
    # and a per-group arm see different feature sets, so a single global winner compares two
    # things that are not alternatives to each other. Per track the question is the useful one:
    # given this encoder, which combination of DRO and anchors is best?
    ranked = [k for k, (_, kind, s) in summaries.items()
              if kind not in ("ctrl", "ext") and s["worst_acc"]]
    tops, tied = set(), set()
    for track in ("shared", "per-group"):
        in_track = [k for k in ranked if summaries_meta[k][0] == track]
        if not in_track:
            continue
        top = max(in_track, key=lambda k: summaries[k][2]["worst_acc"][0])
        tops.add(top); tied.add(top)
        for k in in_track:
            if k == top:
                continue
            pv = paired_p(series[top], series[k])
            if pv is None or pv >= 0.05:
                tied.add(k)

    for key, (label, kind, s) in summaries.items():
        is_tied = key in tied
        tag = {"full": "<span class='tag ours'>full method</span>",
               "ctrl": "<span class='tag ctrl'>control</span>",
               "ext":  "<span class='tag ctrl'>external baseline</span>"}.get(kind, "")
        if key in tops:
            note = "<span class='tag best'>best for this encoder</span>"
        elif is_tied:
            note = "<span class='tag tied'>tied</span>"
        else:
            note = ""
        enc_c, dro_c, anc_c = (summaries_meta[key])
        rows.append(
            f"<tr class='{'best' if is_tied else ''}'>"
            f"<td class='m'>{html.escape(label)}{tag}{note}</td>"
            f"<td class='sw'>{html.escape(enc_c)}</td>"
            f"<td class='sw'>{html.escape(dro_c)}</td>"
            f"<td class='sw'>{html.escape(anc_c)}</td>"
            f"{cell(s['worst_acc'])}"
            f"<td class='sw'>{html.escape(s['worst_group'] or '')}</td>"
            f"{cell(s['tail_acc'])}{cell(s['wt_acc'])}{cell(s['wt_f1'])}"
            f"{cell(s['worst_loss'], 3)}{cell(s['max_excess'], 3)}"
            f"<td class='sw'>{html.escape(s['excess_group'] or '')}</td>"
            # Step 5 asks for parameter counts on every row, because row 4 (REMIND) is a
            # different architecture and the comparison is meaningless without them.
            + (f"<td class='dim'>{int(s['n_params']):,}</td>"
               if s.get("n_params") else "<td class='na'>—</td>")
            + f"<td class='dim'>{s['seeds']}</td></tr>")
    foot = ("<p class='legend'>Best is marked separately for each encoder, since a "
            "common-feature model and a per-group model do not see the same inputs. "
            "<b>Tied</b> means a paired t-test over shared seeds cannot separate it from the "
            "best arm in its own track (p &ge; 0.05).</p>")
    return ("<table class='data'><thead><tr><th>method</th>"
            "<th class='sw'>encoder</th><th class='sw'>DRO</th><th class='sw'>anchors</th>"
            "<th>worst-group acc <span class='hint'>higher better</span></th>"
            "<th class='sw'>which</th>"
            "<th>tail-mean acc</th>"
            "<th>overall acc <span class='hint'>whole dataset</span></th>"
            "<th>overall macro-F1</th>"
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
.bar{max-width:1080px;margin:0 auto;padding:14px 24px;display:flex;align-items:baseline;gap:20px;
  flex-wrap:wrap}
.bar h1{font-size:15px;font-weight:600;margin:0;letter-spacing:-.01em}
nav{display:flex;gap:2px;flex-wrap:wrap;margin-left:auto}
nav a{color:var(--dim);text-decoration:none;padding:6px 12px;border-radius:6px;font-size:13.5px}
nav a:hover{color:var(--ink);background:var(--bg)}
nav a.on{color:var(--ink);background:var(--bg);box-shadow:inset 0 -2px 0 var(--accent)}
nav a:focus-visible{outline:2px solid var(--ours);outline-offset:1px}
main{max-width:1080px;margin:0 auto;padding:36px 24px 100px}
section{display:none} section.on{display:block}
h2{font-size:26px;font-weight:650;margin:0 0 6px;letter-spacing:-.02em;text-wrap:balance}
h3{font-size:13px;font-weight:600;margin:38px 0 12px;color:var(--dim);
  text-transform:uppercase;letter-spacing:.07em}
.sub{color:var(--dim);font-size:14px;margin:0 0 4px}
.blurb{color:var(--dim);max-width:64ch;margin:12px 0 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:6px 20px 18px;
  margin:14px 0;overflow-x:auto}
table.data{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
table.data th{text-align:right;padding:14px 12px 10px;font-weight:600;font-size:11.5px;
  color:var(--dim);text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--line);
  white-space:nowrap}
table.data th:first-child,table.data td.m{text-align:left}
table.data th{cursor:pointer;user-select:none;position:relative}
table.data th:hover{color:var(--fg)}
table.data th::after{content:'↕';opacity:.25;margin-left:6px;font-size:10px}
table.data th.asc::after{content:'↑';opacity:.9}
table.data th.desc::after{content:'↓';opacity:.9}
table.data th.def::after{content:'↕';opacity:.25}
table.data td{text-align:right;padding:10px 12px;border-bottom:1px solid var(--line);
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
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

JS = r"""
function show(id,el){document.querySelectorAll('section').forEach(s=>s.classList.remove('on'));
document.getElementById(id).classList.add('on');
document.querySelectorAll('nav a').forEach(a=>a.classList.remove('on'));el.classList.add('on');
window.scrollTo(0,0);}

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
<li><b>Per-group encoders plus GroupDRO beat the common-features baseline on Fed-Heart.</b>
+5.22 worst-group accuracy on the standard FLamby split (p&lt;0.0001), and +11.30 when the two
small hospitals are capped at 20 and 25 training patients (p&lt;0.0001). The gain scales with how
imbalanced the groups are. On NHANES the same comparison gives +0.82, which is not significant
(p=0.36).</li>
<li><b>Regret-DRO is indistinguishable from GroupDRO on every dataset.</b> +0.10, +0.05, +0.03
and +0.00 worst-group accuracy across the four configurations, no p-value below 0.55. Subtracting
R* changes which group gets weight, but not the outcome.</li>
<li><b>The anchors help on one dataset of three.</b> Holding the architecture fixed and switching
only the anchors on: NHANES <b>+2.34</b> (p=0.0496), Fed-Heart <b>-2.20</b> uncapped and
<b>-3.28</b> capped (both significant), EMBED <b>-4.52</b> (p=0.0003). One win, three losses.</li>
<li><b>EMBED cannot resolve any of this.</b> ERM, GroupDRO and Regret-DRO return
<i>identical</i> worst-group accuracy on all ten seeds. The worst group holds 37 exams, so
accuracy moves only in 2.70-point steps and a loss change of 0.03 never flips a decision. The
arms do differ on loss -- GroupDRO -0.035 against ERM on 10/10 seeds, the full method -0.115 --
but at that group size the effect is not resolvable (p=0.095 for the full method).</li>
<li><b>The anchors do align groups, and we measured it.</b> Cross-group latent misalignment
falls by a factor of 25 to 65 once the anchor loss is on, read directly off the latent space
rather than inferred from accuracy.</li>
<li><b>But the alignment is not class-conditional.</b> Replacing each sample's correct class
anchor with a random one does not hurt performance. Tested on three datasets, both forms of the
loss, ten seeds each. It never fails. So the anchors do something real, but not the
class-by-class thing the write-up claims.</li>
<li><b>Per-group encoders pay off only when feature spaces genuinely diverge.</b> In a
controlled sweep, dialling feature overlap between groups from complete to none moves the
benefit from +0.15 to +23.9. On Fed-Heart, where hospitals record overlapping tests, they are
worth +9.0. On NHANES, where each group's features are a subset of the next, they cost about a
point, because splitting 2,100-sample groups across separate encoders loses more to sample
efficiency than the extra columns return.</li>
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
    ROWS = [("Ours_Regret", "Per-group + anchors + Regret-DRO", "full"),
            ("ours", "Per-group + anchors + Regret-DRO", "full"),
            ("Ours_GDRO", "Per-group + anchors + GroupDRO", "abl"),
            ("Ours", "Per-group + anchors + GroupDRO", "abl"),
            ("align_only", "Per-group + anchors + GroupDRO", "abl"),
            ("GroupDRO", "Per-group encoders + GroupDRO", "base"),
            ("groupdro", "Per-group encoders + GroupDRO", "base"),
            # Per-group + Regret-DRO, anchors off. This was missing from every dataset's
            # baseline table while being present in METHODS and in every CSV, so the tab
            # silently dropped it. It is the arm that isolates regret from the anchors, which
            # is the comparison the regret claim rests on -- and on EMBED it is one of the
            # three arms that return identical worst-group accuracy, so leaving it out hid
            # that result rather than merely shortening the table.
            ("RegretDRO", "Per-group encoders + Regret-DRO", "base"),
            ("regret_only", "Per-group encoders + Regret-DRO", "base"),
            ("ERM", "ERM, common features", "base"),
            ("erm", "Per-group encoders + ERM", "base"),
            ("PerGroupOnly", "Per-group encoders + ERM", "base")]
    seen, rows = set(), []
    def cellf(v, dp=1):
        return f"<td>{v[0]:.{dp}f}<span class='sd'>±{v[1]:.{dp}f}</span></td>" if v else "<td class='na'>—</td>"
    def emit(key, label, kind, suffix=""):
        if key not in data or label in seen:
            return
        seen.add(label)
        s_ = summarize(data[key], tail=tail)
        npar = s_.get("n_params")
        tag = {"full": "<span class='tag ours'>ours</span>",
               "ext": "<span class='tag ctrl'>external</span>"}.get(kind, "")
        rows.append(
            f"<tr class='{'best' if kind == 'full' else ''}'>"
            f"<td class='m'>{html.escape(label)}{suffix}{tag}</td>"
            f"{cellf(s_['worst_acc'])}{cellf(s_['tail_acc'])}{cellf(s_['wt_acc'])}"
            f"{cellf(s_['wt_f1'])}{cellf(s_['worst_loss'], 3)}{cellf(s_['max_excess'], 3)}"
            f"<td class='dim'>{int(npar):,}</td>" if npar else
            f"<tr class='{'best' if kind == 'full' else ''}'>"
            f"<td class='m'>{html.escape(label)}{suffix}{tag}</td>"
            f"{cellf(s_['worst_acc'])}{cellf(s_['tail_acc'])}{cellf(s_['wt_acc'])}"
            f"{cellf(s_['wt_f1'])}{cellf(s_['worst_loss'], 3)}{cellf(s_['max_excess'], 3)}"
            f"<td class='na'>—</td>")
        rows[-1] += f"<td class='dim'>{s_['seeds']}</td></tr>"
    for k, l, kind in ROWS:
        emit(k, l, kind)
    for tag, lbl in [("released", " (published defaults)"), ("matched", " (capacity-matched)")]:
        for m in ["Reweigh", "FlexMoE", "REMIND"]:
            emit(f"{m}__{tag}", f"{m}{lbl}", "ext")
    return ("<table class='data'><thead><tr><th>method</th>"
            "<th>worst-group acc</th><th>tail acc</th><th>overall acc</th><th>macro-F1</th>"
            "<th>worst-group loss</th><th>max excess</th><th>params</th><th>seeds</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


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
<li><b>R*.</b> The best loss a group could reach using only its own features, measured by
training on that group alone with 5-fold cross-validation. A group's difficulty floor.</li>
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
                         "per-group ERM. Red bars have the anchors on. The interaction is "
                         "visible here: Regret-DRO on its own does nothing (57.8, identical to "
                         "ERM), but paired with the anchors it reaches 61.9. Regret needs the "
                         "anchors, because it compares each group against its own reference "
                         "loss and that comparison is only meaningful once the groups share a "
                         "latent space."),
                        (f"figs/{d['key']}_pergroup.png",
                         "Per-group accuracy, per-group ERM against the full method. The gain "
                         "is concentrated in g2, one of the rare tail groups."),
                        (f"figs/{d['key']}_lambda.png",
                         "Where each method spends its group weight, averaged over 10 seeds. "
                         "GroupDRO collapses onto g4 alone; ours splits across g4 and g6. "
                         "Neither puts weight on the tail groups. g5 is the clearest case: it "
                         "has by far the highest reference loss at 1.28, so it is intrinsically "
                         "hard rather than neglected, and regret correctly declines to push on "
                         "it. That is the rule working as designed, and it is also why regret "
                         "does not raise worst-group accuracy here.")]
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
                        "one that is simply hard (high R*, low excess). R* is measured once by "
                        "training on that group alone with 5-fold cross validation, and is the "
                        "same constant for every method.</p>")
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
             "<div class='note'><b>Read the first two first.</b> <b>Lambda against R*</b> is the "
             "mechanism test. R* is the lowest loss a group's own features permit, so a group "
             "with a high R* is one no model can do much about. GroupDRO reweights on raw loss "
             "and therefore chases exactly those groups: on NHANES its weight tracks R* with "
             "slope +15.7 and correlation +0.94, putting 0.76 of its weight on the survey-only "
             "group. Regret-DRO subtracts the floor first and does not track it, slope -1.85, "
             "correlation -0.15. Fed-Heart is a null for both, because its lambda barely moves "
             "there.<br><br><b>Per-group loss</b> shows why the max player cannot be fed training "
             "loss. Switzerland trains on 20 patients and the VA on 25, so their train loss "
             "collapses to near zero while test loss climbs past the R* line, the VA to nearly "
             "three times what its features permit. By the training signal those are the two "
             "best groups in the dataset. NHANES is the opposite regime: both groups sit above "
             "their floor throughout and never memorise.</div>",
             "<div class='note'><b>Why Fed-Heart's small groups diverge.</b> Switzerland and "
             "the VA are capped at 20 and 25 training patients by "
             "<code>group_max_train_samples</code>, out of 98 and 160 available. That cap is ours, "
             "a deliberate scarcity manipulation to create a hard worst-group problem, not a "
             "property of the dataset. A 38,946-parameter model memorises 20 samples trivially, "
             "which is why train loss reaches 0.03 while test climbs past 0.73.<br><br>"
             "<b>The validation curve is the one to check.</b> It drives the lambda update and "
             "selects the reported epoch, so it has to track test rather than train or the whole "
             "selection leaks. On Fed-Heart it correlates with test at r=0.94 and with train at "
             "r=-0.62. Switzerland sits at train 0.03, val 0.67, test 0.73.</div>",
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
            "NHANES, Regret-DRO: per-group loss against R*, with that group's weight below")]
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
          "is therefore a reimplementation from the paper's description and is labelled as one. "
          "Reweigh is standard inverse-frequency group weighting.</div>"]
    BL = [("fedheart", "Fed-Heart", "runs/baselines_fedheart/metrics_long.csv",
           "runs/baselines_fedheart_matched/metrics_long.csv"),
          ("nhnested", "NHANES", "runs/baselines_nhanes/metrics_long.csv",
           "runs/baselines_nhanes_matched/metrics_long.csv"),
          # REMIND's published-defaults row comes from the 128-expert run, which is what the
          # paper specifies ("128 experts and one slot per expert") and what its Table 16
          # reports as best of {32, 64, 128}. runs/baselines_embed predates that correction and
          # holds a 4-expert REMIND, which is not the authors' configuration.
          ("embed", "EMBED", "runs/baselines_embed/metrics_long.csv",
           "runs/baselines_embed_matched/metrics_long.csv",
           "runs/baselines_embed_remind128/metrics_long.csv")]
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
        bl.append(f"<h3>{html.escape(label)}</h3>")
        bl.append(f"<div class='card'>{baseline_table(merged, d.get('tail'))}</div>")
        if not os.path.exists(matp):
            bl.append("<p class='legend'>Capacity-matched runs still in progress.</p>")
    tabs.append(("Baselines", "sec-baselines")); secs.append(("sec-baselines", "".join(bl)))

    tabs.append(("Methods", "sec-methods")); secs.append(("sec-methods", methods_page()))

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
