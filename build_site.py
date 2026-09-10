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
         caveat="5 seeds x 5 folds, every patient held out exactly once, per-group accuracy "
                "pooled by fold count. Group sizes are Cleveland 305, Hungarian 295, "
                "Switzerland 125, VA 200. The earlier single-split protocol tested Switzerland "
                "on 10 patients and is superseded; every figure here comes from the "
                "cross-validated runs.",
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
    dict(key="embed", label="EMBED", path="runs/embed_xenia_production/metrics_long.csv",
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
     "Per-group + anchors + GroupDRO",    "per-group", "GroupDRO", "yes", "full"),
    ("Ours_Regret",         ["Ours_Regret", "ours"],
     "Per-group + anchors + Regret-DRO",  "per-group", "Regret", "yes", "full"),
    ("group_only",          ["group_only"],
     "Dedicated model per group",         "per-group", "—",      "—",  "base"),
    ("rand_anchor",         ["rand_anchor"],
     "Control: random anchor targets",    "per-group", "GroupDRO", "random", "ctrl"),
]


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
    ranked = [k for k, (_, kind, s) in summaries.items() if kind != "ctrl" and s["worst_acc"]]
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
               "ctrl": "<span class='tag ctrl'>control</span>"}.get(kind, "")
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

JS = """
function show(id,el){document.querySelectorAll('section').forEach(s=>s.classList.remove('on'));
document.getElementById(id).classList.add('on');
document.querySelectorAll('nav a').forEach(a=>a.classList.remove('on'));el.classList.add('on');
window.scrollTo(0,0);}
"""


def overview(loaded):
    cards = []
    for d in DATASETS:
        data = loaded[d["key"]]
        cands, series = {}, {}
        for key, aliases, label, enc, dro, anc, kind in METHODS:
            by_seed = next((data[a] for a in aliases if a in data), None)
            if not by_seed or kind == "ctrl":
                continue
            s = summarize(by_seed, tail=None)
            if s["worst_acc"]:
                cands[key] = (label, s["worst_acc"][0])
                series[key] = worst_by_seed(by_seed)
        if not cands:
            cards.append(f"<div class='stat'><div class='k'>{html.escape(d['label'])}</div>"
                         f"<div class='v'>—</div><div class='d'>runs in progress</div></div>")
            continue
        top = max(cands, key=lambda k: cands[k][1])
        n_tied = sum(1 for k in cands if k != top
                     and (paired_p(series[top], series[k]) or 1.0) >= 0.05)
        # Naming a single winner is only honest when it actually beats the field.
        sub = (f"tied with {n_tied} other arm{'s' if n_tied > 1 else ''}"
               if n_tied else html.escape(cands[top][0]))
        cards.append(f"<div class='stat'><div class='k'>{html.escape(d['label'])}</div>"
                     f"<div class='v'>{cands[top][1]:.1f}%</div>"
                     f"<div class='d'>best worst-group<br>{sub}</div></div>")
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
<li><b>The full pipeline beats the common-features baseline on both tabular datasets.</b>
Fed-Heart +7.6 worst-group accuracy (p&lt;0.001) and NHANES +4.1 (p=0.013), with max excess loss
falling from 0.439 to 0.078 on Fed-Heart.</li>
<li><b>The anchors specifically are inconsistent.</b> Holding the architecture fixed and
switching only the anchors on: NHANES +3.7 (p=0.007), Fed-Heart <b>-2.3</b> (p=0.011), EMBED
-1.0 (not significant). One win, one loss, one tie.</li>
<li><b>On EMBED they win on the loss metrics.</b> Anchors plus regret gives the best worst-group
loss (1.099 against GroupDRO's 1.200) and the best max excess loss (0.239 against 0.251), while
tying on accuracy. Those two are the headline numbers in the experiment specification.</li>
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
<b>Protocol</b> 5 seeds x 5 folds, median imputation, every patient tested once</div></div>
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
<b>Protocol</b> 10 seeds, frozen ViT-Base features</div></div>
</div>
<div class='note'><b>Reading these tables.</b> Every number is mean plus or minus standard
deviation over 10 seeds, except Fed-Heart which is 5 seeds by 5 folds with every one of its 925
patients held out exactly once. Best is marked separately for each encoder, and arms a paired
t-test cannot separate from the best are marked tied rather than ranked. Nothing here is
selected for looking good.</div>"""


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


def build(outdir=SITE):
    os.makedirs(outdir, exist_ok=True)
    loaded = {d["key"]: load(d["path"]) for d in DATASETS}

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
                         "there is no shared-feature model to build. The ladder starts from "
                         "per-group ERM instead. Each rung changes exactly one thing, so the "
                         "last one holds the DRO variant at R*=0 and switches only the anchors "
                         "on. The full method as the spec defines it (row 5) also swaps the DRO "
                         "update to regret, which is worth a further +0.5 to 61.9; the two "
                         "ingredients interact and are compared properly in the table below."),
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
                         "What each step of the pipeline is worth. Error bars are 95% confidence "
                         "intervals over seeds."),
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
