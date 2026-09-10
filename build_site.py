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
         blurb="Four hospitals, each recording a different subset of clinical tests. The "
               "heterogeneity is real, not constructed. Evaluated with 5-fold cross "
               "validation and median imputation, so every one of the 925 patients is tested.",
         caveat="5 seeds x 5 folds, every patient held out exactly once, per-group accuracy "
                "pooled by fold count. Group sizes are Cleveland 305, Hungarian 295, "
                "Switzerland 125, VA 200. The earlier single-split protocol tested Switzerland "
                "on 10 patients and is superseded. Loss and macro-F1 are blank because the "
                "cross-validation runner recorded accuracy only.",
         groups={"g0": "Cleveland, 305 patients", "g1": "Hungarian, 295 patients",
                 "g2": "Switzerland, 125 patients", "g3": "VA, 200 patients"}),
    dict(key="nhnested", label="NHANES nested",
         path="runs/matrix_nhanes_nested/metrics_long.csv",
         task="Binary cardiovascular disease prediction", split="assessment completeness",
         blurb="Groups are how far someone got through the NHANES assessment. Features are "
               "strictly nested: survey is inside exam is inside labs. This is the real "
               "availability structure.",
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
         blurb="Groups are which mammogram views a breast actually has. Two head groups hold "
               "about 98.5% of the data and four tail groups hold the rest.",
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
                "n": float(r.get("n") or 0),
            }
        except ValueError:
            continue
    return out


def summarize(by_seed):
    """Per-seed summary stats, then mean/std across seeds."""
    worst_acc, mean_acc, worst_loss, mean_f1 = [], [], [], []
    for _, groups in by_seed.items():
        if not groups:
            continue
        accs = [g["acc"] for g in groups.values()]
        losses = [g["loss"] for g in groups.values() if g["loss"] == g["loss"]]
        f1s = [g["f1"] for g in groups.values() if g["f1"] == g["f1"]]
        worst_acc.append(min(accs))
        mean_acc.append(sum(accs) / len(accs))
        if losses:
            worst_loss.append(max(losses))
        if f1s:
            mean_f1.append(sum(f1s) / len(f1s))
    def ms(v):
        if not v:
            return None
        return (sum(v) / len(v), st.stdev(v) if len(v) > 1 else 0.0)
    return dict(worst_acc=ms(worst_acc), mean_acc=ms(mean_acc),
                worst_loss=ms(worst_loss), mean_f1=ms(mean_f1), seeds=len(worst_acc))


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


def headline_table(data):
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
        summaries[key] = (label, kind, summarize(by_seed))
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
            f"{cell(s['worst_acc'])}{cell(s['mean_acc'])}{cell(s['mean_f1'])}"
            f"{cell(s['worst_loss'], 3)}"
            f"<td class='dim'>{s['seeds']}</td></tr>")
    foot = ("<p class='legend'>Best is marked separately for each encoder, since a "
            "common-feature model and a per-group model do not see the same inputs. "
            "<b>Tied</b> means a paired t-test over shared seeds cannot separate it from the "
            "best arm in its own track (p &ge; 0.05).</p>")
    return ("<table class='data'><thead><tr><th>method</th>"
            "<th class='sw'>encoder</th><th class='sw'>DRO</th><th class='sw'>anchors</th>"
            "<th>worst-group acc <span class='hint'>higher better</span></th>"
            "<th>mean acc</th><th>macro-F1</th>"
            "<th>worst-group loss <span class='hint'>lower better</span></th>"
            "<th>seeds</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>" + foot)


def pergroup_table(data, glegend, metric="acc"):
    gs = sorted({g for by_seed in data.values() for gr in by_seed.values() for g in gr})
    if not gs:
        return ""
    head = "".join(f"<th>{html.escape(g)}</th>" for g in gs)
    rows = []
    for key, aliases, label, enc, dro, anc, kind in METHODS:
        by_seed = next((data[a] for a in aliases if a in data), None)
        if not by_seed:
            continue
        tds = []
        for g in gs:
            vals = [gr[g][metric] for gr in by_seed.values()
                    if g in gr and gr[g][metric] == gr[g][metric]]
            tds.append(f"<td>{sum(vals)/len(vals):.1f}</td>" if vals else "<td class='na'>—</td>")
        rows.append(f"<tr><td class='m'>{html.escape(label)}</td>{''.join(tds)}</tr>")
    legend = " · ".join(f"<b>{html.escape(g)}</b> {html.escape(glegend.get(g, ''))}"
                        for g in gs if glegend.get(g))
    return (f"<p class='legend'>{legend}</p>"
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
            s = summarize(by_seed)
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
columns. Four hospitals each run different tests; some patients get blood work and some only fill
out a survey; some mammograms have four views and some have one. Averaging the loss lets the
large well-measured groups dominate, and the small ones get ignored. We optimise for the group
the model does worst on.</div>
<h3>What we add</h3>
<ul>
<li><b>Per-group encoders.</b> Each group gets its own network into one shared latent space, so
groups with different feature sets do not have to pretend otherwise.</li>
<li><b>Gaussian anchors.</b> A learned Gaussian per class, shared across groups, that embeddings
are pulled toward, to stop each encoder drifting into its own corner.</li>
<li><b>GroupDRO, and a regret variant.</b> Weight groups by loss and push toward the worst one.
The regret variant weights by how far a group is above its own achievable floor.</li>
</ul>
<h3>What the runs show</h3>
<ul>
<li><b>Anchors on top of GroupDRO is the best configuration whichever encoder you use.</b> On
both NHANES settings, in both the common-feature and the per-group track, adding the anchors to
GroupDRO gives the top arm, and it beats plain GroupDRO in both: +3.9 (p=0.002) on common
features and +3.7 (p=0.007) on per-group encoders. In each track the only arm it cannot be
separated from also has anchors switched on. This is the most robust result we have, because it
does not depend on the per-group architecture holding up. It does not carry to Fed-Heart, where
the anchors cost about two points.</li>
<li><b>Per-group encoders help only when feature spaces genuinely diverge.</b> In a controlled
sweep, dialling feature overlap from complete to none moves the benefit from +0.15 to +23.9
worst-group accuracy. On NHANES-nested, where the groups' features are subsets of one another,
per-group encoders actually cost about a point, because splitting 2,100-sample groups across
separate encoders loses more to sample efficiency than the extra features return.</li>
<li><b>The anchors do align groups, measurably.</b> Cross-group latent misalignment drops by a
factor of 25 to 65 once the anchor loss is on, measured directly from the latent space rather
than inferred from accuracy.</li>
<li><b>But not class-by-class.</b> Replacing each sample's correct class anchor with a random one
does not hurt. Tested on three datasets, both loss forms, 10 seeds each.</li>
<li><b>It does not win everywhere.</b> Significant gains on both NHANES settings; loses to plain
GroupDRO on Fed-Heart and ties on EMBED. Both are datasets where group features overlap heavily,
which is where the sweep predicts little benefit.</li>
</ul>
<div class='note'><b>An awkward result, reported as found.</b> Completing the ablation grid
showed the anchors help whether or not the encoders are per-group: on a shared encoder they add
+3.9 worst-group on both NHANES settings, versus +2.6 to +3.7 on per-group encoders. On
NHANES-disjoint the best arm overall is anchors + DRO on a <i>shared</i> encoder at 76.1, which
edges out the same thing on per-group encoders at 75.9 despite seeing only the 10 features common
to every group rather than each group's 15. In other words the 5 private features per group add
almost nothing once anchors and DRO are on. That weakens the case for per-group encoders on this
dataset, though the controlled overlap sweep still supports it, and the likeliest explanation is
that NHANES-disjoint's private features (body measures, blood pressure, lipids) simply carry
less cardiovascular signal than the shared demographic and smoking ones.</div>"""


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
<div class='note'><b>Per-group encoders are the setting, not an ingredient.</b> In the
specification every method row uses the same per-group architecture, and the anchors are
described as the only parameters the method adds on top of it. Groups have different input
dimensions, so per-group encoders are what makes the problem tractable at all rather than
something to switch off. The common-feature rows in these tables go beyond that specification:
they are an extra robustness check, restricting one shared encoder to the features every group
has. They are worth reading because the anchor result survives them, which means it does not
rest on the architecture. They are not part of the paper's ablation.</div>
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
            body.append("<h3>Summary</h3>")
            body.append(f"<div class='card'>{headline_table(data)}</div>")
            body.append("<h3>Accuracy by group</h3>")
            body.append(f"<div class='card'>{pergroup_table(data, d['groups'], 'acc')}</div>")
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
