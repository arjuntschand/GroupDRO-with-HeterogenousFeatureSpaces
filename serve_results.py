"""Local results viewer: renders every results table and figure in one browsable page.

  python serve_results.py            # builds site + serves on http://localhost:8000
  python serve_results.py --build    # build only (site/ directory)
  python serve_results.py --port 8080

Pulls in the markdown result docs (tables rendered as real HTML tables, sortable) and every
figure in documentation/figures/, plus the raw metrics_long.csv files as searchable tables.
"""
from __future__ import annotations
import argparse, csv, html, io, os, re, shutil

# NOTE: DOCS is no longer consumed by build(); tabs are constructed explicitly below.
# Kept only as a record of which documents are paper-facing.
DOCS = [
    ("Results", "documentation/ICML_RESULTS.md"),
    ("Anchor Analysis", "documentation/ANCHOR_RESULTS.md"),
    ("EMBED", "documentation/EMBED_XENIA.md"),
]
CSVS = [
    ("Fed-Heart", "runs/matrix_fedheart/metrics_long.csv"),
    ("NHANES-nested (natural)", "runs/matrix_nhanes_nested/metrics_long.csv"),
    ("NHANES-disjoint (synthetic)", "runs/matrix_nhanes_disjoint/metrics_long.csv"),
    ("EMBED (full, 10 seeds)", "runs/embed_xenia_production/metrics_long.csv"),
]

# ── context shown on the site so tables are self-explanatory ──────────────────
DATASETS_INFO = [
    dict(name="Fed-Heart", task="Binary heart-disease prediction",
         groups=4, split="4 hospitals (natural federated split)",
         detail="Each hospital records a different subset of clinical features, and that is where "
                "the heterogeneity comes from. These groups are real hospitals, not something we "
                "constructed.",
         rows=[("G0 Cleveland", "10 feats", "61 test", "R*=0.499"),
               ("G1 Hungarian", "8 feats", "53 test", "R*=0.592"),
               ("G2 Switzerland", "8 feats", "10 test", "R*=0.424"),
               ("G3 VA", "9 feats", "26 test", "R*=1.480")],
         note="G2 and G3 are capped to 20 and 25 training samples to simulate small sites with "
              "little data. G3 has the highest R* at 1.48, which means it is genuinely the "
              "hardest group to learn on its own."),
    dict(name="NHANES-nested (natural)", task="Binary CVD prediction",
         groups=3, split="assessment completeness",
         detail="This is the real availability structure. If a patient got lab work they also got "
                "the exam and the survey, so each group's features are a subset of the next one "
                "(G0 inside G1 inside G2).",
         rows=[("G0 survey only", "10 feats", "533 test", "R*=0.314"),
               ("G1 + exam", "13 feats", "530 test", "R*=0.276"),
               ("G2 + labs", "20 feats", "2338 test", "R*=0.266")],
         note="This is the natural, real world setting. Nothing here is constructed."),
    dict(name="NHANES-disjoint (synthetic)", task="Binary CVD prediction",
         groups=3, split="constructed feature partition",
         detail="This one we constructed as a stress test. Every group gets 15 features, made up of "
                "10 shared ones plus 5 that only that group has. Unlike nested, each group has "
                "private features nobody else sees.",
         rows=[("G0", "15 feats (10 shared + 5 unique)", "533 test", "R*=0.300"),
               ("G1", "15 feats (10 shared + 5 unique)", "530 test", "R*=0.283"),
               ("G2", "15 feats (10 shared + 5 unique)", "2338 test", "R*=0.265")],
         note="This setup is synthetic. Real NHANES availability is not disjoint. We include it "
              "to test the method when feature spaces are as different as possible."),
    dict(name="EMBED (mammography)", task="4-class BI-RADS breast density",
         groups=6, split="which imaging views are present",
         detail="Groups are defined by which mammogram views a breast actually has. The four "
                "possible views are M1 C-View CC, M2 C-View MLO, M3 FFDM CC and M4 FFDM MLO.",
         rows=[("g1 {M3}", "1 view", "tail", ""), ("g2 {M1,M3}", "2 views", "tail", ""),
               ("g3 {M4}", "1 view", "tail", ""), ("g4 {M3,M4}", "2 views", "HEAD ~57%", ""),
               ("g5 {M1,M3,M4}", "3 views", "tail", ""), ("g6 all four", "4 views", "HEAD ~37%", "")],
         note="Production run in progress on the full 128,680-row dataset."),
]

METHODS_INFO = [
    ("ERM", "One shared encoder, plain cross entropy. The naive baseline with no group awareness."),
    ("GroupDRO", "Per-group encoders plus GroupDRO, which reweights groups by their raw loss. This "
                 "is the standard robustness baseline, and the same thing as setting R* to zero."),
    ("Regret-DRO", "Same idea, but it reweights by regret instead of raw loss. Regret is how far a "
                   "group is above its own floor R*. Groups that are already doing as well as "
                   "they possibly can stop getting pushed."),
    ("Anchors only", "Per-group encoders plus the class anchors, with no DRO at all. This isolates what "
                     "the anchors contribute on their own."),
    ("Ours (anchors+GroupDRO)", "Anchors combined with standard GroupDRO."),
    ("Ours (anchors+regret)", "Anchors combined with regret reweighting. This is the full method."),
]

GLOSSARY = [
    ("group", "The unit this whole method works on. A group is a set of samples that share the "
              "same available features. For example one hospital that records 8 clinical "
              "variables, or patients who only filled out the survey and never got lab work. "
              "Different groups genuinely have different inputs, and that is what we mean by "
              "heterogeneous feature spaces."),
    ("head group", "A common group with lots of samples. Normal training is dominated by these "
                   "because they make up most of the data."),
    ("tail group", "A rare group with few samples. These get ignored by normal training, which "
                   "is the exact problem this method is trying to fix. The names head and tail "
                   "come from the shape of the group size distribution, where a couple of groups "
                   "are huge and the rest have a long thin tail. In EMBED, 2 of the 6 view "
                   "combinations cover about 95 percent of exams and the other 4 are rare."),
    ("worst-group accuracy", "The accuracy of whichever group the model does worst on. This is "
                             "the main number to watch. A model can look great at 90 percent "
                             "average accuracy while one minority group sits at 50 percent, and "
                             "this metric is what catches that. Higher is better."),
    ("overall vs balanced accuracy", "Overall counts every sample equally, so big groups dominate "
                                     "it. Balanced counts every group equally, so a group with 10 "
                                     "samples matters as much as one with 5,000."),
    ("R*_g (reference loss)", "The best loss a group could possibly reach using only its own "
                              "features. We measure it by training a model on that group alone "
                              "with 5-fold cross validation. Think of it as that group's "
                              "difficulty floor. A high R* means the group is just hard, like "
                              "Fed-Heart's VA site at 1.48. A low R* means it should be easy. It "
                              "is a fixed number we compute once, not something the model learns."),
    ("excess loss (L_g minus R*_g)", "How far a group is from its own floor. This separates two "
                                     "very different situations. A group can score badly because "
                                     "it is genuinely hard, which shows up as high R* and low "
                                     "excess, and there is nothing to fix there. Or it can score "
                                     "badly because the shared model is neglecting it, which "
                                     "shows up as low R* and high excess, and that is fixable. "
                                     "Regret optimisation targets excess loss instead of raw "
                                     "loss, so it stops pouring effort into groups that are "
                                     "already doing as well as they possibly can."),
    ("anchors ON / OFF", "ON means the anchor loss weights are 0.1. OFF means 0.001 rather than "
                         "exactly zero, because exactly zero turned out to be numerically "
                         "unstable here and would have made the anchor effect look like +7 when "
                         "the real number is +2.65."),
    ("mean plus/minus std", "Averaged over 10 random seeds, which are different model "
                            "initialisations. The spread tells you whether a difference is real "
                            "or just noise."),
]

FIGDIR = "documentation/figures"
SITE = "site"

CSS = """
:root{--bg:#0f1115;--panel:#171a21;--fg:#e6e9ef;--mut:#9aa4b2;--acc:#e05252;--acc2:#4a9eff;
--bd:#262b36;--good:#38b48b}
@media(prefers-color-scheme:light){:root{--bg:#f6f7f9;--panel:#fff;--fg:#1a1d23;--mut:#5b6472;
--bd:#e2e6ec}}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;z-index:10;background:var(--panel);border-bottom:1px solid var(--bd);
padding:12px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header h1{font-size:16px;margin:0;font-weight:650}
nav{display:flex;gap:6px;flex-wrap:wrap}
nav a{color:var(--mut);text-decoration:none;padding:5px 11px;border-radius:7px;font-size:12.5px;
border:1px solid transparent}
nav a:hover{color:var(--fg);background:var(--bg);border-color:var(--bd)}
nav a.on{color:#fff;background:var(--acc);border-color:var(--acc)}
main{max-width:1400px;margin:0 auto;padding:22px 20px 80px}
section{display:none} section.on{display:block}
h2{font-size:20px;margin:26px 0 12px;padding-bottom:7px;border-bottom:2px solid var(--acc)}
h3{font-size:16px;margin:22px 0 9px;color:var(--acc2)}
h4{font-size:14px;margin:16px 0 7px;color:var(--mut)}
p{color:var(--mut);margin:9px 0}
table{border-collapse:collapse;width:100%;margin:12px 0 20px;font-size:12.5px;background:var(--panel);
border-radius:9px;overflow:hidden;display:block;overflow-x:auto;white-space:nowrap}
th{background:var(--bd);padding:9px 12px;text-align:left;font-weight:640;
position:sticky;top:0;cursor:pointer;user-select:none}
th:hover{color:var(--acc2)}
td{padding:8px 12px;border-top:1px solid var(--bd)}
tbody tr:hover{background:rgba(224,82,82,.06)}
td:first-child,th:first-child{position:sticky;left:0;background:var(--panel);z-index:1}
th:first-child{background:var(--bd)}
code{background:var(--bd);padding:2px 6px;border-radius:4px;font-size:12px}
pre{background:var(--panel);padding:13px;border-radius:9px;overflow-x:auto;border:1px solid var(--bd)}
pre code{background:none;padding:0}
.fig{background:var(--panel);border:1px solid var(--bd);border-radius:11px;padding:14px;margin:16px 0}
.fig img{width:100%;height:auto;border-radius:7px;background:#fff}
.fig .cap{color:var(--mut);font-size:12.5px;margin-top:9px}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(440px,1fr))}
.filter{width:100%;max-width:340px;padding:8px 11px;border-radius:8px;border:1px solid var(--bd);
background:var(--panel);color:var(--fg);margin:8px 0 4px;font-size:13px}
blockquote{border-left:3px solid var(--acc);margin:12px 0;padding:2px 0 2px 14px;color:var(--mut)}
strong{color:var(--fg)}
.meta{color:var(--mut);font-size:12px;margin-bottom:14px}
hr{border:0;border-top:1px solid var(--bd);margin:26px 0}
"""

JS = """
function show(id,el){document.querySelectorAll('section').forEach(s=>s.classList.remove('on'));
document.getElementById(id).classList.add('on');
document.querySelectorAll('nav a').forEach(a=>a.classList.remove('on'));el.classList.add('on');
window.scrollTo(0,0);}
// sortable tables
document.addEventListener('click',e=>{
 if(e.target.tagName!=='TH')return;
 const th=e.target,tb=th.closest('table'),i=[...th.parentNode.children].indexOf(th);
 const body=tb.tBodies[0],rows=[...body.rows];
 const asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
 const val=r=>{const t=r.cells[i]?.innerText.trim()??'';const m=parseFloat(t.replace(/[^0-9.\\-]/g,''));
  return isNaN(m)?t.toLowerCase():m;};
 rows.sort((a,b)=>{const x=val(a),y=val(b);
  if(typeof x==='number'&&typeof y==='number')return asc?x-y:y-x;
  return asc?String(x).localeCompare(y):String(y).localeCompare(x);});
 rows.forEach(r=>body.appendChild(r));});
// csv filter
function filt(inp,tid){const q=inp.value.toLowerCase();
 document.querySelectorAll('#'+tid+' tbody tr').forEach(r=>{
  r.style.display=r.innerText.toLowerCase().includes(q)?'':'none';});}
"""


def _dedash(t):
    """Render em dashes as ordinary punctuation. In headings 'Table 1 — Foo' becomes
    'Table 1. Foo'; elsewhere ' — ' becomes ', '."""
    t = re.sub(r"^(#{1,6}\s*[^—]*?)\s+—\s+", r"\1. ", t)
    t = re.sub(r"^(\*?\*?Table \d+)\s+—\s+", r"\1. ", t)
    t = t.replace(" — ", ", ").replace("—", ", ")
    return t


def md_inline(s):
    s = _dedash(s)
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", s)
    return s


def md_to_html(md):
    md = "\n".join(_dedash(l) for l in md.split("\n"))
    out, i, lines = [], 0, md.split("\n")
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j]); j += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            i = j + 1; continue
        # table
        if ln.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[-: |]+\|\s*$", lines[i + 1]):
            hdr = [c.strip() for c in ln.strip().strip("|").split("|")]
            j = i + 2; rows = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")]); j += 1
            t = ["<table><thead><tr>" + "".join(f"<th>{md_inline(h)}</th>" for h in hdr) + "</tr></thead><tbody>"]
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t)); i = j; continue
        if ln.startswith("#"):
            lv = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{min(lv+1,5)}>{md_inline(ln.lstrip('# ').strip())}</h{min(lv+1,5)}>")
            i += 1; continue
        if ln.startswith(">"):
            out.append(f"<blockquote>{md_inline(ln.lstrip('> '))}</blockquote>"); i += 1; continue
        if re.match(r"^\s*[-*]\s+", ln):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(f"<li>{md_inline(re.sub(r'^\\s*[-*]\\s+','',lines[i]))}</li>"); i += 1
            out.append("<ul>" + "".join(items) + "</ul>"); continue
        if ln.strip():
            out.append(f"<p>{md_inline(ln)}</p>")
        i += 1
    return "\n".join(out)


def csv_section(title, path, tid):
    if not os.path.exists(path):
        return ""
    rows = list(csv.reader(open(path)))
    if not rows:
        return ""
    hdr, body = rows[0], rows[1:]
    h = [f"<h3>{html.escape(title)}</h3>",
         f"<p class='meta'>{len(body)} rows &middot; <code>{html.escape(path)}</code> "
         f"&middot; click a column header to sort</p>",
         f"<input class='filter' placeholder='filter rows…' oninput=\"filt(this,'{tid}')\">",
         f"<table id='{tid}'><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in hdr) + "</tr></thead><tbody>"]
    for r in body:
        h.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>")
    h.append("</tbody></table>")
    return "".join(h)


FIG_CAPS = {
    "icml_fig1_methods": "Figure 1. Method comparison across datasets, scored on worst-group accuracy over 10 seeds. This is the headline result.",
    "icml_fig2_pergroup": "Figure 2. Per-group accuracy, plain ERM against the full method. Shows which groups the gain actually comes from.",
    "icml_fig3_anchor_effect": "Figure 3. What the anchors add, with the encoder and GroupDRO held fixed. Stars mark paired significance, ** is p below 0.01.",
    "icml_fig4_mechanism": "Figure 4. How the anchors work. Panel (a) sweeps the anchor weight, panel (b) shows which of the two anchor losses is doing the work.",
}




def split_results_md(path="documentation/ICML_RESULTS.md"):
    """Split the combined results doc into: headline, one section per dataset, mechanism.
    Tables 1/6 are cross-dataset; Tables 2/3 have per-dataset subsections; 4/5 are analysis."""
    if not os.path.exists(path):
        return {}, {}
    txt = open(path).read()
    # top-level table blocks
    blocks, cur, name = {}, [], "intro"
    for ln in txt.split("\n"):
        if ln.startswith("## "):
            blocks[name] = "\n".join(cur); cur = []; name = ln[3:].strip()
        cur.append(ln)
    blocks[name] = "\n".join(cur)

    cross, per_ds = {}, {}
    for k, v in blocks.items():
        if k.startswith("Table 1") or k.startswith("Table 6") or k == "intro":
            cross[k] = v
        elif k.startswith("Table 4") or k.startswith("Table 5"):
            cross[k] = v
        else:
            # Tables 2/3 contain '### <dataset>' subsections — split them out
            sub, subname = [], None
            for ln in v.split("\n"):
                if ln.startswith("### "):
                    if subname:
                        per_ds.setdefault(subname, []).append((k, "\n".join(sub)))
                    subname = ln[4:].strip(); sub = [ln]
                else:
                    sub.append(ln)
            if subname:
                per_ds.setdefault(subname, []).append((k, "\n".join(sub)))
            else:
                cross[k] = v
    return cross, per_ds


def datasets_section():
    h = ["<h2>What this research is</h2>",
         "<div class='fig'>"
         "<p><strong>The problem.</strong> Real medical data is not uniform. One hospital records "
         "8 clinical variables, another records 12. Some patients get a full lab workup, others "
         "only fill out a questionnaire. Some mammograms have four imaging views, most have two. "
         "Standard machine learning assumes every sample has the same features, so it either "
         "throws away the extra information or throws away the incomplete patients. Either way, "
         "the groups with less data end up with the worst predictions.</p>"
         "<p><strong>The method.</strong> Three pieces that work together. First, a separate "
         "small network per group, so each group can use its own feature set. Second, learned "
         "reference points called class anchors in a shared space, which every group's network "
         "gets pulled toward so the groups become comparable. Third, GroupDRO, which keeps "
         "shifting attention to whichever group is currently doing worst.</p>"
         "<p><strong>The claim.</strong> The contribution is not any single one of those pieces. "
         "It is that the anchors and GroupDRO help each other. The anchors organise the shared "
         "space so that GroupDRO can actually reweight across groups that do not even share the "
         "same inputs. You can see this in Headline, Table 6. On NHANES-nested each piece on its "
         "own gives almost nothing, +0.35 and +1.26, but together they give +4.06.</p>"
         "<p><strong>What success looks like.</strong> Higher worst-group accuracy, meaning the "
         "score of whichever group the model treats worst goes up, without giving up overall "
         "accuracy.</p></div>",
         "<h2>Datasets and setup</h2>",
         "<p>Each dataset below defines groups differently. The group table for each one shows "
         "how many groups there are, how many features each group has, and how big its test set "
         "is. Those differences are exactly what separates one results table from another.</p>",
         "<div class='grid'>"]
    for d in DATASETS_INFO:
        rows = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td>"
                       f"<td>{html.escape(c)}</td><td>{html.escape(e)}</td></tr>"
                       for a, b, c, e in d["rows"])
        h.append(
            f"<div class='fig'><h3 style='margin-top:0'>{html.escape(d['name'])}</h3>"
            f"<p><strong>Task:</strong> {html.escape(d['task'])}<br>"
            f"<strong>Groups:</strong> {d['groups']}, split by {html.escape(d['split'])}</p>"
            f"<p>{html.escape(d['detail'])}</p>"
            f"<table><thead><tr><th>group</th><th>features</th><th>size</th><th>R*_g</th></tr>"
            f"</thead><tbody>{rows}</tbody></table>"
            f"<div class='cap'>{html.escape(d['note'])}</div></div>")
    h.append("</div>")

    h.append("<h2>Methods compared</h2>")
    h.append("<p>All methods share <em>identical model capacity</em> and differ only in the "
             "training objective, so differences are attributable to the objective alone.</p>")
    h.append("<table><thead><tr><th>method</th><th>what it does</th></tr></thead><tbody>")
    for m, desc in METHODS_INFO:
        h.append(f"<tr><td><strong>{html.escape(m)}</strong></td><td>{html.escape(desc)}</td></tr>")
    h.append("</tbody></table>")

    h.append("<h2>Metric glossary</h2>")
    h.append("<table><thead><tr><th>term</th><th>meaning</th></tr></thead><tbody>")
    for t, desc in GLOSSARY:
        h.append(f"<tr><td><strong>{html.escape(t)}</strong></td><td>{html.escape(desc)}</td></tr>")
    h.append("</tbody></table>")

    h.append("<h2>How to read the result tables</h2>"
             "<ul>"
             "<li><strong>Table 1.</strong> One row per method, one column per dataset. Use this "
             "to compare methods against each other.</li>"
             "<li><strong>Table 2.</strong> Every metric for every method, one dataset at a time.</li>"
             "<li><strong>Table 3.</strong> A per group breakdown. This shows which group is "
             "dragging the worst-group number down, and whether that group is just hard (high R*) "
             "or being neglected by the model (high excess loss).</li>"
             "<li><strong>Table 4.</strong> The two by two. Anchors on or off, crossed with regret "
             "on or off, with paired significance tests. This is the table that isolates what the "
             "new parts contribute.</li>"
             "<li><strong>Table 5.</strong> Same method, different group definitions. This is "
             "testing the group structure itself, not hyperparameters.</li>"
             "<li><strong>Table 6.</strong> The synergy check, which is the paper's main "
             "intellectual claim.</li>"
             "</ul>"
             "<p>All numbers are averaged over 10 random seeds and shown as mean plus or minus "
             "standard deviation.</p>")
    return "".join(h)


def build(outdir=SITE):
    os.makedirs(f"{outdir}/figs", exist_ok=True)
    figs = []
    if os.path.isdir(FIGDIR):
        for fn in sorted(os.listdir(FIGDIR)):
            if fn.endswith(".png"):
                shutil.copy(os.path.join(FIGDIR, fn), f"{outdir}/figs/{fn}")
                figs.append(fn)
    # order: icml figs, then paper figs, then anchor, then rest
    def rank(f):
        for i, p in enumerate(["icml_fig", "fig", "anchor_", "pergroup_", "ablation_"]):
            if f.startswith(p):
                return (i, f)
        return (9, f)
    figs.sort(key=rank)

    tabs, secs = [], []

    def figblock(names, heading=None):
        h = [f"<h2>{heading}</h2>"] if heading else []
        h.append("<div class='grid'>")
        for fn in names:
            stem = fn[:-4]
            cap = FIG_CAPS.get(stem, stem.replace("_", " "))
            h.append(f"<div class='fig'><img src='figs/{fn}' alt='{stem}'>"
                     f"<div class='cap'>{html.escape(cap)}</div></div>")
        h.append("</div>")
        return "".join(h)

    cross, per_ds = split_results_md()

    # 1. Overview
    tabs.append(("Overview", "sec-info"))
    secs.append(("sec-info", datasets_section()))

    # 1b. Paper draft: the written-up results, first thing after the overview
    if os.path.exists("documentation/PAPER_RESULTS_DRAFT.md"):
        tabs.append(("Paper Draft", "sec-draft"))
        secs.append(("sec-draft", md_to_html(open("documentation/PAPER_RESULTS_DRAFT.md").read())))

    # 2. Headline (cross-dataset tables + main figure)
    hl = ["<h2>Headline results</h2>"]
    for k in cross:
        if k.startswith("Table 1"):
            hl.append(md_to_html(cross[k]))
    hl.append(figblock([f for f in figs if "fig1" in f]))
    for k in cross:
        if k.startswith("Table 6"):
            hl.append(md_to_html(cross[k]))
    tabs.append(("Headline", "sec-headline")); secs.append(("sec-headline", "".join(hl)))

    # 3. One tab per dataset
    DS_TABS = [("Fed-Heart", "Fed-Heart"), ("NHANES-nested", "NHANES-nested (natural)"),
               ("NHANES-disjoint", "NHANES-disjoint (synthetic)")]
    for label, key in DS_TABS:
        parts = per_ds.get(key)
        if not parts:
            continue
        info = next((d for d in DATASETS_INFO if d["name"].startswith(label)), None)
        body = [f"<h2>{html.escape(key)}</h2>"]
        if info:
            rows = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td>"
                           f"<td>{html.escape(c)}</td><td>{html.escape(e)}</td></tr>"
                           for a, b, c, e in info["rows"])
            body.append(f"<div class='fig'><p><strong>Task:</strong> {html.escape(info['task'])} &nbsp;·&nbsp; "
                        f"<strong>{info['groups']} groups</strong>, split by {html.escape(info['split'])}</p>"
                        f"<p>{html.escape(info['detail'])}</p>"
                        f"<table><thead><tr><th>group</th><th>features</th><th>size</th>"
                        f"<th>R*_g</th></tr></thead><tbody>{rows}</tbody></table>"
                        f"<div class='cap'>{html.escape(info['note'])}</div></div>")
        # Fed-Heart: the cross-validated protocol is the headline result, so it goes first
        # and the older single-split tables are demoted to a collapsed section below it.
        if label == "Fed-Heart" and os.path.exists("documentation/FEDHEART_CV.md"):
            body.append(md_to_html(open("documentation/FEDHEART_CV.md").read()))
            body.append("<hr><details><summary style='cursor:pointer;color:var(--mut);"
                        "padding:8px 0'>Earlier single-split protocol (superseded, kept for "
                        "reference)</summary><div style='opacity:.75'>")
            for tname, content in parts:
                body.append(f"<h3>{html.escape(_dedash(tname))}</h3>")
                body.append(md_to_html(content))
            body.append("</div></details>")
        else:
            for tname, content in parts:
                body.append(f"<h3>{html.escape(_dedash(tname))}</h3>")
                body.append(md_to_html(content))
        pgf = [f for f in figs if "fig2" in f]
        if pgf:
            body.append(figblock(pgf, "Per-group figure (all datasets)"))
        sid = "sec-" + label.lower().replace("-", "")
        tabs.append((label, sid)); secs.append((sid, "".join(body)))

    # 4. Mechanism (anchor analysis + 2x2 + parameter test + figs 3/4)
    mech = ["<h2>Mechanism &amp; ablations</h2>"]
    for k in cross:
        if k.startswith("Table 4") or k.startswith("Table 5"):
            mech.append(md_to_html(cross[k]))
    mech.append(figblock([f for f in figs if "fig3" in f or "fig4" in f]))
    if os.path.exists("documentation/MECHANISM_CONTROLS.md"):
        mech.append("<hr>")
        mech.append(md_to_html(open("documentation/MECHANISM_CONTROLS.md").read()))
    if os.path.exists("documentation/ANCHOR_RESULTS.md"):
        mech.append("<hr>")
        mech.append(md_to_html(open("documentation/ANCHOR_RESULTS.md").read()))
    tabs.append(("Mechanism", "sec-mech")); secs.append(("sec-mech", "".join(mech)))

    # 5. EMBED: production results first, methodology notes collapsed underneath
    emb = []
    if os.path.exists("runs/embed_xenia_production/REPORT.md"):
        emb.append("<h2>EMBED, full dataset (128,680 rows / 22,997 patients)</h2>")
        emb.append("<div class='fig'><p>Frozen ViT-Base backbone, per-view projections into "
                   "per-group MLPs, shared head, four diagonal Gaussian class anchors. Six "
                   "groups defined by which mammogram views a breast has. Two head groups "
                   "(g4, g6) hold about 98.5% of the data; the four tail groups hold 1.5%.</p>"
                   "<p><strong>DRO step size matters enormously here.</strong> At the spec "
                   "default of 0.02 the group weights never move off their initial "
                   "proportions, so every DRO variant trains identically to ERM. These "
                   "results use 0.5 with uniform initialisation, matching what REMIND uses on "
                   "this same dataset.</p></div>")
        emb.append(md_to_html(open("runs/embed_xenia_production/REPORT.md").read()))
    if os.path.exists("documentation/EMBED_XENIA.md"):
        emb.append("<hr><details><summary style='cursor:pointer;color:var(--mut);padding:8px 0'>"
                   "Methodology, data construction, and the earlier 13% pilot</summary>"
                   "<div style='opacity:.8'>")
        emb.append(md_to_html(open("documentation/EMBED_XENIA.md").read()))
        emb.append("</div></details>")
    if emb:
        tabs.append(("EMBED", "sec-embed")); secs.append(("sec-embed", "".join(emb)))

    # run logs
    LOGS = [("Fed-Heart cross validation", "runs/fedheart_cv.log"),
            ("Method matrix, Fed-Heart", "runs/matrix_fedheart.log"),
            ("Method matrix, NHANES nested", "runs/matrix_nhanes_nested.log"),
            ("Method matrix, NHANES disjoint", "runs/matrix_nhanes_disjoint.log")]
    lh = ["<h2>Run logs</h2>",
          "<p>Tail of each experiment's console output, so you can see exactly what was run and "
          "what it printed. Full logs are in the repo under <code>runs/</code>.</p>"]
    any_log = False
    for title, path in LOGS:
        if not os.path.exists(path):
            continue
        any_log = True
        txt = open(path, errors="ignore").read().split("\n")
        keep = [l for l in txt if l.strip() and not l.lstrip().startswith(("Sep method", "warnings.warn"))]
        tail = "\n".join(keep[-45:])
        lh.append(f"<h3>{html.escape(title)}</h3>"
                  f"<p class='meta'><code>{html.escape(path)}</code></p>"
                  f"<pre><code>{html.escape(tail)}</code></pre>")
    if any_log:
        tabs.append(("Run Logs", "sec-logs")); secs.append(("sec-logs", "".join(lh)))

    # csvs
    ch = ["<h2>Raw metrics (metrics_long.csv)</h2>",
          "<p>Per-group rows in the Step-6 schema: accuracy, macro_f1, loss, R_star, excess_loss.</p>"]
    for k, (title, path) in enumerate(CSVS):
        ch.append(csv_section(title, path, f"csv{k}"))
    tabs.append(("Raw CSVs", "sec-csv")); secs.append(("sec-csv", "".join(ch)))

    nav = "".join(f"<a href='#' onclick=\"show('{sid}',this);return false\" "
                  f"class='{'on' if n == 0 else ''}'>{html.escape(t)}</a>"
                  for n, (t, sid) in enumerate(tabs))
    body = "".join(f"<section id='{sid}' class='{'on' if n == 0 else ''}'>{c}</section>"
                   for n, (sid, c) in enumerate(secs))
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GroupDRO with Heterogeneous Feature Spaces</title><style>{CSS}</style></head><body>
<header><h1>GroupDRO with Heterogeneous Feature Spaces</h1><nav>{nav}</nav></header>
<main>{body}</main><script>{JS}</script></body></html>"""
    open(f"{outdir}/index.html", "w").write(page)
    print(f"built {outdir}/index.html  ({len(figs)} figures, {len(tabs)} tabs)")
    return outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--build", action="store_true", help="build only, do not serve")
    args = ap.parse_args()
    d = build()
    if args.build:
        return
    import http.server, socketserver, functools
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=d)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"\n  ➜  http://localhost:{args.port}\n\n(ctrl-C to stop)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
