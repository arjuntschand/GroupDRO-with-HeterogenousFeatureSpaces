"""Local results viewer: renders every results table and figure in one browsable page.

  python serve_results.py            # builds site + serves on http://localhost:8000
  python serve_results.py --build    # build only (site/ directory)
  python serve_results.py --port 8080

Pulls in the markdown result docs (tables rendered as real HTML tables, sortable) and every
figure in documentation/figures/, plus the raw metrics_long.csv files as searchable tables.
"""
from __future__ import annotations
import argparse, csv, html, io, os, re, shutil

DOCS = [
    ("ICML Results", "documentation/ICML_RESULTS.md"),
    ("Anchor Results", "documentation/ANCHOR_RESULTS.md"),
    ("Paper Tables", "documentation/PAPER_TABLES.md"),
    ("Paper Results", "documentation/PAPER_RESULTS.md"),
    ("NHANES Verified", "runs/NHANES_VERIFIED.md"),
    ("EMBED (Xenia spec)", "documentation/EMBED_XENIA.md"),
    ("EMBED Ready", "documentation/EMBED_READY.md"),
]
CSVS = [
    ("Fed-Heart metrics_long", "runs/matrix_fedheart/metrics_long.csv"),
    ("NHANES-disjoint metrics_long", "runs/matrix_nhanes_disjoint/metrics_long.csv"),
    ("NHANES-nested metrics_long", "runs/matrix_nhanes_nested/metrics_long.csv"),
    ("NHANES-expanded metrics_long", "runs/matrix_nhanes_expanded/metrics_long.csv"),
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


def md_inline(s):
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", s)
    return s


def md_to_html(md):
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
    "icml_fig1_methods": "Method comparison across all datasets (worst-group accuracy, 10 seeds).",
    "icml_fig2_pergroup": "Per-group accuracy: ERM vs the full method.",
    "icml_fig3_regret": "Achieved loss vs per-group reference loss R*_g. Points below the dashed line beat the group's own dedicated model (positive transfer).",
    "icml_fig4_2x2": "2x2 interaction: anchors x regret.",
    "icml_fig5_groupdef": "Parameter test: same method under different group definitions.",
    "icml_fig6_losses": "Per-group loss vs the achievable floor R*_g.",
    "fig1_headline": "Headline: full method vs naive baseline.",
    "fig2_ablation_grid": "Full 2x2x2 ablation (encoder x GroupDRO x anchors).",
    "fig3_pergroup": "Per-group accuracy, baseline vs ours.",
    "fig4_anchor_effect": "Anchor contribution with paired significance (** p<0.01, * p<0.05).",
    "fig5_anchor_mechanism": "Anchor mechanism: weight sweep and fit-vs-sep decomposition.",
    "anchor_specificity": "Anchor effect across datasets/feature modes.",
    "anchor_weight_sweep": "Worst-group accuracy vs anchor weight.",
    "anchor_fit_vs_sep": "Which anchor loss drives the gain.",
}


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

    # figures section
    fh = ["<h2>Figures</h2>", "<div class='grid'>"]
    for fn in figs:
        stem = fn[:-4]
        cap = FIG_CAPS.get(stem, stem.replace("_", " "))
        fh.append(f"<div class='fig'><img src='figs/{fn}' alt='{stem}'>"
                  f"<div class='cap'><strong>{html.escape(stem)}</strong> — {html.escape(cap)}</div></div>")
    fh.append("</div>")
    tabs.append(("Figures", "sec-figs"))
    secs.append(("sec-figs", "".join(fh)))

    # markdown docs
    for i, (title, path) in enumerate(DOCS):
        if not os.path.exists(path):
            continue
        sid = f"sec-doc{i}"
        body = f"<p class='meta'>source: <code>{html.escape(path)}</code></p>" + md_to_html(open(path).read())
        tabs.append((title, sid)); secs.append((sid, body))

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
<title>GroupDRO Heterogeneous Features — Results</title><style>{CSS}</style></head><body>
<header><h1>GroupDRO with Heterogeneous Feature Spaces — Results</h1><nav>{nav}</nav></header>
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
