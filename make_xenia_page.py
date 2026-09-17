"""Render documentation/XENIA_CHECKS_2026-09-17.md as one self-contained page for sharing.

Figures named in the text are embedded as data URIs after the paragraph that names them, tables
get a horizontal-scroll container, certificate verdicts become pass/fail pills, and the h2s form a
table of contents. Output: site/xenia_checks_2026-09-17.html (site/ is not tracked).

  python make_xenia_page.py
"""
import base64, html, re
import markdown

SRC = "documentation/XENIA_CHECKS_2026-09-17.md"
OUT = "site/xenia_checks_2026-09-17.html"
FIGS = {
    "fig11_latent_scatter.png": ("NHANES: every test patient in each model's latent space (first two principal components). Top: coloured by group, ringed markers are group centroids. Bottom: the same points coloured by outcome, stars are the learnt anchors. Axes are scaled per panel.", "figs/paper/fig11_latent_scatter.png"),
    "fig10_recommended_dynamics_ours_regret.png": ("Full method: held-out loss per group (dashed = that group's floor) and group weight against epoch, the paper's λ setting beside γ 0.5 per-step. Dotted line = the epoch that gets reported.", "figs/paper/fig10_recommended_dynamics_ours_regret.png"),
    "fig8_gamma_lambda.png": ("Fed-Heart: group-weight trajectories by step size γ, per-step refresh (mean of 3 seeds, fold 0). A flat line at 0.25 means the max player never engaged.", "figs/paper/fig8_gamma_lambda.png"),
    "fig9_gamma_lambda_nhanes.png": ("NHANES, train-batch signal every step: group-weight trajectories by step size γ (mean of 3 seeds). At γ ≥ 0.5 the raw-loss weights swing one-hot between groups from epoch to epoch.", "figs/paper/fig9_gamma_lambda_nhanes.png"),
    "fig7_latent_w2.png": ("NHANES: 2-Wasserstein geometry of the latent space, 10 seeds, normalised by latent scale.", "figs/paper/fig7_latent_w2.png"),
}


def data_uri(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


def main():
    md = open(SRC).read()
    for name, (cap, path) in FIGS.items():
        idx = md.find(name)
        if idx < 0:
            continue
        end = md.find("\n\n", idx); end = len(md) if end < 0 else end
        tag = f'\n\n<figure><img src="{data_uri(path)}" alt="{html.escape(cap)}"><figcaption>{html.escape(cap)}</figcaption></figure>\n\n'
        md = md[:end] + tag + md[end:]
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    body = re.sub(r"<td>(?:<strong>)?(fail[^<]*)(?:</strong>)?</td>", lambda m: f'<td><span class="pill fail">{m.group(1)}</span></td>', body, flags=re.I)
    body = body.replace("<td>pass</td>", '<td><span class="pill pass">pass</span></td>')
    body = body.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")
    heads = re.findall(r"<h2>(.*?)</h2>", body)
    toc = "".join(f'<li><a href="#s{i}">{h}</a></li>' for i, h in enumerate(heads))
    it = iter(range(len(heads)))
    body = re.sub(r"<h2>", lambda m: f'<h2 id="s{next(it)}">', body)
    css = open(__file__).read().split("CSS = '''")[1].split("'''")[0]
    page = (f"<title>Xenia Checks, 17 September</title>\n"
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">\n'
            f"<style>{css}</style>\n<main>\n"
            '<p class="lede">GroupDRO with heterogeneous feature spaces · overnight checks against the ICLR draft, 17 September 2026 · every number regenerates from <code>python xenia_checks.py</code></p>\n'
            f'<nav class="toc"><ol>{toc}</ol></nav>\n{body}\n</main>')
    open(OUT, "w").write(page)
    print(f"  wrote {OUT}  ({len(page)/1e6:.1f} MB, {len(heads)} sections, {page.count('<table>')} tables)")


CSS = '''
:root{--bg:#f5f6f8;--surface:#ffffff;--ink:#16181d;--ink2:#5b5f6a;--rule:#dcdfe6;--accent:#2a78d6;--pass:#0ca30c;--fail:#d03b3b;--code:#eef0f4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141518;--surface:#1c1e23;--ink:#ececec;--ink2:#b8bbc4;--rule:#33363d;--accent:#3987e5;--pass:#2fbf2f;--fail:#e66767;--code:#23262c}}
:root[data-theme="dark"]{--bg:#141518;--surface:#1c1e23;--ink:#ececec;--ink2:#b8bbc4;--rule:#33363d;--accent:#3987e5;--pass:#2fbf2f;--fail:#e66767;--code:#23262c}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Serif",Georgia,"Times New Roman",serif;font-size:16px;line-height:1.55;padding-inline:16px;padding-block:24px 64px}
main{max-width:76ch;margin:0 auto}
h1,h2,h3{font-family:"IBM Plex Sans Condensed","Arial Narrow",Arial,sans-serif;font-weight:600;line-height:1.15;text-wrap:balance;color:var(--ink)}
h1{font-size:2.1rem;margin:0 0 .35rem}
h2{font-size:1.45rem;margin:2.6rem 0 .8rem;padding-top:1.2rem;border-top:1px solid var(--rule)}
h3{font-size:1.1rem;margin:1.6rem 0 .5rem;color:var(--ink2)}
p{margin:0 0 1rem} li{margin:.25rem 0}
a{color:var(--accent);text-decoration:none} a:hover,a:focus-visible{text-decoration:underline;outline:none}
.lede{color:var(--ink2);font-style:italic;margin-bottom:1.4rem}
nav.toc{font-family:"IBM Plex Sans Condensed",Arial,sans-serif;font-size:.95rem;background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:.8rem 1.1rem;margin:1.2rem 0 2rem}
nav.toc ol{margin:0;padding-left:1.2rem;columns:2;column-gap:2rem} nav.toc li{margin:.15rem 0;break-inside:avoid}
.tbl{overflow-x:auto;margin:0 0 1.2rem;border:1px solid var(--rule);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-family:"IBM Plex Mono",Menlo,Consolas,monospace;font-size:.82rem;font-variant-numeric:tabular-nums}
th,td{padding:.42rem .65rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--rule);white-space:nowrap}
th{font-family:"IBM Plex Sans Condensed",Arial,sans-serif;font-size:.8rem;letter-spacing:.03em;text-transform:uppercase;color:var(--ink2);background:var(--code)}
tr:last-child td{border-bottom:0}
td:first-child{font-family:"IBM Plex Serif",Georgia,serif;white-space:normal;min-width:9ch}
code{font-family:"IBM Plex Mono",Menlo,monospace;font-size:.85em;background:var(--code);padding:.05em .35em;border-radius:3px}
pre{background:var(--code);padding:.8rem 1rem;overflow-x:auto;border-radius:6px;font-size:.82rem}
strong{font-weight:600}
.pill{display:inline-block;font-family:"IBM Plex Sans Condensed",Arial,sans-serif;font-size:.74rem;letter-spacing:.04em;text-transform:uppercase;padding:.1em .5em;border-radius:999px;color:#fff}
.pill.pass{background:var(--pass)} .pill.fail{background:var(--fail)}
figure{margin:1.2rem 0 1.6rem;background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:.6rem}
figure img{display:block;width:100%;height:auto}
figcaption{font-family:"IBM Plex Sans Condensed",Arial,sans-serif;font-size:.86rem;color:var(--ink2);margin-top:.5rem}
blockquote{border-left:3px solid var(--accent);margin:1rem 0;padding:.2rem 1rem;color:var(--ink2)}
@media (max-width:520px){nav.toc ol{columns:1} body{font-size:15px}}
@media (prefers-reduced-motion: reduce){*{scroll-behavior:auto}}
'''

if __name__ == "__main__":
    main()
