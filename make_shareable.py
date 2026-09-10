"""Turn the local site into one self-contained page that can be published and shared.

The local viewer writes site/index.html plus a site/figs/ directory. A shared page cannot
reach those files, and the publishing target also blocks any external request, so every
figure has to be embedded as a data URI and the doctype/html/head/body wrapper has to go
(the host supplies its own).

Usage: python make_shareable.py            -> writes site/shareable.html
"""
from __future__ import annotations
import base64, mimetypes, os, re, sys

SRC = "site/index.html"
FIGS = "site/figs"
OUT = "site/shareable.html"

# The local viewer is dark-first: the bare :root holds the dark palette and light lives only
# inside a prefers-color-scheme query. That is fine in a browser tab but breaks when the host
# stamps an explicit data-theme on the root element, because a stamped light theme on a dark
# OS (or the reverse) never matches the query and the page renders one theme's text on the
# other theme's ground. This re-states the same palette across all three states the viewer can
# be in: stamped dark, stamped light, and unstamped (system).
#
# Palette is unchanged from serve_results.py, with two fixes for the light ground: the red
# accent is darkened for contrast on white, and the link blue moved off #4a9eff, which is too
# pale to read on a white panel.
THEME_FIX = """
<style>
/* dark: the committed default, and what an explicit dark stamp gets */
:root, :root[data-theme="dark"]{
  --bg:#0f1115; --panel:#171a21; --fg:#e6e9ef; --mut:#9aa4b2;
  --acc:#e05252; --acc2:#4a9eff; --bd:#262b36; --good:#38b48b;
}
@media (prefers-color-scheme: light){
  :root:not([data-theme="dark"]){
    --bg:#f6f7f9; --panel:#ffffff; --fg:#1a1d23; --mut:#5b6472;
    --acc:#c23b3b; --acc2:#1c6fd4; --bd:#e2e6ec; --good:#1f8a68;
  }
}
:root[data-theme="light"]{
  --bg:#f6f7f9; --panel:#ffffff; --fg:#1a1d23; --mut:#5b6472;
  --acc:#c23b3b; --acc2:#1c6fd4; --bd:#e2e6ec; --good:#1f8a68;
}
/* the host paints its own ground behind the page, so this must be explicit */
body{background:var(--bg); color:var(--fg);}
/* figures are rendered on white by matplotlib, so keep a white plate under them in both themes */
.fig img{background:#fff;}
/* keyboard focus was invisible in the local build */
nav a:focus-visible, th:focus-visible, .filter:focus-visible{
  outline:2px solid var(--acc2); outline-offset:2px;
}
@media (prefers-reduced-motion: reduce){ *{animation:none !important; transition:none !important;} }
/* digits in these tables are meant to be compared down the column */
td, th{font-variant-numeric:tabular-nums;}
</style>
"""


def main():
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} not found; run serve_results.py --build first")
    doc = open(SRC, encoding="utf-8").read()

    # strip the page wrapper; keep <style> and <script>, which are inline already
    style = "".join(re.findall(r"<style>.*?</style>", doc, re.S))
    body = re.search(r"<body>(.*)</body>", doc, re.S)
    body = body.group(1) if body else doc

    # inline every figure as a data URI
    embedded, missing = 0, []
    def sub(m):
        nonlocal embedded
        fn = m.group(1)
        path = os.path.join(FIGS, os.path.basename(fn))
        if not os.path.exists(path):
            missing.append(fn); return m.group(0)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        embedded += 1
        return f'src="data:{mime};base64,{b64}"'
    body = re.sub(r'src=["\'](figs/[^"\']+)["\']', sub, body)

    # the publisher reads <title> out of the file to name the tab and gallery card
    title = "<title>GroupDRO with Heterogeneous Feature Spaces</title>"
    open(OUT, "w", encoding="utf-8").write(title + "\n" + style + THEME_FIX + "\n" + body)
    size = os.path.getsize(OUT)
    print(f"wrote {OUT}  ({size/1e6:.1f} MB, {embedded} figures embedded)")
    if missing:
        print("  missing figures:", ", ".join(missing))
    if size > 16e6:
        print("  WARNING: over the 16 MB publish limit")


if __name__ == "__main__":
    main()
