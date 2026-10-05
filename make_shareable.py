"""Turn the local site into one self-contained page that can be published and shared.

The local viewer writes site/index.html plus a site/figs/ directory. A shared page cannot
reach those files, and the publishing target also blocks any external request, so every
figure has to be embedded as a data URI and the doctype/html/head/body wrapper has to go
(the host supplies its own).

Usage: python make_shareable.py            -> writes site/shareable.html
"""
from __future__ import annotations
import base64, json, mimetypes, os, re, sys

SRC = "site/index.html"
FIGS = "site/figs"
OUT = "site/shareable.html"
FILES_OUT = "site/shareable_files.json"
EXTERNAL = "--external" in sys.argv     # figures as separate published files instead of data URIs



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
        path = os.path.join(os.path.dirname(SRC), fn)
        if not os.path.exists(path):
            missing.append(fn); return m.group(0)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        embedded += 1
        return f'src="data:{mime};base64,{b64}"'
    if EXTERNAL:
        # keep the relative figure paths and list them for the publisher's `files` map; the page
        # itself then stays well under the 16 MB limit (figures were 8 MB of data URIs)
        figs = sorted(set(re.findall(r'src=["\'](figs/[^"\']+)["\']', body)))
        files = {fn: os.path.join(os.path.dirname(SRC), fn) for fn in figs if os.path.exists(os.path.join(os.path.dirname(SRC), fn))}
        json.dump(files, open(FILES_OUT, "w"), indent=1)
        print(f"wrote {FILES_OUT}  ({len(files)} figures to publish as files)")
    else:
        body = re.sub(r'src=["\'](figs/[^"\']+)["\']', sub, body)

    # the publisher reads <title> out of the file to name the tab and gallery card
    title = "<title>GroupDRO with Heterogeneous Feature Spaces</title>"
    open(OUT, "w", encoding="utf-8").write(title + "\n" + style + "\n" + body)
    size = os.path.getsize(OUT)
    print(f"wrote {OUT}  ({size/1e6:.1f} MB, {embedded} figures embedded)")
    if missing:
        print("  missing figures:", ", ".join(missing))
    if embedded == 0:
        print("  (no figures in this build; tables are generated from CSV)")
    if size > 16e6:
        print("  WARNING: over the 16 MB publish limit")


if __name__ == "__main__":
    main()
