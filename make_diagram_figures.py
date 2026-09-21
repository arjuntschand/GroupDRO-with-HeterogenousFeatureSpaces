"""Hand-laid-out SVG diagrams for the paper (no data involved).

  figs/paper/fig_intro_problem.svg        introduction: why no common model fits clinical data
  figs/paper/fig_setup_architecture.svg   setup: components of the method and how they connect

  python make_diagram_figures.py          writes both SVGs and, if rsvg-convert is installed, PNG + PDF

Every shape and label stays an editable object when the SVG is imported into Figma.
Style: flat pastel fills, thin dark outlines, short noun labels; sentences belong in the caption.
"""
import os, random, shutil, subprocess

OUT = "figs/paper"
INK, DIM, LINE, PANEL = "#141413", "#6b6b6b", "#c9c8c3", "#f7f6f2"
GROUPS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]           # same slots as the results figures
PASTEL = ["#d9e8fa", "#fbe1d5", "#d3f0e5", "#fbedc8"]
NEG, POS, RED = "#9a9a94", "#e87ba4", "#b03a3a"


class Svg:
    def __init__(self, w, h, label):
        self.w, self.h, self.b = w, h, []
        self.head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" xml:space="preserve" role="img" '
                     f'aria-label="{label}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="{INK}">'
                     '<defs>' + "".join(
                         f'<marker id="a{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                         f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for k, c in [("k", INK), ("g", DIM), ("r", RED)])
                     + f'<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                       f'<rect width="6" height="6" fill="#ffffff"/><line x1="0" y1="0" x2="0" y2="6" stroke="{RED}" stroke-width="1.6"/></pattern>'
                     + '</defs>' + f'<rect width="{w}" height="{h}" fill="#ffffff"/>')

    def add(self, s): self.b.append(s)

    def rect(self, x, y, w, h, fill="#ffffff", stroke=INK, rx=4, dash=None, sw=1.1):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=13, anchor="start", weight="400", fill=INK, italic=False):
        it = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{it}>{s}</text>')

    def line(self, x1, y1, x2, y2, col=INK, arrow=True, dash=None, sw=1.2):
        k = {INK: "k", DIM: "g", RED: "r"}.get(col, "k")
        m = f' marker-end="url(#a{k})"' if arrow else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{sw}"{m}{d}/>')

    def path(self, d, col=INK, arrow=True, dash=None, sw=1.2, fill="none"):
        k = {INK: "k", DIM: "g", RED: "r"}.get(col, "k")
        m = f' marker-end="url(#a{k})"' if arrow else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{col}" stroke-width="{sw}"{m}{ds}/>')

    def poly(self, pts, fill, stroke=INK, sw=1.1):
        self.add(f'<polygon points="{" ".join(f"{x},{y}" for x, y in pts)}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def circle(self, x, y, r, fill, stroke="none", sw=1, op=1.0):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" fill-opacity="{op}"/>')

    def ellipse(self, x, y, rx, ry, fill="none", stroke=INK, dash=None, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill}" fill-opacity="{op}" stroke="{stroke}" stroke-width="1.2"{d}/>')

    def star(self, x, y, r, fill):
        import math
        pts = [(x + (r if i % 2 == 0 else r * 0.42) * math.sin(i * math.pi / 5), y - (r if i % 2 == 0 else r * 0.42) * math.cos(i * math.pi / 5)) for i in range(10)]
        self.poly([(round(a, 1), round(b, 1)) for a, b in pts], fill, INK, 1.2)

    def cross(self, x, y, r=9):
        self.circle(x, y, r, "#ffffff", RED, 1.4)
        self.line(x - 4, y - 4, x + 4, y + 4, RED, False, sw=1.8); self.line(x - 4, y + 4, x + 4, y - 4, RED, False, sw=1.8)

    def tick(self, x, y, r=9):
        self.circle(x, y, r, "#ffffff", "#1a7f4b", 1.4)
        self.add(f'<path d="M{x-4},{y} L{x-1},{y+4} L{x+5},{y-4}" fill="none" stroke="#1a7f4b" stroke-width="1.9"/>')

    def save(self, stem):
        os.makedirs(OUT, exist_ok=True)
        p = f"{OUT}/{stem}.svg"
        open(p, "w").write(self.head + "".join(self.b) + "</svg>")
        if shutil.which("rsvg-convert"):
            subprocess.run(["rsvg-convert", "-w", str(self.w * 2), p, "-o", f"{OUT}/{stem}.png"], check=True)
            subprocess.run(["rsvg-convert", "-f", "pdf", p, "-o", f"{OUT}/{stem}.pdf"], check=True)
        print("wrote", p)


def sub(base, s, size=13, italic=True):
    """base with a real subscript (tspan dy), which survives rsvg, browsers and Figma import."""
    it = ' font-style="italic"' if italic else ""
    return f'<tspan{it}>{base}</tspan><tspan dy="{size*0.3:.1f}" font-size="{size*0.72:.1f}">{s}</tspan><tspan dy="-{size*0.3:.1f}">\u200b</tspan>'


# which measurement blocks each clinical group carries (columns: questionnaire, body, blood pressure, labs, imaging)
COLS = ["Questionnaire", "Body measures", "Blood pressure", "Lab panel", "Imaging"]
SITES = [("Community survey", [1, 0, 0, 0, 0], 9), ("Clinic visit", [1, 1, 0, 0, 0], 6),
         ("Hospital work-up", [1, 1, 1, 1, 0], 4), ("Imaging centre", [0, 0, 0, 0, 1], 2)]


def mini_table(S, x, y, cw, ch, mode):
    """Small copy of the group x measurement table. mode: 'common' | 'impute'."""
    for r, (_, have, _) in enumerate(SITES):
        for c in range(5):
            xx, yy = x + c * cw, y + r * ch
            if mode == "common":
                S.rect(xx, yy, cw - 3, ch - 3, "#ecebe6" if have[c] else "#ffffff", LINE, 2, None if have[c] else "3 2", 0.9)
            else:
                if have[c]:
                    S.rect(xx, yy, cw - 3, ch - 3, PASTEL[r], GROUPS[r], 2, None, 0.9)
                else:
                    S.rect(xx, yy, cw - 3, ch - 3, "url(#hatch)", RED, 2, None, 0.9)


# ---------------------------------------------------------------------------------------------
# Figure 1 (introduction), v2. Canvas is 792 px = 2x the ICLR text width (5.5 in = 396 pt), so a
# 14 px label prints at 7 pt. Nothing in the drawing is smaller than 13 px.
# ---------------------------------------------------------------------------------------------
TESTS = ["Survey", "Exam", "Labs", "Imaging"]
HAVE = [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]]      # nested groups 1-3, group 4 shares nothing
NPAT = [8, 5, 3, 2]
GREEN = "#1a7f4b"


def mark(S, x, y, ok):
    if ok:
        S.add(f'<path d="M{x-6},{y} L{x-2},{y+5} L{x+7},{y-6}" fill="none" stroke="{GREEN}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>')
    else:
        S.add(f'<path d="M{x-5},{y-5} L{x+5},{y+5} M{x-5},{y+5} L{x+5},{y-5}" fill="none" stroke="{RED}" stroke-width="2.4" stroke-linecap="round"/>')


def pict(S, x, y, kind):
    """4x4 thumbnail of what an approach does to the group x test table."""
    cw, ch = 11, 7
    for r in range(4):
        for c in range(4):
            xx, yy, h = x + c * cw, y + r * ch, HAVE[r][c]
            if kind == "drop":
                S.rect(xx, yy, cw - 2, ch - 2, "#dddbd4" if h else "#ffffff", LINE, 1, None, 0.7)
            elif kind == "impute":
                S.rect(xx, yy, cw - 2, ch - 2, PASTEL[r] if h else "#f0b9b9", GROUPS[r] if h else RED, 1, None, 0.7)
            elif kind == "fusion":
                S.rect(xx, yy, cw - 2, ch - 2, "#cfcdc6" if h else "#ffffff", LINE, 1, None, 0.7)
            else:
                if h:
                    S.rect(xx, yy, cw - 2, ch - 2, PASTEL[r], GROUPS[r], 1, None, 0.8)
    if kind == "drop":
        S.line(x - 2, y + 4 * ch - 1, x + 4 * cw, y - 1, RED, False, sw=2)
    if kind == "fusion":
        for c in range(4):
            S.rect(x + c * cw - 1, y - 1.5, cw, 4 * ch + 1, "none", INK, 1.5, None, 0.9)
    if kind == "separate":
        for r in range(1, 4):
            S.line(x - 2, y + r * ch - 1, x + 4 * cw, y + r * ch - 1, INK, False, sw=0.9)


def intro():
    S = Svg(792, 598, "Patients carry different sets of tests, so no single input space exists. Dropping to shared columns, imputing, one model "
                      "per group and modality fusion each give up a property; the proposed method keeps all measured data, imputes nothing and "
                      "couples groups even when they share no test, by giving each group its own encoder into one shared latent space with "
                      "class anchors. ERM and GroupDRO each miss a property of the objective; the proposed objective protects the group "
                      "furthest from its own achievable loss rather than the group with the highest loss.")
    T, B, SM = 16, 14, 13                                          # title / body / smallest label, in px (= 8 / 7 / 6.5 pt in print)
    # ---------------- (a) ----------------
    S.rect(6, 6, 296, 258, PANEL, LINE, 8)
    S.text(18, 30, "(a)  Patients carry different tests", T, weight="700")
    x0, y0, cw, ch = 96, 62, 50, 35
    for c, name in enumerate(TESTS):
        S.text(x0 + c * cw + (cw - 5) / 2, y0 - 9, name, SM, "middle", fill=DIM)
    for r in range(4):
        yy = y0 + r * ch
        S.circle(24, yy + 10, 5.5, GROUPS[r]); S.text(34, yy + 15, f"Group {r+1}", B, weight="600")
        for i in range(NPAT[r]):
            S.circle(21 + i * 8, yy + 25, 2.4, GROUPS[r], op=0.75)
        for c in range(4):
            if HAVE[r][c]:
                S.rect(x0 + c * cw, yy, cw - 5, ch - 6, PASTEL[r], GROUPS[r], 4)
            else:
                S.rect(x0 + c * cw, yy, cw - 5, ch - 6, "#ffffff", LINE, 4, "4 3")
    S.text(154, 220, "Group 4 shares no test with the others", SM, "middle", fill=RED, weight="600")
    ky = 238
    S.rect(18, ky, 18, 12, PASTEL[0], GROUPS[0], 3); S.text(41, ky + 11, "measured", SM, fill=DIM)
    S.rect(108, ky, 18, 12, "#ffffff", LINE, 3, "4 3"); S.text(131, ky + 11, "never ordered", SM, fill=DIM)
    S.circle(226, ky + 6, 2.4, GROUPS[0]); S.circle(234, ky + 6, 2.4, GROUPS[0]); S.text(242, ky + 11, "patients", SM, fill=DIM)

    # ---------------- (b) how to share ----------------
    S.rect(310, 6, 476, 258, PANEL, LINE, 8)
    S.text(322, 30, "(b)  How to share: one model for everyone", T, weight="700")
    cx0, cwid = 524, 64
    heads = [("keeps", "all data"), ("imputes", "nothing"), ("groups", "share"), ("no overlap", "needed")]
    for c, (h1, h2) in enumerate(heads):
        S.text(cx0 + c * cwid + cwid / 2, 54, h1, SM, "middle", fill=DIM); S.text(cx0 + c * cwid + cwid / 2, 69, h2, SM, "middle", fill=DIM)
    rows = [("Shared columns only", "drop", [0, 1, 1, 0]), ("Imputation", "impute", [1, 0, 1, 1]),
            ("One model per group", "separate", [1, 1, 0, 0]), ("Modality fusion (MoE)", "fusion", [1, 1, 1, 0]),
            ("Ours", "ours", [1, 1, 1, 1])]
    for r, (name, kind, props) in enumerate(rows):
        yy = 78 + r * 36
        ours = kind == "ours"
        S.rect(318, yy, 460, 32, "#e2f3ea" if ours else "#ffffff", GREEN if ours else LINE, 5, None, 1.4 if ours else 0.9)
        pict(S, 328, yy + 2.5, kind)
        S.text(382, yy + 21, name, B, weight="700" if ours else "400")
        for c, ok in enumerate(props):
            mark(S, cx0 + c * cwid + cwid / 2, yy + 16, ok)

    # ---------------- (c) ----------------
    S.rect(6, 272, 396, 320, PANEL, LINE, 8)
    S.text(18, 296, "(c)  Ours: share in a latent space", T, weight="700")
    ys = [334, 382, 430, 478]
    cyc = 414
    for r in range(4):
        y = ys[r]; n = sum(HAVE[r])
        for k in range(n):
            S.rect(18 + k * 19, y, 16, 16, PASTEL[r], GROUPS[r], 3)
        S.line(77, y + 8, 92, y + 8, DIM)
        hin = 8 + n * 8
        S.poly([(96, y + 8 - hin / 2), (150, y - 1), (150, y + 17), (96, y + 8 + hin / 2)], PASTEL[r], GROUPS[r])
        S.add(f'<text x="124" y="{y+13}" font-size="{B}" text-anchor="middle">{sub("φ", str(r+1), B)}</text>')
        S.path(f"M152,{y+8} C180,{y+8} 172,{cyc + (r-1.5)*14:.0f} 194,{cyc + (r-1.5)*14:.0f}", DIM)
    S.ellipse(262, cyc, 66, 70, "#ffffff", INK)
    rnd = random.Random(5)
    for ax, ay, col in [(240, cyc - 24, NEG), (286, cyc + 24, POS)]:
        S.ellipse(ax, ay, 32, 28, col, col, "5 3", 0.14)
        for g in range(4):
            for _ in range(5):
                S.circle(ax + rnd.gauss(0, 11), ay + rnd.gauss(0, 9), 2.8, GROUPS[g], op=0.9)
        S.star(ax, ay, 9, col)
    S.line(330, cyc, 344, cyc)
    S.rect(348, cyc - 20, 40, 40, "#ecebe6", INK, 5)
    S.add(f'<text x="368" y="{cyc+6}" font-size="17" text-anchor="middle" font-style="italic">ψ</text>')
    S.text(124, 524, "one encoder", SM, "middle", fill=DIM); S.text(124, 539, "per group", SM, "middle", fill=DIM)
    S.text(262, 524, "one shared", SM, "middle", fill=DIM); S.text(262, 539, "space", SM, "middle", fill=DIM)
    S.text(368, 524, "one head", SM, "middle", fill=DIM); S.text(368, 539, "for all", SM, "middle", fill=DIM)
    S.star(98, 569, 6.5, NEG); S.text(109, 574, "class anchor", SM, fill=DIM)
    for g in range(4):
        S.circle(222 + g * 7, 569, 2.8, GROUPS[g])
    S.text(252, 574, "patients, by group", SM, fill=DIM)

    # ---------------- (d) what to optimise ----------------
    S.rect(410, 272, 376, 320, PANEL, LINE, 8)
    S.text(422, 296, "(d)  What to optimise: the largest gap", T, weight="700")
    base, scale, ax = 436, 104, 462
    S.line(ax, base, ax, base - scale - 12, DIM, True); S.text(ax - 7, base - scale - 14, "loss", SM, "end", fill=DIM)
    for v in (0, 0.5, 1.0):
        S.line(ax - 4, base - v * scale, ax, base - v * scale, DIM, False); S.text(ax - 7, base - v * scale + 4.5, f"{v:g}", SM, "end", fill=DIM)
    S.line(ax, base, 776, base, DIM, False)
    for i, (lab, loss, floor, who, col, pas) in enumerate([("Group A", 0.90, 0.85, "GroupDRO targets", GROUPS[0], PASTEL[0]),
                                                           ("Group B", 0.60, 0.30, "we target", GROUPS[1], PASTEL[1])]):
        bx = 486 + i * 156
        S.rect(bx, base - loss * scale, 66, loss * scale, pas, col, 3)
        S.text(bx + 33, base - loss * scale - 6, f"loss {loss:.2f}", B, "middle", weight="600")
        S.line(bx - 8, base - floor * scale, bx + 74, base - floor * scale, INK, False, "5 3", 1.5)
        S.text(bx + 33, base - floor * scale + 15, f"best {floor:.2f}", SM, "middle", fill=DIM)
        S.path(f"M{bx+72},{base - loss*scale} v{(loss-floor)*scale}", RED, False, sw=3.2)
        gy = base - (loss + floor) / 2 * scale
        S.text(bx + 80, gy + 4.5, f"gap {loss-floor:.2f}", SM, fill=RED, weight="700")
        S.text(bx + 33, base + 16, lab, B, "middle", weight="600")
        S.text(bx + 33, base + 32, who, SM, "middle", fill=(RED if i else DIM), weight=("700" if i else "400"))
    # objective checklist, same visual language as (b)
    ox0, owid = 560, 74
    for c, (h1, h2) in enumerate([("robust to", "group mix"), ("protects", "worst group"), ("fair to", "hard groups")]):
        S.text(ox0 + c * owid + owid / 2, 490, h1, SM, "middle", fill=DIM); S.text(ox0 + c * owid + owid / 2, 505, h2, SM, "middle", fill=DIM)
    for r, (name, props) in enumerate([("ERM", [0, 0, 0]), ("GroupDRO", [1, 1, 0]), ("Ours: regret", [1, 1, 1])]):
        yy = 512 + r * 26
        ours = r == 2
        S.rect(418, yy, 360, 23, "#e2f3ea" if ours else "#ffffff", GREEN if ours else LINE, 5, None, 1.4 if ours else 0.9)
        S.text(430, yy + 16.5, name, B, weight="700" if ours else "400")
        for c, ok in enumerate(props):
            mark(S, ox0 + c * owid + owid / 2, yy + 11.5, ok)
    S.save("fig_intro_problem")


def architecture():
    S = Svg(1240, 640, "System view of the method: per-group encoders map group-specific inputs into a shared latent space with learnt class "
                       "anchors, a single head predicts for all groups, alignment and separation losses shape the latent space, and group "
                       "weights driven by each group's excess loss over an offline reference feed the weighted training objective.")
    # lanes
    S.rect(14, 14, 1212, 372, PANEL, LINE, 8); S.text(30, 38, "MODEL  (min player)", 11.5, weight="700", fill=DIM)
    S.rect(14, 398, 596, 228, PANEL, LINE, 8); S.text(30, 422, "LATENT-SPACE LOSSES", 11.5, weight="700", fill=DIM)
    S.rect(622, 398, 604, 228, PANEL, LINE, 8); S.text(638, 422, "GROUP WEIGHTS  (max player)", 11.5, weight="700", fill=DIM)

    # inputs
    S.text(30, 66, "Group inputs", 13.5, weight="700")
    ys, ncell, names = [96, 186, 296], [3, 5, 8], ["1", "2", "G"]
    for r in range(3):
        y = ys[r]
        S.add(f'<text x="30" y="{y-8}" font-size="12" fill="{DIM}">group {names[r]}:  <tspan font-style="italic">x</tspan> ∈ {sub("X", names[r], 12)}</text>')
        for k in range(8):
            if k < ncell[r]:
                S.rect(30 + k * 19, y, 16, 22, PASTEL[r], GROUPS[r], 2)
            else:
                S.rect(30 + k * 19, y, 16, 22, "#ffffff", LINE, 2, "3 2")
        S.add(f'<text x="30" y="{y+40}" font-size="11" fill="{DIM}">{sub("S", names[r], 11)}: {ncell[r] if r < 2 else "…"} measurements</text>')
    S.text(104, 262, "⋮", 16, "middle", fill=DIM)

    # encoders
    S.text(250, 66, "Per-group encoders", 13.5, weight="700")
    for r in range(3):
        y = ys[r] + 11
        hin = 14 + ncell[r] * 6
        S.line(186, y, 246, y, DIM)
        S.poly([(250, y - hin / 2), (340, y - 14), (340, y + 14), (250, y + hin / 2)], PASTEL[r], GROUPS[r])
        S.add(f'<text x="295" y="{y+5}" font-size="14" text-anchor="middle">{sub("φ", names[r], 14)}</text>')
        S.path(f"M342,{y} C392,{y} 392,{210 + (r-1)*34} 436,{210 + (r-1)*34}", DIM)
    S.text(295, 364, "no shared parameters", 11, "middle", fill=DIM)

    # latent space
    S.text(440, 66, "Shared latent space", 13.5, weight="700")
    S.add(f'<text x="700" y="66" font-size="12.5" text-anchor="end" fill="{DIM}"><tspan font-style="italic">Z</tspan> = ℝ<tspan dy="-5" font-size="9">d</tspan></text>')
    S.rect(440, 80, 262, 262, "#ffffff", INK, 6)
    rnd = random.Random(7)
    anchors = [(516, 160, NEG, "1"), (626, 262, POS, "2")]
    for ax, ay, col, c in anchors:
        S.ellipse(ax, ay, 58, 46, col, col, "5 3", 0.12)
        for g in range(3):
            for _ in range(9):
                S.circle(ax + rnd.gauss(0, 21), ay + rnd.gauss(0, 16), 3.1, GROUPS[g], op=0.85)
        S.star(ax, ay, 11, col)
    S.add(f'<text x="452" y="104" font-size="12">{sub("μ", "1", 12)} = N({sub("m", "1", 12)}, {sub("Σ", "1", 12)})</text>')
    S.add(f'<text x="690" y="330" font-size="12" text-anchor="end">{sub("μ", "2", 12)} = N({sub("m", "2", 12)}, {sub("Σ", "2", 12)})</text>')
    S.path("M558,196 L586,226", RED, True, sw=1.4); S.path("M586,226 L558,196", RED, True, sw=1.4)
    S.text(690, 104, "learnt class anchors", 10.5, "end", fill=DIM)

    # head, prediction, losses
    S.text(752, 66, "Shared head", 13.5, weight="700")
    S.line(704, 210, 750, 210)
    S.rect(754, 176, 84, 68, "#ecebe6", INK, 6)
    S.add(f'<text x="796" y="207" font-size="16" text-anchor="middle" font-style="italic">ψ</text>')
    S.text(796, 228, "one for all", 10.5, "middle", fill=DIM)
    S.add(f'<text x="796" y="268" font-size="11.5" text-anchor="middle" fill="{DIM}">{sub("f", "g", 11.5)} = ψ∘{sub("φ", "g", 11.5)}</text>')
    S.line(840, 210, 884, 210)
    S.add(f'<text x="902" y="215" font-size="15" text-anchor="middle" font-style="italic">ŷ</text>')
    S.text(960, 66, "Per-group losses", 13.5, weight="700")
    for r in range(3):
        y = [130, 196, 262][r]
        S.path(f"M916,210 C936,210 936,{y+16} 956,{y+16}", DIM)
        S.rect(960, y, 120, 32, PASTEL[r], GROUPS[r], 5)
        S.add(f'<text x="1020" y="{y+21}" font-size="13" text-anchor="middle">{sub("L", names[r], 13)}(θ)</text>')
    S.text(1020, 320, "cross-entropy, per group", 11, "middle", fill=DIM)

    # ---- latent-space losses lane ----
    S.rect(34, 440, 270, 86, "#ffffff", INK, 6)
    S.text(48, 462, "Alignment", 12.5, weight="700")
    S.add(f'<text x="48" y="488" font-size="13">{sub("L", "align", 13)} = Σ<tspan dy="4" font-size="9">g,c</tspan><tspan dy="-4"> </tspan>{sub("W", "2", 13)}<tspan dy="-5" font-size="9">2</tspan><tspan dy="5">(</tspan>{sub("ν", "g,c", 13)}, {sub("μ", "c", 13)})</text>')
    S.text(48, 512, "each group's class cloud → its anchor", 10.5, fill=DIM)
    S.rect(320, 440, 270, 86, "#ffffff", INK, 6)
    S.text(334, 462, "Separation", 12.5, weight="700")
    S.add(f'<text x="334" y="488" font-size="13">{sub("L", "sep", 13)}: keeps {sub("μ", "c", 13)} and {sub("μ", "c′", 13)} apart</text>')
    S.text(334, 512, "anchors of different classes", 10.5, fill=DIM)
    S.path("M500,342 C500,392 170,392 170,438", DIM, True, "4 3")
    S.path("M600,342 C600,400 455,400 455,438", DIM, True, "4 3")
    S.text(60, 560, "Closed form for diagonal Gaussians:", 10.5, fill=DIM)
    S.add(f'<text x="60" y="580" font-size="12">{sub("W", "2", 12)}<tspan dy="-5" font-size="8.5">2</tspan><tspan dy="5"> = ‖</tspan>{sub("m", "1", 12)} − {sub("m", "2", 12)}‖² + ‖{sub("σ", "1", 12)} − {sub("σ", "2", 12)}‖²</text>')

    # ---- group weights lane ----
    S.rect(642, 440, 176, 86, "#ffffff", INK, 6, "5 3")
    S.text(656, 462, "Reference (offline)", 12.5, weight="700")
    S.add(f'<text x="656" y="488" font-size="13">{sub("R̃", "g", 13)}: best loss group</text>')
    S.add(f'<text x="656" y="506" font-size="13"><tspan font-style="italic">g</tspan> can reach alone</text>')
    S.line(820, 483, 850, 483)
    S.rect(854, 440, 150, 86, "#ffffff", RED, 6)
    S.text(868, 462, "Excess", 12.5, weight="700", fill=RED)
    S.add(f'<text x="868" y="492" font-size="13.5">{sub("Δ", "g", 13.5)} = {sub("L", "g", 13.5)} − {sub("R̃", "g", 13.5)}</text>')
    S.text(868, 514, "signed, per group", 10.5, fill=DIM)
    S.line(1006, 483, 1036, 483, RED)
    S.rect(1040, 440, 170, 86, "#ffffff", INK, 6)
    S.text(1054, 462, "Weight update", 12.5, weight="700")
    S.add(f'<text x="1054" y="492" font-size="13.5">{sub("λ", "g", 13.5)} ∝ {sub("λ", "g", 13.5)} exp(γ {sub("Δ", "g", 13.5)})</text>')
    S.text(1054, 514, "on the simplex", 10.5, fill=DIM)
    S.path("M1086,130 h10 v164 h-10", DIM, False); S.path("M1096,212 H1140 V392 H929 V438", DIM, True, "4 3")
    # objective and feedback
    S.rect(642, 548, 568, 58, "#ecebe6", INK, 6)
    S.text(656, 570, "Training objective", 12.5, weight="700")
    S.add(f'<text x="656" y="594" font-size="13.5">min<tspan dy="4" font-size="9">θ</tspan><tspan dy="-4">  Σ</tspan><tspan dy="4" font-size="9">g</tspan><tspan dy="-4"> </tspan>{sub("λ", "g", 13.5)} {sub("L", "g", 13.5)}(θ)  +  {sub("λ", "fit", 13.5)} {sub("L", "align", 13.5)}  +  {sub("λ", "sep", 13.5)} {sub("L", "sep", 13.5)}</text>')
    S.line(1125, 528, 1125, 546)
    S.path("M170,526 V538 H604", DIM, False); S.path("M455,526 V538", DIM, False); S.path("M604,538 V577 H640", DIM, True)
    S.path("M1210,577 H1219 V388", INK, True, "6 3", 1.4)
    S.text(1198, 570, "gradients to encoders, head, anchors  →", 10.5, "end", fill=DIM)
    S.save("fig_setup_architecture")


if __name__ == "__main__":
    intro(); architecture()
