"""'V3 Updated Baselines' tab: protocol v4 results (documentation/PROTOCOL_V4_2026-09-22.md).

Reads runs/v4/summary.json and runs/v4/curves.json (written by report_v4.py) and returns the tab's
HTML. Every family is rendered once per selection view; a selector at the top of the tab shows one
view at a time. Tables, heat maps and curve plots are all computed from the same per-seed series.
"""
import html, json, os
import numpy as np

PRIMARY = "fixed10"
def _fx(t, e): return f"fixed budget, no early stopping (tabular: epoch {t} | EMBED: epoch {e})"
VIEWS = [("fixed10", _fx(10, 10)), ("fixed5", _fx(5, 5)), ("fixed15", _fx(15, 15)), ("fixed20", _fx(20, 20)), ("fixed25", _fx(25, 20)), ("fixed", _fx(30, 20)),
         ("excess", "early stopping on max excess"), ("worst", "early stopping on worst-group loss"),
         ("overall", "early stopping on overall loss"), ("groupavg", "early stopping on group-averaged loss")]
VIEWS_V3 = [("fixed10", _fx(10, 10)), ("fixed5", _fx(5, 5)), ("fixed15", _fx(10, 15)), ("fixed", _fx(10, 20)),
            ("excess", "early stopping on max excess"), ("worst", "early stopping on worst-group loss"),
            ("overall", "early stopping on overall loss"), ("groupavg", "early stopping on group-averaged loss")]
FAMS = [("nhanes", "NHANES", "v4-NHANES"), ("fedheart", "Fed-Heart", "v4-Fed-Heart"), ("embed", "EMBED", "v4-EMBED"),
        ("nhanes_nooverlap", "NHANES with no common information", "v4-nocommon"),
        ("fedheart_nooverlap", "Fed-Heart with no common information", "v4-nocommon-fh"),
        ("embed_disj", "EMBED with no common information", "v4-nocommon-em")]
ROWS_TAB = [("Ours_Regret", "Per-group + anchors + Regret-DRO", "full"), ("Ours_GDRO", "Per-group + anchors + GroupDRO", "abl"),
            ("RegretDRO", "Per-group encoders + Regret-DRO", "base"), ("GroupDRO", "Per-group encoders + GroupDRO", "base"),
            ("AnchorsOnly", "Per-group encoders + anchors + ERM", "base"), ("PerGroupOnly", "Per-group encoders + ERM", "base"),
            ("Independent", "Dedicated model per group", "base"),
            ("Shared_Anchors_GDRO", "Anchors + GroupDRO, common features", "base"), ("Shared_GDRO", "GroupDRO, common features", "base"),
            ("ERM", "ERM, common features", "base")]
ROWS_EM = [("ours", "Per-group + anchors + Regret-DRO", "full"), ("align_only", "Per-group + anchors + GroupDRO", "abl"),
           ("regret_only", "Per-group encoders + Regret-DRO", "base"), ("groupdro", "Per-group encoders + GroupDRO", "base"),
           ("anchors_only", "Per-group encoders + anchors + ERM", "base"), ("erm", "Per-group encoders + ERM", "base"),
           ("dedicated", "Dedicated model per group", "base")]
BASE = [("Reweigh", "Reweigh"), ("FlexMoE", "Flex-MoE"), ("REMIND", "REMIND (gamma chosen on validation)"), ("REMIND_pub", "REMIND (published gamma 0.02)")]
GROUP_LABELS = {"nhanes": ["G0 survey", "G1 + exam", "G2 + labs"], "nhanes_nooverlap": ["G0 survey", "G1 body + HbA1c/HDL", "G2 BP + lipids"],
                "fedheart": ["Cleveland", "Hungarian", "Switzerland", "VA"], "fedheart_nooverlap": ["Cleveland", "Hungarian", "Switzerland", "VA"]}
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def paired_p(a, b):
    from scipy import stats
    k = sorted(set(a) & set(b))
    if len(k) < 3:
        return None
    x = np.array([a[s] for s in k]); y = np.array([b[s] for s in k])
    if np.allclose(x, y):
        return None
    return float(stats.ttest_rel(x, y).pvalue)


def fmt(v, dp=1):
    return "—" if v is None or v != v else f"{v:.{dp}f}"


def table(tab, rows, full_key):
    ent = [(k, lab, kind) for k, lab, kind in rows if k in tab] + [(k, lab, "ext") for k, lab in BASE if k in tab]
    if not ent:
        return "<p class='na'>no runs</p>"
    cols = ["worst_acc", "overall_acc", "worst_loss", "worst_excess", "worst_auroc"]
    best = {}
    for c in cols:
        vals = [(tab[k][c], k) for k, _, _ in ent if tab[k].get(c) == tab[k].get(c)]
        if vals:
            best[c] = (max if c in ("worst_acc", "overall_acc", "worst_auroc") else min)(vals)[1]
    out = ["<table class='data'><thead><tr><th>method</th><th>worst-group acc</th><th>overall acc</th><th>worst-group loss</th>"
           "<th>worst-group excess</th><th>worst-group AUROC</th><th>step size</th><th>mean epoch</th><th>params</th><th>seeds</th></tr></thead><tbody>"]
    for k, lab, kind in ent:
        t = tab[k]
        tag = {"full": "<span class='tag ours'>ours</span>", "ext": "<span class='tag ctrl'>external</span>"}.get(kind, "")
        style = " style='background:rgba(31,159,110,.14)'" if kind == "full" else ""
        def cell(c, dp):
            v = t.get(c); s = fmt(v, dp)
            return f"<td>{'<b>' + s + '</b>' if best.get(c) == k else s}</td>"
        out.append(f"<tr{style}><td class='m'>{html.escape(lab)}{tag}</td>{cell('worst_acc', 1)}{cell('overall_acc', 1)}{cell('worst_loss', 3)}"
                   f"{cell('worst_excess', 3)}{cell('worst_auroc', 3)}<td class='dim'>{fmt(t.get('step'), 2) if t.get('step') else '—'}</td>"
                   f"<td class='dim'>{fmt(t.get('mean_epoch'), 1)}</td><td class='dim'>{int(t['n_params']):,}</td><td class='dim'>{t['n_seeds']}</td></tr>"
                   if t.get("n_params") == t.get("n_params") else
                   f"<tr{style}><td class='m'>{html.escape(lab)}{tag}</td>{cell('worst_acc', 1)}{cell('overall_acc', 1)}{cell('worst_loss', 3)}"
                   f"{cell('worst_excess', 3)}{cell('worst_auroc', 3)}<td class='dim'>{fmt(t.get('step'), 2) if t.get('step') else '—'}</td>"
                   f"<td class='dim'>{fmt(t.get('mean_epoch'), 1)}</td><td class='na'>—</td><td class='dim'>{t['n_seeds']}</td></tr>")
    out.append("</tbody></table><p class='legend'>Bold = best in the column. Mean over 10 seeds (Fed-Heart: seeds x 5 real folds, every patient tested once). "
               "Step size and mean selected epoch are the values behind the row; the step size is chosen on the same validation rule as the epoch.</p>")
    return "".join(out)


def heat(tab, full_key):
    if full_key not in tab:
        return ""
    f = tab[full_key]["per_seed"]
    out = ["<div class='card'><table class='data'><thead><tr><th class='it'>full method</th><th class='it'>metric</th>"]
    bl = [(k, lab) for k, lab in BASE if k in tab]
    out.append("".join(f"<th>vs {html.escape(lab)}</th>" for _, lab in bl) + "</tr></thead><tbody>")
    for mi, (name, key, higher, pct) in enumerate([("worst-group acc", "worst_acc", True, False), ("worst-group loss", "worst_loss", False, True),
                                                   ("worst-group excess", "worst_excess", False, True)]):
        cells = []
        for k, _ in bl:
            a, b = f[key], tab[k]["per_seed"][key]
            seeds = sorted(set(a) & set(b)); ma = np.mean([a[s] for s in seeds]); mb = np.mean([b[s] for s in seeds])
            p = paired_p(a, b); sig = p is not None and p < 0.05
            if pct:
                d = (mb - ma) / mb * 100; better = d > 0; txt = f"{abs(d):.1f}% {'lower' if better else 'higher'}"; sub = f"{ma:.3f} vs {mb:.3f}"
            else:
                d = (ma - mb) * 100; better = d > 0; txt = f"{d:+.1f} pts"; sub = f"{ma*100:.1f} vs {mb*100:.1f}"
            col = "#1a7f4b" if better else "#b03a3a"
            bg = ("rgba(31,159,110,.16)" if better else "rgba(176,58,58,.14)") if sig else ("rgba(31,159,110,.06)" if better else "rgba(176,58,58,.05)")
            ptxt = "n.s." if not sig else ("p&lt;0.001" if p < 0.001 else f"p={p:.3f}")
            cells.append(f"<td style='background:{bg};color:{col};font-weight:600'>{txt}<span class='hint' style='color:var(--dim);font-weight:400'>{sub} · {ptxt}</span></td>")
        out.append(f"<tr><td class='it'>{'full method' if mi == 0 else ''}</td><td class='it'>{name}</td>{''.join(cells)}</tr>")
    out.append("</tbody></table></div><p class='legend'>Paired over seeds. Strong shading = p &lt; 0.05, pale = not separable.</p>")
    return "".join(out)


def overview(summ, v):
    """One heat map for the whole tab: every family x every baseline x (acc, loss, excess), full method vs baseline."""
    fams = [(k, lab) for k, lab, _ in FAMS if k in summ and v in summ[k]]
    if not fams:
        return ""
    out = ["<div class='card'><table class='data'><thead><tr><th class='it'>full method vs</th>"]
    for _, lab in BASE[:3]:
        out.append(f"<th colspan='3' style='text-align:center'>{html.escape(lab)}</th>")
    out.append("</tr><tr><th></th>" + "".join("<th>worst acc</th><th>worst loss</th><th>worst excess</th>" for _ in BASE[:3]) + "</tr></thead><tbody>")
    for k, lab in fams:
        tab = summ[k][v]; full = "ours" if k.startswith("embed") else "Ours_Regret"
        if full not in tab:
            continue
        cells = []
        for bk, _ in BASE[:3]:
            for key, hi, pct in [("worst_acc", True, False), ("worst_loss", False, True), ("worst_excess", False, True)]:
                if bk not in tab:
                    cells.append("<td class='na'>n/a</td>"); continue
                a_, b_ = tab[full]["per_seed"][key], tab[bk]["per_seed"][key]
                seeds = sorted(set(a_) & set(b_)); ma = np.mean([a_[x] for x in seeds]); mb = np.mean([b_[x] for x in seeds])
                p = paired_p(a_, b_); sig = p is not None and p < 0.05
                if pct:
                    d = (mb - ma) / mb * 100; better = d > 0; txt = f"{abs(d):.0f}% {'lower' if better else 'higher'}"
                else:
                    d = (ma - mb) * 100; better = d > 0; txt = f"{d:+.1f}"
                col = "#1a7f4b" if better else "#b03a3a"
                bg = ("rgba(31,159,110,.16)" if better else "rgba(176,58,58,.14)") if sig else ("rgba(31,159,110,.06)" if better else "rgba(176,58,58,.05)")
                cells.append(f"<td style='background:{bg};color:{col};font-weight:600'>{txt}{'*' if sig else ''}</td>")
        out.append(f"<tr><td class='it'>{html.escape(lab.replace('with no common information', ': no common info'))}</td>{''.join(cells)}</tr>")
    out.append("</tbody></table></div><p class='legend'>Full method (per-group encoders + anchors + Regret-DRO) against each published baseline at the selected rule. "
               "Accuracy in percentage points, loss and excess as relative reduction (baseline minus ours, over baseline). Green = ours better, red = ours worse; "
               "strong shading and * = paired t-test p &lt; 0.05 over seeds. Detailed tables per dataset follow.</p>")
    return "".join(out)


def svg_curves(curves, fam, arms, glabels, metric, title, ymax=None):
    """Small-multiples SVG: one panel per arm, one line per group, epochs on x."""
    W, H, PW, PH = 900, 210, 200, 150
    arms = [a for a in arms if a in curves]
    if not arms:
        return ""
    W = 40 + len(arms) * (PW + 20)
    allv = [r[metric] for a in arms for r in curves[a] if r[metric] == r[metric] and r["epoch"] >= 1]
    if not allv:
        return ""
    lo, hi = (0.0, 1.0) if metric == "weight" else (min(allv) * 0.95, (ymax or max(allv)) * 1.05)
    ep_max = max(r["epoch"] for a in arms for r in curves[a])
    s = [f"<svg viewBox='0 0 {W} {H}' style='max-width:100%;height:auto;font-family:inherit'>",
         f"<text x='10' y='16' font-size='13' font-weight='600' fill='currentColor'>{html.escape(title)}</text>"]
    for i, a in enumerate(arms):
        x0, y0 = 40 + i * (PW + 20), 34
        s.append(f"<rect x='{x0}' y='{y0}' width='{PW}' height='{PH}' fill='none' stroke='#c9c8c3'/>")
        s.append(f"<text x='{x0 + PW/2}' y='{y0 - 6}' font-size='11' text-anchor='middle' fill='currentColor'>{html.escape(a.split('|')[0])}</text>")
        for gi, g in enumerate(sorted({r['group'] for r in curves[a]})):
            pts = [(r["epoch"], r[metric]) for r in curves[a] if r["group"] == g and r[metric] == r[metric] and r["epoch"] >= (0 if metric == "weight" else 1)]
            if not pts:
                continue
            path = " ".join(f"{x0 + (e / ep_max) * PW:.1f},{y0 + PH - (min(max(v, lo), hi) - lo) / (hi - lo) * PH:.1f}" for e, v in pts)
            s.append(f"<polyline points='{path}' fill='none' stroke='{COLS[gi % 6]}' stroke-width='1.6'/>")
        s.append(f"<text x='{x0}' y='{y0 + PH + 14}' font-size='10' fill='#6b6b6b'>0</text><text x='{x0 + PW}' y='{y0 + PH + 14}' font-size='10' text-anchor='end' fill='#6b6b6b'>epoch {ep_max}</text>")
        s.append(f"<text x='{x0 - 4}' y='{y0 + 10}' font-size='10' text-anchor='end' fill='#6b6b6b'>{hi:.2f}</text><text x='{x0 - 4}' y='{y0 + PH}' font-size='10' text-anchor='end' fill='#6b6b6b'>{lo:.2f}</text>")
    leg = "".join(f"<tspan fill='{COLS[i % 6]}'>&#9632;</tspan> {html.escape(l)}  " for i, l in enumerate(glabels))
    s.append(f"<text x='40' y='{H - 4}' font-size='11' fill='currentColor'>{leg}</text></svg>")
    return "".join(s)


def page(root="runs/v4", title="V3 Updated Baselines (protocol v4a: equal-group batches, constant lr, 10 epochs)", intro=None, toc="v4"):
    if not os.path.exists(f"{root}/summary.json"):
        return f"<h2>{html.escape(title)}</h2><p class='na'>{root}/summary.json not built yet (python report_v4.py {root})</p>"
    summ = json.load(open(f"{root}/summary.json")); curves = json.load(open(f"{root}/curves.json")) if os.path.exists(f"{root}/curves.json") else {}
    if root != "runs/v4" and os.path.exists("runs/v4/summary.json"):     # EMBED families live under runs/v4
        s0 = json.load(open("runs/v4/summary.json")); c0 = json.load(open("runs/v4/curves.json"))
        for k in ("embed", "embed_disj"):
            if k in s0:
                summ[k] = s0[k]; curves[k] = c0.get(k, {})
    P = toc
    side = [f"<aside class='toc' id='{P}-toc'><div class='toc-t'>On this page</div><a href='#{P}-what' data-t='{P}-what'>Protocol</a>"
            f"<a href='#{P}-overview' data-t='{P}-overview'>Heat-map overview</a>"]
    side += [f"<a href='#{aid.replace('v4-', P + '-')}' data-t='{aid.replace('v4-', P + '-')}'>{html.escape(lab.replace('with no common information', ': no common info'))}</a>" for k, lab, aid in FAMS if k in summ]
    side.append("</aside>")
    # V3 (runs/v4) trained 10 epochs on the tabular datasets, so only the views that exist there are offered
    views = VIEWS if root != "runs/v4" else VIEWS_V3
    primary = PRIMARY
    sel = (f"<div class='note' id='{P}-sel' style='position:sticky;top:0;z-index:5'><b>Selection rule for every table, heat map and number on this tab:</b> "
           f"<select id='{P}-view' onchange=\"document.querySelectorAll('.{P}v').forEach(e=>e.style.display=(e.dataset.v===this.value?'':'none'))\" style='font-size:14px;padding:4px 8px;margin-left:8px'>"
           + "".join(f"<option value='{v}'>{html.escape(l)}</option>" for v, l in views) + "</select>"
           "<span class='hint' style='margin-left:12px'>The reported epoch is chosen per run on the VALIDATION split by this rule; the test metrics of that epoch are reported. "
           "The step size of every DRO arm is chosen the same way. Same rule for every method.</span></div>")
    out = ["".join(side), f"<h2 id='{P}-what'>{html.escape(title)}</h2>", intro or "",
           "<p class='sub'>Every method, ours and the published baselines, trained under one protocol declared before any run "
           "(<code>documentation/PROTOCOL_V4_2026-09-22.md</code>): equal-group batches (the same number of samples from every group in every step), "
           "uniform initial group weights logged as epoch 0, a fixed budget of 10 epochs with no early stopping, and every epoch's validation and test metrics stored, "
           "so the five selection rules above are applied afterwards to the same runs. Our tabular arms follow Algorithm 1 of the draft literally "
           "(alignment inside the weighted objective, weights driven by a running training-batch average, signed excess, refresh every N steps).</p>",
           "<div class='note'><b>Read with the earlier tabs in mind.</b> Under equal-group batches 'ERM' is group-balanced ERM, and every group is sampled equally, "
           "so much of what group weighting did before is now done by the sampler for every method. The tabular budget is 10 epochs at a constant learning rate; "
           "the earlier tabs trained up to 100 epochs with early stopping. Numbers here are not comparable to those tabs, only to each other.</div>", sel]
    out.append(f"<h3 id='{P}-overview' style='font-size:20px;text-transform:none;letter-spacing:0;color:var(--ink);margin-top:32px'>Heat-map overview: full method against the published baselines, all datasets</h3>")
    for v, _ in views:
        out.append(f"<div class='{P}v' data-v='{v}'{'' if v == primary else ' style=display:none'}>{overview(summ, v)}</div>")
    for k, lab, aid in FAMS:
        if k not in summ:
            continue
        rows = ROWS_EM if k.startswith("embed") else ROWS_TAB
        full = "ours" if k.startswith("embed") else "Ours_Regret"
        out.append(f"<h3 id='{aid.replace('v4-', P + '-')}' style='font-size:20px;text-transform:none;letter-spacing:0;color:var(--ink);margin-top:44px'>{html.escape(lab)}</h3>")
        for v, _ in views:
            if v not in summ[k]:
                continue
            tab = summ[k][v]
            out.append(f"<div class='{P}v' data-v='{v}'{'' if v == primary else ' style=display:none'}>")
            out.append(f"<div class='card'>{table(tab, rows, full)}</div>")
            out.append(heat(tab, full))
            out.append("</div>")
        # curves are selection-independent
        cv = curves.get(k, {})
        if cv:
            gl = GROUP_LABELS.get(k, sorted({r['group'] for a in cv.values() for r in a}))
            def pick(names):
                got = []
                for n in names:
                    cands = [a for a in cv if a.split("|")[0] == n]
                    if cands:
                        # the step size the primary view chose, else the first
                        st = summ[k].get(PRIMARY, {}).get(n, {}).get("step")
                        best = next((a for a in cands if st is not None and abs(float(a.split("|")[1]) - st) < 1e-9), cands[0])
                        got.append(best)
                return got
            arms_w = pick([full, "Ours_GDRO" if not k.startswith("embed") else "align_only", "GroupDRO" if not k.startswith("embed") else "groupdro",
                           "RegretDRO" if not k.startswith("embed") else "regret_only", "REMIND"])
            arms_l = pick([full, "GroupDRO" if not k.startswith("embed") else "groupdro", "Independent" if not k.startswith("embed") else "dedicated", "REMIND", "Reweigh"])
            out.append("<div class='fig'>" + svg_curves(cv, k, arms_w, gl, "weight", "Group weight per epoch (mean over seeds; epoch 0 = initial, uniform)") + "</div>")
            out.append("<div class='fig'>" + svg_curves(cv, k, arms_l, gl, "val_loss", "Per-group validation loss per epoch (mean over seeds)") + "</div>")
            out.append("<div class='fig'>" + svg_curves(cv, k, arms_l, gl, "test_loss", "Per-group test loss per epoch (mean over seeds; shown for the dynamics only, never used to select)") + "</div>")
    out.append("<p class='legend'>Per-seed series behind every number: runs/v4/summary.json; per-epoch logs: runs/v4/&lt;family&gt;/epochs and the EMBED curve_*.json files.</p>")
    return "".join(out)
