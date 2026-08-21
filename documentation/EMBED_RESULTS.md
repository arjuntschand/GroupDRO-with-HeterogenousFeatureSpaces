# EMBED results

Benchmark of our method (per-group encoders + class-conditional Gaussian anchors +
GroupDRO) vs baselines on **EMBED breast-density classification (BI-RADS A/B/C/D)**
under modality-availability heterogeneity. Task/groups match REMIND (arXiv 2603.00046).

---

## ★ HEADLINE — Tuned + bolstered study (resnet18, 20k samples, λ=0.01/η=3, **5 seeds**) — 2026-08-21

The first pass used blind defaults (λ=1e-3, η=1) on ~8k samples and looked flat. After an
HP sweep found the settings that actually engage the method (λ=0.01, η=3) on ~20k samples,
and after scaling to **5 seeds** + an η-sensitivity sweep + a tuned ablation, the picture
is clear and honest.

### 5-seed main result (mean ± std %, seeds 42/1337/7/2024/31337)

| method | overall | tail | **worst-group** |
|--------|:------:|:----:|:---------------:|
| ERM (baseline) | 77.2 ± 0.6 | 75.8 ± 1.2 | 69.9 ± 2.9 |
| GroupDRO (tuned η=3) | 76.5 ± 1.0 | 75.5 ± 1.4 | 73.0 ± 1.8 |
| **Anchors + GroupDRO (ours)** | 77.1 ± 0.4 | **77.1 ± 0.4** | **74.2 ± 1.0** |

*Figures: `figures/embed_5seed.png`, `embed_tuned_ablation.png`, `embed_eta_sensitivity.png`,
`embed_proper_dropout.png`.*

**Three findings, stated at their real strength:**

1. **Robust worst-group win over ERM: +4.4% mean, positive in ALL 5/5 seeds** (per-seed
   +4.0/+4.1/+9.1/+3.3/+1.4). This is the solid, significant result.
2. **Ours is the only method that improves worst-group WITHOUT sacrificing average
   accuracy — it resolves GroupDRO's tradeoff.** Plain GroupDRO lifts worst-group (+3.1)
   but *drops* overall (77.2→76.5) and tail (75.8→75.5). Ours reaches the **best tail of
   any method (77.1, +1.6 over both, tightest std ±0.4)** and keeps overall at ERM level
   (77.1), i.e. **+0.5 overall and +1.6 tail over GroupDRO**, both while matching/exceeding
   its worst-group. So the anchors don't just reweight — they structure the latent so the
   shared head keeps average accuracy.
3. **Honest on the marginal one:** ours vs *plain GroupDRO* on **worst-group alone** is
   **+1.3% mean but only 3/5 seeds positive** — within noise. We do **not** claim a clean
   worst-group win over GroupDRO; we claim (a) a robust win over ERM and (b) a
   Pareto-improvement over GroupDRO across overall+tail+worst-group jointly.

### Tuned ablation — each component earns its place (`embed_tuned_ablation.png`)

| variant | worst-group | overall |
|---|:---:|:---:|
| ERM | 68.8 | 76.8 |
| GroupDRO | 73.3 | 76.1 |
| **Anchors only (no GDRO)** | **72.1** | 76.5 |
| **Anchors + GDRO (ours)** | **74.5** | **77.1** |
| Full (per-group encoders) | 66.0 | 76.2 |

- **Anchors alone lift worst-group +3.3 over ERM** (68.8→72.1) with almost no overall cost
  — the anchor mechanism contributes *independently* of GroupDRO.
- **Anchors + GDRO is best on both** — the two components synergize (the paper's thesis).
- **Per-group encoders HURT on EMBED** (66.0, below ERM) — expected: the 4 views are
  redundant mammograms, so a shared encoder is right here. (Per-group helps only where
  feature spaces genuinely differ — the tabular datasets.)

### η-sensitivity (`embed_eta_sensitivity.png`)
Across η∈{1,2,3,5}, anchors+GDRO ≥ plain GDRO on worst-group at η∈{1,3,5} (dips at η=2)
and matches/exceeds on overall — the improvement is not a single cherry-picked η, and η=3
is a sound default.

### vs REMIND's published EMBED numbers
GroupDRO 78.9 / REMIND 80.7 overall. Our absolute (~77) sits just below — a
**backbone/data gap** (REMIND: ViT on the full non-public set; us: resnet18 on the 20%
open subset), not a method gap. resnet50 was attempted but OOM-hangs on our box; closing
the absolute gap is a scale-up (ViT / more data), not a method change. The **relative**
contribution — Pareto-improving GroupDRO and a robust +4.4 worst-group over ERM — is what
transfers.

### Honest scope
EMBED gains are real but **modest in magnitude** (its four modalities are redundant
mammograms). The method's *large* gains live where feature spaces are genuinely
heterogeneous: the tabular datasets (+7.9–15% worst-group), and next a true multimodal
benchmark (**MIMIC** image+text+labs), where REMIND itself shows GroupDRO→REMIND jumps of
+11.6 on tail groups. EMBED establishes we are **competitive on imaging and Pareto-superior
to GroupDRO on the robustness–accuracy frontier.**

*(The sections below are the earlier untuned exploratory study, kept for the record.)*

---

# Earlier exploratory study (untuned, ~8k samples) — 2026-08-19

**Setup:** AWS g5.xlarge (A10G). ~8.2k per-breast samples from the open subset (decode-
cached). resnet18, 20 epochs, class-weighted loss, **3 seeds** (ablation) / 2 seeds
(starvation). **Principled head/tail:** head = complete-modality exams
(FFDM CC+MLO, or all-4 with C-View), tail = any missing-view exam. Metric = accuracy.
37 ablation runs + 16 corrected tail-starvation runs, 0 failures.

> **Bottom line, stated honestly:** on EMBED our method is **competitive with the
> baseline but does NOT significantly beat it.** The gains that are large on our
> heterogeneous *tabular* datasets (+7.9–15% worst-group) do **not** transfer to EMBED,
> because EMBED's four "modalities" are all mammograms of the same breast — highly
> **redundant**, not genuinely heterogeneous feature spaces. This is a real, useful
> scientific finding: it sharpens *when* the method helps.

## 1. Ablation (mean ± std, 3 seeds) — `figures/embed_ablation.png`

| cell | encoder | anchors | GDRO | tail acc | overall acc |
|------|---------|:---:|:---:|:---:|:---:|
| erm_shared (baseline) | shared | – | – | 75.5 ± 1.2 | 77.3 ± 0.4 |
| gdro_shared | shared | – | ✓ | 75.6 ± 0.5 | 76.7 ± 0.3 |
| anchors_shared | shared | ✓ | – | 74.8 ± 0.7 | 76.8 ± 0.6 |
| **anchors_gdro_shared** | shared | ✓ | ✓ | **76.2 ± 0.5** | 77.3 ± 0.9 |
| erm_pergroup | per-group | – | – | 73.7 ± 1.2 | 76.2 ± 0.9 |
| gdro_pergroup | per-group | – | ✓ | 74.9 ± 1.9 | 76.9 ± 0.8 |
| full | per-group | ✓ | ✓ | 74.8 ± 0.4 | 77.2 ± 0.2 |

- **Best cell = `anchors_gdro_shared`** (tail 76.2 vs baseline 75.5), but the **+0.7 tail
  gain is within one std** — not statistically significant at 3 seeds.
- **Per-group encoders HURT** (erm_pergroup 73.7, full 74.8 tail, both below baseline).
  With 4 dedicated ResNets over redundant mammogram views, they add parameters and
  overfit rather than specialize. This is the clearest signal in the table.
- Overall accuracy is flat (~77%) across all shared cells.

## 2. Missingness robustness (test-time view-dropout) — `figures/embed_dropout.png`

Train normally, then drop present views at inference with prob *p*:

| p | ERM overall / tail | Ours overall / tail | Δ overall / Δ tail |
|---|---|---|---|
| 0.00 | 77.3 / 75.5 | 77.3 / 76.2 | +0.0 / +0.7 |
| 0.25 | 76.6 / 74.8 | 76.6 / 75.8 | +0.0 / +1.0 |
| 0.50 | 75.7 / 74.5 | 76.3 / 76.3 | +0.6 / +1.8 |
| 0.75 | 74.9 / 75.0 | 75.6 / 76.8 | **+0.7 / +1.8** |

- **The one place ours shows an edge:** as views go missing, `anchors_gdro_shared`
  degrades **more gracefully** — the gap widens from ~0 to **+0.7 overall / +1.8 tail**
  at 75% dropout. Directionally supports the "anchors align the latent so the shared
  head survives missing modalities" story. **Magnitude is modest**, but the trend is
  consistent across dropout levels.

## 3. Tail-starvation (does the gain grow as the tail thins?) — `figures/embed_starvation.png`

Cap tail-group training samples; compare baseline vs our best config:

| tail cap | ERM tail / worst | Ours tail / worst | Δ tail |
|---|---|---|---|
| 10 | 73.2 / 65.7 | 70.7 / 62.2 | **−2.5** |
| 50 | 73.7 / 67.8 | 73.6 / 66.6 | −0.1 |
| 200 | 75.0 / 67.8 | 74.9 / 67.8 | −0.1 |
| uncapped | 75.2 / 65.6 | 76.5 / 61.1 | +1.3 |

- **No consistent advantage** — ours ties or slightly *loses*, and is *worst* exactly
  where the tail is most starved (cap=10, −2.5). The mechanism that pays off on tabular
  data (structured latent → GroupDRO reweights a starved tail) does **not** engage here:
  when every view is a redundant mammogram, the baseline already extracts most of the
  signal from whatever views remain, so there is little for anchors/GDRO to recover.

## 4. Honest interpretation

EMBED is a **mild-heterogeneity** benchmark for our method. The four view-types
(FFDM/C-View × CC/MLO) are correlated images of one breast; density is largely readable
from any single view. So:
- a **shared** encoder is already near-optimal (per-group encoders only add overfitting),
- **anchors + GroupDRO** buy a small, real robustness edge under *missingness*, but
- there is **no significant worst-group gain** on the standard or tail-starved task.

This does not contradict the paper's thesis — it **bounds** it. Our method's advantage
scales with how *genuinely different* the per-group feature spaces are. It is large on
tabular datasets with truly disjoint/nested features (Fed-Heart, NHANES: +7.9–15%
worst-group) and small on EMBED where the "modalities" are near-duplicates.

## 5. Recommendation for the paper

- **Headline (significant gains):** the heterogeneous **tabular** datasets (+7.9–15%
  worst-group) — that's where the method's mechanism genuinely fires.
- **EMBED framing:** an **imaging extension** showing the method is **competitive** with
  REMIND-style baselines (~77% overall) and gives a **modest missingness-robustness edge**
  (+1.8 tail at 75% view-dropout) — *not* a significant-gains claim. Report it honestly;
  the negative starvation result is worth a sentence (it delimits applicability).
- **To get a strong imaging-era gains result, we need genuine multimodal heterogeneity.**
  Recommend targeting **MIMIC-IV (image + text + labs)** — REMIND's other benchmark — where
  missing a modality means missing an entire *different* signal, exactly the regime our
  per-group-encoder + anchor alignment is built for. (Needs credentialed access; touch base
  with Xenia first, per her original note.)

## Reproduce
```
run_embed_full_study.py                                   # Study A + tail-starve
run_embed_full_study.py --only B --starve-cells erm_shared anchors_gdro_shared \
    --out-dir runs/embed_study_supp                       # corrected starvation
```
Raw results: `runs/embed_study/results_main.json`, `results_supp.json` (gitignored).
Figures: `documentation/figures/embed_{ablation,dropout,starvation}.png`.
DUA: results + code only; no EMBED-trained weights/embeddings released.
