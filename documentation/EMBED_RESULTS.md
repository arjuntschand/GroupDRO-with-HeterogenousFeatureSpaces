# EMBED results — full overnight study (2026-08-19)

Benchmark of our method (per-group encoders + class-conditional Gaussian anchors +
GroupDRO) vs baselines on **EMBED breast-density classification (BI-RADS A/B/C/D)**
under modality-availability heterogeneity. Task/groups match REMIND (arXiv 2603.00046).

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
