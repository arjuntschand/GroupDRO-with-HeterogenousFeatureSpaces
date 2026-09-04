# EMBED — REMIND-faithful experimental setup + thorough reporting (plan)

Goal: mimic REMIND's (arXiv 2603.00046) EMBED experimental protocol and produce a
paper-grade comparison + thorough per-group/per-epoch reporting, plus a novel
**Regret-DRO** variant Xenia asked for.

## 1. Faithful setup (from the paper)
- **Modalities:** M1=C-View CC, M2=C-View MLO, M3=FFDM CC, M4=FFDM MLO (== our 4 view-types).
- **Groups:** modality-combination groups; **Tail = <15% frequency**.
- **Metrics:** Accuracy + F1, per group + "Entire Dataset", mean±std over seeds.
- **Encoder (paper):** ViT-Base. **Ours:** resnet18 primary (fair, stable, fast) +
  resnet50 for the winning methods (scaling / absolute-gap check). ViT skipped (OOM risk).
- **Train:** Adam, lr 5e-5, wd 5e-5, batch 32, ~50 epochs, ≥3 seeds (we do more).
- **Missingness:** (a) NATURAL EMBED availability (headline); (b) SIMULATED random
  masking of full-modality exams (ablation, matches paper's MIMIC protocol).

## 2. Methods to run on OUR pipeline (same encoder/data/seeds = fair comparison)
Reported in one self-contained table. (REMIND's SoftMoE/FuseMoE/FlexMoE = their
architecture; we CITE their published numbers, cannot match their impl.)
- **ERM** (Trans/baseline) — plain CE
- **Reweighting** — inverse group-frequency loss weights
- **FairBatch** — group-balanced sampling (equal draw per group per batch)
- **FairMixup** — mixup across group pairs to smooth the group manifold
- **GroupDRO** — exp-reweight worst group (have)
- **⭐ Regret-DRO** (Xenia's ask) — see §3
- **Ours** — Anchors + GroupDRO (our contribution)

## 3. Regret-DRO (novel variant)
Standard GroupDRO upweights by raw group loss R_g = L_g. Problem: an intrinsically-hard
group keeps getting upweighted even after it hits its floor. **Regret-DRO** upweights by
*regret* instead:
- **Optimal loss L*_g** = best loss a group can achieve with its OWN dedicated model
  (train a per-group "personal" model on that group's data → its converged loss).
- **Regret R_g = max(0, L_g − L*_g)** — how far the shared model is from that group's
  achievable best.
- GroupDRO q-update uses R_g (regret) instead of L_g. Groups at their optimum get ~0
  regret → stop being upweighted; effort flows to genuinely under-optimized groups.
- Implementation: Phase A trains K per-group models (small, cheap) → {L*_g}; Phase B runs
  GroupDRO with regret. Also report a "regret table" per group.

## 4. Metrics & reporting (thorough)
Per EPOCH, logged to CSV + live terminal table + training-curve plots:
- Overall: loss, accuracy, macro-F1, worst-group acc, tail acc, avg (balanced) acc.
- Per group g: loss, accuracy, F1, count, and (for DRO methods) q_g weight + regret R_g.
- Group-DRO regret loss (sum_g q_g R_g) as a tracked quantity.
- Final deliverables: (1) per-group ACC/F1 table like REMIND Table 1 (methods × groups);
  (2) per-group loss/regret table; (3) training curves (overall + per-group loss/acc vs
  epoch); (4) mean±std over seeds.

## 5. Ablations (structural, not HP)
- Group definition: natural availability vs simulated random masking (both settings).
- Components: ERM / +anchors / +GroupDRO / +both (ours) / per-group encoder.
- Tail threshold sensitivity (10% / 15% / 20%) — how head/tail split changes results.
- (Encoder scaling: resnet18 vs resnet50 on winning methods.)

## 6. Files
- `dro_hetero_anchors/src/model/groupdro.py` — add regret (optimal_losses) support.
- `dro_hetero_anchors/src/train_embed.py` — method modes (reweight/fairbatch/fairmixup/
  regret_dro), per-group/per-epoch CSV + terminal metrics, F1.
- `dro_hetero_anchors/src/datasets_embed.py` — simulated-masking option.
- `run_embed_paper.py` — driver: per-group optimal-loss phase, all methods × seeds,
  writes per-group tables + curves.
- Analysis/plots → documentation/figures/embed_paper_*.png + CSVs.
