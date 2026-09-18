# Final protocol and results (frozen 2026-09-18)

What was run for the numbers in `runs/FINAL_TABLES.txt`, the results site and the paper tables. Everything below comes from committed run families; nothing is typed in by hand.

## Method as run

Per-group encoders φ_g (one MLP per group over that group's own feature set) → shared latent space (dim 64 on all three datasets) → shared classifier head → per-class Gaussian anchors. Loss = CE + λ_fit · W₂-fit + λ_sep · L_sep, with group weights λ_g on the CE term updated by exponentiated gradient on a running per-group loss (GroupDRO) or on the excess over a per-group optimal-loss floor R̃_g (Regret-DRO).

| | Fed-Heart, NHANES | EMBED |
|---|---|---|
| classifier head | MLP, 32 hidden units | linear |
| anchor covariance | full (L_c L_cᵀ + εI) | diagonal |
| anchor moments | per class, pooled over groups | per (group, class) |
| anchor weight λ_fit = λ_sep | 0.1 (0.001 in arms without anchors) | 1.0 |
| λ initialisation | uniform, 1/G | uniform, 1/G |
| λ signal | held-out per-group CE (the validation split carved from train) | running training loss + λ_fit · alignment, EMA ρ = 0.9 |
| λ refresh | every step | every 50 steps (draft Algorithm 1) |
| λ step size γ | 0.1 | 2.0 |
| optimal-loss floors R̃_g | eq. 13: min{group fit, constant predictor} − bootstrap margin c_g (`runs/rstar_*_eq13.json`) | 5-fold out-of-fold min{group fit, joint fit} (`runs/final_embed/rstar.json`); eq. 13 floors tested and rejected, see below |
| epoch selection | validation worst-group accuracy | validation (worst-group loss for DRO arms, average loss otherwise) |
| seeds | 10 (42 1337 7 2024 31337 11 22 33 44 55) | 10 (0 1 42 1337 7 2024 31337 11 22 33) |
| splits | Fed-Heart 5-fold CV, every patient tested once, uncapped; NHANES fixed split (`data_split_seed 100`) | one patient-level split (`split_seed 0`) |
| configs | `experiments/fedheart_final.yaml`, `experiments/nhanes_final.yaml` | flags recorded in `runs/final_embed/PROTOCOL.md` |
| families | `runs/final_fedheart` (from `runs/gamma_sweep_fh10/g0.1_s1`), `runs/final_nhanes` (from `runs/gamma_sweep_nh_val10/g0.1_s1`) | `runs/final_embed` (from `runs/embed_final10`, jobs `base` and `g2.0`) |

The tabular architecture follows the November 2025 draft; the ICLR draft's design (linear head, diagonal anchors, per-(group, class) moments, alignment in the λ signal) was implemented and tested at 10 seeds and is worse for the full method on both tabular datasets (−2.75 on Fed-Heart, p = 0.011; anchors' accuracy edge on NHANES disappears). The paper should describe the tabular variant as built and put the architecture table in the appendix. EMBED already follows the ICLR draft's design.

## Where the protocol departs from the ICLR draft, and why

1. **λ signal on tabular is held-out loss, not training loss.** Training loss is memorised on the small groups (Switzerland's training CE goes to 0 while its held-out CE rises), so a training-loss signal hands the vulnerable group *less* weight. EMBED keeps the draft's training-loss schedule because the held-out signal fails there: λ piles onto the 40-row groups, whose validation loss then rises from epoch 0, and validation selection always returns the epoch-0 model.
2. **γ and refresh cadence.** At the draft's γ = 0.02 with one refresh per epoch, λ had moved 0.00–0.02 by the reported epoch on every dataset, which the draft's own Appendix E rule calls an invalid DRO run. γ = 0.1 refreshed every step engages λ (0.4–0.7 L1 by the reported epoch). On NHANES this is worth +2 worst-group accuracy for every DRO arm (p < 0.03) with AUROC, class-balanced accuracy and loss unchanged; on Fed-Heart nothing significant changes; γ = 2.0 raises NHANES accuracy further only through majority-class drift (class-balanced accuracy −11 for GroupDRO, p = 0.001) and is not used on tabular. On EMBED γ = 2.0 is needed because the excess signal is small.
3. **eq. 13 floors on EMBED.** g5 has 51 rows, so its bootstrap margin is 0.65 and its floor 0.24, which gives it a permanent excess; λ and the worst group move onto its 14 test exams and the regret arms lose 1–6 points. EMBED keeps the plain out-of-fold floors.
4. **Alignment loss in the λ signal** (draft Algorithm 1) was tested on both tabular datasets and is neutral (all changes within ±1, none significant). It is available as `dro_align_in_signal: true` and off in the frozen configs.

## Headline numbers (worst-group accuracy ± sd | worst-group loss)

| arm | Fed-Heart | NHANES | EMBED |
|---|---|---|---|
| ERM, common features | 68.28 ± 1.0 \| 0.638 | 69.28 ± 1.5 \| 0.496 | 62.16 ± 3.4 \| 1.350 |
| per-group + GroupDRO | 72.99 ± 1.2 \| 0.583 | 71.35 ± 1.9 \| 0.507 | 62.70 ± 2.9 \| 1.051 |
| per-group + Regret-DRO | 73.18 ± 1.3 \| 0.576 | 70.84 ± 2.0 \| 0.527 | 61.02 ± 3.6 \| 1.344 |
| per-group + anchors + GroupDRO | 71.15 ± 2.3 \| 0.591 | 74.30 ± 4.0 \| 0.524 | (rerun under frozen schedule, see FINAL_TABLES) |
| **full method** (per-group + anchors + Regret-DRO) | **72.25 ± 1.5 \| 0.570** | **74.67 ± 3.7 \| 0.522** | **62.74 ± 3.1 \| 1.066** |
| Reweigh / Flex-MoE / REMIND | 72.4 / 72.4 / 73.0 \| 0.64 / 0.79 / 0.66 | 72.8 / 72.7 / 72.3 \| 0.68 / 0.61 / 0.66 | 59.3 / 62.3 / 62.1 \| 2.12 / 1.33 / 1.18 |

Paired over seeds: full method vs common-features ERM, Fed-Heart +3.96 (p < 0.001), NHANES +5.39 (p = 0.001), EMBED +0.58 (p = 0.68) and loss −0.28 (p < 0.001). Against the three published baselines the full method wins on worst-group loss 9/9 and ties on accuracy 9/9. Regret-DRO vs GroupDRO is a tie on accuracy everywhere. Anchors: NHANES +3.83 over Regret-DRO (p = 0.004); Fed-Heart −0.94 (n.s.) and −1.84 vs GroupDRO (p = 0.03); EMBED +0.02 loss vs GroupDRO (n.s.).

## Mechanism evidence

- Latent alignment (scale-normalised W₂ between group centroids, no anchors → anchors): NHANES 2.57 → 0.15, EMBED 1.03 → 0.19 (random anchors 0.69), Fed-Heart 0.65 → 0.42. Figures `figs/paper/fig11_latent_scatter*`.
- No-common-information test (NHANES re-partitioned so groups share no columns, `runs/matrix_nhanes_partition`, `runs/baselines_nhanes_partition`): shared-encoder ERM and all three published baselines predict the majority class on the two smaller groups (AUROC 0.50); the per-group arms keep AUROC 0.55–0.71.
- Dynamics: `figs/paper/fig5_dynamics_*` (mean of 10 seeds on tabular, seed 0 on EMBED) show λ moving under the frozen protocol.

## Superseded families kept for the record

`runs/fedheart_cv` (capped scarcity study, old schedule), `runs/fedheart_uncapped`, `runs/matrix_nhanes_nested`, `runs/embed_fix_final` (old schedule, λ frozen); `runs/gamma_sweep_*` (the γ sweeps), `runs/draft_arch/*` (architecture check), `runs/align_signal/*`, `runs/embed_final10` (all EMBED jobs including eq. 13 floors and γ 0.5). The report page for Xenia's 2026-09-17 request list (`documentation/XENIA_CHECKS_2026-09-17.md`) has every table behind these choices.
