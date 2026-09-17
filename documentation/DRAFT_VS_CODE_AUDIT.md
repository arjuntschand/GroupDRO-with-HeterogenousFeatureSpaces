# Algorithm 1 of the ICLR draft vs. the code, line by line

Audit date 2026-09-17, against `ICLR2027 (1).pdf` (Sections 3.1–4.2, Appendices C, E, F, G).
"Tabular" is `train_fedheart.py` / `train_nhanes.py`; "EMBED" is `train_embed_xenia.py`.
Each row says what the draft specifies, what the code does, and whether they agree.

## Stage 1: reference values (Algorithm 1 lines 1–5, Section 4.2, eq. 13)

| Draft | Code | Status |
|---|---|---|
| R̂_g = out-of-fold loss of a model cross-fitted on group g's data alone (line 2) | `estimate_rstar_v2.py`: 5-fold OOF, min over {group-only, pooled} × {linear, MLP-32, MLP-64} × {weighted, unweighted} plus a joint fit | **Superset.** The draft fits group g alone; the code also allows pooled and joint fits. Every candidate is a valid upper bound, so taking the min is sound and tighter. |
| Certificates (i) base rate, (ii) ordering checked before training (line 3, Prop. 3) | Constant-predictor bound and fit-only estimate are now written to every `rstar_*.json`; `xenia_checks.py` prints Tables 1 and 2 of Appendix F | **Implemented 2026-09-17.** Previously not computed. Certificate (iii) (attainment) was already printed as "still above an achieved loss". |
| R̃_g = min{R̂_g, R^const_g} − c_g, c_g a bootstrap margin on pooled OOF losses, union-bounded over G (line 4, eq. 13, App. F) | Implemented 2026-09-17: constant candidate in the pool; c_g = half-width of the bootstrap percentile interval of the mean of the per-sample OOF losses at level 0.05/G | **Matches.** Note: bootstrapping fold means instead of per-sample losses gives c_g = 0 for a constant predictor; the per-sample form is the one that "grows as groups shrink". |
| Certificate outcome on our data | Fed-Heart: Switzerland fails (i) (fit 0.341 > const 0.254), VA fails (i) (0.598 > 0.591). EMBED fails (ii) on g1⊂g2 and g2⊂g5 (spread ≥ 0.195). NHANES passes all. | The draft's Section 4.2 prediction ("biased against precisely the groups the objective is meant to protect") is what the Fed-Heart result shows. |

## Stage 2: training (Algorithm 1 lines 6–19)

| Line | Draft | Tabular | EMBED | Status |
|---|---|---|---|---|
| 6 | λ_g ← 1/G, L̄_g ← R̃_g. "Uniform, not p; see Appendix E" | λ init = p_g (`groupdro_uniform_init` exists, default False). EMA init = R* for regret arms, first loss otherwise | λ init = p_g (`--uniform-lambda-init` exists, default off). EMA init = R* | **Deviation on λ init, all datasets.** App. E: undoing a p_g init needs ~γ⁻¹ log(N_max/N_min) refreshes; at EMBED's 1:1000 and γ = 0.02 that is ~345, more than the run. Overnight runs use 1/G. |
| 10 | L_g = mean loss over the group-g samples in the batch | `GroupDRO.forward`: per-group mean cross-entropy over the batch, with optional class weights | per-group CE on a group-stratified batch, weighted by λ_g | **Matches**, with two additions the draft does not have: NHANES uses inverse-frequency class weights (`class_weight: auto`); EMBED samples group-stratified batches (min(batch, n_g) per group). |
| 11 | L^align_g = eq. 6, skipping classes with < n_min samples in the batch | `per_class_batch_moments` skips classes with < 2 samples | same via anchor fit | **Matches.** |
| 12 | L̄_g ← ρ L̄_g + (1−ρ)(L_g + α_align L^align_g) | EMA over **CE only**; the alignment term is added to the training loss after `groupdro.forward` and never reaches the EMA | EMA over L_task + λ_fit · L_fit | **Deviation, tabular only.** The tabular max player does not see the alignment loss. |
| 14–15 | J = Σ λ_g[(L_g − R̃_g) + α_align L^align_g] + α_sep L_sep; gradient step in θ | weighted per-group CE (λ_g) + λ_fit · L_fit + λ_sep · L_sep. R̃ is a constant so it does not enter ∇θ, as the draft notes | task + λ_fit · L_fit inside the λ-weighted bracket, + λ_sep · L_sep outside | **Matches** in gradient; tabular puts L_fit outside the λ bracket (draft: inside). Same ∇θ up to the group weighting of the alignment term. |
| 16–17 | every N steps: λ_g ← λ_g exp(γ [L̄_g − R̃_g]₊), renormalise | `update_weights`, mode `exp`: EMA, clamp at 0 for regret arms, q ← q·exp(η·excess), clamp ≥1e-8, renormalise; `update_every` = N | same rule, N = 50 | **Matches.** Fed-Heart config sets N = 1 but drives the update from the held-out loss once per epoch (~8 steps), so the effective N ≈ steps/epoch. NHANES had no held-out path until 2026-09-17 and updated from the train-batch loss every step. |
| App. E | log the λ trajectory; a trace that never leaves its initialisation invalidates the run; sweep γ over three orders of magnitude | λ logged per epoch in `metrics.csv` (`groupdro_weights`). γ was never swept on the tabular datasets | γ = 0.02 only; λ logged per epoch | **Validity rule fails on Fed-Heart:** λ moves 0.005 (L1) by the reported epoch with p_g init, 0.05–0.08 with 1/G. Overnight: γ ∈ {0.02, 0.1, 0.5, 2.0} × {per-epoch, per-step} on both datasets. |
| App. E | clamp [·]₊ before the exponent | `torch.clamp(ema − R*, min=0)` | `torch.clamp(ema − rstar, min=0)` | **Matches.** |

## Model (Section 3.1, Appendix C, eq. 6)

| Draft | Tabular | EMBED | Status |
|---|---|---|---|
| Per-group encoders φ_g with no shared parameters | per-group MLPs | per-group MLPs over per-view projections | Matches. |
| Per-measurement projection p_v shared across groups (App. C) | none: each group's MLP reads its raw columns | `self.proj[v]` shared across groups | **Tabular lacks the shared per-measurement projection.** For tabular data a "measurement" is a column, so this is the difference between shared and per-group first layers. |
| Head ψ is a single linear map (App. C: a nonlinear head absorbs misalignment and makes the anchor ablation uninterpretable) | `head_hidden: 32` in both paper configs → `MLPHead` | `nn.Linear(latent_dim, C)` | **Deviation, tabular.** |
| Anchors μ_c = N(m_c, Σ_c) with Σ_c diagonal (eq. 6) | full Σ_c = L_c L_cᵀ + εI | diagonal via softplus(ρ) | **Deviation, tabular.** |
| L^align_g = (1/C) Σ_c W₂²(ν^c_g, μ_c), moments per (group, class) (App. G) | `per_group_fit` config flag: per-(group, class) moments when True, **pooled over all groups when False. Neither paper config sets it, so the paper runs used pooled moments.** | per-group W₂ to anchors | **Deviation, tabular.** With pooled moments only the global class-c cloud is pulled to anchor c; no group is constrained on its own, so cross-group alignment is a side effect (the code comment at `train_nhanes.py:664` says exactly this). Overnight runs add `per_group_fit: true`. |
| L_sep = (1/C) Σ_c E_{z∼μ_c}[L(ψ; z, c)], sampled with the reparameterisation trick | `anchor_sep_loss(sep_method="classifier")`: J reparameterised samples per anchor scored by the head | same | **Matches.** |
| Latent dimension k | 64 | 64 | Matches. |

## Summary of deviations, ordered by how much they matter

1. **λ initialised at p_g on every dataset** (draft: 1/G). This alone can disable the max player at realistic imbalance (App. E). Fixed by config; overnight runs use 1/G.
2. **NHANES updated λ from the train-batch loss** while Fed-Heart used the held-out loss; the draft is silent on which, but the train signal is the memorised one. `dro_signal: val` is now implemented for NHANES.
3. **Tabular EMA excludes the alignment loss** (line 12).
4. **Tabular head is an MLP**, not linear (App. C).
5. **Tabular anchors have full covariance**, not diagonal (eq. 6).
6. **eq. 13 was absent** (no constant candidate, no margin). Now implemented.
7. **Tabular alignment moments are pooled over groups** (`per_group_fit` unset), so the alignment term as run is not eq. 6's per-(group, class) W₂. Overnight runs add `per_group_fit: true` to measure the difference.
8. Tabular has no shared per-measurement projection (App. C); NHANES adds class weighting; EMBED adds group-stratified batching. These are additions rather than contradictions, but the paper describes one method.

What is right: the per-group loss, the EMA form, the clamp, the multiplicative refresh with renormalisation, the separation loss, the latent size, the n_min guard, and the EMBED head and anchors all match Algorithm 1. The weight update itself is implemented as written; what has been wrong is what it was fed (p_g init, a floor above the achievable loss, and on NHANES the train signal).
