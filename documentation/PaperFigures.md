# Paper figures: plan, style and status

Working document for the figures in the ICLR draft ("Learning across heterogeneous feature
spaces with group robustness"). The draft currently has two tables, one algorithm and **no
figures**. Corrections and decisions go at the bottom under "Change log".

## 1. Which figures the paper needs

| # | Where | Job | Status |
|---|---|---|---|
| 1 | Introduction (page 1-2) | Make the problem visible before any notation: clinical groups with different tests, why no common model fits (shared columns, imputation, one model per group), the idea in one strip, and loss vs regret | draft v1: `figs/paper/fig_intro_problem.{svg,pdf,png}` |
| 2 | Setup, next to eq. 5-6 and eq. 12 | Map the notation onto a picture: encoders, shared latent space with anchors, one head, the regret-driven weight update | draft v2 (system view, three lanes: model / latent-space losses / group weights): `figs/paper/fig_setup_architecture.{svg,pdf,png}`; v1 was `fig2_method` |
| 3 | Results: latent space | Evidence that the anchors align groups (and what the random-anchor control shows) | done: `figs/paper/fig11_latent_scatter_main` (NHANES, 2-column, main text); 3-column versions for NHANES, Fed-Heart, EMBED for the appendix |
| 4 | Results: training dynamics | Per-group loss (train/val/test) and group weight against epoch, GroupDRO vs Regret-DRO | done: `figs/paper/fig5_dynamics_{nhanes,fedheart,embed}_*` |
| 5 | Near the intro or in Section 4.2 (Xenia's suggestion) | Estimated optimal loss against group sample size: the upward bias that motivates eq. 13 | not started; needs no training (`estimate_rstar_v2.py --group-cap N0 N1 N2`) |
| 6 | Appendix | EMBED scatter with the anchor learning rate raised (anchors separate, numbers do not move) | done: `figs/paper/fig11_latent_scatter_embed_main_anchorlr` |

Figures 1 and 2 are separate on purpose. An intro figure that shows Wasserstein anchors and
lambda updates loses the reader; a setup figure cluttered with patient records loses the
notation. Build Figure 2 first (the method section cannot do without it), then Figure 1.

## 2. Figure 1 (introduction): content

Small, one row, no symbols.

- (a) A few patient records as rows of cells with different measured tests, so "no common
  feature space" is visible. Include one pair of groups that share nothing.
- (b) The two standard fixes failing: restricting to common columns (nothing left when groups
  share nothing) and imputing (filling in a test that was never ordered).
- (c) Loss vs regret: two groups, one at loss 0.9 with a floor of 0.85, one at loss 0.6 with a
  floor of 0.3. Raw loss picks the first; regret picks the second. These are the numbers
  Section 3.2 of the draft already uses.
- Optional (d): the R*-vs-sample-size curve (figure 5 above), or keep that in Section 4.2.

## 3. Figure 2 (setup): content

Every symbol in the figure must appear in the equations on the same page.

- Left: G groups drawn as rows of measurement cells (filled = measured, dashed = not measured),
  including a pair with no measurement in common. S_g ⊆ [V].
- Encoders φ_g : X_g → Z, one per group, drawn with different input widths, no shared
  parameters.
- Shared latent space Z: the class anchors μ_c = N(m_c, Σ_c) as a star (mean) plus a dashed
  ellipse (Gaussian); points from every group landing on the same anchors; arrows for
  L_align (W₂² between each group's class-c cloud ν_g^c and μ_c) and a double arrow for L_sep.
- One head ψ : Z → Y → ŷ, f_g = ψ ∘ φ_g.
- The max player: per-group loss L_g with the estimated floor R̃_g (eq. 13) as a dashed line,
  regret Δ_g = L_g − R̃_g as the accent-coloured gap, the update
  λ_g ← λ_g exp(γ[L̄_g − R̃_g]₊), and λ feeding back into the min player's objective (eq. 12).

Draft v1 has all of the above in one two-row figure: top row = setup, bottom row = regret vs
loss. It splits cleanly into Figure 2 (top) and the core of Figure 1 (bottom).

## 4. Style

Reference: the block diagrams in MLPerf Power (arXiv 2410.12032, Fig. 3). Borrow its visual
language, not its form.

Adopt:
- flat pastel fills, thin dark outlines, no gradients or shadows
- one encoding that carries meaning, explained in a small key box
- short noun labels only; no sentences inside the drawing (explanations go in the caption)
- even spacing on a grid

Differ where the content needs it: a systems diagram is boxes inside boxes; ours must also show
points from different groups landing on shared class Gaussians and a loss bar above its floor,
so the latent-space panel and the bar chart stay, drawn in the same flat style.

Colours (already used in every results figure, colour-blind-safe): groups `#2a78d6` blue,
`#eb6834` orange, `#1baf7a` green (then `#eda100`, `#e87ba4`, `#008300`); classes grey `#9a9a9a`
(negative) and pink `#e87ba4` (positive); one accent, red `#b03a3a`, reserved for regret.

Format: vector PDF in Overleaf; at most \textwidth (about 5.5 in) wide; text at least 8 pt at
print size; font to match the paper (ICLR is Times; the v1 SVG uses Helvetica).

## 5. Known weaknesses of draft v1 (to fix in v2)

1. Too many explanatory sentences inside the drawing; move them to the caption.
2. No key box: add one (filled cell = measured, dashed cell = not measured, star = anchor mean,
   dashed ellipse = anchor Gaussian, dashed line = estimated floor, red bracket = regret).
3. Box treatment is inconsistent: give encoders, head and the two "picks" boxes the same pastel
   fill and thin outline.
4. Subscripts are typed as `λ_g`, `R̃_g` (plain SVG text cannot do real subscripts); set them
   properly in Figma.
5. One figure, should be two.

## 6. Workflow

1. The SVG is written by hand in code (exact layout, colours, notation), rendered with
   `rsvg-convert` to PDF and PNG for review.
2. Import the SVG into Figma (File → Import, or drag onto the canvas); every shape and label
   stays editable. Polish spacing, fonts and subscripts there.
3. Export PDF from Figma into `figs/paper/` and into Overleaf.

Re-render after editing the SVG:

    rsvg-convert -w 2480 figs/paper/fig2_method.svg -o figs/paper/fig2_method.png
    rsvg-convert -f pdf  figs/paper/fig2_method.svg -o figs/paper/fig2_method.pdf

## 7. Draft text issues found while reading the paper (for Xenia)

- Section 6.2, "Anchors": "With group DRO, alignment lowers worst-group loss on all three
  datasets" is not what Table 2 shows. With per-group encoders and GroupDRO the anchors raise
  worst-group loss on all three (0.507→0.524, 0.583→0.591, 1.051→1.134). With **regret** they
  lower it on all three (0.527→0.522, 0.576→0.570, 1.344→1.066). One-word fix: regret.
- Not yet in the paper: the no-common-information experiment (the abstract's "groups that share
  no measurements" claim rests on it), the group-size sweep, the λ-schedule finding, the
  latent-space evidence and the random-anchor control's result, a conclusion / limitations
  section, and the appendices the text references.

## Change log

- 2026-09-20: document created; Figure 2 draft v1 committed.
- 2026-09-21: both figures redrawn as separate files, generated by `python make_diagram_figures.py`
  (layout lives in code; re-run after any edit, then import the SVG into Figma for the final pass).
  Figure 1 = four panels: (a) four clinical groups x five measurement blocks, one pair sharing
  nothing; (b) the three standard options failing; (c) the idea in one strip; (d) loss vs regret
  with the Section 3.2 numbers. Figure 2 = system architecture in three lanes. The weight update is
  drawn with the signed excess (no clamp), matching the corrected code; the draft's eq. still has
  the [.]+ clamp and needs the same change.
- 2026-09-21 (later): Figure 1 v2. Redrawn at print scale (canvas 792 px = 2x the 5.5 in text
  width; smallest label 13 px = 6.5 pt). Groups are measurement patterns within one institution,
  not sites, as in the intro. Panel (b) is now a properties checklist that includes modality
  fusion / MoE (the family Flex-MoE and REMIND belong to) and an "Ours" row. Panel (d) says who
  targets which group. To confirm with Xenia: the "invents nothing" tick for modality fusion
  (Flex-MoE uses a learnt missing-modality embedding, which is a placeholder, not an imputed value).

  Proposed caption: "Figure 1: (a) Patients receive different sets of tests, so groups of patients
  occupy different feature spaces, and some groups share no test with any other. (b) Existing ways
  of serving all of them with one model each give something up: restricting to shared columns
  discards data and fails without overlap; imputation invents measurements that were never ordered;
  separate models share nothing; modality fusion shares only through common modalities. (c) We give
  each group its own encoder into one latent space, where learnt class anchors align the groups
  under a single head. (d) Groups differ in the loss their measurements permit (dashed), so we
  train against the largest gap to that level rather than the largest loss."
- 2026-09-21 (v4 of Figure 1): panel (b) columns are now keeps all data / no imputation / one model /
  aligns groups. "No overlap needed" was dropped because our own no-overlap experiment shows
  REMIND-style fusion runs fine without overlap; what it lacks is an alignment mechanism. "One model"
  is the axis on which dedicated per-group models lose (the intro's maintenance argument). Panel (c)
  labels the two anchors by outcome and the legend says "one per outcome". Panel (d) uses badges
  ("GroupDRO picks A" / "Ours picks B") with the reason under each. Every tick is a design property,
  not a performance claim; the caption should stay that way (the no-overlap results do not show a
  performance win for latent sharing over dedicated models).
  Caption, updated sentence for (b): "...separate models share nothing and multiply what has to be
  deployed; modality-fusion mixtures of experts (e.g. Flex-MoE, REMIND) keep one model but have no
  mechanism that aligns groups."

