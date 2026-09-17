# The method, and what the evidence supports

## The problem

A model normally assumes every training example has the same input columns.
Real clinical data does not work that way. Four hospitals each record a
different subset of the same workup. Some survey participants give blood and
others only answer the questionnaire. Some mammograms have four views and some
have one. The usual fix is to keep only the columns every group shares and
train one model on those. Averaging the loss then lets the large,
well-measured groups dominate, and the small, sparsely measured groups are
quietly ignored. We care about the group the model does worst on, because in
a clinical setting that group is real patients.

## Three components

Each can be switched on independently, which is what the ablation grid does.

**Per-group encoders.** Group g gets its own small network phi_g that maps
its own feature set into one shared latent space of dimension 64. A site with
8 tests and a site with 13 tests no longer have to pretend they share an
input layer. On tabular data the encoder is an MLP with layer norm; on EMBED
it is a per-group MLP over the views a breast actually has.

**Gaussian class anchors.** Per-group encoders create a new problem: nothing
forces them to agree on where things sit in the latent space, so one shared
classifier may not serve all of them. Each class c gets a learnable Gaussian
N(m_c, S_c), shared across groups, and two loss terms act on it:

```
L_total = L_task + lambda_fit * L_fit + lambda_sep * L_sep
```

`L_fit` is the squared 2-Wasserstein distance between the batch's class-c
moments and anchor c, computed per group in the GroupDRO setting (each
group's class-c cloud is pulled to anchor c separately). `L_sep` draws samples
from each anchor, passes them through the shared head, and requires them to
be classified correctly, which stops all anchors from collapsing onto one
point. "Anchors off" sets both weights to 0.001, which is numerically stable
and effectively off; exactly zero is unstable on NHANES.

**GroupDRO, and a regret variant.** Groups carry weights lambda_g on the
simplex. The weights move toward whichever group is doing worst, and the
objective is the weighted sum of group losses:

```
min over parameters, max over lambda:  sum_g lambda_g [ (L_g - R*_g) + lambda_fit * L_fit_g ] + lambda_sep * L_sep
```

Standard GroupDRO weights by raw loss L_g. The regret variant (Agarwal and
Zhang, 2022, "MERO") subtracts each group's reference loss R*_g, the best
loss that group can reach on its own, so groups already at their floor stop
being pushed. Setting R* to zero recovers plain GroupDRO. The weight update
is an exponential step on an exponential moving average of the excess loss,
renormalised each time.

**R*_g.** Estimated as an upper bound on group g's Bayes risk: a dedicated
model is trained on group g alone and its out-of-fold cross-entropy is
recorded, and the same is done for a model pooled over every patient with
group g's feature set. The smaller of the two is used. R* must not depend on
how much data we chose to train our own model on, which an earlier estimator
got wrong (see `docs/results.md`).

## The ablation grid

The three components are separable, so every tabular result is a point in a
full grid: encoder in {shared, per-group}, DRO in {off, GroupDRO, regret},
anchors in {off, on}.

| arm | encoder | DRO | anchors |
|---|---|---|---|
| ERM | shared, common features only | off | off |
| PerGroupOnly | per-group | off | off |
| Shared_GDRO | shared | GroupDRO | off |
| GroupDRO | per-group | GroupDRO | off |
| Shared_Anchors | shared | off | on |
| AnchorsOnly | per-group | off | on |
| Shared_Anchors_GDRO | shared | GroupDRO | on |
| Ours (Ours_GDRO) | per-group | GroupDRO | on |
| RegretDRO | per-group | regret | off |
| Ours_Regret | per-group | regret | on |

Reading down the per-group column gives an additive decomposition: ERM to
PerGroupOnly is the value of the architecture, PerGroupOnly to GroupDRO is
the value of reweighting, GroupDRO to Ours is the value of the anchors. On
EMBED no common-features baseline exists (the intersection of the six view
sets is empty), so its grid starts from per-group ERM. Arms are ranked
separately per encoder, and arms a paired t-test cannot separate from the
best are marked tied rather than ranked.

## Baselines

Three methods from the REMIND paper (arXiv 2603.00046), reimplemented in
`dro_hetero_anchors/src/model/baselines.py` and run on our data, seeds and
splits: Reweigh (fixed inverse-frequency group weights), Flex-MoE (shared
block projections with learnable stand-ins for absent blocks and Soft MoE
fusion), and REMIND itself at its published 128 experts. Each is run at its
released size and at a size matched to ours.

## What the evidence supports

This is the honest accounting, measured rather than estimated. The numbers
behind it are in `runs/FINAL_TABLES.txt` and on the results site.

Supported:

- Per-group encoders into a shared latent space give large worst-group gains
  when the groups' feature spaces genuinely differ. In a controlled synthetic
  sweep the gain rises from about zero at full feature overlap to about 24
  points at zero overlap. Fed-Heart, whose sites record different tests,
  supports the architecture; NHANES, whose feature sets nest, does not once
  the shared baseline is given the same features.
- The anchor loss measurably aligns the per-group latent distributions.
  Cross-group latent misalignment falls by a factor of 25 to 65 when it is
  on, read directly off the latent space.
- The gain is not explained by parameter count or by a generic latent
  regulariser; both controls pass.
- On worst-group loss and maximum excess loss, our full method is ahead of
  all three baselines on all three datasets, nine comparisons out of nine.

Not supported:

- That the alignment is specifically class-conditional. Replacing each
  sample's class anchor with a permuted one does not hurt, on three datasets,
  under both forms of the fit loss, at ten seeds. The anchors align groups;
  they do not do the class-by-class thing the original write-up claimed.
- That the anchors help everywhere. They give a significant worst-group
  accuracy gain on NHANES and cost accuracy on Fed-Heart and EMBED. The
  pattern is that they help where groups' representations are genuinely
  under-aligned and hurt where features already overlap heavily.
- That regret improves worst-group accuracy. It matches GroupDRO on accuracy
  and improves the excess-loss objective it actually optimises.
- That we beat the baselines on worst-group accuracy. We match them, with one
  significant loss (Fed-Heart against REMIND).

## Closest prior work

Rakotomamonjy et al., 2023, "Personalised Federated Learning on Heterogeneous
Feature Spaces" (FLIC), uses per-client encoders into a shared latent space
with Gaussian, label-keyed anchor distributions and Wasserstein alignment. The
representation-learning half of our method is the same family. What FLIC does
not have is any worst-group objective or a per-group reference loss; our
contribution is the robust objective on top of that alignment. Other
reference points are GroupDRO (Sagawa et al., 2020), MERO (Agarwal and Zhang,
2022), REMIND (arXiv 2603.00046) and Flex-MoE (NeurIPS 2024).
