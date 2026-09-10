# What we claim, and what the evidence actually supports

This page tracks each claim in the write-up against the runs that test it. It exists because two
of the claims turned out to be stated more strongly than the data supports, and one of the
baselines was not what its label said. Everything here is measured, not estimated.

## The claim as written

From Section 2.2 of `ADDEDDRO_GroupDRO_with_HeterogeneousFeatureSpaces.pdf`, titled
"Class-Conditional Anchor Distributions":

> "To ensure that embeddings from different groups are geometrically consistent in the latent
> space, we associate each semantic class with a global latent anchor distribution."

Broken into links:

1. Per-group encoders phi_g handle heterogeneous input spaces
2. A shared class anchor mu_c pulls every group's class-c embeddings to the same region
3. Because of 2, groups become geometrically consistent, so one shared classifier works

GroupDRO comes in later, in Section 6, for worst-group robustness. It is layered on top and is
not part of the core alignment claim.

## Status of each link

| link | status | evidence |
|---|---|---|
| 1. per-group encoders help | supported | synthetic overlap sweep, +0.15 at full overlap rising to +23.94 at zero overlap |
| 2. the alignment is class-conditional | **not supported, settled** | random anchor targets match real ones on 3 datasets, both loss forms, 10 seeds each |
| 3. groups become geometrically consistent | **supported and now measured** | cross-group latent misalignment falls from 2.24 to 0.16 over 5 seeds |

## Link 3: the alignment is real and we can measure it

Rather than infer the mechanism from accuracy, we read the latent space directly.
NHANES-disjoint, averaged over 5 seeds:

| configuration | anchor_sep | class_sep | group_align (lower is better) | latent_scale | worst-group |
|---|---|---|---|---|---|
| no anchors | 4.447 | 0.007 | 2.239 | 15.821 | 73.28 |
| real anchors | 0.550 | 0.054 | **0.158** | 0.341 | 76.48 |
| random anchors | 0.386 | 0.053 | 0.120 | 0.258 | 77.31 |
| fit only (real) | 0.350 | 0.062 | 0.170 | 0.355 | 75.78 |
| sep only | 4.673 | 0.052 | 2.487 | 3.855 | 71.14 |

`group_align` is the mean distance between the per-group latent means. Low means the groups sit
on top of each other in the latent space, which is exactly the geometric consistency the paper
asks for. It drops by a factor of 14 when the anchor loss is on.

Two things worth noting. The `sep only` row keeps the anchors far apart (4.673) but never aligns
the groups (2.487) and has the worst accuracy of any arm, so separation alone is not the useful
part. And the anchor loss also shrinks `latent_scale` by a factor of 46, which is a confound we
have not fully closed: the earlier plain-L2 control may have been too weak to match that shrink,
so "it is mostly a scale constraint" is not yet ruled out.

## Link 2: the class-conditional part does not survive its control

The control swaps each sample's own class anchor for a randomly chosen one. Same loss, same
parameters, same magnitude, but the class structure is destroyed. If the class structure is what
matters, this should break the method.

| dataset | fit loss form | real anchors | random anchors | difference | p |
|---|---|---|---|---|---|
| NHANES-disjoint | pooled (eq. 13) | 75.05 | 75.20 | +0.15 | 0.81 |
| NHANES-nested | pooled (eq. 13) | 74.14 | 73.23 | +0.91 | 0.16 |
| EMBED worst-group | per-group (eq. 18) | 61.10 | 59.80 | +1.30 | 0.35 |
| EMBED tail-mean | per-group (eq. 18) | 67.00 | 67.50 | -0.50 | 0.21 |

Real anchors lead on three of the four rows but nothing reaches significance, and on EMBED tail
the random anchors are actually ahead. **We cannot claim class-conditional structure is the
operative mechanism.** The write-up should say the anchors align group latent distributions and
drop the "class-conditional" qualifier. The pending runs have now finished and did not change
the picture (see below).

## An implementation gap that partly explains this

The write-up specifies the anchor-fit loss twice, and the two forms are not equivalent.

**Eq. 11-13**, Section 3.2, the centralized version. Class moments are pooled over every group:
> "we estimate the empirical latent distribution for each class c by pooling all samples of that
> class from every group"

**Eq. 18**, Section 6.1, the GroupDRO version, which is the setting we actually run:

    L_fit^(g) = sum_c W2^2( N(m_hat_{g,c}, S_hat_{g,c}), N(m_c, S_c) )

computed per group, per class.

Only eq. 18 performs the operation Section 2.2 describes. Pooled moments pull the global class-c
cloud toward anchor c, so no individual group is ever constrained and cross-group alignment is
only a by-product of shrinking the space. Eq. 18 pulls each group's class-c cloud to anchor c
separately, which is what actually forces group 1's class c onto group 2's class c.

`train_embed_xenia.py` had eq. 18 from the start, because its fit loss is computed inside the
per-group loop. The three tabular trainers were all using the pooled form. This matters for the
control: under the pooled form, scrambling labels still leaves a loss that collapses the latent
space, so alignment survives and the control cannot tell the two hypotheses apart.

Eq. 18 is now available on the tabular trainers behind `per_group_fit` (default off, so every
earlier result still reproduces).

**The 10-seed answer: eq. 18 does not rescue the class-conditional claim.** A single-seed check
had suggested it might, showing real minus random flipping from -1.17 to +0.32. That was noise.
At 10 seeds on both NHANES modes:

| mode | fit form | real | random | real - random | p |
|---|---|---|---|---|---|
| disjoint | pooled | 75.79 | 76.54 | -0.75 | 0.16 |
| disjoint | **per-group (eq. 18)** | 76.54 | 76.73 | **-0.19** | 0.68 |
| nested | pooled | 74.51 | 73.45 | +1.06 | 0.11 |
| nested | **per-group (eq. 18)** | 74.33 | 73.66 | **+0.68** | 0.25 |

The class-conditional claim is now tested on three datasets, under both forms of the loss, at
10 seeds each, and it fails every time. We should treat this as settled and state it plainly in
the paper rather than leave it open.

Eq. 18 is still worth keeping for two reasons that have nothing to do with the control. It is
what the GroupDRO section of the write-up actually specifies, so the implementation should match
it. And it measurably improves both alignment and accuracy on disjoint: `group_align` 0.152 ->
0.121 and worst-group 75.79 -> 76.54.

The alignment effect, meanwhile, is enormous and holds at 10 seeds. With anchors off,
`group_align` is 3.030 on disjoint and 7.651 on nested. With them on it is about 0.12 in both.
That is a 25 to 65 fold improvement, and it is the part of the claim the evidence supports.

## A labelling problem in the baselines

"ERM" did not mean the same thing across datasets.

| dataset | what "ERM" was | per-group encoders? |
|---|---|---|
| NHANES, Fed-Heart | `common_encoder=True`, one shared encoder | no |
| EMBED | `XeniaEmbedModel`, always per-group MLP_g | yes |

So on the tabular datasets the ERM-to-Ours gap bundled three changes at once: adding per-group
encoders, adding anchors, and adding GroupDRO. Since the overlap sweep shows per-group encoding
alone is worth up to +23.94 at zero feature overlap, folding it into the baseline gap both
overstates the anchors and understates the architecture.

A `PerGroupOnly` arm (per-group encoders, shared head, no anchors, no DRO) has been added to the
tabular matrix so the decomposition is additive and readable:

    ERM -> PerGroupOnly       value of the per-group architecture
    PerGroupOnly -> GroupDRO  value of DRO reweighting
    GroupDRO -> Ours_GDRO     value of the anchors

It also makes the tabular baseline match EMBED's, so the two can finally be read side by side.
Those runs are in progress.

## What we can say today

Supported: per-group encoders into a shared latent space give large worst-group gains that grow
as feature spaces diverge; the anchor loss measurably aligns the per-group latent distributions;
the full method gives significant worst-group gains on both NHANES settings; the gain is not
explained by parameter count.

Not supported: that the alignment is specifically class-conditional; that regret improves
worst-group accuracy (it does improve excess loss); that the method beats plain GroupDRO on
Fed-Heart or EMBED.

Still open: whether the anchor loss is doing something beyond constraining latent scale, and
whether eq. 18 rescues the class-conditional claim on data with genuinely private features.

## The ablation grid

The method adds three separable things, so the ablation is a full 2x2x2 over them:

| # | encoder | GroupDRO | anchors | arm | what it isolates |
|---|---|---|---|---|---|
| 1 | shared | off | off | ERM | the naive baseline |
| 2 | per-group | off | off | PerGroupOnly | the per-group architecture alone |
| 3 | shared | on | off | Shared_GDRO | GroupDRO as published, on a shared model |
| 4 | per-group | on | off | GroupDRO | DRO on top of per-group encoders |
| 5 | shared | off | on | Shared_Anchors | anchors without per-group encoders |
| 6 | per-group | off | on | AnchorsOnly | anchors without DRO |
| 7 | shared | on | on | Shared_Anchors_GDRO | everything except per-group encoders |
| 8 | per-group | on | on | Ours_GDRO | the full method |

Regret is a fourth, separate axis. It does not replace GroupDRO, it changes what drives the
weight update: raw loss L_g (standard GroupDRO, equivalently R*=0) versus excess loss
L_g - R*_g. `RegretDRO` is cell 4 with regret, `Ours_Regret` is cell 8 with regret.

Only cells 1, 4, 6 and 8 existed before. Everything except ERM was per-group, so the encoder
axis was never varied independently of the other two, and three of the eight cells had no run
behind them.

Cell 3 is the important omission. Our "GroupDRO" baseline is cell 4, which already contains the
per-group architecture, and the architecture is one of our own contributions. GroupDRO as
published (Sagawa et al.) reweights a shared model, which is cell 3. Without it we never showed
the literature's actual baseline, and we could not separate "DRO helps" from "DRO helps once you
have per-group encoders". Cells 3, 5 and 7 are running now.
