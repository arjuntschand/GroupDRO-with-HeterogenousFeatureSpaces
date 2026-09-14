# FLIC overlaps with our method more than we have been assuming

`documentation/2301.11447v1 (5).pdf` is Rakotomamonjy, Vono, Medina Ruiz and Ralaivola, 2023,
*Personalised Federated Learning On Heterogeneous Feature Spaces*. It has been sitting in our
documentation folder the whole time. Reading it properly changes what we can claim as new.

## What FLIC does

From their abstract and Section 1:

> "we propose a general framework coined FLIC that maps client's data onto a common feature
> space via **local embedding functions**. The common feature space is learnt in a federated
> manner using **Wasserstein barycenters** while the local embedding functions are trained on
> each client via **distribution alignment**."

And on the anchors:

> "data related to the same semantic information (e.g. label) have to be embedded in the same
> region of the latent space. To ensure this property, we align clients' embedded feature
> distributions via a **latent anchor distribution** that is shared across clients."

The word "anchor" appears 25 times in that paper and "Gaussian" 40 times.

## Component by component

| component | FLIC 2023 | ours |
|---|---|---|
| per-group encoders into a shared latent space | yes, "local embedding functions" | yes |
| a shared latent anchor distribution | yes | yes |
| anchors are Gaussian | yes | yes |
| anchors tied to the label | yes, "same semantic information (e.g. label)" | yes |
| alignment by Wasserstein distance | yes, barycenters | yes, W2 to the anchor |
| worst-group / DRO objective | **no** — "worst" appears 0 times, "robust" twice | **yes** |
| regret against a per-group floor R* | **no** | **yes** |
| setting | federated, privacy-constrained | centralised |

## What this means for the paper

The first five rows of that table are prior work. Per-group encoders mapping into a shared
latent space, aligned to a shared Gaussian anchor distribution keyed on the label, with
Wasserstein as the alignment metric, is FLIC. We should not present any of it as novel, and a
reviewer who knows this literature will recognise it immediately.

What is left as genuinely ours is the last two rows: putting a **worst-group objective** on top
of that architecture, and the **regret variant** that measures each group against its own
achievable floor rather than against raw loss.

That is a narrower contribution than "per-group encoders plus anchors plus DRO", but it is
defensible and it is the part no one else has done.

## It also reframes our awkward results

Two findings that looked like problems now read differently.

Our random-target control showed the anchors align groups but not class-by-class. FLIC's
framing is distribution alignment to a shared anchor, with the label mentioned as motivation
rather than as a hard per-class constraint. Our result is consistent with theirs: the mechanism
is distribution alignment, and the class-conditional part was our over-claim, not something the
prior work asserts either.

The anchors costing 1.2 points on Fed-Heart is a result about an existing technique, not a
failure of something we invented. "Alignment of this kind helps when group feature spaces are
complementary and hurts when they overlap" is a finding about FLIC-style alignment generally.

## What to do

1. Cite FLIC prominently and position against it, rather than risk a reviewer finding it.
2. State the contribution as the DRO layer on top of FLIC-style alignment, not the alignment.
3. Consider running FLIC as a baseline. It is the closest prior method and its absence from our
   comparison is more conspicuous than any of the three baselines we did add.
4. Ask Xenia whether she was already positioning against this. The paper is in the folder she
   shared, so she likely knows it; our write-up simply does not cite it.
