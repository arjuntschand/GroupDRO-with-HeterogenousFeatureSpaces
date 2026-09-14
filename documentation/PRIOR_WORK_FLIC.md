# FLIC: component-by-component comparison

Rakotomamonjy, Vono, Medina Ruiz and Ralaivola, 2023, *Personalised Federated Learning On
Heterogeneous Feature Spaces*, arXiv 2301.11447. PDF is in this folder.

**This is not a new discovery.** `documentation/archive/PROJECT_CONTEXT.md` section 9 already
lists it as "closest prior work; the method to distinguish ourselves from", and the PDF has been
in the repo since February. What follows is the component-level detail, which was not written
down anywhere, so that the positioning argument can be made precisely rather than in general
terms.

## What FLIC does, in their words

> "maps client's data onto a common feature space via local embedding functions. The common
> feature space is learnt in a federated manner using Wasserstein barycenters while the local
> embedding functions are trained on each client via distribution alignment."

> "data related to the same semantic information (e.g. label) have to be embedded in the same
> region of the latent space. To ensure this property, we align clients' embedded feature
> distributions via a latent anchor distribution that is shared across clients."

"anchor" appears 25 times in that paper, "Gaussian" 40 times, "worst" zero times and "robust"
twice.

## Component by component

| component | FLIC 2023 | ours |
|---|---|---|
| per-group encoders into a shared latent space | yes | yes |
| a shared latent anchor distribution | yes | yes |
| anchors Gaussian | yes | yes |
| anchors keyed on the label | yes | yes |
| Wasserstein alignment | barycenters | W2 to the anchor |
| **worst-group objective** | **no** | **yes** |
| **regret against a per-group floor R\*** | **no** | **yes** |
| setting | federated, privacy-constrained | centralised |

The overlap is in the representation-learning half. The differentiator is the robust objective,
which is exactly where PROJECT_CONTEXT said we would distinguish ourselves, and the comparison
above confirms that holds: FLIC has no worst-group component at all.

## Two things this is useful for

**Sharpening the positioning.** "We extend FLIC-style alignment with a worst-group objective" is
a more precise claim than "per-group encoders plus anchors plus DRO", and it survives a reviewer
who knows this literature.

**Reframing two awkward results.** Our random-target control found the alignment is not
class-conditional. FLIC describes its mechanism as distribution alignment and mentions the label
as motivation rather than as a hard per-class constraint, so our result sits consistently beside
theirs rather than contradicting the field. And the anchors costing about a point on Fed-Heart
reads as a finding about when alignment of this family helps, rather than as a failure of the
method.

## Open question for Xenia

FLIC is the closest prior method and we are not currently running it as a baseline, while we are
running three baselines from the REMIND paper. Whether that asymmetry matters is her call: FLIC
is federated and ours is centralised, so a direct comparison needs care about what is being held
constant. Worth raising rather than deciding unilaterally.
