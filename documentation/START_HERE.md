# Start here

## The problem in one paragraph

Normally a machine learning model assumes every training example has the same input columns.
Real data often does not work that way. Four hospitals each record a different subset of clinical
tests. Some survey participants got blood drawn and others only answered questions. Some
mammograms have four imaging views and some have one. Standard training handles this badly: it
optimises average accuracy, so the large well-measured groups dominate and the small
sparsely-measured ones get quietly ignored. We care about the group the model does **worst** on,
because in a clinical setting that group is real patients.

## What we built

Three pieces, which can be switched on independently:

1. **Per-group encoders.** Each group gets its own small network that maps its own feature set
   into one shared latent space. A hospital with 8 tests and a hospital with 13 tests no longer
   have to pretend they have the same inputs.
2. **Gaussian anchors.** A learned Gaussian per class, shared across all groups, that the
   embeddings are pulled toward. The intent is to stop each group's encoder from drifting into
   its own private corner of the latent space.
3. **GroupDRO, with an optional regret variant.** Instead of averaging loss over groups, weight
   the groups and push the weight toward whichever group is doing worst. The regret variant
   weights by how far a group is above its own best achievable loss, so groups that are already
   maxed out stop being pushed.

## What we found

**Per-group encoders are the big effect, and it scales with how different the groups are.**
In a controlled sweep where we dial feature overlap between groups from complete to none, the
benefit goes from +0.15 to +23.94 worst-group accuracy. This is the most robust result we have.

**The anchors measurably align the groups.** With the anchor loss off, the per-group latent
clouds sit far apart (a misalignment score of 3.03 on NHANES-disjoint, 7.65 on nested). With it
on, they collapse onto each other (about 0.12). That is a 25 to 65 fold improvement, and it is
measured directly from the latent space rather than inferred from accuracy.

**But the alignment is not class-conditional, and we can show that.** If we replace each sample's
correct class anchor with a randomly chosen one, performance does not drop. We tested this on
three datasets, under both versions of the loss, at 10 seeds each. It never fails. So the anchors
are doing something real, but not the specific class-by-class thing the write-up claims. We
report this rather than bury it.

**The method does not win everywhere, and the pattern is predictable.** It gives significant
worst-group gains on both NHANES settings. It loses to plain GroupDRO on Fed-Heart and ties on
EMBED. Both of those are datasets where the groups' features heavily overlap, which is exactly
where the overlap sweep says the architecture should not help much. The failures agree with the
theory.

## How to read the rest of this site

- **Results** has every dataset, with a short description of what its groups are before each
  table.
- **What holds up** is the honest accounting: which claims the runs support, which they do not,
  and what is still open.
- **Reference** explains the terms (worst-group accuracy, head and tail, R*, the ablation grid)
  and describes each dataset's group structure.
- **Paper draft** is the written-up results section.
- **Data** is the raw per-seed numbers if you want to check anything yourself.

Everything is averaged over 10 random seeds and shown as mean plus or minus standard deviation.
Numbers on this site come from committed runs only.
