# Mechanism controls: what is actually causing the gain?

A skeptical reviewer has three cheap alternative explanations for why per-group encoders plus
anchors beat a shared ERM baseline. Each one gets a matched control arm. All numbers are
worst-group accuracy on NHANES-disjoint, 10 seeds, paired t-tests against the stated baseline.

## Results

| arm | worst-group | what it tests |
|---|---|---|
| shared ERM | 70.08 ± 0.99 | baseline |
| shared ERM, widened to matched capacity | 70.52 ± 1.15 | is it just more parameters? |
| per-group + GroupDRO | 73.22 ± 1.35 | |
| per-group + GroupDRO + anchors | 75.87 ± 1.70 | the method |
| per-group + GroupDRO + plain L2 on latent | 73.14 ± 1.34 | is it just a regularizer? |
| per-group + GroupDRO + **random** anchor targets | 76.02 ± 2.25 | is the class structure needed? |

Parameter counts: shared 11,746, per-group 30,946, shared-widened 30,854 (hidden=132). The
widened shared encoder is within 0.3% of the per-group budget.

| comparison | delta | p |
|---|---|---|
| widening shared encoder to matched capacity | +0.44 | 0.273 |
| adding real anchors to per-group GroupDRO | **+2.65** | **0.0022** |
| adding a plain L2 penalty instead | −0.08 | 0.889 |
| adding **random-target** anchors instead | **+2.80** | **0.0048** |
| real anchors vs random anchors | +0.15 | 0.814 |

## What passes

**It is not extra parameters.** Widening the shared encoder until it has the same parameter
count as the four per-group encoders buys only +0.44 (p=0.27). The architecture matters, not
the budget.

**It is not generic regularization.** Replacing the anchor loss with a plain L2 penalty on the
latent gives −0.08 (p=0.89), i.e. nothing. Any-old-penalty does not reproduce the effect.

## What fails, and we should say so

**The class-conditional structure is NOT what makes the anchors work.** Assigning each sample a
random anchor instead of its class anchor performs just as well (76.02 vs 75.87, difference
+0.15, p=0.81). If class-conditional alignment were the mechanism, destroying it should have
removed the gain. It did not.

We verified this is a real result and not a wiring bug: `random_anchor_targets` replaces `y`
only where the batch moments are computed, so the per-class moment sets genuinely become random
partitions of the batch.

The likely explanation is that with random labels every "class" moment converges to the same
global batch mean and covariance, so the anchor-fit loss degenerates into a **global latent
distribution-matching constraint** (pull the latent cloud toward a fixed Gaussian of learned
mean and scale). That constraint is apparently what helps, and it is a stronger and differently
shaped constraint than plain L2, which explains why L2 does nothing while this does.

## Honest implication for the paper

The defensible claim is:

> The anchor loss provides a significant worst-group gain (+2.65, p=0.002) that is not
> explained by parameter count or by generic weight regularization. Its benefit comes from
> constraining the geometry of the shared latent space, not specifically from class-conditional
> alignment: a random-target ablation reproduces the gain.

The claim we should **not** make is that class-conditional alignment is the operative mechanism.
On this dataset it is not. Either the anchors should be re-described as a latent-geometry
constraint, or we need a setting where the class-conditional version measurably beats the random
one. Running the same control on NHANES-nested (where the synergy effect is strongest) is the
obvious next check.

## Reproduce

```bash
python run_mechanism_controls.py --dataset nhanes \
    --base experiments/nhanes_disjoint_pergroup_gdro.yaml --tag nhanes_disjoint
```
