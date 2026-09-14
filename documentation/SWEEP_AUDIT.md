# What we swept, against what the comparison papers swept

Honest accounting, because the sweeps we have done are real but scattered across datasets
rather than systematic, and a reviewer will check this.

## What REMIND documents (their Table 16, EMBED only)

| hyperparameter | values tried | chosen |
|---|---|---|
| group-DRO step size eta | 0.1, 0.5 | not stated explicitly |
| fusion experts E | 32, 64, 128 | 128 (avg 80.7 vs 80.6 and 80.2) |
| group-DRO sharpness gamma | 0.5, 0.1, 0.02 | 0.02 |
| imputation / init scheme | learnable vs zero emb, sinusoidal residual routing | learnable |

Fixed by them: 128 experts, one slot per expert, embedding 768, 8 heads, AdamW lr 1e-4,
weight decay 5e-5, batch 32, max 20 epochs, 3 seeds.

They do NOT document an equivalent sweep for FlexMoE, Reweigh, FuseMoE or FairBatch. Their own
method is swept; their baselines are not. Worth knowing, because it means their reported margin
carries the same asymmetry we are trying to avoid.

## What we have swept

| hyperparameter | values | datasets covered | gap |
|---|---|---|---|
| lambda_fit (anchor weight) | 0.05, 0.1, 0.3, 0.5 | Fed-Heart (CV) | not swept on NHANES under CV; EMBED swept at 0.1, 1, 10 |
| dro_gamma | 0.02, 0.5 | EMBED | never swept on tabular |
| capacity (latent/hidden width) | 32, 64, 128, 256 | NHANES | never swept on Fed-Heart or EMBED |
| dro_signal (train vs val) | both | EMBED | never tried on tabular |
| anchors on/off, DRO variant | full 2x2x2 | all three | complete |

So the ablation grid is complete everywhere, but the *hyperparameter* sweeps are patchy: each
knob has been swept on one dataset and assumed to transfer. That assumption already failed once,
when Fed-Heart inherited NHANES's lambda_fit of 0.1 and lost 1.2 points to it.

## The protocol problem, which matters more than any missing sweep

`run_method_matrix.py:105` and `run_fedheart_cv.py:56` select the reported epoch by

    best = max(rows, key=test_worst_group_acc)

and neither tabular loader builds a validation split at all. Every tabular number we report is
therefore a best-of-about-40 on the test set. The baselines do the same
(`run_baselines_tabular.py:208`), so the comparison between methods is at least consistent, and
EMBED is clean because `train_embed_xenia.py:330` selects on a real validation split. But the
absolute tabular numbers are optimistically biased and a reviewer will say so.

Standard wording we would want to be able to write, and currently cannot:

> "Hyperparameters for all methods were selected on the validation set using comparable search
> budgets. The test set was used only for final evaluation."

## Priority order

1. **Validation split on NHANES and Fed-Heart**, and select on it. Without this the tabular
   results are not defensible regardless of how much sweeping we add.
2. **REMIND at 128 experts.** We had been running it at 4, which is a fraction of its specified
   capacity. Fixed, needs re-running.
3. **Fill the sweep matrix**: lambda_fit on NHANES, capacity on Fed-Heart and EMBED, dro_gamma
   on tabular.
4. **Sensitivity appendix**, separate from tuning: show performance across each swept value so
   the method is visibly not fragile, which is what their Table 16 does.
