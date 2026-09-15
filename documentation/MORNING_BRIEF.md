# Overnight, 2026-09-15

Short version: the headline moved from "we beat the baselines" to "we match them on accuracy and
beat them on every loss metric, nine times out of nine." That is a narrower claim and a much
better supported one. Three protocol bugs were found and fixed getting there, and two of them
were changing results.

## Results as they now stand

Every dataset is on a matched evaluation protocol, and REMIND runs at its published 128 experts.

| metric | NHANES | Fed-Heart | EMBED | total | sign test | significant W/T/L |
|---|---|---|---|---|---|---|
| worst-group accuracy | 2/3 | 1/3 | 2/3 | 5/9 | p=1.000 | 0/8/**1** |
| worst-group loss | 3/3 | 3/3 | 3/3 | **9/9** | **p=0.004** | 6/3/0 |
| max excess loss | 3/3 | 3/3 | 3/3 | **9/9** | **p=0.004** | 6/3/0 |

Read it as: on worst-group accuracy we are indistinguishable from Reweigh, Flex-MoE and REMIND,
with one real loss. On both loss metrics we are ahead everywhere, and the consistency across nine
independent comparisons is itself significant even where individual gaps are not.

The one loss is Fed-Heart against REMIND on worst-group accuracy, −2.43 at p=0.0010. It should be
reported, not buried inside 5/9.

## Why the metric choice is defensible rather than shopping

Picking the metric after seeing which one we win is exactly what a reviewer looks for. The reason
it holds here is that Xenia's theory never mentions accuracy. Proposition 1 is about Bayes risk.
Proposition 2 gives a closed-form ratio of worst-group **excess risk**. Our objective is a sum of
risks. Excess loss is the quantity the theory predicts, so reporting it is testing a stated
prediction.

That argument only works if the paper establishes it **before** the results section. After the
tables it reads as an excuse regardless of being true.

Lead with worst-group loss rather than max excess: it needs no R\*, and R\* is still shaky
(below).

## Bugs found, and what each one cost

**Fed-Heart baselines were scored on a fifth of the data.** Our arms use 5-fold CV over all 925
patients; the baselines were on a single 20% split, 185 patients. Switzerland: 125 for us, 25 for
them. Fixing it turned unresolvable ties into six significant wins on the loss metrics, and also
turned one tie into a genuine loss. Neither direction was visible before.

**Fed-Heart's ten seeds were one initialisation ten times.** `train_fedheart` calls `set_seed`
before `build_fedheart_loaders`, which calls `torch.manual_seed` internally with the split seed
and overwrites it. Under CV that split seed is `1000+fold`, identical across experiment seeds, so
every seed built identical weights. Seed-to-seed variation was coming only from `subsample_seed`.
Fixed by reseeding after the loaders. NHANES was checked and is clean; EMBED does not use these
loaders.

**The shared encoder only ever saw common features.** `shared_common_features` defaults to true in
`train_nhanes`, restricting the shared encoder to the intersection of all groups' features. So
"per-group vs shared" was a feature-access comparison, not an architecture one. With the shared
encoder given the same padded features, the per-group advantage on NHANES-disjoint falls from
+3.92 (p=0.005) to +1.02 (p=0.22), and three of four cells turn negative. Per-group encoders are
supported by Fed-Heart and not by NHANES in either feature mode.

**REMIND was running at 4 experts, not 128.** Correcting it to the paper's specification cost us a
win: worst-group loss on EMBED went from p=0.029 to p=0.816. What survives is max excess at
p=0.036, and they need 1,288,580 parameters against our 276,228.

## The sweep: do not use it

Twelve configurations, three seeds, chosen on validation.

| | current default | val-chosen | gain | p | does val rank like test? |
|---|---|---|---|---|---|
| NHANES | 71.42 | 74.96 | +3.54 | 0.146 | **ρ=+0.23, p=0.47** |
| Fed-Heart | 68.70 | 70.09 | +1.39 | 0.077 | ρ=+0.78, p=0.003 |

On NHANES validation barely ranks configurations like test, so the +3.54 is a lucky draw from a
grid spanning 4.23 points, not tuning that worked. Fed-Heart's rank correlation is real but its
gain is marginal.

The one solid finding: Fed-Heart's best configuration sets the anchor weight to 0.01, effectively
off. That independently reproduces the direct result that anchors hurt on Fed-Heart (−2.30,
p=0.016), from a completely separate analysis.

## Decisions waiting on you and Xenia

**1. Re-run our Fed-Heart arms under the fixed seeding?** Current error bars measure resampling
only. Not wrong, but less than "ten seeds" implies. About 25 minutes.

**2. Do the training caps stay?** `group_max_train_samples: [None, None, 20, 25]` caps Switzerland
at 20 training patients and VA at 25, against 98 and 160 available. It arrives silently from the
YAML and `run_fedheart_cv` never mentions it. The baselines inherit it so the comparison is fair,
but Fed-Heart then demonstrates robustness under scarcity we imposed, not the dataset's natural
heterogeneity. It is also the direct cause of VA's R\*=1.480.

**3. R\* needs a held-out split that is not test.** Pooling the fit across every patient with a
group's feature set drops VA from 1.480 to 0.609, but raises Switzerland from 0.424 to 0.737,
because pooling trades scarcity for distribution shift. `min(group-only, pooled)` is the right
combination, and both are valid upper bounds. Even then all four sites still sit above an achieved
loss, so the estimates remain loose. Tightening them means flooring R\* at the best validation
loss across a pool of models, and validation is what was reverted. This is a genuine tension, not
an oversight.

**4. What does "shared encoder" mean in the paper?** Common-features-only and all-features-padded
give materially different baselines, and only the first has ever been reported on NHANES.

## Housekeeping

All of this is on branch `embed-benchmark`; `main` is still at `22035a1c`.

Superseded results are preserved rather than deleted, so every correction above can be quantified:
`runs/baselines_fedheart_SINGLESPLIT`, `runs/baselines_fedheart_NOSEEDVAR`,
`runs/matrix_nhanes_nested_VALSEL`, `runs/fedheart_cv_VALSEL`, `runs/baselines_embed`
(4-expert REMIND).

Unrelated but worth having: batching the Soft MoE experts took REMIND-128 from 63.4 ms/step to
2.17 ms/step on the A10G, 29x, turning a 32-hour run into two hours. The loop version had the GPU
at 13% utilisation because it was launch-bound, not compute-bound.
