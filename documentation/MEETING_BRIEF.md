# Meeting brief, Thursday 10 Sept 10:30 EST

Everything here is checked against the run files. Numbers are worst-group accuracy unless
stated otherwise.

---

## Part 1. Agenda

**1. Results, 5 minutes.** She has the dashboard. Do not walk her through tables she can read.
Say the three things she cannot get from looking:

- Full method beats the ERM baseline on all three datasets, +7.8 / +4.1 / +4.1, all significant
  at 10 seeds.
- Against the GroupDRO baseline the anchors are +3.7 on NHANES, -1.8 on Fed-Heart, -1.0 on
  EMBED. One clear win, one clear loss, one tie.
- On EMBED the two ingredients depend on each other: regret alone does literally nothing (57.8,
  identical to ERM), regret plus anchors is 61.9.

**2. The tension, 10 minutes.** This is the real agenda item. Our own ablation says the anchors,
the novel component, subtract on two of three datasets. And the random-target control says the
alignment is not class-conditional. Both are in the tables, both are things a reviewer reaches
in about five minutes. Ask her directly: does she want to defend the anchors, or reframe?

**3. The framing decision, 15 minutes.**

| option | claim | odds | what it needs |
|---|---|---|---|
| Method paper | we propose this, it beats baselines, each part matters | low | the ablation currently contradicts the last clause |
| Characterization | alignment helps in proportion to feature non-redundancy | better | a redundancy statistic, about a day |
| Workshop | either of the above | high | nothing extra |

Recommend characterization. The Fed-Heart and EMBED nulls stop being embarrassments and become
confirmations, because those are the datasets with redundant features.

**4. What to run next, 10 minutes.** Two candidates and only time for one:

- **REMIND, row 4.** Her doc says use their released code, and Step 6 says use our R\* values for
  every method including REMIND, so their published numbers cannot be dropped in. Needs their
  code obtained and run on our pipeline. Days, not hours. Flag that it has not started.
- **Redundancy statistic.** Measure how much each group's non-shared features add beyond the
  shared ones, plot against anchor gain. Turns the characterization claim from a story into a
  prediction. About a day.

Which matters depends on which framing she picks. Method paper needs REMIND. Characterization
needs the statistic.

**5. Ask for a decision.** Two weeks and two days to the deadline. Enough to write one paper
well or two badly.

---

## Part 2. The method

Three components, shared across all three datasets.

**Per-group encoders.** Each group g gets its own network phi_g mapping its own feature set
into one shared 64-dimensional latent space. This exists because groups have different input
dimensions and you cannot feed 8 features and 13 features into the same linear layer. Without
it your only option is to discard every column the groups do not share.

**Gaussian class anchors.** Per-group encoders create a new problem: nothing forces them to
agree on where things sit in the latent space. Each class c gets a learnable Gaussian
N(m_c, S_c) and two loss terms pull embeddings onto it:

    L_total = L_clf + lambda_fit * L_fit + lambda_sep * L_sep

- `L_fit` is the squared 2-Wasserstein distance between the batch's class-c moments and anchor c.
- `L_sep` samples from each anchor, pushes the samples through the shared head, and requires
  correct classification. Without it, collapsing all anchors to one point would satisfy L_fit
  perfectly and be useless.
- "Anchors off" means lambda_fit = lambda_sep = 0, which leaves plain cross-entropy.

**GroupDRO, and the regret variant.** Weight each group by lambda_g on the simplex and push
weight toward whichever group is doing worst. Regret weights by excess over that group's own
achievable floor R\*_g instead of by raw loss, so groups already at their limit stop being
pushed. Setting R\* to zero recovers standard GroupDRO exactly.

**A point worth being ready for.** With the anchors off, the groups are still aligned, just
implicitly: every group's latents pass through the same shared classifier, so gradient descent
has to arrange them for one decision boundary. The anchors add an explicit constraint on top of
that implicit one. Measured on NHANES: cross-group latent misalignment is 4.40 without anchors
and 0.16 with them, a 27-fold difference, while latent scale shrinks 45-fold. On Fed-Heart that
extra constraint over-tightens and costs accuracy.

---

## Part 3. The three datasets

### Fed-Heart

**Data.** 925 patients, four cardiology sites from the UCI heart disease archive. Binary label,
presence of heart disease.

**Groups and features.** Cleveland 305 patients / 10 features, Hungarian 295 / 8,
Switzerland 125 / 8, VA 200 / 9. Feature subsets are set by `feature_mask` in the config and
`true_hetero_input_dim: true` gives each encoder an input dimension matching its real feature
count rather than padding to 13.

**Deliberate scarcity.** `group_max_train_samples: [null, null, 20, 25]` caps Switzerland to 20
and VA to 25 training samples. If she asks whether the heterogeneity is real: the feature
differences are real, the sample scarcity is imposed to simulate small sites.

**Protocol.** 5-fold cross validation with median imputation, 10 seeds, so 500 fold-runs across
10 arms. Every patient is held out exactly once per seed. Median imputation follows FLamby and
keeps every patient; without it the loader dropped any row with a missing value, which removed
63% of Switzerland because it lacks serum cholesterol on most records.

**Why not a single split.** The old protocol tested Switzerland on 10 patients, so worst-group
accuracy moved in 10-point steps and half the seeds landed on identical values. That is
superseded and nothing on the dashboard uses it.

**Architecture.** MLP with layer norm per group, latent 64, head hidden 32, Adam, lr 1e-3,
weight decay 1e-4, batch 64, up to 100 epochs with early stopping at 30, cosine schedule.

**Results.** ERM 65.8, per-group ERM 74.8, per-group + GroupDRO 75.4, full method 73.6.
Per-group encoders are worth +9.0. GroupDRO adds +0.9 on top. The anchors cost 1.8 (p=0.001).

### NHANES

**Data.** 17,005 US adults, CDC survey 2017-2020 and 2021-2023. Binary cardiovascular disease
label, 10.5% positive, so the loss is inverse-frequency class weighted.

**Groups.** Defined by how much of the assessment a participant completed, which is a
logistical fact about how the survey runs rather than a split we chose.

- G0, about 2,100 people, 10 features: age, sex, race one-hot, education, income-to-poverty,
  ever smoked.
- G1, about 2,100, 13: adds BMI, weight, height from the physical exam.
- G2, about 9,400, 20: adds systolic and diastolic BP, HbA1c, HDL, total cholesterol,
  triglycerides, LDL.

The sets nest because the assessment is sequential. Nobody gives blood without first answering
the questionnaire. G2 is larger than the other two combined, which is why ordinary training
neglects G0 and G1.

**Protocol.** 10 seeds, fixed train/test split (`data_split_seed: 100`) so the seed varies model
initialisation only, 80/20 split, stratified batching to preserve group proportions.

**Architecture.** Same as Fed-Heart: per-group MLP with layer norm, latent 64, head hidden 32,
Adam 1e-3, batch 128, up to 100 epochs, early stopping 25.

**Results.** ERM 70.1, per-group ERM 69.0, per-group + GroupDRO 70.4, full method 74.1.
Per-group encoders alone COST 1.1 here, because splitting 2,100-sample groups across separate
encoders loses more to sample efficiency than the extra columns return. The anchors are worth
+3.7 (p=0.007) and carry the entire gain.

**If she asks about disjoint or expanded.** Two other feature modes exist. Expanded is 15/18/25
and still nested, so it tests nothing new; the poster used it. Disjoint gave each group 10
shared plus 5 private features, was invented in an earlier session, is in neither of her
documents, and its private sets turn out to be substitutes for each other (G0's self-reported
hypertension versus G2's measured blood pressure), so it cannot test what it was built to test.
Both are cut. Runs are kept for an appendix.

### EMBED

**Data.** 128,680 breast-exam records from 22,997 patients, Emory mammography archive. Four
ordered BI-RADS density classes.

**Groups.** Which of four imaging views a breast has: C-View CC, C-View MLO, FFDM CC, FFDM MLO.
g1 {FFDM CC} 3.0%, g2 {C-View CC, FFDM CC} 0.6%, g3 {FFDM MLO} 1.2%, g4 {FFDM CC, FFDM MLO}
57.8%, g5 {C-View CC, FFDM CC, FFDM MLO} 0.3%, g6 all four 37.1%. Two head groups hold about
95%. Most imbalanced of the three.

**No common-features baseline exists here.** g1 is {FFDM CC} and g3 is {FFDM MLO}, so the
intersection over all six groups is empty. This is why EMBED's tables and ladder start from
per-group ERM. It is also a good motivating point: the heterogeneity is not always reducible by
throwing columns away.

**Architecture, her Step 3 exactly.** Frozen ViT-Base, cached 768-d CLS embedding per image, one
`Linear(768 -> 64)` per view shared across groups, per-group `MLP_g` with input `|g| x 64`,
hidden 64, output 64, shared head `Linear(64 -> 4)`, four diagonal Gaussian anchors with
`s_c = softplus(rho_c) + 1e-4`. 276,228 parameters. AdamW lr 5e-5, weight decay 5e-5, batch 32,
20 epochs, StepLR(5, 0.1). Checkpoint selection per row: rows 1 and 6 on average loss, row 2 on
worst-group loss, rows 5 and 7 on max excess loss.

**Results.** ERM 57.8, GroupDRO 62.4, regret alone 57.8, anchors + GroupDRO 61.4,
anchors + regret 61.9. Ours has the best worst-group loss, 1.099 against GroupDRO's 1.200,
p=0.015, and worst-group loss is one of the two numbers Step 6 calls headline.

---

## Part 4. The ablation

Every arm is a point in the same grid: encoder in {shared, per-group}, DRO in {off, GroupDRO,
regret}, anchors in {off, on}. Ten arms on the tabular datasets, seven on EMBED.

Her spec's 2x2 is rows 2, 5, 6, 7: anchors on or off crossed with R\* used or set to zero. That
is what separates her two named ingredients. The shared-encoder rows are our addition beyond
the spec, kept because the anchor result survives them, so it does not rest on the architecture.

Ranking on the dashboard is done separately per encoder, since a common-feature model and a
per-group model do not see the same inputs, and arms a paired t-test cannot separate from the
best are marked tied rather than ranked.

---

## Part 5. Things she may push on

**"Why do the anchors hurt on Fed-Heart?"** Best answer: the shared head already aligns groups
implicitly, and Fed-Heart's four sites record heavily overlapping tests, so there is little
cross-group inconsistency left for the anchors to fix. The explicit constraint then over-tightens
and destroys useful per-site signal. The synthetic overlap sweep supports this: dialling feature
overlap from complete to none moves the benefit from +0.15 to +23.9.

**"Is the alignment really class-conditional?"** No, and we can show it. Replacing each sample's
own class anchor with a permuted one does not hurt: NHANES +0.55, p=0.41, winning on 6 of 10
seeds. Tested on three datasets and both forms of the loss. The first version of this control
used uniform random labels, which on a 90/10 task also changed class proportions to 50/50 and
confounded the result; permuting real labels preserves every class count and isolates the
structure. The conclusion survived the fix.

**"Did you tune lambda_fit?"** Yes, all three values her table names. Worst-group accuracy for
anchors+regret: 56.8 at 0.1, 62.2 at 1.0, 42.9 at 10. For anchors+GroupDRO: 58.6, 60.1, 59.5.
The 1.0 default is best for both.

**"Why does GroupDRO put all its weight on g4?"** Because the lambda update reads training loss
and g5 has about 40 training rows, so it gets memorised, its excess goes to zero, and its weight
never grows. We tested the obvious fix, driving lambda from validation excess instead, and it is
worse across the board: groupdro 62.2 to 55.9, ours 62.2 to 54.6. The tail validation sets are
too small to steer with. So the spec's choice of training loss is empirically right even though
the resulting weight distribution looks degenerate.

**Deviations from her spec, disclose these.**

1. lambda initialised uniform rather than at group proportions p_g. At p_g the tail groups start
   near 0.3% and never move.
2. dro_gamma 0.5 rather than 0.02. Sanctioned; her table says always sweep 0.02, 0.1, 0.5.
3. Density joined on accession alone, not accession and laterality as her Step 1 says. Joining
   on both lost about 75% of labels, because `side` marks per-finding laterality and is roughly
   48% missing while each accession has exactly one `tissueden`. This is the deviation most
   worth raising, since she explicitly warned against it.
4. Row 4, REMIND, not run.
5. Fine-tuned backbone was explored and dropped. Her spec says frozen; everything reported is
   frozen.
