# NHANES: nested versus disjoint, and why both are in the paper

Both settings use the same 17,005 people, the same CVD label, the same 3 groups and the same
train/test split. The only thing that changes is **which features each group is allowed to see**.
That is deliberate: it makes the comparison between them a clean experiment about feature
structure rather than about data.

## Nested: the real availability structure

Groups are defined by how far a participant got through the NHANES assessment.

| group | who they are | features | train size |
|---|---|---|---|
| G0 | filled out the survey only | 10 | ~2,100 |
| G1 | survey + physical exam | 13 | ~2,100 |
| G2 | survey + exam + blood pressure + labs | 20 | ~9,400 |

The key property is that **each group's features are a subset of the next**: G0 is inside G1 is
inside G2. That is not a design choice, it is how the data comes. If someone got blood drawn,
they also got the body measurements and they also answered the questionnaire. Nobody has labs
without a survey.

This is the honest, real-world setting. It is what the method would meet in an actual clinical
deployment, where the heterogeneity comes from missing assessments rather than from genuinely
different instruments.

The catch: because the features are nested, a shared encoder can do fairly well by simply
zero-filling what is absent. There is no feature any group has that another group could not in
principle also have. So nested is a realistic test but a gentle one.

## Disjoint: the stress test we constructed

Every group gets 15 features: the same 10 shared survey items, plus 5 that are private to that
group and appear nowhere else.

| group | shared | private to that group |
|---|---|---|
| G0 | 10 survey items | 5 extra questionnaire items |
| G1 | 10 survey items | BMI, weight, height, HbA1c, HDL |
| G2 | 10 survey items | systolic BP, diastolic BP, total cholesterol, triglycerides, LDL |

(Feature construction is in `datasets_nhanes.py`, `feature_mode == "disjoint"`.)

Now no single encoder can see everything. Two thirds of each group's inputs are invisible to the
other two groups, so a shared encoder cannot represent all of them at once and per-group encoders
become necessary rather than merely helpful.

**This setting is synthetic and we say so on the site and in the paper.** Real NHANES availability
is nested. We built disjoint by partitioning the real measurements, so the values are real
patient data, but the partition is ours.

## Why we report both

They test different things, and reporting only one would be misleading in opposite directions.

- **Nested alone** would understate the method. It is the easy end of the heterogeneity range,
  where a shared encoder is already a decent approximation, so it makes the architecture look
  less necessary than it is.
- **Disjoint alone** would overstate it. It is a partition we chose, and a reviewer would
  reasonably ask whether we constructed the setting that our method happens to win on.

Together they are one point each on the axis the synthetic overlap sweep varies continuously,
from fully shared features to fully private ones. The sweep shows per-group encoders going from
+0.15 at complete overlap to +23.94 at zero overlap. Nested sits toward the low end of that
curve, disjoint toward the high end, and both land where the curve predicts. That agreement is
the argument: the benefit tracks feature heterogeneity, and we can show it on real data at two
different points rather than asserting it from one.

Practical read: **nested is the headline real-world result, disjoint is the mechanism result.**
If we ever have to cut one for space, nested stays in the main paper and disjoint moves to the
appendix alongside the overlap sweep, since those two make the same argument.

## What about "expanded"?

There was a third mode, `expanded` (15/18/25 features, still nested). It is archived and not
reported. It adds more survey features but keeps the nested structure, so it tests nothing that
nested does not already test and only adds a column readers have to interpret.
