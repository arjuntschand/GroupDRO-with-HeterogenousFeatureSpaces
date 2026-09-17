# EMBED

Four-class BI-RADS breast-density classification on the Emory EMBED
mammography archive, with groups defined by which imaging views a breast has.
This is the dataset the REMIND paper (arXiv 2603.00046) uses, so it is where
our method is compared against theirs on their own ground.

## Access and what is committed

EMBED is distributed through the AWS Open Data programme
(`s3://embed-dataset-open`, us-west-2) under a research use agreement that
must be signed first. That agreement forbids redistributing the images, any
index derived from the tables, cached embeddings, or model weights trained
on the data. None of those are in this repository, and `.gitignore` treats
`runs/embed_*` as an allowlist so they cannot be added by accident. What is
committed: the code, and metric summaries only (`metrics_long.csv`,
`rstar.json`, and per-epoch `curve_*.json` files with group weights and
accuracies).

## Data

The open subset holds 23,256 patients, 72,770 exams and 480,323 images.
After keeping the canonical views and exams with a density label there are
128,680 breast-level records from 22,997 patients. Density is recorded per
exam and broadcast to both breasts; the clinical table's `side` column is
per-finding and must not be used for the density join.

Four view types, called modalities in the REMIND paper:

| code | view |
|---|---|
| M1 | C-View CC (synthetic 2D from tomosynthesis) |
| M2 | C-View MLO |
| M3 | FFDM CC (full-field digital mammogram) |
| M4 | FFDM MLO |

A breast's group is the set of views it has. Two combinations cover about
95% of records and the other four are rare, which makes this the most
imbalanced of the three datasets.

| group | views | share |
|---|---|---|
| g1 | M3 | 0.9% |
| g2 | M1, M3 | 0.1% |
| g3 | M4 | 0.4% |
| g4 | M3, M4 (head) | 56.8% |
| g5 | M1, M3, M4 | 0.1% |
| g6 | all four (head) | 41.7% |

Classes: A almost entirely fatty 9.9%, B scattered 41.9%, C heterogeneously
dense 42.8%, D extremely dense 5.4%. No common-features baseline exists here
because g1 and g3 share no view, so the ablation starts from per-group ERM.

## Pipeline

Everything downstream of feature extraction runs on cached vectors, so the
experiments themselves take minutes on a CPU.

1. **Tables.** `aws s3 cp s3://embed-dataset-open/tables/ datasets/embed/tables/ --recursive`
2. **Index.** `python -m dro_hetero_anchors.src.tools.build_embed_xenia_index --mode full`
   joins the metadata and clinical tables into the six-group breast-level
   index (`datasets/embed/index_xenia_6group.parquet`). Groups are assigned
   from each breast's true view set in the full metadata.
3. **Features.** A frozen ViT-Base encodes every referenced image once and
   its 768-dimensional CLS embedding is cached
   (`python -m dro_hetero_anchors.src.extract_vit_embeddings --images-root datasets/embed`).
   `stream_embed_features.py` does the same without ever holding the 1.8 TB
   of DICOMs on disk: download a chunk, decode, embed, delete, repeat.
   `download_embed.py` fetches only the images an index references.
4. **Train.** `python -m dro_hetero_anchors.src.train_embed_xenia --out runs/embed_xenia_production --seeds 0 1 42 7 11 22 33 1337 2024 31337`
5. **Report.** `python -m dro_hetero_anchors.src.report_embed_xenia --run runs/embed_xenia_production`
6. **Baselines.** `python run_baselines_embed.py --index datasets/embed/index_xenia_6group.parquet --cache datasets/embed/vit_cache`
   runs Reweigh, Flex-MoE and REMIND (128 experts) on the same cached
   features, split and R* estimates.

`datasets_embed.py` holds the schema logic (which `FinalImageType` and
`ViewPosition` tokens count as canonical views) and an `--inspect` mode
that prints the real value counts.

## Model and objective

Per-view linear projection 768 to 64, shared across groups; per-group MLP
over the concatenation of the views that group has (input 64 times the
number of views, hidden 64, output 64, GELU); shared linear head 64 to 4;
four diagonal Gaussian class anchors. 276,228 parameters. AdamW at learning
rate 5e-5 and weight decay 5e-5, batch 32, 20 epochs, step schedule (5,
0.1). Patient-level 70/10/20 split with a fixed split seed; ten seeds.

The objective is the same as on the tabular datasets: a min-max over group
weights of each group's task loss minus its reference loss R*_g plus the
anchor fit term, with the separation term outside the max. R*_g comes from
a dedicated per-group model by 5-fold out-of-fold cross-entropy. Checkpoints
are selected on validation: ERM and align-only on average loss, GroupDRO on
worst-group loss, the regret arms on maximum excess loss.

Methods in `runs/embed_fix_final/metrics_long.csv`: `erm`, `groupdro`
(R* = 0), `regret_only`, `align_only` (anchors, no regret), `ours` (anchors
and regret). The group-only dedicated models supply R*.

One thing to know when reading the EMBED rows of `runs/FINAL_TABLES.txt`:
with the specification's update rule (step size 0.02 on an exponential
moving average of the excess loss) and the weights initialised at group
proportions, the two head groups start with 98.5% of the weight and the
tail groups at a fraction of a percent, and 20 epochs are not enough to
move them. The DRO arms then coincide with ERM. The trainer exposes
`--dro-gamma` and `--uniform-lambda-init` for the larger, REMIND-scale step
size, under which GroupDRO separates clearly from ERM; those runs are kept
alongside so the step-size sensitivity can be reported rather than hidden.

## Results

| folder | what it holds |
|---|---|
| `runs/embed_fix_final/` | the canonical run the tables and site read: `metrics_long.csv`, `rstar.json` |
| `runs/baselines_embed*/` | Reweigh, Flex-MoE, REMIND at released and matched sizes; `_remind128` is the 128-expert run |
| `runs/embed_xenia_production/` | the full production run with per-epoch `curve_*.json` (group weights, accuracies) and `REPORT.md` |
| `runs/embed_valsignal_final/`, `runs/embed_uniform_final/`, `runs/embed_lamfit_*/` | variants: validation-driven weights, uniform initialisation, anchor weight 0.1 and 10 |

On EMBED the anchors cost worst-group accuracy and improve worst-group loss;
the four views are images of the same breast, so there is little cross-group
misalignment for the anchors to remove. The tail groups are small enough to
be memorised during training, which starves any loss-driven reweighting of
signal; that is a property of the data, and it is why the tabular datasets
carry the robustness story.
