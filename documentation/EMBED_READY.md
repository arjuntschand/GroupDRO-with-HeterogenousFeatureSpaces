# EMBED full-data run — ready to fire the moment access returns

**Status:** blocked on EMBED S3 access. Emory migrated to a new portal (data.hitilab.com)
and **rescinded all existing grants**; re-application submitted. Access grants last 90 days.
Test with: `aws s3 ls s3://embed-dataset-open/tables/` — when it lists files, we're live.

## Why this is now a ~4 hour job, not a week

The full 6-group index references **363,098 images (~1.5 TB of DICOMs)** — but Xenia's
frozen-ViT architecture only needs the **768-d CLS embedding per image = 558 MB total**.
`stream_embed_features.py` never stores the images:

```
for each chunk of 2000:
    parallel download from S3  →  decode DICOM  →  frozen ViT-Base
    →  append embedding  →  DELETE the DICOMs  →  next chunk
```

Peak disk stays a few GB. Fully **resumable** (skips already-cached paths), so an
interrupted run or a dropped SSH costs nothing.

## The exact sequence (all scripted, nothing left to build)

```bash
# 0. verify access is back (locally)
aws s3 ls s3://embed-dataset-open/tables/

# 1. start the box + open SSH for the current IP
aws ec2 start-instances --instance-ids i-REDACTED --region us-west-2
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id sg-REDACTED \
    --protocol tcp --port 22 --cidr ${MYIP}/32 --region us-west-2
# update ~/.ssh/config host `embed` HostName to the new public IP

# 2. sync code + creds, then stream-extract the full dataset (~3-4 h, resumable)
rsync -az dro_hetero_anchors/src/ embed:~/GroupDRO/dro_hetero_anchors/src/
scp datasets/embed/index_xenia_6group.parquet embed:~/GroupDRO/datasets/embed/
ssh embed 'cd ~/GroupDRO && setsid bash -c "/opt/pytorch/bin/python -m \
  dro_hetero_anchors.src.stream_embed_features \
  --index datasets/embed/index_xenia_6group.parquet \
  --out datasets/embed/vit_cache_full --chunk 2000 --workers 16 \
  > ~/embed_stream.log 2>&1" </dev/null >/dev/null 2>&1 &'

# 3. pull the 558 MB cache, stop the box
scp embed:~/GroupDRO/datasets/embed/vit_cache_full/* datasets/embed/vit_cache_full/
aws ec2 stop-instances --instance-ids i-REDACTED --region us-west-2

# 4. run the full experiment LOCALLY (minutes — tiny MLPs on cached vectors)
python -m dro_hetero_anchors.src.train_embed_xenia \
    --index datasets/embed/index_xenia_6group.parquet \
    --cache datasets/embed/vit_cache_full --out runs/embed_xenia_full \
    --methods erm groupdro align_only regret_only ours group_only \
    --seeds 0 1 42 --epochs 20 --rstar-folds 5
python -m dro_hetero_anchors.src.report_embed_xenia --run runs/embed_xenia_full
```

## What the full run fixes vs the 13% pilot

| | 13% pilot (current) | full data |
|---|---|---|
| rows | 16,575 | **128,680** |
| g1 / g3 (mid tails) | 801 / 411 | **1,158 / 543** |
| g4 / g6 (heads) | 7,485 / 7,732 | **73,040 / 53,687** |
| absolute accuracy | ~73% | expected ~78-80% → **REMIND-comparable** |
| tail memorization | severe (tiny tails → λ≈0) | much reduced → regret-DRO can actually grip |

The pilot's core limitation was that the tails were so small the model memorized them
(train loss→0), which neutralizes the DRO/regret mechanism. Full data is the real test.

## Everything already built and validated

- `stream_embed_features.py` — streaming extractor (this doc's step 2)
- `train_embed_xenia.py` — Xenia's spec exactly (per-view proj + per-group MLPs + shared
  head + diagonal anchors; regret-DRO with 5-fold R*_g; all 7 methods)
- `report_embed_xenia.py` — tables + R*-vs-λ plot + metrics_long.csv
- `tools/build_embed_xenia_index.py` — rebuilds the index (`--mode full|offline`)
- Full index already built locally: `datasets/embed/index_xenia_6group.parquet`
  (128,680 rows / 22,997 patients, 6 groups, per-exam density labels)

## DUA reminders

Publish results/metrics/figures freely. **Never** share the dataset, the cached embeddings,
or trained weights (Research Use Agreement §4-6). All EMBED artifacts are gitignored.
