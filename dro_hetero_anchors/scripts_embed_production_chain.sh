#!/bin/bash
# Wait for extraction to finish, then build the full index and run the production
# 10-seed experiment. Everything ready by morning.
cd /home/ubuntu/GroupDRO
LOG=/home/ubuntu/production_chain.log
echo "[chain] waiting for extraction to complete..." > $LOG

while pgrep -f "[s]tream_embed_features" >/dev/null; do sleep 120; done
echo "[chain] extraction finished at $(date)" >> $LOG

# build the full usable index from whatever the cache actually contains
/opt/pytorch/bin/python - >> $LOG 2>&1 <<'PY'
import pandas as pd, json
idx = pd.read_parquet("datasets/embed/index_xenia_6group.parquet")
cached = set(json.load(open("datasets/embed/vit_cache_full/paths.json")).keys())
def imgs(r):
    d = r if isinstance(r, dict) else dict(r)
    return [p for p in d.values() if isinstance(p, str)]
ok = idx[idx["paths"].apply(lambda d: all(p in cached for p in imgs(d)))].reset_index(drop=True)
ok.to_parquet("datasets/embed/index_production.parquet")
print(f"[chain] production index: {len(ok)} rows, {ok.empi_anon.nunique()} patients")
print("[chain] per group:", ok.group.value_counts().sort_index().to_dict())
PY

echo "[chain] launching 10-seed production experiment at $(date)" >> $LOG
/opt/pytorch/bin/python -m dro_hetero_anchors.src.train_embed_xenia \
  --index datasets/embed/index_production.parquet \
  --cache datasets/embed/vit_cache_full \
  --out runs/embed_xenia_production \
  --methods erm groupdro align_only regret_only ours group_only \
  --seeds 0 1 42 7 1337 2024 31337 11 22 33 \
  --epochs 20 --rstar-folds 5 >> $LOG 2>&1
echo "[chain] PRODUCTION RUN COMPLETE at $(date)" >> $LOG
