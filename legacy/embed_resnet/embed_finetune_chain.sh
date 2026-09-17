#!/bin/bash
# Track B: fine-tuned backbone (REMIND-comparable). Waits for the image cache AND for the
# frozen-backbone production run to free the GPU, then fine-tunes, re-extracts, and runs
# the full method grid on the improved features.
cd /home/ubuntu/GroupDRO
LOG=/home/ubuntu/finetune_chain.log
echo "[chain] waiting for image cache..." > $LOG
while pgrep -f "[c]ache_embed_images" >/dev/null; do sleep 120; done
echo "[chain] image cache done $(date)" >> $LOG

echo "[chain] waiting for frozen production run to free the GPU..." >> $LOG
while pgrep -f "[t]rain_embed_xenia" >/dev/null; do sleep 120; done
echo "[chain] GPU free $(date)" >> $LOG

echo "[chain] fine-tuning ViT-Base $(date)" >> $LOG
/opt/pytorch/bin/python -m dro_hetero_anchors.src.finetune_embed_vit \
  --index datasets/embed/index_production.parquet \
  --img-cache datasets/embed/img_cache \
  --out-cache datasets/embed/vit_cache_finetuned \
  --ckpt runs/embed_vit_finetuned.pt --epochs 5 --batch 64 >> $LOG 2>&1
echo "[chain] fine-tune + extract done $(date)" >> $LOG

echo "[chain] running method grid on fine-tuned features $(date)" >> $LOG
/opt/pytorch/bin/python -m dro_hetero_anchors.src.train_embed_xenia \
  --index datasets/embed/index_production.parquet \
  --cache datasets/embed/vit_cache_finetuned \
  --out runs/embed_xenia_finetuned \
  --methods erm groupdro align_only regret_only ours group_only \
  --seeds 0 1 42 7 1337 2024 31337 11 22 33 \
  --epochs 20 --rstar-folds 5 --dro-gamma 0.5 --uniform-lambda-init >> $LOG 2>&1
echo "[chain] FINE-TUNED TRACK COMPLETE $(date)" >> $LOG
