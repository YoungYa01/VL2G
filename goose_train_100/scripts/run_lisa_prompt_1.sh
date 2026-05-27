#!/usr/bin/env bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate lisa
cd ~/projects/LISA

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT_ROOT="/mnt/d/research_vl2g/goose_train_100"
mkdir -p "$OUT_ROOT/preds/lisa/traversable" "$OUT_ROOT/logs"

echo "[LISA] Running prompt 1: traversable"

CUDA_VISIBLE_DEVICES=0 python chat.py \
  --version='xinlai/LISA-13B-llama2-v1' \
  --precision='fp16' \
  --load_in_4bit \
  --vis_save_path "$OUT_ROOT/preds/lisa/traversable" \
  < "/mnt/d/research_vl2g/goose_train_100/lisa_inputs/lisa_prompt_1_traversable.txt" \
  | tee "$OUT_ROOT/logs/lisa_1_traversable.log"

echo "[DONE] LISA prompt 1 finished."
