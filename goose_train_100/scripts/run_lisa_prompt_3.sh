#!/usr/bin/env bash
set -e

conda activate lisa
cd ~/projects/LISA

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT_ROOT="/mnt/d/research_vl2g/goose_train_100"
mkdir -p "$OUT_ROOT/preds/lisa/water_risk" "$OUT_ROOT/logs"

echo "[LISA] Running prompt 3: water_risk"

CUDA_VISIBLE_DEVICES=0 python chat.py \
  --version='xinlai/LISA-13B-llama2-v1' \
  --precision='fp16' \
  --load_in_4bit \
  --vis_save_path "$OUT_ROOT/preds/lisa/water_risk" \
  < "/mnt/d/research_vl2g/goose_train_100/lisa_inputs/lisa_prompt_3_water_risk.txt" \
  | tee "$OUT_ROOT/logs/lisa_3_water_risk.log"

echo "[DONE] LISA prompt 3 finished."
