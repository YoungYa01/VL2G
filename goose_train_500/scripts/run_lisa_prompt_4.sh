#!/usr/bin/env bash
set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate lisa
cd ~/projects/LISA

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT_ROOT="/mnt/d/research_vl2g/goose_train_500"
mkdir -p "$OUT_ROOT/preds/lisa/unsafe_proxy" "$OUT_ROOT/logs"

echo "[LISA] Running prompt 4: unsafe_proxy"

CUDA_VISIBLE_DEVICES=0 python chat.py \
  --version='xinlai/LISA-13B-llama2-v1' \
  --precision='fp16' \
  --load_in_4bit \
  --vis_save_path "$OUT_ROOT/preds/lisa/unsafe_proxy" \
  < "/mnt/d/research_vl2g/goose_train_500/lisa_inputs/lisa_prompt_4_unsafe_proxy.txt" \
  | tee "$OUT_ROOT/logs/lisa_4_unsafe_proxy.log"

echo "[DONE] LISA prompt 4 finished."
