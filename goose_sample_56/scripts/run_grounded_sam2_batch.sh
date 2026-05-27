#!/usr/bin/env bash
set -e

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate localretro
cd ~/projects/Grounded-SAM-2

OUT_ROOT="/mnt/d/research_vl2g/goose_sample_56"
IMG_DIR="$OUT_ROOT/images"
PRED_ROOT="$OUT_ROOT/preds/grounded_sam2"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$PRED_ROOT" "$LOG_DIR"

run_one_prompt () {
  PROMPT_TYPE="$1"
  PROMPT="$2"

  for img in "$IMG_DIR"/*; do
    [ -f "$img" ] || continue
    base=$(basename "$img")
    sample="${base%.*}"
    out="$PRED_ROOT/$PROMPT_TYPE/$sample"
    mkdir -p "$out"

    echo "[GSAM2] $sample | $PROMPT_TYPE | $PROMPT"

    python grounded_sam2_hf_model_demo.py \
      --img-path "$img" \
      --text-prompt "$PROMPT" \
      --output-dir "$out" \
      >> "$LOG_DIR/grounded_sam2_${PROMPT_TYPE}.log" 2>&1
  done
}

run_one_prompt "road" "road. dirt road. path. gravel. soil."
run_one_prompt "obstacle" "rock. obstacle. vehicle. person. tree. barrier."
run_one_prompt "water_risk" "water. puddle. mud. snow. high grass. rock."

echo "[DONE] Grounded-SAM2 batch finished."
