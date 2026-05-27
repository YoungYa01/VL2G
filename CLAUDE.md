# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research project on **Vision-Language to Grounding (VL2G)** for off-road driving perception. Compares open-vocabulary segmentation models (Grounded-SAM2, LISA) on the GOOSE dataset to evaluate their suitability as pseudo-label teachers, then distills a lightweight SegFormer student model from the best teacher's pseudo-labels.

Key finding: LISA (LLM-guided reasoning segmentation) significantly outperforms Grounded-SAM2 for traversable area segmentation (IoU 0.703 vs 0.139). A SegFormer student trained on LISA pseudo-labels achieves IoU ~0.69, close to the GT-supervised upper bound (~0.76).

## Pipelines

### 1. GOOSE data preparation + teacher inference + evaluation

`scripts/goose_vl2g_pipeline.py` is the main orchestration script. Two subcommands:

```bash
# Prepare: sample images from GOOSE, generate task masks, write run scripts
python scripts/goose_vl2g_pipeline.py prepare \
  --goose-root /path/to/goose-dataset \
  --out-root /mnt/d/research_vl2g/goose_sample_NN \
  --num 100 --seed 42 --splits train,val

# Evaluate: compute per-sample and summary metrics for all models against GT
python scripts/goose_vl2g_pipeline.py evaluate \
  --out-root /mnt/d/research_vl2g/goose_sample_NN
```

`prepare` writes self-contained bash scripts into `<out_root>/scripts/` that run Grounded-SAM2 and LISA inference, plus a thin `evaluate_predictions.py` wrapper.

### 2. Pseudo-label generation (teacher outputs → binary masks)

```bash
# LISA pseudo-labels
python student_segformer/prepare_lisa_pseudo_labels.py \
  --root /mnt/d/research_vl2g/goose_train_100 --threshold 50

# Grounded-SAM2 pseudo-labels
python student_segformer/prepare_gsam2_pseudo_labels.py \
  --root /mnt/d/research_vl2g/goose_train_100 --prompt-type road
```

These decode JSON/mask outputs from each teacher into `.png` binary masks at `<root>/pseudo_labels/`.

### 3. Student (SegFormer) training

```bash
python student_segformer/train_student_segformer.py \
  --train-img-dir /mnt/d/research_vl2g/goose_train_500/images \
  --train-mask-dir /mnt/d/research_vl2g/goose_train_500/pseudo_labels/lisa_traversable_thr50 \
  --val-img-dir /mnt/d/research_vl2g/goose_sample_56/images \
  --val-mask-dir /mnt/d/research_vl2g/goose_sample_56/gt_tasks/traversable_strict \
  --out-dir /mnt/d/research_vl2g/student_results/segformer_b0_lisa_pseudo_500 \
  --epochs 30 --batch-size 4 --lr 5e-5 --image-size 512
```

Uses SegFormer (MiT-B0) with 2-class semantic segmentation (background/traversable). Loss = CrossEntropy + Dice. Saves best model by validation IoU, outputs per-mask predictions to `pred_masks/`.

### 4. Analysis scripts (student_segformer/)

- `analyze_student_per_image.py` — per-sample IoU metrics for student models vs GT on the 56-image held-out set
- `analyze_lisa_pseudo_quality.py` — quality assessment of LISA pseudo-labels vs GT on training data
- `plot_student_results.py` / `plot_student_results_v2.py` — bar charts comparing student variants
- `plot_scale_trend.py` — line plot of IoU vs training data size (100→500)
- `make_student_visual_comparison.py` / `make_visual_comparison_100_500.py` — side-by-side overlay visualizations

## Repository structure conventions

- `images/` — 56 raw GOOSE off-road images (held-out evaluation set)
- `goose_sample_56/` — prepared 56-image set: images, labels, gt_tasks (GT binary masks per task), predictions, metrics
- `goose_train_100/` / `goose_train_500/` — training sets (100/500 images) with teacher predictions and pseudo_labels
- `student_results/segformer_b0_*/` — trained student checkpoints, validation predictions (`pred_masks/`), `final_metrics.json`, `metrics_log.jsonl`
- `frozen_results/` — frozen experiment snapshots with metrics
- `final_tables/` — exported CSVs for paper figures
- `final_figures/` — exported PNGs for paper figures

All scripts use hardcoded absolute paths under `/mnt/d/research_vl2g/` (WSL/Cygwin mount). Update these if running on a different machine.

## Category taxonomy

GOOSE class names are mapped to task masks: `TRAVERSABLE_NAMES` (asphalt, gravel, soil, etc.), `LOW_CONF_TRAVERSABLE_NAMES` (low_grass, moss, snow), `OBSTACLE_NAMES` (rocks, vehicles, persons, barriers, etc.), `WATER_RISK_NAMES` (water, mud, high_grass, bush, etc.), `VOID_NAMES` (ego_vehicle, sky, outlier). Derived masks: `traversable_strict`, `traversable_loose`, `obstacle`, `water_risk`, `non_traversable`, `unsafe_proxy` (union of obstacle + water_risk + non_traversable).

## Dependencies

Python packages: `torch`, `transformers`, `PIL`/`numpy`/`pandas`/`matplotlib`, `pycocotools` (for decoding GSAM2 JSON RLE masks). External models run in their own conda environments (see generated bash scripts): `localretro` for Grounded-SAM2, `lisa` for LISA.
