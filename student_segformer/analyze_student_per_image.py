from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

gt_dir = Path("/mnt/d/research_vl2g/goose_sample_56/gt_tasks/traversable_strict")

models = {
    "student_gs100": Path("/mnt/d/research_vl2g/student_results/segformer_b0_gsam2_road_100/pred_masks"),
    "student_lisa100": Path("/mnt/d/research_vl2g/student_results/segformer_b0_lisa_pseudo_100/pred_masks"),
    "student_lisa500": Path("/mnt/d/research_vl2g/student_results/segformer_b0_lisa_pseudo_500/pred_masks"),
    "student_gt500": Path("/mnt/d/research_vl2g/student_results/segformer_b0_goose_gt_500/pred_masks"),
}

out_dir = Path("/mnt/d/research_vl2g/final_tables")
out_dir.mkdir(parents=True, exist_ok=True)

def load_mask(path):
    return np.array(Image.open(path).convert("L")) > 127

def metrics(pred, gt):
    if pred.shape != gt.shape:
        pred_img = Image.fromarray(pred.astype(np.uint8) * 255)
        pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
        pred = np.array(pred_img) > 127

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    inter = tp
    union = np.logical_or(pred, gt).sum()

    iou = inter / union if union > 0 else 0
    dice = 2 * tp / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    pred_area = pred.mean()
    gt_area = gt.mean()

    return iou, dice, precision, recall, pred_area, gt_area

rows = []

for gt_path in sorted(gt_dir.glob("*.png")):
    sample_id = gt_path.stem
    gt = load_mask(gt_path)

    for model_name, pred_dir in models.items():
        pred_path = pred_dir / f"{sample_id}.png"
        if not pred_path.exists():
            continue

        pred = load_mask(pred_path)
        iou, dice, precision, recall, pred_area, gt_area = metrics(pred, gt)

        rows.append({
            "sample_id": sample_id,
            "model": model_name,
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "pred_area": pred_area,
            "gt_area": gt_area,
            "area_ratio": pred_area / gt_area if gt_area > 0 else 0,
        })

df = pd.DataFrame(rows)
out_path = out_dir / "student_per_image_metrics.csv"
df.to_csv(out_path, index=False)

print("saved:", out_path)

for model in models:
    sub = df[df["model"] == model].sort_values("iou")
    print("\nMODEL:", model)
    print("worst 5:")
    print(sub.head(5)[["sample_id", "iou", "precision", "recall", "area_ratio"]].to_string(index=False))
    print("best 5:")
    print(sub.tail(5)[["sample_id", "iou", "precision", "recall", "area_ratio"]].to_string(index=False))
