from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

root = Path("/mnt/d/research_vl2g/goose_train_500")

pseudo_dir = root / "pseudo_labels/lisa_traversable_thr50"
gt_dir = root / "gt_tasks/traversable_strict"

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

    return iou, dice, precision, recall, pred.mean(), gt.mean()

rows = []

for pseudo_path in sorted(pseudo_dir.glob("*.png")):
    sample_id = pseudo_path.stem
    gt_path = gt_dir / f"{sample_id}.png"

    if not gt_path.exists():
        continue

    pseudo = load_mask(pseudo_path)
    gt = load_mask(gt_path)

    iou, dice, precision, recall, pseudo_area, gt_area = metrics(pseudo, gt)

    rows.append({
        "sample_id": sample_id,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "pseudo_area": pseudo_area,
        "gt_area": gt_area,
        "area_ratio": pseudo_area / gt_area if gt_area > 0 else 0,
    })

df = pd.DataFrame(rows)
out_path = out_dir / "lisa_pseudo_quality_train500.csv"
df.to_csv(out_path, index=False)

print("saved:", out_path)
print("mean:")
print(df[["iou", "dice", "precision", "recall", "area_ratio"]].mean())

print("\nworst 10:")
print(df.sort_values("iou").head(10)[["sample_id", "iou", "precision", "recall", "area_ratio"]].to_string(index=False))

print("\nbest 10:")
print(df.sort_values("iou").tail(10)[["sample_id", "iou", "precision", "recall", "area_ratio"]].to_string(index=False))
