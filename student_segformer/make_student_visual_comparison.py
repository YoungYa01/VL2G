from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import random

image_dir = Path("/mnt/d/research_vl2g/goose_sample_56/images")
gt_dir = Path("/mnt/d/research_vl2g/goose_sample_56/gt_tasks/traversable_strict")

gs_dir = Path("/mnt/d/research_vl2g/student_results/segformer_b0_gsam2_road_100/pred_masks")
lisa_dir = Path("/mnt/d/research_vl2g/student_results/segformer_b0_lisa_pseudo_100/pred_masks")
gt_student_dir = Path("/mnt/d/research_vl2g/student_results/segformer_b0_goose_gt_100/pred_masks")

out_dir = Path("/mnt/d/research_vl2g/final_figures/student_visual_cases")
out_dir.mkdir(parents=True, exist_ok=True)

case_ids = [p.stem for p in sorted(image_dir.iterdir()) if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

# 先随机选 8 张；后面你可以手动改成你觉得典型的样本
random.seed(42)
case_ids = random.sample(case_ids, min(8, len(case_ids)))

def load_rgb(path, size=(320, 240)):
    img = Image.open(path).convert("RGB")
    return img.resize(size)

def load_mask_as_overlay(img_path, mask_path, size=(320, 240)):
    img = Image.open(img_path).convert("RGB").resize(size)

    if not mask_path.exists():
        return img

    mask = Image.open(mask_path).convert("L").resize(size, Image.NEAREST)
    img_arr = np.array(img).astype(np.float32)
    mask_arr = np.array(mask) > 127

    overlay = img_arr.copy()
    overlay[mask_arr] = overlay[mask_arr] * 0.45 + np.array([255, 0, 0]) * 0.55

    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

def find_image(sample_id):
    for ext in [".png", ".jpg", ".jpeg"]:
        p = image_dir / f"{sample_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(sample_id)

for sample_id in case_ids:
    img_path = find_image(sample_id)

    panels = [
        ("Image", load_rgb(img_path)),
        ("GOOSE GT", load_mask_as_overlay(img_path, gt_dir / f"{sample_id}.png")),
        ("Student-GS", load_mask_as_overlay(img_path, gs_dir / f"{sample_id}.png")),
        ("Student-LISA", load_mask_as_overlay(img_path, lisa_dir / f"{sample_id}.png")),
        ("Student-GT", load_mask_as_overlay(img_path, gt_student_dir / f"{sample_id}.png")),
    ]

    w, h = panels[0][1].size
    title_h = 36
    canvas = Image.new("RGB", (w * len(panels), h + title_h), "white")
    draw = ImageDraw.Draw(canvas)

    for i, (title, panel) in enumerate(panels):
        x = i * w
        canvas.paste(panel, (x, title_h))
        draw.text((x + 10, 10), title, fill=(0, 0, 0))

    out_path = out_dir / f"{sample_id}_comparison.jpg"
    canvas.save(out_path)

print("saved to:", out_dir)
