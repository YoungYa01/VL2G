import argparse
import csv
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pycocotools.mask as mask_util
except Exception:
    mask_util = None


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


# 这些类别来自 GOOSE class definitions，可根据你自己的机器人/车辆设定微调
TRAVERSABLE_NAMES = {
    "asphalt", "cobble", "gravel", "soil",
    "bikeway", "sidewalk", "pedestrian_crossing",
}

LOW_CONF_TRAVERSABLE_NAMES = {
    "low_grass", "moss", "snow",
}

OBSTACLE_NAMES = {
    "animal", "person", "rider",
    "obstacle", "rock", "barrel", "pipe",
    "debris", "fence", "guard_rail", "wall", "wire",
    "traffic_cone", "road_block", "boom_barrier",
    "tree_trunk", "tree_root",
    "bicycle", "bus", "car", "caravan", "heavy_machinery",
    "kick_scooter", "motorcycle", "on_rails", "trailer", "truck",
}

WATER_RISK_NAMES = {
    "water", "snow", "high_grass", "bush", "crops",
    "forest", "hedge", "tree_root", "rock",
}

VOID_NAMES = {
    "ego_vehicle", "outlier", "undefined", "sky",
}


GSAM_PROMPTS = [
    {
        "prompt_type": "road",
        "prompt": "road. dirt road. path. gravel. soil.",
        "target_gt": "traversable_strict",
    },
    {
        "prompt_type": "obstacle",
        "prompt": "rock. obstacle. vehicle. person. tree. barrier.",
        "target_gt": "obstacle",
    },
    {
        "prompt_type": "water_risk",
        "prompt": "water. puddle. mud. snow. high grass. rock.",
        "target_gt": "water_risk",
    },
]

LISA_PROMPTS = [
    {
        "prompt_type": "traversable",
        "prompt": "Please segment the traversable road area in this image and explain why.",
        "target_gt": "traversable_strict",
    },
    {
        "prompt_type": "obstacle",
        "prompt": "Please segment the obstacle blocking the road.",
        "target_gt": "obstacle",
    },
    {
        "prompt_type": "water_risk",
        "prompt": "Please segment the muddy or waterlogged region that may affect driving.",
        "target_gt": "water_risk",
    },
    {
        "prompt_type": "unsafe_proxy",
        "prompt": "Please segment the area that is unsafe for a vehicle to drive through and explain why.",
        "target_gt": "unsafe_proxy",
    },
]


def read_csv_mapping(mapping_path: Path):
    """
    尽量兼容不同版本的 goose_label_mapping.csv。
    输出：
    - id_to_name: {int_id: class_name}
    - color_to_id: {(r,g,b): int_id}，如果 CSV 有颜色列
    """
    if not mapping_path.exists():
        raise FileNotFoundError(f"Cannot find mapping file: {mapping_path}")

    rows = []
    with mapping_path.open("r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"Empty mapping csv: {mapping_path}")

    columns = list(rows[0].keys())
    lower_cols = {c.lower(): c for c in columns}

    id_col = None
    for key in ["id", "label_id", "class_id", "train_id", "classid"]:
        if key in lower_cols:
            id_col = lower_cols[key]
            break

    name_col = None
    for key in ["name", "class", "class_name", "label", "label_name"]:
        if key in lower_cols:
            name_col = lower_cols[key]
            break

    # fallback：找第一个能转 int 的列作为 id，找第一个非数字字符串列作为 name
    if id_col is None:
        for c in columns:
            ok = True
            for r in rows[:20]:
                try:
                    int(str(r[c]).strip())
                except Exception:
                    ok = False
                    break
            if ok:
                id_col = c
                break

    if name_col is None:
        for c in columns:
            if c == id_col:
                continue
            vals = [str(r[c]).strip() for r in rows[:20]]
            if any(v and not v.replace("_", "").replace("-", "").isdigit() for v in vals):
                name_col = c
                break

    if id_col is None or name_col is None:
        raise RuntimeError(
            "Cannot infer id/name columns from goose_label_mapping.csv. "
            f"Columns are: {columns}"
        )

    # 尝试推断 RGB 颜色列
    color_candidates = []
    for triplet in [
        ("r", "g", "b"),
        ("red", "green", "blue"),
        ("color_r", "color_g", "color_b"),
        ("rgb_r", "rgb_g", "rgb_b"),
    ]:
        if all(k in lower_cols for k in triplet):
            color_candidates = [lower_cols[k] for k in triplet]
            break

    id_to_name = {}
    color_to_id = {}

    for r in rows:
        try:
            cid = int(str(r[id_col]).strip())
        except Exception:
            continue

        name = str(r[name_col]).strip()
        name = name.replace(" ", "_").replace("-", "_").lower()
        id_to_name[cid] = name

        if color_candidates:
            try:
                rgb = tuple(int(str(r[c]).strip()) for c in color_candidates)
                color_to_id[rgb] = cid
            except Exception:
                pass

    return id_to_name, color_to_id, columns, id_col, name_col


def find_mapping_file(goose_root: Path):
    candidates = list(goose_root.rglob("goose_label_mapping.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find goose_label_mapping.csv under {goose_root}"
        )
    return candidates[0]


def find_image_label_pairs(goose_root: Path, splits=("train", "val")):
    """
    Fast GOOSE pairing.

    GOOSE image names usually contain camera/sensor suffixes such as:
        xxx_windshield_vis.png

    The official example derives the label basename by removing the last two
    underscore-separated fields from the image stem, then checking:
        base_color.png
        base_instanceids.png
        base_labelids.png

    For semantic segmentation evaluation, we use:
        base_labelids.<ext>
    """
    pairs = []

    for split in splits:
        img_root = goose_root / "images" / split
        lab_root = goose_root / "labels" / split

        if not img_root.exists() or not lab_root.exists():
            print(f"[WARN] skip split={split}, missing {img_root} or {lab_root}")
            continue

        print(f"[INFO] indexing labelids for split={split} ...")

        labelids_by_stem = {}
        for lab in lab_root.rglob("*"):
            if lab.suffix.lower() not in IMAGE_EXTS:
                continue
            if "labelids" not in lab.stem.lower():
                continue
            labelids_by_stem[lab.stem] = lab

        print(f"[INFO] labelids indexed for split={split}: {len(labelids_by_stem)}")

        img_count = 0
        pair_count = 0

        for img in img_root.rglob("*"):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue

            img_count += 1
            stem = img.stem
            parts = stem.split("_")

            candidates = []

            # GOOSE official-style rule:
            # remove last two tokens, then append _labelids
            if len(parts) > 2:
                base = "_".join(parts[:-2])
                candidates.append(base + "_labelids")

            # fallback rules for variants
            candidates.append(stem.replace("_windshield_vis", "_labelids"))
            candidates.append(stem.replace("_vis", "_labelids"))
            candidates.append(stem + "_labelids")

            label = None
            for cand in candidates:
                if cand in labelids_by_stem:
                    label = labelids_by_stem[cand]
                    break

            if label is not None:
                pairs.append((split, img, label))
                pair_count += 1

            if img_count % 1000 == 0:
                print(f"[INFO] split={split}: scanned images={img_count}, pairs={pair_count}")

        print(f"[INFO] split={split}: total images scanned={img_count}, pairs={pair_count}")

    if not pairs:
        raise RuntimeError(
            "No image-label pairs found. This usually means the image/label naming "
            "rule differs from the expected GOOSE pattern. Please print a few image "
            "and label filenames and inspect their suffixes."
        )

    return pairs

def load_label_as_id_mask(label_path: Path, color_to_id):
    arr = np.array(Image.open(label_path))

    # 单通道 ID mask
    if arr.ndim == 2:
        return arr.astype(np.int32)

    # RGB/RGBA color mask
    if arr.ndim == 3:
        rgb = arr[:, :, :3]
        h, w, _ = rgb.shape
        out = np.full((h, w), 255, dtype=np.int32)

        if not color_to_id:
            raise RuntimeError(
                f"{label_path} looks like RGB label, but mapping CSV has no RGB columns."
            )

        for color, cid in color_to_id.items():
            mask = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
            out[mask] = cid

        return out

    raise RuntimeError(f"Unsupported label shape {arr.shape}: {label_path}")


def make_task_masks(label_id, id_to_name):
    name_to_ids = {}
    for cid, name in id_to_name.items():
        name_to_ids.setdefault(name, []).append(cid)

    def ids_for(names):
        ids = []
        for n in names:
            ids.extend(name_to_ids.get(n, []))
        return ids

    traversable_ids = ids_for(TRAVERSABLE_NAMES)
    low_conf_ids = ids_for(LOW_CONF_TRAVERSABLE_NAMES)
    obstacle_ids = ids_for(OBSTACLE_NAMES)
    water_risk_ids = ids_for(WATER_RISK_NAMES)
    void_ids = ids_for(VOID_NAMES)

    traversable_strict = np.isin(label_id, traversable_ids)
    traversable_loose = np.isin(label_id, traversable_ids + low_conf_ids)
    obstacle = np.isin(label_id, obstacle_ids)
    water_risk = np.isin(label_id, water_risk_ids)
    void = np.isin(label_id, void_ids)

    valid = ~void
    non_traversable = valid & (~traversable_strict)

    # unsafe_proxy 是可自动评价的“危险/不可通行代理标签”，不是严格人工风险标签
    unsafe_proxy = obstacle | water_risk | non_traversable

    return {
        "traversable_strict": traversable_strict,
        "traversable_loose": traversable_loose,
        "obstacle": obstacle,
        "water_risk": water_risk,
        "non_traversable": non_traversable,
        "unsafe_proxy": unsafe_proxy,
        "valid": valid,
    }


def save_binary_mask(mask, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def overlay_mask(image_path, mask, out_path):
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img).astype(np.float32)

    mask = mask.astype(bool)

    # GOOSE image and label resolution may differ.
    # Resize mask to image size only for visualization.
    if mask.shape != img_arr.shape[:2]:
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
        mask_img = mask_img.resize((img_arr.shape[1], img_arr.shape[0]), Image.NEAREST)
        mask = np.array(mask_img) > 127

    overlay = img_arr.copy()
    overlay[mask] = overlay[mask] * 0.45 + np.array([255, 0, 0]) * 0.55

    out = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def cmd_prepare(args):
    goose_root = Path(args.goose_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    random.seed(args.seed)

    out_images = out_root / "images"
    out_labels = out_root / "labels"
    out_gt = out_root / "gt_tasks"
    out_vis = out_root / "gt_vis"
    out_scripts = out_root / "scripts"
    out_prompts = out_root / "prompts"

    for p in [out_images, out_labels, out_gt, out_vis, out_scripts, out_prompts]:
        p.mkdir(parents=True, exist_ok=True)

    mapping_path = find_mapping_file(goose_root)
    id_to_name, color_to_id, columns, id_col, name_col = read_csv_mapping(mapping_path)

    print(f"[INFO] mapping: {mapping_path}")
    print(f"[INFO] inferred id column: {id_col}, name column: {name_col}")
    print(f"[INFO] classes found: {len(id_to_name)}")

    pairs = find_image_label_pairs(goose_root, splits=tuple(args.splits.split(",")))
    print(f"[INFO] image-label pairs found: {len(pairs)}")

    n = min(args.num, len(pairs))
    sampled = random.sample(pairs, n)

    shutil.copy2(mapping_path, out_root / "goose_label_mapping.csv")

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "split", "src_image", "src_label",
            "image_path", "label_path"
        ])

        for i, (split, img, label) in enumerate(sampled, start=1):
            sample_id = f"goose_{i:03d}"
            img_ext = img.suffix.lower()
            lab_ext = label.suffix.lower()

            dst_img = out_images / f"{sample_id}{img_ext}"
            dst_lab = out_labels / f"{sample_id}{lab_ext}"

            shutil.copy2(img, dst_img)
            shutil.copy2(label, dst_lab)

            label_id = load_label_as_id_mask(dst_lab, color_to_id)
            task_masks = make_task_masks(label_id, id_to_name)

            for task, mask in task_masks.items():
                save_binary_mask(mask, out_gt / task / f"{sample_id}.png")

            # 可视化几个关键 GT
            for task in ["traversable_strict", "obstacle", "water_risk", "unsafe_proxy"]:
                overlay_mask(dst_img, task_masks[task], out_vis / task / f"{sample_id}.jpg")

            writer.writerow([
                sample_id, split, str(img), str(label), str(dst_img), str(dst_lab)
            ])

    write_prompt_files(out_prompts)
    write_gsam_script(out_root, out_scripts)
    write_lisa_scripts(out_root, out_scripts)
    write_eval_script(out_root, out_scripts)

    print(f"[DONE] sampled {n} image-label pairs")
    print(f"[DONE] output root: {out_root}")
    print(f"[NEXT] Grounded-SAM2: bash {out_scripts / 'run_grounded_sam2_batch.sh'}")
    print(f"[NEXT] LISA prompt 1: bash {out_scripts / 'run_lisa_prompt_1.sh'}")
    print(f"[NEXT] evaluate: python {out_scripts / 'evaluate_predictions.py'}")


def write_prompt_files(out_prompts: Path):
    with (out_prompts / "grounded_sam2_prompts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_type", "prompt", "target_gt"])
        writer.writeheader()
        writer.writerows(GSAM_PROMPTS)

    with (out_prompts / "lisa_prompts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_type", "prompt", "target_gt"])
        writer.writeheader()
        writer.writerows(LISA_PROMPTS)


def write_gsam_script(out_root: Path, out_scripts: Path):
    script = f"""#!/usr/bin/env bash
set -e

conda activate localretro
cd ~/projects/Grounded-SAM-2

OUT_ROOT="{out_root}"
IMG_DIR="$OUT_ROOT/images"
PRED_ROOT="$OUT_ROOT/preds/grounded_sam2"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$PRED_ROOT" "$LOG_DIR"

run_one_prompt () {{
  PROMPT_TYPE="$1"
  PROMPT="$2"

  for img in "$IMG_DIR"/*; do
    [ -f "$img" ] || continue
    base=$(basename "$img")
    sample="${{base%.*}}"
    out="$PRED_ROOT/$PROMPT_TYPE/$sample"
    mkdir -p "$out"

    echo "[GSAM2] $sample | $PROMPT_TYPE | $PROMPT"

    python grounded_sam2_hf_model_demo.py \\
      --img-path "$img" \\
      --text-prompt "$PROMPT" \\
      --output-dir "$out" \\
      >> "$LOG_DIR/grounded_sam2_${{PROMPT_TYPE}}.log" 2>&1
  done
}}

run_one_prompt "road" "road. dirt road. path. gravel. soil."
run_one_prompt "obstacle" "rock. obstacle. vehicle. person. tree. barrier."
run_one_prompt "water_risk" "water. puddle. mud. snow. high grass. rock."

echo "[DONE] Grounded-SAM2 batch finished."
"""
    path = out_scripts / "run_grounded_sam2_batch.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def write_lisa_scripts(out_root: Path, out_scripts: Path):
    lisa_input_dir = out_root / "lisa_inputs"
    lisa_input_dir.mkdir(parents=True, exist_ok=True)

    img_dir = out_root / "images"
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    for idx, item in enumerate(LISA_PROMPTS, start=1):
        prompt_type = item["prompt_type"]
        prompt = item["prompt"]
        input_file = lisa_input_dir / f"lisa_prompt_{idx}_{prompt_type}.txt"

        with input_file.open("w", encoding="utf-8") as f:
            for img in images:
                f.write(prompt + "\n")
                f.write(str(img) + "\n")

        script = f"""#!/usr/bin/env bash
set -e

conda activate lisa
cd ~/projects/LISA

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT_ROOT="{out_root}"
mkdir -p "$OUT_ROOT/preds/lisa/{prompt_type}" "$OUT_ROOT/logs"

echo "[LISA] Running prompt {idx}: {prompt_type}"

CUDA_VISIBLE_DEVICES=0 python chat.py \\
  --version='xinlai/LISA-13B-llama2-v1' \\
  --precision='fp16' \\
  --load_in_4bit \\
  --vis_save_path "$OUT_ROOT/preds/lisa/{prompt_type}" \\
  < "{input_file}" \\
  | tee "$OUT_ROOT/logs/lisa_{idx}_{prompt_type}.log"

echo "[DONE] LISA prompt {idx} finished."
"""
        script_path = out_scripts / f"run_lisa_prompt_{idx}.sh"
        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o755)

    run_all = "#!/usr/bin/env bash\nset -e\n"
    for idx in range(1, len(LISA_PROMPTS) + 1):
        run_all += f'bash "{out_scripts / f"run_lisa_prompt_{idx}.sh"}"\n'
    run_all_path = out_scripts / "run_lisa_all_prompts.sh"
    run_all_path.write_text(run_all, encoding="utf-8")
    run_all_path.chmod(0o755)


def load_binary_mask(path: Path, threshold=127):
    arr = np.array(Image.open(path).convert("L"))
    return arr > threshold


def decode_gsam_json(json_path: Path):
    if mask_util is None:
        raise RuntimeError("pycocotools is required to decode Grounded-SAM2 JSON RLE masks.")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    h = int(data["img_height"])
    w = int(data["img_width"])
    pred = np.zeros((h, w), dtype=bool)

    for ann in data.get("annotations", []):
        rle = ann.get("segmentation")
        if rle is None:
            continue
        m = mask_util.decode(rle).astype(bool)
        pred |= m

    return pred


def metrics(pred, gt, valid=None):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if valid is not None:
        valid = valid.astype(bool)
        pred = pred & valid
        gt = gt & valid

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    inter = tp
    union = np.logical_or(pred, gt).sum()

    iou = inter / union if union > 0 else np.nan
    dice = 2 * tp / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    return {
        "iou": float(iou) if not np.isnan(iou) else "",
        "dice": float(dice) if not np.isnan(dice) else "",
        "precision": float(precision) if not np.isnan(precision) else "",
        "recall": float(recall) if not np.isnan(recall) else "",
        "pred_pixels": int(pred.sum()),
        "gt_pixels": int(gt.sum()),
    }


def write_eval_script(out_root: Path, out_scripts: Path):
    this_script = Path(__file__).resolve()
    # 生成一个薄包装脚本，实际调用本文件 evaluate 子命令
    script = f"""import subprocess
from pathlib import Path

subprocess.run([
    "python",
    "{this_script}",
    "evaluate",
    "--out-root",
    "{out_root}"
], check=True)
"""
    path = out_scripts / "evaluate_predictions.py"
    path.write_text(script, encoding="utf-8")


def cmd_evaluate(args):
    out_root = Path(args.out_root).expanduser().resolve()
    manifest_path = out_root / "manifest.csv"

    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    gt_root = out_root / "gt_tasks"
    valid_root = gt_root / "valid"

    eval_items = []

    # Grounded-SAM2 自动评估
    for item in GSAM_PROMPTS:
        model = "grounded_sam2"
        prompt_type = item["prompt_type"]
        target_gt = item["target_gt"]

        for r in rows:
            sid = r["sample_id"]
            json_path = out_root / "preds" / model / prompt_type / sid / "grounded_sam2_hf_model_demo_results.json"
            gt_path = gt_root / target_gt / f"{sid}.png"
            valid_path = valid_root / f"{sid}.png"

            if not json_path.exists() or not gt_path.exists():
                eval_items.append({
                    "sample_id": sid, "model": model, "prompt_type": prompt_type,
                    "target_gt": target_gt, "status": "missing_pred_or_gt"
                })
                continue

            pred = decode_gsam_json(json_path)
            gt = load_binary_mask(gt_path)
            valid = load_binary_mask(valid_path) if valid_path.exists() else None

            # 尺寸不一致时，将 pred resize 到 GT 尺寸
            if pred.shape != gt.shape:
                pred_img = Image.fromarray((pred.astype(np.uint8) * 255))
                pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
                pred = np.array(pred_img) > 127

            m = metrics(pred, gt, valid)
            eval_items.append({
                "sample_id": sid, "model": model, "prompt_type": prompt_type,
                "target_gt": target_gt, "status": "ok", **m
            })

    # LISA 自动评估
    for item in LISA_PROMPTS:
        model = "lisa"
        prompt_type = item["prompt_type"]
        target_gt = item["target_gt"]

        for r in rows:
            sid = r["sample_id"]
            pred_path = out_root / "preds" / model / prompt_type / f"{sid}_mask_0.jpg"
            if not pred_path.exists():
                pred_path = out_root / "preds" / model / prompt_type / f"{sid}_mask_0.png"

            gt_path = gt_root / target_gt / f"{sid}.png"
            valid_path = valid_root / f"{sid}.png"

            if not pred_path.exists() or not gt_path.exists():
                eval_items.append({
                    "sample_id": sid, "model": model, "prompt_type": prompt_type,
                    "target_gt": target_gt, "status": "missing_pred_or_gt"
                })
                continue

            pred = load_binary_mask(pred_path, threshold=50)
            gt = load_binary_mask(gt_path, threshold=127)
            valid = load_binary_mask(valid_path, threshold=127) if valid_path.exists() else None

            if pred.shape != gt.shape:
                pred_img = Image.fromarray((pred.astype(np.uint8) * 255))
                pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
                pred = np.array(pred_img) > 127

            m = metrics(pred, gt, valid)
            eval_items.append({
                "sample_id": sid, "model": model, "prompt_type": prompt_type,
                "target_gt": target_gt, "status": "ok", **m
            })

    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    detail_path = metrics_dir / "metrics_detail.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id", "model", "prompt_type", "target_gt", "status",
            "iou", "dice", "precision", "recall", "pred_pixels", "gt_pixels"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in eval_items:
            writer.writerow(item)

    # 汇总平均值
    summary = {}
    for item in eval_items:
        if item.get("status") != "ok":
            continue
        key = (item["model"], item["prompt_type"], item["target_gt"])
        summary.setdefault(key, {"count": 0, "iou": [], "dice": [], "precision": [], "recall": []})
        summary[key]["count"] += 1
        for k in ["iou", "dice", "precision", "recall"]:
            if item.get(k) != "":
                summary[key][k].append(float(item[k]))

    summary_path = metrics_dir / "metrics_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "prompt_type", "target_gt", "count", "mean_iou", "mean_dice", "mean_precision", "mean_recall"])
        for (model, prompt_type, target_gt), vals in sorted(summary.items()):
            writer.writerow([
                model, prompt_type, target_gt, vals["count"],
                np.mean(vals["iou"]) if vals["iou"] else "",
                np.mean(vals["dice"]) if vals["dice"] else "",
                np.mean(vals["precision"]) if vals["precision"] else "",
                np.mean(vals["recall"]) if vals["recall"] else "",
            ])

    print(f"[DONE] detail: {detail_path}")
    print(f"[DONE] summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("prepare")
    p1.add_argument("--goose-root", required=True, help="Path to goose-dataset root")
    p1.add_argument("--out-root", default="/mnt/d/research_vl2g/goose_sample")
    p1.add_argument("--num", type=int, default=56)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--splits", default="train,val")
    p1.set_defaults(func=cmd_prepare)

    p2 = sub.add_parser("evaluate")
    p2.add_argument("--out-root", default="/mnt/d/research_vl2g/goose_sample")
    p2.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
