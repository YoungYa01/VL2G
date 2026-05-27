import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    tensor = torch.from_numpy(arr)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    return (tensor - mean) / std


def build_model():
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
    except Exception:
        config = SegformerConfig(
            num_labels=2,
            depths=[2, 2, 2, 2],
            hidden_sizes=[32, 64, 160, 256],
            decoder_hidden_size=256,
        )
        model = SegformerForSemanticSegmentation(config)

    return model


@torch.no_grad()
def predict_prob(model, img_path, image_size, device):
    img = Image.open(img_path).convert("RGB")
    original_size = img.size

    resized = img.resize((image_size, image_size), Image.BILINEAR)
    tensor = pil_to_tensor(resized).unsqueeze(0).to(device)

    outputs = model(pixel_values=tensor)
    logits = outputs.logits
    logits = F.interpolate(
        logits,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
    prob_img = Image.fromarray((prob * 255).astype(np.uint8))
    prob_img = prob_img.resize(original_size, Image.BILINEAR)

    return np.array(prob_img).astype(np.float32) / 255.0


def load_gt(mask_path, size):
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(size, Image.NEAREST)
    return np.array(mask) > 127


def metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    iou = tp / union if union > 0 else 0.0
    dice = 2 * tp / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return iou, dice, precision, recall


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--gt-mask-dir", required=True)
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--vl2g-ckpt", required=True)
    parser.add_argument("--baseline-threshold", type=float, required=True)
    parser.add_argument("--vl2g-threshold", type=float, required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--image-size", type=int, default=512)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    baseline = build_model()
    baseline.load_state_dict(torch.load(args.baseline_ckpt, map_location=device))
    baseline.to(device)
    baseline.eval()

    vl2g = build_model()
    vl2g.load_state_dict(torch.load(args.vl2g_ckpt, map_location=device))
    vl2g.to(device)
    vl2g.eval()

    img_dir = Path(args.img_dir)
    gt_dir = Path(args.gt_mask_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    img_paths = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])

    for img_path in img_paths:
        sid = img_path.stem
        gt_path = gt_dir / f"{sid}.png"
        if not gt_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        gt = load_gt(gt_path, img.size)

        prob_base = predict_prob(baseline, img_path, args.image_size, device)
        prob_vl2g = predict_prob(vl2g, img_path, args.image_size, device)

        pred_base = prob_base > args.baseline_threshold
        pred_vl2g = prob_vl2g > args.vl2g_threshold

        b_iou, b_dice, b_prec, b_rec = metrics(pred_base, gt)
        v_iou, v_dice, v_prec, v_rec = metrics(pred_vl2g, gt)

        rows.append({
            "id": sid,
            "baseline_iou": b_iou,
            "vl2g_iou": v_iou,
            "delta_iou": v_iou - b_iou,
            "baseline_dice": b_dice,
            "vl2g_dice": v_dice,
            "delta_dice": v_dice - b_dice,
            "baseline_precision": b_prec,
            "vl2g_precision": v_prec,
            "baseline_recall": b_rec,
            "vl2g_recall": v_rec,
        })

    rows = sorted(rows, key=lambda x: x["delta_iou"], reverse=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "baseline_iou",
                "vl2g_iou",
                "delta_iou",
                "baseline_dice",
                "vl2g_dice",
                "delta_dice",
                "baseline_precision",
                "vl2g_precision",
                "baseline_recall",
                "vl2g_recall",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", out_csv)
    print("Top improved:")
    for r in rows[:10]:
        print(r["id"], "delta_iou=", round(r["delta_iou"], 4))

    print("Top worse:")
    for r in rows[-10:]:
        print(r["id"], "delta_iou=", round(r["delta_iou"], 4))


if __name__ == "__main__":
    main()