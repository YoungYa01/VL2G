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
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=2,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    return model


@torch.no_grad()
def predict_prob(model, img_path, image_size, device):
    img = Image.open(img_path).convert("RGB")
    original_size = img.size

    resized = img.resize((image_size, image_size), Image.BILINEAR)
    x = pil_to_tensor(resized).unsqueeze(0).to(device)

    outputs = model(pixel_values=x)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--image-size", type=int, default=512)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    img_dir = Path(args.img_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        prob = predict_prob(model, img_path, args.image_size, device)
        pred = prob > args.threshold

        rows.append({
            "id": img_path.stem,
            "pred_area_ratio": float(pred.mean()),
            "mean_score": float(prob.mean()),
            "max_score": float(prob.max()),
        })

    ratios = np.array([r["pred_area_ratio"] for r in rows])
    scores = np.array([r["mean_score"] for r in rows])

    print("checkpoint:", args.checkpoint)
    print("threshold:", args.threshold)
    print("num:", len(rows))
    print("pred area mean:", ratios.mean())
    print("pred area median:", np.median(ratios))
    print("pred area p25:", np.percentile(ratios, 25))
    print("pred area p75:", np.percentile(ratios, 75))
    print("mean score:", scores.mean())

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "pred_area_ratio", "mean_score", "max_score"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("saved:", out_csv)


if __name__ == "__main__":
    main()
