import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    tensor = torch.from_numpy(arr)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    return (tensor - mean) / std


def build_model():
    return SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=2,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )


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


def load_binary_mask(path, size):
    if not path.exists():
        return None
    mask = Image.open(path).convert("L")
    mask = mask.resize(size, Image.NEAREST)
    return np.array(mask) > 127


def safe_mean(arr, mask):
    if mask is None or mask.sum() == 0:
        return None
    return float(arr[mask].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--safe-mask-dir", required=True)
    parser.add_argument("--risk-mask-dirs", nargs="+", required=True)
    parser.add_argument("--risk-names", nargs="+", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--image-size", type=int, default=512)

    args = parser.parse_args()

    if len(args.risk_mask_dirs) != len(args.risk_names):
        raise ValueError("risk-mask-dirs and risk-names must have the same length.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    img_dir = Path(args.img_dir)
    safe_dir = Path(args.safe_mask_dir)
    risk_dirs = [Path(x) for x in args.risk_mask_dirs]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        sid = img_path.stem
        img = Image.open(img_path).convert("RGB")
        size = img.size

        prob = predict_prob(model, img_path, args.image_size, device)

        safe_mask = load_binary_mask(safe_dir / f"{sid}.png", size)
        safe_score = safe_mean(prob, safe_mask)

        row = {
            "id": sid,
            "safe_score": safe_score if safe_score is not None else "",
            "mean_score": float(prob.mean()),
        }

        for risk_name, risk_dir in zip(args.risk_names, risk_dirs):
            risk_mask = load_binary_mask(risk_dir / f"{sid}.png", size)
            risk_score = safe_mean(prob, risk_mask)

            if risk_score is None:
                row[f"{risk_name}_score"] = ""
                row[f"{risk_name}_gap"] = ""
            else:
                row[f"{risk_name}_score"] = risk_score
                row[f"{risk_name}_gap"] = (
                    safe_score - risk_score if safe_score is not None else ""
                )

        rows.append(row)

    fieldnames = ["id", "safe_score", "mean_score"]
    for risk_name in args.risk_names:
        fieldnames += [f"{risk_name}_score", f"{risk_name}_gap"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("saved:", out_csv)

    # Summary
    print("\nSummary:")

    def col_float(col):
        vals = []
        for r in rows:
            v = r.get(col, "")
            if v != "":
                vals.append(float(v))
        return np.array(vals, dtype=float)

    safe_vals = col_float("safe_score")
    print("safe_score mean:", safe_vals.mean() if len(safe_vals) else "NA")

    for risk_name in args.risk_names:
        risk_vals = col_float(f"{risk_name}_score")
        gap_vals = col_float(f"{risk_name}_gap")

        print(f"{risk_name}_score mean:", risk_vals.mean() if len(risk_vals) else "NA")
        print(f"{risk_name}_gap mean:", gap_vals.mean() if len(gap_vals) else "NA")


if __name__ == "__main__":
    main()
