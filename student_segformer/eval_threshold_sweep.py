import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    tensor = torch.from_numpy(arr)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    return (tensor - mean) / std


def load_mask(path: Path, image_size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    mask = mask.resize((image_size, image_size), Image.NEAREST)
    arr = np.array(mask)
    return torch.from_numpy((arr > 127).astype(np.int64))


class BinaryValDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_size=512):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        image_paths = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])

        self.items = []
        for img_path in image_paths:
            sid = img_path.stem
            mask_path = self.mask_dir / f"{sid}.png"
            if mask_path.exists():
                self.items.append((img_path, mask_path))

        if len(self.items) == 0:
            raise RuntimeError(f"No matched validation pairs found: {img_dir}, {mask_dir}")

        print(f"Val image dir: {self.img_dir}")
        print(f"Val mask dir: {self.mask_dir}")
        print(f"Matched val pairs: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]

        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        img_tensor = pil_to_tensor(img)
        mask = load_mask(mask_path, self.image_size)

        return img_tensor, mask, img_path.stem


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


def compute_metrics(pred, gt):
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


@torch.no_grad()
def collect_probs_and_gts(model, loader, device):
    model.eval()

    all_probs = []
    all_gts = []

    for imgs, masks, names in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        outputs = model(pixel_values=imgs)
        logits = outputs.logits
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        probs = torch.softmax(logits, dim=1)[:, 1]

        for i in range(probs.shape[0]):
            all_probs.append(probs[i].cpu().numpy())
            all_gts.append(masks[i].cpu().numpy())

    return all_probs, all_gts


def evaluate_thresholds(all_probs, all_gts, thresholds):
    rows = []

    for thr in thresholds:
        metrics = []

        for prob, gt in zip(all_probs, all_gts):
            pred = prob > thr
            iou, dice, precision, recall = compute_metrics(pred, gt)
            metrics.append([iou, dice, precision, recall])

        arr = np.array(metrics)

        rows.append({
            "threshold": thr,
            "iou": float(arr[:, 0].mean()),
            "dice": float(arr[:, 1].mean()),
            "precision": float(arr[:, 2].mean()),
            "recall": float(arr[:, 3].mean()),
        })

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-img-dir", required=True)
    parser.add_argument("--val-mask-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = build_model()
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    ds = BinaryValDataset(
        img_dir=args.val_img_dir,
        mask_dir=args.val_mask_dir,
        image_size=args.image_size,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    thresholds = [round(x, 2) for x in np.arange(0.1, 0.91, 0.05)]

    all_probs, all_gts = collect_probs_and_gts(model, loader, device)
    rows = evaluate_thresholds(all_probs, all_gts, thresholds)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "threshold_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threshold", "iou", "dice", "precision", "recall"],
        )
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda x: x["iou"])

    with open(out_dir / "best_threshold.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    print("Saved:", csv_path)
    print("Best threshold by IoU:")
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()