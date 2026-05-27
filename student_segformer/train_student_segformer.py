import argparse
from pathlib import Path
import time
import json

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import SegformerForSemanticSegmentation, SegformerConfig


IMAGE_SIZE = 512


def pil_to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    tensor = torch.from_numpy(arr)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    return (tensor - mean) / std


class BinarySegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_size=512):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        self.images = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])

        self.items = []
        for img_path in self.images:
            sample_id = img_path.stem
            mask_path = self.mask_dir / f"{sample_id}.png"
            if mask_path.exists():
                self.items.append((img_path, mask_path))

        if len(self.items) == 0:
            raise RuntimeError(f"No matched image-mask pairs found: {img_dir}, {mask_dir}")

        print(f"Dataset: {img_dir}")
        print(f"Matched pairs: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        img_tensor = pil_to_tensor(img)
        mask_arr = np.array(mask)
        mask_tensor = torch.from_numpy((mask_arr > 127).astype(np.int64))

        return img_tensor, mask_tensor, img_path.stem


def dice_loss(logits, target, eps=1e-6):
    probs = torch.softmax(logits, dim=1)[:, 1]
    target = target.float()

    inter = (probs * target).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def compute_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    inter = tp
    union = np.logical_or(pred, gt).sum()

    iou = inter / union if union > 0 else 0.0
    dice = 2 * tp / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return iou, dice, precision, recall


def build_model():
    try:
        print("Trying to load pretrained nvidia/mit-b0...")
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        print("Loaded pretrained nvidia/mit-b0")
    except Exception as e:
        print("Failed to load pretrained nvidia/mit-b0. Use random init.")
        print("Reason:", repr(e))

        config = SegformerConfig(
            num_labels=2,
            depths=[2, 2, 2, 2],
            hidden_sizes=[32, 64, 160, 256],
            decoder_hidden_size=256,
        )
        model = SegformerForSemanticSegmentation(config)

    return model


@torch.no_grad()
def evaluate(model, loader, device, out_dir=None):
    model.eval()

    all_metrics = []

    if out_dir is not None:
        out_dir = Path(out_dir)
        pred_dir = out_dir / "pred_masks"
        pred_dir.mkdir(parents=True, exist_ok=True)

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

        preds = torch.argmax(logits, dim=1)

        for i in range(preds.shape[0]):
            pred_np = preds[i].cpu().numpy().astype(np.uint8)
            gt_np = masks[i].cpu().numpy().astype(np.uint8)

            iou, dice, precision, recall = compute_metrics(pred_np, gt_np)
            all_metrics.append([iou, dice, precision, recall])

            if out_dir is not None:
                Image.fromarray(pred_np * 255).save(pred_dir / f"{names[i]}.png")

    arr = np.array(all_metrics)
    mean_metrics = {
        "iou": float(arr[:, 0].mean()),
        "dice": float(arr[:, 1].mean()),
        "precision": float(arr[:, 2].mean()),
        "recall": float(arr[:, 3].mean()),
    }

    return mean_metrics


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = BinarySegDataset(args.train_img_dir, args.train_mask_dir, args.image_size)
    val_ds = BinarySegDataset(args.val_img_dir, args.val_mask_dir, args.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = build_model()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss()

    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start = time.time()

        for imgs, masks, _ in train_loader:
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

            loss_ce = ce_loss(logits, masks)
            loss_dice = dice_loss(logits, masks)
            loss = loss_ce + loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)

        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"time={time.time() - start:.1f}s"
        )

        with open(out_dir / "metrics_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train_loss": train_loss,
                **val_metrics,
            }, ensure_ascii=False) + "\n")

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print("Saved best model:", out_dir / "best_model.pt")

    print("Loading best model for final prediction...")
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    final_metrics = evaluate(model, val_loader, device, out_dir=out_dir)

    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print("Final metrics:")
    print(final_metrics)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-img-dir", required=True)
    parser.add_argument("--train-mask-dir", required=True)
    parser.add_argument("--val-img-dir", required=True)
    parser.add_argument("--val-mask-dir", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--image-size", type=int, default=512)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
