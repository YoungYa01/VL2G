import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
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


def load_mask(path: Path, image_size: int, long: bool = False) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    mask = mask.resize((image_size, image_size), Image.NEAREST)
    arr = np.array(mask)

    if long:
        return torch.from_numpy((arr > 127).astype(np.int64))

    return torch.from_numpy((arr > 127).astype(np.float32))


class VL2GTrainDataset(Dataset):
    """
    label_root structure:

    label_root/
      traversable/
        xxx.png
      boundary/
        xxx.png
      uncertainty/
        xxx.png
      risk/
        xxx.png
    """

    def __init__(self, img_dir, label_root, image_size=512):
        self.img_dir = Path(img_dir)
        self.label_root = Path(label_root)
        self.image_size = image_size

        self.trav_dir = self.label_root / "traversable"
        self.boundary_dir = self.label_root / "boundary"
        self.uncertainty_dir = self.label_root / "uncertainty"
        self.risk_dir = self.label_root / "risk"

        for d in [self.trav_dir, self.boundary_dir, self.uncertainty_dir, self.risk_dir]:
            if not d.exists():
                raise RuntimeError(f"Missing label directory: {d}")

        image_paths = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])

        self.items = []
        for img_path in image_paths:
            sid = img_path.stem
            paths = {
                "traversable": self.trav_dir / f"{sid}.png",
                "boundary": self.boundary_dir / f"{sid}.png",
                "uncertainty": self.uncertainty_dir / f"{sid}.png",
                "risk": self.risk_dir / f"{sid}.png",
            }

            if all(p.exists() for p in paths.values()):
                self.items.append((img_path, paths))

        if len(self.items) == 0:
            raise RuntimeError(f"No matched VL2G samples found: {img_dir}, {label_root}")

        print(f"Train image dir: {self.img_dir}")
        print(f"VL2G label root: {self.label_root}")
        print(f"Matched train pairs: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, paths = self.items[idx]

        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        img_tensor = pil_to_tensor(img)

        traversable = load_mask(paths["traversable"], self.image_size, long=True)
        boundary = load_mask(paths["boundary"], self.image_size, long=False)
        uncertainty = load_mask(paths["uncertainty"], self.image_size, long=False)
        risk = load_mask(paths["risk"], self.image_size, long=False)

        return {
            "image": img_tensor,
            "mask": traversable,
            "boundary": boundary,
            "uncertainty": uncertainty,
            "risk": risk,
            "name": img_path.stem,
        }


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

        mask = load_mask(mask_path, self.image_size, long=True)

        return img_tensor, mask, img_path.stem


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


def dice_loss(logits, target, eps=1e-6):
    probs = torch.softmax(logits, dim=1)[:, 1]
    target = target.float()

    inter = (probs * target).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def weighted_ce_loss(logits, target, boundary, uncertainty, boundary_weight=2.0, uncertainty_weight=1.0):
    """
    边界和不确定区域附近加权 CE。
    目的：让模型不要只学大面积道路内部，而是更重视边界。
    """
    ce = F.cross_entropy(logits, target, reduction="none")
    weight = 1.0 + boundary_weight * boundary + uncertainty_weight * uncertainty
    return (ce * weight).mean()


def soft_morphological_boundary(prob, kernel_size=3):
    """
    Differentiable-ish soft boundary from probability map.

    prob: [B, H, W]
    return: [B, H, W]
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    x = prob.unsqueeze(1)
    dilated = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    eroded = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    edge = (dilated - eroded).squeeze(1)
    return edge.clamp(0.0, 1.0)


def boundary_bce_loss(prob, boundary, eps=1e-6):
    """
    让预测概率图的边缘接近 teacher mask 生成的边界图。
    因为 boundary 像素很少，所以手动加一个 positive weight。
    """
    pred_boundary = soft_morphological_boundary(prob, kernel_size=3)

    target = boundary.float()
    pos = target.sum()
    neg = target.numel() - pos
    pos_weight = (neg / (pos + eps)).clamp(1.0, 20.0)

    pred_boundary = pred_boundary.clamp(eps, 1.0 - eps)

    loss_pos = -pos_weight * target * torch.log(pred_boundary)
    loss_neg = -(1.0 - target) * torch.log(1.0 - pred_boundary)

    return (loss_pos + loss_neg).mean()


def smoothness_loss(prob, boundary=None):
    """
    Total variation smoothness.
    目标：让可通行性 score map 不要碎片化。
    但边界处不强行平滑，否则会把边界抹掉。
    """
    dx = torch.abs(prob[:, :, 1:] - prob[:, :, :-1])
    dy = torch.abs(prob[:, 1:, :] - prob[:, :-1, :])

    if boundary is not None:
        bx = torch.maximum(boundary[:, :, 1:], boundary[:, :, :-1])
        by = torch.maximum(boundary[:, 1:, :], boundary[:, :-1, :])

        dx = dx * (1.0 - bx)
        dy = dy * (1.0 - by)

    return dx.mean() + dy.mean()


def risk_suppression_loss(prob, risk, eps=1e-6):
    """
    风险区域抑制项：
    risk 区域里，traversability score 应该低。
    如果当前 v1 risk 全 0，该 loss 自动接近 0。
    """
    numerator = (prob * risk).sum()
    denominator = risk.sum() + eps
    return numerator / denominator


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
def evaluate(model, loader, device, out_dir=None):
    model.eval()

    all_metrics = []

    pred_dir = None
    score_dir = None

    if out_dir is not None:
        out_dir = Path(out_dir)
        pred_dir = out_dir / "pred_masks"
        score_dir = out_dir / "score_maps"
        pred_dir.mkdir(parents=True, exist_ok=True)
        score_dir.mkdir(parents=True, exist_ok=True)

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
        preds = (probs > 0.5).long()

        for i in range(preds.shape[0]):
            pred_np = preds[i].cpu().numpy().astype(np.uint8)
            gt_np = masks[i].cpu().numpy().astype(np.uint8)

            iou, dice, precision, recall = compute_metrics(pred_np, gt_np)
            all_metrics.append([iou, dice, precision, recall])

            if out_dir is not None:
                Image.fromarray(pred_np * 255).save(pred_dir / f"{names[i]}.png")

                score_np = (probs[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(score_np).save(score_dir / f"{names[i]}.png")

    arr = np.array(all_metrics)

    return {
        "iou": float(arr[:, 0].mean()),
        "dice": float(arr[:, 1].mean()),
        "precision": float(arr[:, 2].mean()),
        "recall": float(arr[:, 3].mean()),
    }


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = VL2GTrainDataset(
        img_dir=args.train_img_dir,
        label_root=args.train_label_root,
        image_size=args.image_size,
    )

    val_ds = BinaryValDataset(
        img_dir=args.val_img_dir,
        mask_dir=args.val_mask_dir,
        image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_iou = -1.0

    config = vars(args)
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()

        total_loss = 0.0
        total_ce = 0.0
        total_dice = 0.0
        total_boundary = 0.0
        total_smooth = 0.0
        total_risk = 0.0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            boundary = batch["boundary"].to(device)
            uncertainty = batch["uncertainty"].to(device)
            risk = batch["risk"].to(device)

            outputs = model(pixel_values=imgs)
            logits = outputs.logits
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            probs = torch.softmax(logits, dim=1)[:, 1]

            loss_ce = weighted_ce_loss(
                logits=logits,
                target=masks,
                boundary=boundary,
                uncertainty=uncertainty,
                boundary_weight=args.boundary_ce_weight,
                uncertainty_weight=args.uncertainty_ce_weight,
            )

            loss_dice = dice_loss(logits, masks)
            loss_boundary = boundary_bce_loss(probs, boundary)
            loss_smooth = smoothness_loss(probs, boundary)
            loss_risk = risk_suppression_loss(probs, risk)

            loss = (
                loss_ce
                + args.lambda_dice * loss_dice
                + args.lambda_boundary * loss_boundary
                + args.lambda_smooth * loss_smooth
                + args.lambda_risk * loss_risk
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_dice += loss_dice.item()
            total_boundary += loss_boundary.item()
            total_smooth += loss_smooth.item()
            total_risk += loss_risk.item()

        n = max(len(train_loader), 1)

        train_log = {
            "epoch": epoch,
            "train_loss": total_loss / n,
            "loss_ce": total_ce / n,
            "loss_dice": total_dice / n,
            "loss_boundary": total_boundary / n,
            "loss_smooth": total_smooth / n,
            "loss_risk": total_risk / n,
        }

        val_metrics = evaluate(model, val_loader, device)

        log_record = {
            **train_log,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "time_sec": time.time() - start,
        }

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_log['train_loss']:.4f} | "
            f"ce={train_log['loss_ce']:.4f} | "
            f"dice={train_log['loss_dice']:.4f} | "
            f"bd={train_log['loss_boundary']:.4f} | "
            f"sm={train_log['loss_smooth']:.4f} | "
            f"risk={train_log['loss_risk']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"time={log_record['time_sec']:.1f}s"
        )

        with open(out_dir / "metrics_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print("Saved best model:", out_dir / "best_model.pt")

    print("Loading best model for final prediction...")
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))

    final_metrics = evaluate(model, val_loader, device, out_dir=out_dir)

    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    print("Final metrics:")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-img-dir", required=True)
    parser.add_argument("--train-label-root", required=True)
    parser.add_argument("--val-img-dir", required=True)
    parser.add_argument("--val-mask-dir", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--lambda-dice", type=float, default=1.0)
    parser.add_argument("--lambda-boundary", type=float, default=0.2)
    parser.add_argument("--lambda-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-risk", type=float, default=0.2)

    parser.add_argument("--boundary-ce-weight", type=float, default=2.0)
    parser.add_argument("--uncertainty-ce-weight", type=float, default=1.0)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()