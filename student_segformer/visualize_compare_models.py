import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

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
def predict_prob(model, image_path, image_size, device):
    img = Image.open(image_path).convert("RGB")
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


def load_mask(mask_path, size):
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(size, Image.NEAREST)
    return np.array(mask) > 127


def mask_to_rgb(mask, color):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[mask] = color
    return Image.fromarray(out)


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.45):
    img = image.convert("RGB")
    overlay = mask_to_rgb(mask, color).convert("RGB")
    return Image.blend(img, overlay, alpha)


def score_to_heatmap(prob):
    """
    简单热力图，不依赖 opencv/matplotlib。
    低分偏暗，高分偏亮。
    """
    p = np.clip(prob, 0, 1)
    r = (p * 255).astype(np.uint8)
    g = (np.sqrt(p) * 255).astype(np.uint8)
    b = ((1 - p) * 80).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1))


def add_title(img, title, height=36):
    img = img.convert("RGB")
    canvas = Image.new("RGB", (img.width, img.height + height), (255, 255, 255))
    canvas.paste(img, (0, height))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), title, fill=(0, 0, 0))

    return canvas


def make_grid(images, cols=3, pad=8):
    widths = [im.width for im in images]
    heights = [im.height for im in images]

    w = max(widths)
    h = max(heights)

    rows = int(np.ceil(len(images) / cols))

    canvas = Image.new(
        "RGB",
        (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad),
        (245, 245, 245),
    )

    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        canvas.paste(im.resize((w, h)), (x, y))

    return canvas


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--gt-mask-dir", required=True)
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--vl2g-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--baseline-threshold", type=float, default=0.9)
    parser.add_argument("--vl2g-threshold", type=float, default=0.85)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=20)

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
    gt_mask_dir = Path(args.gt_mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])

    saved = 0

    for img_path in img_paths:
        sid = img_path.stem
        gt_path = gt_mask_dir / f"{sid}.png"

        if not gt_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        gt = load_mask(gt_path, image.size)

        prob_base = predict_prob(baseline, img_path, args.image_size, device)
        prob_vl2g = predict_prob(vl2g, img_path, args.image_size, device)

        pred_base = prob_base > args.baseline_threshold
        pred_vl2g = prob_vl2g > args.vl2g_threshold

        panels = [
            add_title(image, "Image"),
            add_title(overlay_mask(image, gt, color=(0, 255, 0)), "GT traversable"),
            add_title(overlay_mask(image, pred_base, color=(255, 0, 0)), "Pseudo-only pred"),
            add_title(overlay_mask(image, pred_vl2g, color=(0, 0, 255)), "VL2G pred"),
            add_title(score_to_heatmap(prob_base), "Pseudo-only score"),
            add_title(score_to_heatmap(prob_vl2g), "VL2G score"),
        ]

        grid = make_grid(panels, cols=3)
        grid.save(out_dir / f"{sid}_compare.png")

        saved += 1
        if saved >= args.max_samples:
            break

    print("Saved visualizations:", saved)
    print("Output dir:", out_dir)


if __name__ == "__main__":
    main()