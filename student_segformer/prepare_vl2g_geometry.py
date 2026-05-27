import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def ensure_odd(k: int) -> int:
    if k < 1:
        return 1
    return k if k % 2 == 1 else k + 1


def load_binary_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return arr > 127


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (mask.astype(np.uint8) * 255)
    Image.fromarray(out).save(path)


def pil_morph(mask: np.ndarray, k: int, mode: str) -> np.ndarray:
    """
    mode:
      - max: dilation
      - min: erosion
    """
    k = ensure_odd(k)
    img = Image.fromarray(mask.astype(np.uint8) * 255)

    if mode == "max":
        filtered = img.filter(ImageFilter.MaxFilter(k))
    elif mode == "min":
        filtered = img.filter(ImageFilter.MinFilter(k))
    else:
        raise ValueError(f"Unknown morph mode: {mode}")

    return np.array(filtered) > 127


def make_boundary(mask: np.ndarray, width: int) -> np.ndarray:
    """
    Morphological gradient:
      boundary = dilation(mask) XOR erosion(mask)

    width 越大，边界区域越宽。
    """
    width = ensure_odd(width)
    dilated = pil_morph(mask, width, "max")
    eroded = pil_morph(mask, width, "min")
    boundary = np.logical_xor(dilated, eroded)
    return boundary


def make_uncertainty(boundary: np.ndarray, width: int) -> np.ndarray:
    """
    在边界周围膨胀出一圈 uncertainty 区域。
    这个区域训练时会被加权，让模型重点学习边界附近。
    """
    width = ensure_odd(width)
    return pil_morph(boundary, width, "max")


def find_risk_mask(risk_dir: Path | None, sample_id: str, shape: tuple[int, int]) -> np.ndarray:
    """
    v1 阶段 risk 可以没有。
    后面你可以用 Grounded-SAM2 / LISA / 其他 VLM prompt 生成：
      pothole, ditch, puddle, obstacle, blocked area, muddy unsafe region
    然后放进 risk_dir。
    """
    if risk_dir is None:
        return np.zeros(shape, dtype=bool)

    candidates = [
        risk_dir / f"{sample_id}.png",
        risk_dir / f"{sample_id}_mask_0.png",
        risk_dir / f"{sample_id}_mask_0.jpg",
        risk_dir / f"{sample_id}_mask_0.jpeg",
    ]

    for path in candidates:
        if path.exists():
            risk = load_binary_mask(path)
            if risk.shape != shape:
                risk_img = Image.fromarray(risk.astype(np.uint8) * 255)
                risk_img = risk_img.resize((shape[1], shape[0]), Image.NEAREST)
                risk = np.array(risk_img) > 127
            return risk

    return np.zeros(shape, dtype=bool)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mask-dir",
        required=True,
        help="Existing traversable pseudo label directory, e.g. goose_train_500/pseudo_labels/lisa_traversable_thr50",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output VL2G geometry label root, e.g. goose_train_500/pseudo_labels/vl2g_lisa_thr50",
    )
    parser.add_argument(
        "--risk-dir",
        default=None,
        help="Optional risk mask directory. If omitted, risk masks are all zeros.",
    )
    parser.add_argument("--boundary-width", type=int, default=7)
    parser.add_argument("--uncertainty-width", type=int, default=15)

    args = parser.parse_args()

    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    risk_dir = Path(args.risk_dir) if args.risk_dir else None

    traversable_out = out_dir / "traversable"
    boundary_out = out_dir / "boundary"
    uncertainty_out = out_dir / "uncertainty"
    risk_out = out_dir / "risk"

    for d in [traversable_out, boundary_out, uncertainty_out, risk_out]:
        d.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(mask_dir.glob("*.png"))
    if len(mask_paths) == 0:
        raise RuntimeError(f"No masks found in {mask_dir}")

    stats = {
        "mask_dir": str(mask_dir),
        "risk_dir": str(risk_dir) if risk_dir else None,
        "out_dir": str(out_dir),
        "boundary_width": args.boundary_width,
        "uncertainty_width": args.uncertainty_width,
        "num_samples": len(mask_paths),
        "empty_traversable": 0,
        "empty_risk": 0,
    }

    for mask_path in mask_paths:
        sample_id = mask_path.stem

        traversable = load_binary_mask(mask_path)
        boundary = make_boundary(traversable, args.boundary_width)
        uncertainty = make_uncertainty(boundary, args.uncertainty_width)
        risk = find_risk_mask(risk_dir, sample_id, traversable.shape)

        if traversable.sum() == 0:
            stats["empty_traversable"] += 1
        if risk.sum() == 0:
            stats["empty_risk"] += 1

        save_binary_mask(traversable, traversable_out / f"{sample_id}.png")
        save_binary_mask(boundary, boundary_out / f"{sample_id}.png")
        save_binary_mask(uncertainty, uncertainty_out / f"{sample_id}.png")
        save_binary_mask(risk, risk_out / f"{sample_id}.png")

    with open(out_dir / "geometry_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Saved VL2G geometry labels to:", out_dir)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()