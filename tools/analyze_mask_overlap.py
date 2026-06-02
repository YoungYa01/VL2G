import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path: Path):
    arr = np.array(Image.open(path).convert("L"))
    return arr > 127


def resize_mask_to(mask: np.ndarray, target_shape):
    """
    Resize binary mask to target shape.

    target_shape: (H, W)
    """
    target_h, target_w = target_shape
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((target_w, target_h), Image.NEAREST)
    return np.array(img) > 127


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-a-dir", required=True)
    parser.add_argument("--mask-b-dir", required=True)
    parser.add_argument("--name-a", default="a")
    parser.add_argument("--name-b", default="b")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument(
        "--resize-b-to-a",
        action="store_true",
        help="Resize mask B to mask A resolution when shapes differ.",
    )

    args = parser.parse_args()

    a_dir = Path(args.mask_a_dir)
    b_dir = Path(args.mask_b_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    resized_count = 0
    skipped_shape_mismatch = 0

    for a_path in sorted(a_dir.glob("*.png")):
        sid = a_path.stem
        b_path = b_dir / f"{sid}.png"
        if not b_path.exists():
            continue

        a = load_mask(a_path)
        b = load_mask(b_path)

        if a.shape != b.shape:
            if args.resize_b_to_a:
                b = resize_mask_to(b, a.shape)
                resized_count += 1
            else:
                skipped_shape_mismatch += 1
                continue

        inter = np.logical_and(a, b)
        union = np.logical_or(a, b)

        row = {
            "id": sid,
            f"{args.name_a}_area": float(a.mean()),
            f"{args.name_b}_area": float(b.mean()),
            "overlap_area": float(inter.mean()),
            "overlap_over_a": float(inter.sum() / (a.sum() + 1e-6)),
            "overlap_over_b": float(inter.sum() / (b.sum() + 1e-6)),
            "iou": float(inter.sum() / (union.sum() + 1e-6)),
        }
        rows.append(row)

    keys = [
        "id",
        f"{args.name_a}_area",
        f"{args.name_b}_area",
        "overlap_area",
        "overlap_over_a",
        "overlap_over_b",
        "iou",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print("num matched:", len(rows))
    print("resized b to a:", resized_count)
    print("skipped shape mismatch:", skipped_shape_mismatch)

    if len(rows) == 0:
        print("No matched masks. Please check file names.")
        return

    arr_overlap = np.array([r["overlap_area"] for r in rows], dtype=float)
    arr_over_a = np.array([r["overlap_over_a"] for r in rows], dtype=float)
    arr_over_b = np.array([r["overlap_over_b"] for r in rows], dtype=float)
    arr_iou = np.array([r["iou"] for r in rows], dtype=float)

    print("mean overlap area:", float(arr_overlap.mean()))
    print("mean overlap over a:", float(arr_over_a.mean()))
    print("mean overlap over b:", float(arr_over_b.mean()))
    print("mean iou:", float(arr_iou.mean()))

    print("p75 overlap over a:", float(np.percentile(arr_over_a, 75)))
    print("p90 overlap over a:", float(np.percentile(arr_over_a, 90)))
    print("max overlap over a:", float(arr_over_a.max()))

    print("saved:", out_csv)


if __name__ == "__main__":
    main()
