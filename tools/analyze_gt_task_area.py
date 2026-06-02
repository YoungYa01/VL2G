import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    mask_dir = Path(args.mask_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for p in sorted(mask_dir.glob("*.png")):
        arr = np.array(Image.open(p).convert("L"))
        mask = arr > 127
        rows.append({
            "id": p.stem,
            "area_ratio": float(mask.mean()),
            "pixels": int(mask.sum()),
            "empty": int(mask.sum() == 0),
        })

    ratios = np.array([r["area_ratio"] for r in rows], dtype=float)
    non_empty = ratios[ratios > 0]

    print("mask_dir:", mask_dir)
    print("num:", len(rows))
    print("empty:", sum(r["empty"] for r in rows))
    print("non_empty:", len(non_empty))

    if len(non_empty) > 0:
        print("mean area all:", ratios.mean())
        print("mean area non-empty:", non_empty.mean())
        print("median non-empty:", np.median(non_empty))
        print("p75 non-empty:", np.percentile(non_empty, 75))
        print("max:", ratios.max())
    else:
        print("No non-empty masks.")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "area_ratio", "pixels", "empty"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("saved:", out_csv)


if __name__ == "__main__":
    main()
