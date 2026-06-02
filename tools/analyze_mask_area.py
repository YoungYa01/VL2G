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
        area_ratio = float(mask.mean())

        rows.append({
            "id": p.stem,
            "area_ratio": area_ratio,
            "empty": int(mask.sum() == 0),
        })

    ratios = np.array([r["area_ratio"] for r in rows], dtype=float)

    print("mask_dir:", mask_dir)
    print("num:", len(rows))
    print("empty:", sum(r["empty"] for r in rows))
    print("mean:", ratios.mean())
    print("std:", ratios.std())
    print("min:", ratios.min())
    print("p25:", np.percentile(ratios, 25))
    print("median:", np.percentile(ratios, 50))
    print("p75:", np.percentile(ratios, 75))
    print("max:", ratios.max())

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "area_ratio", "empty"])
        writer.writeheader()
        writer.writerows(rows)

    print("saved:", out_csv)


if __name__ == "__main__":
    main()
