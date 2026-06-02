import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


# Based on goose_label_mapping.csv
# Clear obstacle / blocking / risky classes.
HARD_RISK_IDS = {
    1,   # traffic_cone
    4,   # obstacle
    10,  # road_block
    12,  # car
    13,  # bicycle
    14,  # person
    15,  # bus
    20,  # motorcycle
    25,  # boom_barrier
    29,  # debris
    32,  # rider
    33,  # animal
    34,  # truck
    35,  # on_rails
    36,  # caravan
    37,  # trailer
    40,  # rock
    41,  # fence
    42,  # guard_rail
    48,  # barrier_tape
    49,  # kick_scooter
    54,  # water
    57,  # heavy_machinery
    58,  # container
    60,  # barrel
    61,  # pipe
    63,  # military_vehicle
}


def load_labelids(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))

    # Labelids should usually be single-channel.
    # If it is accidentally RGB/RGBA, use the first channel.
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    return arr.astype(np.int64)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
    ])

    stats = {
        "num_images": len(image_paths),
        "num_written": 0,
        "num_missing_label": 0,
        "num_empty_risk": 0,
        "risk_ids": sorted(list(HARD_RISK_IDS)),
        "area_ratios": [],
    }

    for img_path in image_paths:
        sid = img_path.stem

        # In our curated subsets, labels should be renamed to match images.
        label_path = label_dir / f"{sid}.png"

        if not label_path.exists():
            stats["num_missing_label"] += 1
            continue

        label = load_labelids(label_path)
        risk = np.isin(label, list(HARD_RISK_IDS))

        if risk.sum() == 0:
            stats["num_empty_risk"] += 1

        stats["area_ratios"].append(float(risk.mean()))

        save_mask(risk, out_dir / f"{sid}.png")
        stats["num_written"] += 1

    ratios = np.array(stats["area_ratios"], dtype=float)

    if len(ratios) > 0:
        stats["area_mean"] = float(ratios.mean())
        stats["area_median"] = float(np.median(ratios))
        stats["area_p75"] = float(np.percentile(ratios, 75))
        stats["area_max"] = float(ratios.max())
    else:
        stats["area_mean"] = None
        stats["area_median"] = None
        stats["area_p75"] = None
        stats["area_max"] = None

    with open(out_dir / "risk_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
