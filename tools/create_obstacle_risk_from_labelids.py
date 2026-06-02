import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


# Obstacle-like hard risk classes from GOOSE label mapping.
# Exclude water to avoid confusing broad water_risk proxy.
OBSTACLE_RISK_IDS = {
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
    36,  # caravan
    37,  # trailer
    40,  # rock
    41,  # fence
    42,  # guard_rail
    48,  # barrier_tape
    49,  # kick_scooter
    57,  # heavy_machinery
    58,  # container
    60,  # barrel
    61,  # pipe
    63,  # military_vehicle
}


def load_labelids(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int64)


def save_mask(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


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

    stats = {
        "num_images": 0,
        "num_written": 0,
        "num_missing_label": 0,
        "num_empty_risk": 0,
        "risk_ids": sorted(list(OBSTACLE_RISK_IDS)),
        "area_ratios": [],
    }

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]:
            continue

        stats["num_images"] += 1
        sid = img_path.stem
        label_path = label_dir / f"{sid}.png"

        if not label_path.exists():
            stats["num_missing_label"] += 1
            continue

        label = load_labelids(label_path)
        risk = np.isin(label, list(OBSTACLE_RISK_IDS))

        if risk.sum() == 0:
            stats["num_empty_risk"] += 1

        stats["area_ratios"].append(float(risk.mean()))
        save_mask(risk, out_dir / f"{sid}.png")
        stats["num_written"] += 1

    arr = np.array(stats["area_ratios"], dtype=float)
    stats["area_mean"] = float(arr.mean()) if len(arr) else None
    stats["area_median"] = float(np.median(arr)) if len(arr) else None
    stats["area_p75"] = float(np.percentile(arr, 75)) if len(arr) else None
    stats["area_max"] = float(arr.max()) if len(arr) else None

    with open(out_dir / "obstacle_risk_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
