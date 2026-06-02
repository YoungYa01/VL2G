import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_binary(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return arr > 127


def resize_to(mask: np.ndarray, target_shape):
    h, w = target_shape
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((w, h), Image.NEAREST)
    return np.array(img) > 127


def save_binary(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traversable-dir", required=True)
    parser.add_argument("--risk-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    trav_dir = Path(args.traversable_dir)
    risk_dir = Path(args.risk_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "num": 0,
        "missing_risk": 0,
        "empty_before": 0,
        "empty_after": 0,
        "risk_area_before": [],
        "risk_area_after": [],
        "removed_area": [],
    }

    for trav_path in sorted(trav_dir.glob("*.png")):
        sid = trav_path.stem
        risk_path = risk_dir / f"{sid}.png"

        if not risk_path.exists():
            stats["missing_risk"] += 1
            continue

        trav = load_binary(trav_path)
        risk = load_binary(risk_path)

        if risk.shape != trav.shape:
            risk = resize_to(risk, trav.shape)

        clean = np.logical_and(risk, ~trav)

        stats["num"] += 1
        stats["risk_area_before"].append(float(risk.mean()))
        stats["risk_area_after"].append(float(clean.mean()))
        stats["removed_area"].append(float(np.logical_and(risk, trav).mean()))

        if risk.sum() == 0:
            stats["empty_before"] += 1
        if clean.sum() == 0:
            stats["empty_after"] += 1

        save_binary(clean, out_dir / f"{sid}.png")

    for key in ["risk_area_before", "risk_area_after", "removed_area"]:
        arr = np.array(stats[key], dtype=float)
        stats[key + "_mean"] = float(arr.mean()) if len(arr) else None
        stats[key + "_median"] = float(np.median(arr)) if len(arr) else None
        stats[key + "_p75"] = float(np.percentile(arr, 75)) if len(arr) else None
        stats[key + "_max"] = float(arr.max()) if len(arr) else None

    with open(out_dir / "clean_risk_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
