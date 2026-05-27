from pathlib import Path
from PIL import Image
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="goose_train_100 or goose_train_500 root")
    parser.add_argument("--threshold", type=int, default=50)
    args = parser.parse_args()

    root = Path(args.root)
    pred_dir = root / "preds/lisa/traversable"
    out_dir = root / f"pseudo_labels/lisa_traversable_thr{args.threshold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = sorted(pred_dir.glob("*_mask_0.*"))
    print("found masks:", len(masks))

    empty = 0

    for mask_path in masks:
        sample_id = mask_path.name.split("_mask_0")[0]
        arr = np.array(Image.open(mask_path).convert("L"))

        bin_mask = (arr > args.threshold).astype(np.uint8) * 255

        if bin_mask.sum() == 0:
            empty += 1

        Image.fromarray(bin_mask).save(out_dir / f"{sample_id}.png")

    print("saved:", len(masks))
    print("empty:", empty)
    print("output:", out_dir)

if __name__ == "__main__":
    main()
