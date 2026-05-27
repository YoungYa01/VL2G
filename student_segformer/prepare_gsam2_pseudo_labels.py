from pathlib import Path
from PIL import Image
import numpy as np
import json
import pycocotools.mask as mask_util
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--prompt-type", default="road")
    args = parser.parse_args()

    root = Path(args.root)
    pred_root = root / f"preds/grounded_sam2/{args.prompt_type}"
    img_dir = root / "images"
    out_dir = root / f"pseudo_labels/grounded_sam2_{args.prompt_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    empty = 0

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        sample_id = img_path.stem
        json_path = pred_root / sample_id / "grounded_sam2_hf_model_demo_results.json"

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        mask = np.zeros((h, w), dtype=bool)

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for ann in data.get("annotations", []):
                rle = ann.get("segmentation")
                if rle is None:
                    continue
                m = mask_util.decode(rle).astype(bool)

                if m.shape != mask.shape:
                    m_img = Image.fromarray(m.astype(np.uint8) * 255)
                    m_img = m_img.resize((w, h), Image.NEAREST)
                    m = np.array(m_img) > 127

                mask |= m

        if mask.sum() == 0:
            empty += 1

        Image.fromarray(mask.astype(np.uint8) * 255).save(out_dir / f"{sample_id}.png")
        count += 1

    print("saved:", count)
    print("empty:", empty)
    print("output:", out_dir)

if __name__ == "__main__":
    main()
