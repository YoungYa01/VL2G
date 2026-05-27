from pathlib import Path
from PIL import Image
import numpy as np
import shutil

ROOT = Path("/mnt/d/research_vl2g/goose_train_100")  # 后面可以改成 goose_train_500
PRED_DIR = ROOT / "preds/lisa/traversable"
OUT_DIR = ROOT / "pseudo_labels/lisa_traversable_thr50"
OUT_DIR.mkdir(parents=True, exist_ok=True)

threshold = 50

count = 0
empty = 0

for mask_path in sorted(PRED_DIR.glob("*_mask_0.*")):
    sample_id = mask_path.name.split("_mask_0")[0]
    arr = np.array(Image.open(mask_path).convert("L"))
    bin_mask = (arr > threshold).astype(np.uint8) * 255

    if bin_mask.sum() == 0:
        empty += 1

    Image.fromarray(bin_mask).save(OUT_DIR / f"{sample_id}.png")
    count += 1

print("saved:", count)
print("empty masks:", empty)
print("output:", OUT_DIR)
