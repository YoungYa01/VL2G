import argparse
import csv
import random
import shutil
from pathlib import Path


def is_image_file(p: Path):
    return p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def get_base_id_from_image_stem(stem: str):
    """
    GOOSE image:
      xxx_windshield_vis
      xxx_windshield_nir

    We need base:
      xxx
    """
    if stem.endswith("_windshield_vis"):
        return stem[: -len("_windshield_vis")]
    if stem.endswith("_windshield_nir"):
        return stem[: -len("_windshield_nir")]
    return stem


def get_base_id_from_label_stem(stem: str):
    """
    GOOSE label:
      xxx_color
      xxx_instanceids
      xxx_labelids

    We need base:
      xxx
    """
    for suffix in ["_labelids", "_color", "_instanceids"]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def list_vis_images(source_img_dir: Path):
    return sorted([
        p for p in source_img_dir.rglob("*_windshield_vis.png")
        if is_image_file(p)
    ])


def list_labelids(source_label_dir: Path):
    return sorted([
        p for p in source_label_dir.rglob("*_labelids.png")
        if is_image_file(p)
    ])


def image_stems_from_flat_dir(image_dir: Path):
    if not image_dir.exists():
        return set()

    stems = set()
    for p in image_dir.iterdir():
        if is_image_file(p):
            stems.add(p.stem)
    return stems


def base_ids_from_flat_image_dir(image_dir: Path):
    """
    For existing goose_train_500/images.
    If files are original GOOSE names:
      xxx_windshield_vis -> xxx
    If files were renamed:
      goose_001 -> goose_001
    """
    return {
        get_base_id_from_image_stem(stem)
        for stem in image_stems_from_flat_dir(image_dir)
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source-root", required=True)
    parser.add_argument("--existing-train-root", required=True)
    parser.add_argument("--exclude-test-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--target-num", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    source_root = Path(args.source_root)
    existing_train_root = Path(args.existing_train_root)
    exclude_test_root = Path(args.exclude_test_root)
    out_root = Path(args.out_root)

    source_img_dir = source_root / "images"
    source_label_dir = source_root / "labels"

    existing_img_dir = existing_train_root / "images"
    test_img_dir = exclude_test_root / "images"

    out_img_dir = out_root / "images"
    out_label_dir = out_root / "labels"

    vis_images = list_vis_images(source_img_dir)
    labelids = list_labelids(source_label_dir)

    image_by_base = {
        get_base_id_from_image_stem(p.stem): p
        for p in vis_images
    }

    label_by_base = {
        get_base_id_from_label_stem(p.stem): p
        for p in labelids
    }

    existing_base_ids = base_ids_from_flat_image_dir(existing_img_dir)
    test_base_ids = base_ids_from_flat_image_dir(test_img_dir)

    print("source vis images:", len(vis_images))
    print("source labelids:", len(labelids))
    print("image base ids:", len(image_by_base))
    print("label base ids:", len(label_by_base))
    print("matched image-label base ids:", len(set(image_by_base) & set(label_by_base)))
    print("existing train base ids:", len(existing_base_ids))
    print("test base ids:", len(test_base_ids))

    # Existing 500 can only be inherited if base ids match original GOOSE names.
    valid_existing = sorted([
        bid for bid in existing_base_ids
        if bid in image_by_base and bid in label_by_base
    ])

    print("valid existing inherited:", len(valid_existing))

    candidate_base_ids = sorted([
        bid for bid in image_by_base.keys()
        if bid in label_by_base
        and bid not in set(valid_existing)
        and bid not in test_base_ids
    ])

    need_extra = args.target_num - len(valid_existing)

    if need_extra < 0:
        selected_base_ids = valid_existing[: args.target_num]
    else:
        if len(candidate_base_ids) < need_extra:
            raise RuntimeError(
                f"Not enough candidates. Need extra {need_extra}, got {len(candidate_base_ids)}"
            )

        random.seed(args.seed)
        extra = random.sample(candidate_base_ids, need_extra)
        selected_base_ids = sorted(valid_existing + extra)

    if len(selected_base_ids) != args.target_num:
        raise RuntimeError(f"Expected {args.target_num}, got {len(selected_base_ids)}")

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for bid in selected_base_ids:
        img_src = image_by_base[bid]
        label_src = label_by_base[bid]

        # Keep original image filename.
        img_dst = out_img_dir / img_src.name

        # Save label with image stem name, so later scripts can match image stem directly.
        # image: xxx_windshield_vis.png
        # label: xxx_windshield_vis.png
        label_dst = out_label_dir / img_src.name

        shutil.copy2(img_src, img_dst)
        shutil.copy2(label_src, label_dst)

        rows.append({
            "base_id": bid,
            "image_name": img_dst.name,
            "label_name": label_dst.name,
            "from_existing_500": int(bid in valid_existing),
            "source_image": str(img_src),
            "source_labelids": str(label_src),
        })

    # Create project-style dirs
    for d in [
        "lisa_inputs",
        "logs",
        "preds/lisa/traversable",
        "preds/grounded_sam2/road",
        "preds/grounded_sam2/obstacle",
        "preds/grounded_sam2/water_risk",
        "prompts",
        "pseudo_labels",
        "scripts",
        "gt_tasks",
        "gt_vis",
    ]:
        (out_root / d).mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "base_id",
                "image_name",
                "label_name",
                "from_existing_500",
                "source_image",
                "source_labelids",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    selected_set = set(selected_base_ids)
    overlap_test = selected_set & test_base_ids

    print("Saved:", out_root)
    print("selected:", len(selected_base_ids))
    print("from existing 500:", sum(r["from_existing_500"] for r in rows))
    print("new samples:", len(selected_base_ids) - sum(r["from_existing_500"] for r in rows))
    print("test overlap:", len(overlap_test))
    print("manifest:", manifest_path)

    if overlap_test:
        print("Overlap examples:", sorted(list(overlap_test))[:20])
        raise RuntimeError("Train/test overlap detected!")


if __name__ == "__main__":
    main()
