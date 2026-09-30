import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def inspect_image(path):
    try:
        with Image.open(path) as image:
            image.load()
            return image.size, image.mode, None
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return None, None, str(error)


def validate_dataset(root):
    root = Path(root).resolve()
    with (root / "samples.json").open(encoding="utf-8") as file:
        samples = json.load(file)["samples"]

    categories = {}
    errors = []
    for sample in samples:
        category = sample["category"]["label"]
        stats = categories.setdefault(category, {"count": 0, "sizes": Counter(), "modes": Counter()})
        stats["count"] += 1

        image_path = (root / sample["filepath"]).resolve()
        try:
            image_path.relative_to(root)
        except ValueError:
            errors.append(f"{category}: image path escapes dataset root: {sample['filepath']}")
            continue
        if not image_path.is_file():
            errors.append(f"{category}: missing image: {image_path}")
            continue
        size, mode, error = inspect_image(image_path)
        if error:
            errors.append(f"{category}: corrupt image {image_path}: {error}")
            continue
        stats["sizes"][size] += 1
        stats["modes"][mode] += 1

        if sample["defect"]["label"] == "good":
            continue
        mask = sample.get("defect_mask", {}).get("mask_path")
        if not mask:
            errors.append(f"{category}: missing mask reference for {image_path}")
            continue
        mask_path = (root / mask).resolve()
        try:
            mask_path.relative_to(root)
        except ValueError:
            errors.append(f"{category}: mask path escapes dataset root: {mask}")
            continue
        if not mask_path.is_file():
            errors.append(f"{category}: missing mask: {mask_path}")
            continue
        mask_size, _, error = inspect_image(mask_path)
        if error:
            errors.append(f"{category}: corrupt mask {mask_path}: {error}")
        elif mask_size != size:
            errors.append(f"{category}: mask/image size mismatch: {mask_path} {mask_size} != {image_path} {size}")

    return len(samples), categories, errors


def main():
    parser = argparse.ArgumentParser(description="Decode and validate MVTec images and defect masks.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/mvtec-ad",
    )
    args = parser.parse_args()
    total, categories, errors = validate_dataset(args.dataset_root)
    print(f"Images: {total} | Categories: {len(categories)} | Errors: {len(errors)}")
    for category, stats in sorted(categories.items()):
        sizes = ", ".join(f"{w}x{h}: {n}" for (w, h), n in sorted(stats["sizes"].items())) or "none"
        modes = ", ".join(f"{mode}: {n}" for mode, n in sorted(stats["modes"].items())) or "none"
        print(f"{category}: {stats['count']} images | sizes [{sizes}] | modes [{modes}]")
    for error in errors:
        print(f"ERROR: {error}")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
