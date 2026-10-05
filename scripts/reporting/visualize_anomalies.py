"""Render deterministic MVTec test examples beside masks and anomaly overlays.

Run from the repository root: python -m scripts.reporting.visualize_anomalies
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from anomaly.data.mvtec_dataset import MVTecDataset
from anomaly.model.patchcore import PatchCore
from anomaly.settings import settings


def render_category(dataset_root, output_root, category, image_size, device):
    bank_path = output_root / "patchcore" / f"{category}.pt"
    if not bank_path.is_file():
        raise FileNotFoundError(f"Missing memory bank: {bank_path}")
    model = PatchCore.load(bank_path, device=device)
    if image_size is not None and image_size != model.image_size:
        raise ValueError("--image-size must match the loaded model image_size")
    image_size = model.image_size
    dataset = MVTecDataset(dataset_root, category, "test", image_size)
    good = sorted(
        (
            i
            for i, sample in enumerate(dataset.samples)
            if sample["defect"]["label"] == "good"
        ),
        key=lambda i: dataset.samples[i]["filepath"],
    )
    defective = sorted(
        (
            i
            for i, sample in enumerate(dataset.samples)
            if sample["defect"]["label"] != "good"
        ),
        key=lambda i: dataset.samples[i]["filepath"],
    )
    if len(good) < 2 or len(defective) < 2:
        raise ValueError(
            f"{category}: need at least two good and two defective test "
            f"samples"
        )

    samples = [dataset[i] for i in good[:2] + defective[:2]]
    images = torch.stack([sample["image"] for sample in samples])
    scores, maps = model.predict(images)
    low, high = maps.min().item(), maps.max().item()

    canvas = Image.new(
        "RGB", (3 * image_size, 4 * (image_size + 24) + 24), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(("Image", "Ground truth", "Anomaly overlay")):
        draw.text((col * image_size + 5, 5), title, fill="black")
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    for row, sample in enumerate(samples):
        y = 24 + row * (image_size + 24)
        image = (sample["image"] * std + mean).clamp(0, 1) * 255
        rgb = Image.fromarray(image.permute(1, 2, 0).byte().numpy(), "RGB")
        mask = Image.fromarray(sample["mask"][0].byte().numpy() * 255, "L")
        intensity = (
            ((maps[row] - low) / (high - low)).clamp(0, 1)
            if high > low
            else torch.zeros_like(maps[row])
        )
        alpha = Image.fromarray((intensity * 170).byte().numpy(), "L")
        overlay = Image.composite(
            Image.new("RGB", rgb.size, "red"), rgb, alpha
        )
        for col, panel in enumerate((rgb, mask, overlay)):
            canvas.paste(panel, (col * image_size, y))
        label = "good" if sample["label"] == 0 else "defective"
        draw.text(
            (5, y + image_size + 4),
            f"{label}: {Path(sample['path']).name}  "
            f"score={scores[row].item():.3f}",
            fill="black",
        )

    destination = output_root / "visualizations" / f"{category}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=settings.dataset_root
    )
    parser.add_argument(
        "--output-root", type=Path, default=settings.artifacts_root
    )
    parser.add_argument(
        "--category", help="Render one category; default is all categories"
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.image_size is not None and args.image_size <= 0:
        parser.error("--image-size must be positive")
    with (args.dataset_root / "samples.json").open(encoding="utf-8") as file:
        samples = json.load(file)["samples"]
    categories = (
        [args.category]
        if args.category
        else sorted({sample["category"]["label"] for sample in samples})
    )
    if not categories or any(
        not category.isidentifier() for category in categories
    ):
        parser.error("category names must be nonempty, safe path components")
    for category in categories:
        print(
            render_category(
                args.dataset_root,
                args.output_root,
                category,
                args.image_size,
                args.device,
            )
        )


if __name__ == "__main__":
    main()
