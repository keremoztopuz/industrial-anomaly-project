import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as F

def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class MVTecDataset(Dataset):
    """Load one MVTec category from FiftyOne metadata, without changing source files."""

    def __init__(self, root, category, split, image_size=256):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        self.root = Path(root).resolve()
        self.image_size = image_size
        with (self.root / "samples.json").open(encoding="utf-8") as file:
            samples = json.load(file)["samples"]
        if category not in {sample["category"]["label"] for sample in samples}:
            raise ValueError(f"Unknown category: {category}")
        self.samples = [
            sample for sample in samples
            if sample["category"]["label"] == category
            and sample["split"] == split
            and (split != "train" or sample["defect"]["label"] == "good")
        ]
        if not self.samples:
            raise ValueError(f"No usable samples for {category}/{split}")
        self.transform = build_transform(image_size)

    def __len__(self):
        return len(self.samples)

    def _path(self, relative_path):
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Path escapes dataset root: {relative_path}")
        return path

    def __getitem__(self, index):
        sample = self.samples[index]
        path = self._path(sample["filepath"])
        with Image.open(path) as source:
            original_size = source.size
            image = self.transform(source.convert("RGB"))

        label = int(sample["defect"]["label"] != "good")
        if label:
            mask_reference = sample.get("defect_mask", {}).get("mask_path")
            if not mask_reference:
                raise ValueError(f"Missing mask reference for {path}")
            mask_path = self._path(mask_reference)
            with Image.open(mask_path) as source:
                if source.size != original_size:
                    raise ValueError(f"Mask/image size mismatch: {mask_path} and {path}")
                mask = F.pil_to_tensor(F.resize(
                    source.convert("L"),
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.NEAREST,
                )) > 0
        else:
            mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.bool)
        return {"image": image, "mask": mask, "label": label, "path": str(path)}


def main():
    parser = argparse.ArgumentParser(description="Check the first preprocessed batch of each MVTec category.")
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/mvtec-ad")
    parser.add_argument("--category", help="Omit to check all categories")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    try:
        with (args.dataset_root / "samples.json").open(encoding="utf-8") as file:
            samples = json.load(file)["samples"]
        categories = [args.category] if args.category else sorted({s["category"]["label"] for s in samples})
        for category in categories:
            dataset = MVTecDataset(args.dataset_root, category, args.split, args.image_size)
            batch = next(iter(DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False)))
            print(f"{category}/{args.split}: {len(dataset)} samples | "
                  f"images {tuple(batch['image'].shape)} {batch['image'].dtype} | "
                  f"masks {tuple(batch['mask'].shape)} {batch['mask'].dtype}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
