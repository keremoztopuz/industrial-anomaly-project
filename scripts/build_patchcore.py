"""Build one normal patch memory bank per MVTec category.

Run from the repository root: python -m scripts.build_patchcore
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mvtec_dataset import MVTecDataset
from patchcore import PatchCore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/mvtec-ad"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/patchcore"))
    parser.add_argument("--category", help="Build only this category; default is all categories")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-patches", type=int, default=2048)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection", choices=("random", "coreset"), default="random")
    parser.add_argument("--neighborhood", type=int, default=1,
                        help="Odd feature-averaging window; PatchCore uses 3")
    parser.add_argument("--border", type=int, default=0,
                        help="Outer patch rings ignored by image scores")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    with (args.dataset_root / "samples.json").open(encoding="utf-8") as file:
        samples = json.load(file)["samples"]
    categories = [args.category] if args.category else sorted({
        sample["category"]["label"] for sample in samples
    })
    if any(not isinstance(category, str) or not category.isidentifier() for category in categories):
        parser.error("category names must be safe path components")
    model = PatchCore(args.device, args.max_patches, args.projection_dim, args.seed,
                      args.selection, args.neighborhood, args.border)
    for category in categories:
        dataset = MVTecDataset(args.dataset_root, category, "train", args.image_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        model.fit(loader)
        path = args.output_dir / f"{category}.pt"
        model.save(path)
        print(f"{category}: {len(dataset)} normal images, {len(model.memory_bank)} patches -> {path}")


if __name__ == "__main__":
    main()
