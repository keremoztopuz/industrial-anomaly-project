"""Run PatchCore and anomaly evaluation for one or all MVTec categories."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from prefect import flow, task
from torch.utils.data import DataLoader

from anomaly_metrics import evaluate_category
from mvtec_dataset import MVTecDataset
from patchcore import PatchCore


def categories_from_metadata(dataset_root):
    with (dataset_root / "samples.json").open(encoding="utf-8") as file:
        categories = {sample["category"]["label"] for sample in json.load(file)["samples"]}
    if not categories or any(not isinstance(category, str) or not category.isidentifier()
                             for category in categories):
        raise ValueError("Dataset categories must be nonempty, safe directory names")
    return sorted(categories)


@task(task_run_name="{category}")
def run_category(category, dataset_root, output_root, image_size, batch_size,
                 device, max_patches, projection_dim, seed):
    train = MVTecDataset(dataset_root, category, "train", image_size)
    model = PatchCore(device=device, max_patches=max_patches,
                      projection_dim=projection_dim, seed=seed)
    model.fit(DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0))
    model.save(output_root / "patchcore" / f"{category}.pt")

    test = MVTecDataset(dataset_root, category, "test", image_size)
    scores, maps, labels, masks = [], [], [], []
    for batch in DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0):
        batch_scores, batch_maps = model.predict(batch["image"])
        scores.append(batch_scores.detach().cpu())
        maps.append(batch_maps.detach().cpu())
        labels.append(batch["label"])
        masks.append(batch["mask"])
    return evaluate_category(torch.cat(scores), torch.cat(maps),
                             torch.cat(labels), torch.cat(masks))


@flow(name="mvtec-patchcore")
def run_pipeline(dataset_root, output_root, image_size=256, batch_size=8,
                 device="cpu", max_patches=2048, projection_dim=256, seed=42,
                 category=None):
    if image_size <= 0 or batch_size <= 0:
        raise ValueError("image_size and batch_size must be positive")
    dataset_root, output_root = Path(dataset_root).resolve(), Path(output_root).resolve()
    categories = categories_from_metadata(dataset_root)
    if category is not None:
        if category not in categories:
            raise ValueError(f"Unknown category: {category}. Available: {', '.join(categories)}")
        categories = [category]
    output_root.mkdir(parents=True, exist_ok=True)
    results = {
        category: run_category(category, dataset_root, output_root, image_size,
                               batch_size, device, max_patches, projection_dim, seed)
        for category in categories
    }
    summary = output_root / "metrics.json"
    temporary = summary.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary)

    try:
        git = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        git_commit = git.stdout.strip() if git.returncode == 0 else None
    except OSError:
        git_commit = None
    with (dataset_root / "samples.json").open("rb") as file:
        samples_json_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "samples_json_sha256": samples_json_sha256,
        "categories": categories,
        "image_size": image_size,
        "batch_size": batch_size,
        "device": device,
        "max_patches": max_patches,
        "projection_dim": projection_dim,
        "seed": seed,
        "metrics_file": summary.name,
        "bank_files": [str(Path("patchcore") / f"{name}.pt") for name in categories],
    }
    destination = output_root / "manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path(__file__).resolve().parent / "data/mvtec-ad")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--category", help="Run one category; omit to run all categories")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-patches", type=int, default=2048)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(**vars(args))


if __name__ == "__main__":
    main()
