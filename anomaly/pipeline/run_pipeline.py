"""Run PatchCore and anomaly evaluation for one or all MVTec categories.

Run from the repository root: python -m anomaly.pipeline.run_pipeline
"""

import mlflow

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from prefect import flow, task
from torch.utils.data import DataLoader

from anomaly.evaluation.metrics import evaluate_category
from anomaly.data.mvtec_dataset import MVTecDataset
from anomaly.model.patchcore import PatchCore
from anomaly.settings import TRAINING_EXPERIMENT, settings


def categories_from_metadata(dataset_root):
    with (dataset_root / "samples.json").open(encoding="utf-8") as file:
        categories = {
            sample["category"]["label"]
            for sample in json.load(file)["samples"]
        }
    if not categories or any(
        not isinstance(category, str) or not category.isidentifier()
        for category in categories
    ):
        raise ValueError(
            "Dataset categories must be nonempty, safe directory names"
        )
    return sorted(categories)


@task(task_run_name="{category}")
def run_category(
    category,
    dataset_root,
    output_root,
    image_size,
    batch_size,
    device,
    max_patches,
    projection_dim,
    seed,
    selection,
    neighborhood,
    border,
):
    train = MVTecDataset(dataset_root, category, "train", image_size)
    model = PatchCore(
        device=device,
        max_patches=max_patches,
        projection_dim=projection_dim,
        seed=seed,
        selection=selection,
        neighborhood=neighborhood,
        border=border,
        image_size=image_size,
    )
    model.fit(
        DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0)
    )
    model.save(output_root / "patchcore" / f"{category}.pt")

    test = MVTecDataset(dataset_root, category, "test", image_size)
    scores, maps, labels, masks = [], [], [], []
    for batch in DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=0
    ):
        batch_scores, batch_maps = model.predict(batch["image"])
        scores.append(batch_scores.detach().cpu())
        maps.append(batch_maps.detach().cpu())
        labels.append(batch["label"])
        masks.append(batch["mask"])
    return evaluate_category(
        torch.cat(scores), torch.cat(maps), torch.cat(labels), torch.cat(masks)
    )


@flow(name="mvtec-patchcore")
def run_pipeline(
    dataset_root,
    output_root,
    image_size=256,
    batch_size=8,
    device="cpu",
    max_patches=2048,
    projection_dim=256,
    seed=42,
    category=None,
    selection="random",
    neighborhood=1,
    border=0,
    tracking_uri=settings.mlflow_tracking_uri,
):
    if image_size <= 0 or batch_size <= 0:
        raise ValueError("image_size and batch_size must be positive")
    dataset_root, output_root = (
        Path(dataset_root).resolve(),
        Path(output_root).resolve(),
    )
    categories = categories_from_metadata(dataset_root)
    if category is not None:
        if category not in categories:
            raise ValueError(
                f"Unknown category: {category}. Available: "
                f"{', '.join(categories)}"
            )
        categories = [category]
    output_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(TRAINING_EXPERIMENT)
    with mlflow.start_run(run_name=output_root.name):
        mlflow.log_params(
            {
                "dataset_root": str(dataset_root),
                "output_root": str(output_root),
                "image_size": image_size,
                "batch_size": batch_size,
                "device": device,
                "max_patches": max_patches,
                "projection_dim": projection_dim,
                "seed": seed,
                "selection": selection,
                "neighborhood": neighborhood,
                "border": border,
            }
        )
        results = {
            category: run_category(
                category,
                dataset_root,
                output_root,
                image_size,
                batch_size,
                device,
                max_patches,
                projection_dim,
                seed,
                selection,
                neighborhood,
                border,
            )
            for category in categories
        }

        for category, metrics in results.items():
            mlflow.log_metric(
                f"image_auroc/{category}", metrics["image_auroc"]
            )
            mlflow.log_metric(
                f"pixel_auroc/{category}", metrics["pixel_auroc"]
            )
        mlflow.log_metric(
            "image_auroc_mean",
            sum(m["image_auroc"] for m in results.values()) / len(results),
        )
        mlflow.log_metric(
            "pixel_auroc_mean",
            sum(m["pixel_auroc"] for m in results.values()) / len(results),
        )

        summary = output_root / "metrics.json"
        temporary = summary.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(summary)

        source_root = Path(__file__).resolve().parents[2]
        try:
            git = subprocess.run(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            git_commit = git.stdout.strip() if git.returncode == 0 else None
            git_dirty = None
            if git_commit:
                status = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(source_root),
                        "status",
                        "--porcelain",
                        "--untracked-files=no",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if status.returncode == 0:
                    git_dirty = bool(status.stdout.strip())
        except OSError:
            git_commit = git_dirty = None
        with (dataset_root / "samples.json").open("rb") as file:
            samples_json_sha256 = hashlib.file_digest(
                file, "sha256"
            ).hexdigest()
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "samples_json_sha256": samples_json_sha256,
            "categories": categories,
            "image_size": image_size,
            "batch_size": batch_size,
            "device": str(device),
            "max_patches": max_patches,
            "projection_dim": projection_dim,
            "seed": seed,
            "selection": selection,
            "neighborhood": neighborhood,
            "border": border,
            "metrics_file": summary.name,
            "bank_files": [
                str(Path("patchcore") / f"{name}.pt") for name in categories
            ],
        }
        destination = output_root / "manifest.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

        mlflow.set_tags(
            {
                "git_commit": git_commit,
                "git_dirty": str(git_dirty),
                "samples_json_sha256": samples_json_sha256,
            }
        )

        mlflow.log_artifact(str(summary))
        mlflow.log_artifact(str(destination))
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=settings.dataset_root
    )
    parser.add_argument(
        "--output-root", type=Path, default=settings.artifacts_root
    )
    parser.add_argument(
        "--category", help="Run one category; omit to run all categories"
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-patches", type=int, default=2048)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection", choices=("random", "coreset"), default="random"
    )
    parser.add_argument(
        "--neighborhood",
        type=int,
        default=1,
        help="Odd feature-averaging window; PatchCore uses 3",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=0,
        help="Outer patch rings ignored by image scores",
    )
    parser.add_argument(
        "--tracking-uri",
        default=settings.mlflow_tracking_uri,
        help="MLflow tracking URI",
    )
    args = parser.parse_args()
    run_pipeline(**vars(args))


if __name__ == "__main__":
    main()
