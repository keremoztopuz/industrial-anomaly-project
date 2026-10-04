"""Pick one is_anomaly threshold per category from held-out normal training images.

MVTec AD has no validation split, so a fraction of each category's normal training
images is held out. A bank with the deployed bank's settings is fit on the rest, and
the threshold is the highest held-out score: no held-out normal image is flagged.
The test split is only used afterwards, to report how the thresholds behave.

Run from the repository root: python -m scripts.calibrate_thresholds
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from mvtec_dataset import MVTecDataset
from patchcore import PatchCore


def split_indices(count, holdout_fraction, seed):
    """Return (fit, holdout) index lists from a seeded shuffle; both are nonempty."""
    if holdout_fraction <= 0 or holdout_fraction >= 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if count < 2:
        raise ValueError("need at least two images to hold some out")
    holdout_size = min(count - 1, max(1, math.ceil(count * holdout_fraction)))
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
    return sorted(order[holdout_size:]), sorted(order[:holdout_size])


def confusion(scores, labels, threshold):
    """Count image decisions where score > threshold means anomalous."""
    predicted = torch.as_tensor(scores) > threshold
    labels = torch.as_tensor(labels).bool()
    counts = {
        "true_positive": int((predicted & labels).sum()),
        "false_positive": int((predicted & ~labels).sum()),
        "true_negative": int((~predicted & ~labels).sum()),
        "false_negative": int((~predicted & labels).sum()),
    }
    positives = counts["true_positive"] + counts["false_negative"]
    negatives = counts["false_positive"] + counts["true_negative"]
    counts["recall"] = counts["true_positive"] / positives if positives else None
    counts["false_positive_rate"] = counts["false_positive"] / negatives if negatives else None
    return counts


def score(model, dataset, batch_size):
    scores, labels = [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        batch_scores, _ = model.predict(batch["image"])
        scores.append(batch_scores)
        labels.append(torch.as_tensor(batch["label"]))
    return torch.cat(scores), torch.cat(labels)


@torch.inference_mode()
def calibrate_category(category, args, backbone):
    deployed = PatchCore.load(args.model_dir / f"{category}.pt", args.device, backbone)
    train = MVTecDataset(args.dataset_root, category, "train", args.image_size)
    fit, holdout = split_indices(len(train), args.holdout_fraction, args.seed)

    model = PatchCore(args.device, deployed.max_patches, deployed.projection_dim,
                      deployed.seed, deployed.selection, deployed.neighborhood,
                      deployed.border, backbone=deployed.backbone)
    model.fit(DataLoader(Subset(train, fit), batch_size=args.batch_size,
                         shuffle=False, num_workers=0))
    holdout_scores, _ = score(model, Subset(train, holdout), args.batch_size)
    threshold = float(holdout_scores.max())

    test_scores, test_labels = score(deployed, MVTecDataset(
        args.dataset_root, category, "test", args.image_size), args.batch_size)
    return deployed.backbone, {
        "threshold": threshold,
        "fit_images": len(fit),
        "holdout_images": len(holdout),
        "holdout_score_mean": float(holdout_scores.mean()),
        "holdout_score_max": threshold,
    }, confusion(test_scores, test_labels, threshold)


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/mvtec-ad"))
    parser.add_argument("--model-dir", type=Path,
                        default=Path("artifacts/border-exclusion/coreset-16384-n3-b2/patchcore"))
    parser.add_argument("--category", help="Calibrate only this category; default is all banks")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()
    categories = [args.category] if args.category else sorted(
        path.stem for path in args.model_dir.glob("*.pt"))
    if not categories:
        parser.error(f"no .pt banks in {args.model_dir}")

    thresholds, evaluation, backbone = {}, {}, None
    for category in categories:
        backbone, thresholds[category], evaluation[category] = calibrate_category(
            category, args, backbone)
        result = evaluation[category]
        print(f"{category}: threshold {thresholds[category]['threshold']:.4f}, "
              f"test recall {result['recall']:.3f}, "
              f"false positive rate {result['false_positive_rate']:.3f}", flush=True)

    settings = {"method": "max held-out normal score", "holdout_fraction": args.holdout_fraction,
                "seed": args.seed, "image_size": args.image_size}
    write_json(args.model_dir / "thresholds.json", {**settings, "categories": thresholds})
    write_json(args.model_dir / "threshold_evaluation.json", {**settings, "categories": evaluation})


if __name__ == "__main__":
    main()
