"""Pick one is_anomaly threshold per category from held-out normal training images.

MVTec AD has no validation split, so a fraction of each category's normal training
images is held out. A bank with the deployed bank's settings is fit on the rest, and
the threshold is the highest held-out score: no held-out normal image is flagged.
The test split is only used afterwards, to report how the thresholds behave.

Run from the repository root: python -m scripts.modeling.calibrate_thresholds
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from anomaly.data.mvtec_dataset import MVTecDataset
from anomaly.evaluation.scoring import confusion, score
from anomaly.evaluation.splits import split_indices
from anomaly.model.patchcore import PatchCore
from anomaly.utils import default_device, write_json


@torch.inference_mode()
def calibrate_category(category, args, backbone):
    deployed = PatchCore.load(args.model_dir / f"{category}.pt", args.device, backbone)
    train = MVTecDataset(args.dataset_root, category, "train", deployed.image_size)
    fit, holdout = split_indices(len(train), args.holdout_fraction, args.seed)

    model = PatchCore(args.device, deployed.max_patches, deployed.projection_dim,
                      deployed.seed, deployed.selection, deployed.neighborhood,
                      deployed.border, backbone=deployed.backbone,
                      image_size=deployed.image_size)
    model.fit(DataLoader(Subset(train, fit), batch_size=args.batch_size,
                         shuffle=False, num_workers=0))
    holdout_scores, _ = score(model, Subset(train, holdout), args.batch_size)
    threshold = float(holdout_scores.max())

    test_scores, test_labels = score(deployed, MVTecDataset(
        args.dataset_root, category, "test", deployed.image_size), args.batch_size)
    return deployed.backbone, {
        "threshold": threshold,
        "image_size": deployed.image_size,
        "fit_images": len(fit),
        "holdout_images": len(holdout),
        "holdout_score_mean": float(holdout_scores.mean()),
        "holdout_score_max": threshold,
        "holdout_scores": holdout_scores.tolist(),
    }, confusion(test_scores, test_labels, threshold)


def merge_categories(path, settings, results):
    """Return the file's contents with `results` replacing only those categories."""
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    categories = {**existing.get("categories", {}), **results}
    return {**existing, **settings, "categories": categories}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/mvtec-ad"))
    parser.add_argument("--model-dir", type=Path,
                        default=Path("artifacts/border-exclusion/coreset-16384-n3-b2/patchcore"))
    parser.add_argument("--category", help="Calibrate only this category; default is all banks")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
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
                "seed": args.seed}
    for name, results in (("thresholds.json", thresholds),
                          ("threshold_evaluation.json", evaluation)):
        path = args.model_dir / name
        write_json(path, merge_categories(path, settings, results))


if __name__ == "__main__":
    main()
