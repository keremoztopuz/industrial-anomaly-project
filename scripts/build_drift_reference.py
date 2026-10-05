"""Build the reference that drift checks compare live predictions against.

For each category: the held-out normal training scores from thresholds.json, and the
grayscale brightness and contrast of its normal training images, measured the same
way the service logs them. Each signal gets warning and alarm PSI cut-offs calibrated
for the check window. Writes drift_reference.json next to the banks.

Run from the repository root: python -m scripts.build_drift_reference
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageStat

from anomaly.drift import BINS, EXPECTED_FALSE_POSITIVE_RATE, WINDOW, calibrate_psi_thresholds
from anomaly.mvtec_dataset import MVTecDataset
from scripts.calibrate_thresholds import write_json


def image_stats(dataset):
    brightness, contrast = [], []
    for sample in dataset.samples:
        with Image.open(dataset.root / sample["filepath"]) as image:
            stats = ImageStat.Stat(image.convert("RGB").convert("L"))
        brightness.append(round(stats.mean[0], 1))
        contrast.append(round(stats.stddev[0], 1))
    return brightness, contrast


def signal(values, window):
    warn, alarm = calibrate_psi_thresholds(values, window=window)
    return {"reference": values, "warn": warn, "alarm": alarm}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/mvtec-ad"))
    parser.add_argument("--model-dir", type=Path,
                        default=Path("artifacts/border-exclusion/coreset-16384-n3-b2/patchcore"))
    parser.add_argument("--model-version", default="1",
                        help="Registry version these banks are deployed as")
    parser.add_argument("--window", type=int, default=WINDOW)
    args = parser.parse_args()

    calibration = json.loads((args.model_dir / "thresholds.json").read_text(encoding="utf-8"))
    categories = {}
    for category, values in sorted(calibration["categories"].items()):
        brightness, contrast = image_stats(
            MVTecDataset(args.dataset_root, category, "train"))
        categories[category] = {
            "anomaly_score": signal(values["holdout_scores"], args.window),
            "brightness": signal(brightness, args.window),
            "contrast": signal(contrast, args.window),
        }
        score = categories[category]["anomaly_score"]
        print(f"{category}: {len(values['holdout_scores'])} scores, {len(brightness)} images, "
              f"score PSI warn {score['warn']:.3f} alarm {score['alarm']:.3f}", flush=True)
    write_json(args.model_dir / "drift_reference.json", {
        "model_version": args.model_version, "window": args.window, "bins": BINS,
        "expected_false_positive_rate": EXPECTED_FALSE_POSITIVE_RATE,
        "categories": categories})


if __name__ == "__main__":
    main()
