"""Send a scenario of test images to the live service to exercise the drift checks.

Scenarios, all on one category's test images:
- normal:  50 defect-free images, unchanged. The checks should stay stable.
- dark:    the same images at 60% brightness, like a failing lamp.
- blur:    the same images with a Gaussian blur, like a camera out of focus.
- defects: 40 defective and 10 defect-free images at normal lighting, a defect wave.
           The alarm rate should fire while brightness and contrast stay stable.

Run from the repository root, then wait about 30 seconds for the logs and run
check_drift: python -m scripts.simulate_drift --scenario dark
"""

import argparse
import random
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, ImageEnhance, ImageFilter

from anomaly.mvtec_dataset import MVTecDataset

SCENARIOS = ("normal", "dark", "blur", "defects")
DARK_FACTOR = 0.6
BLUR_RADIUS = 4


def pick_samples(dataset, scenario, count, seed):
    """Choose the test samples for a scenario, in a seeded random order."""
    rng = random.Random(seed)
    good = [s for s in dataset.samples if s["defect"]["label"] == "good"]
    defective = [s for s in dataset.samples if s["defect"]["label"] != "good"]
    rng.shuffle(good)
    rng.shuffle(defective)
    if scenario == "defects":
        defect_count = round(count * 0.8)
        chosen = defective[:defect_count] + good[:count - defect_count]
        rng.shuffle(chosen)
    else:
        chosen = good[:count]
    if len(chosen) < count:
        raise ValueError(f"only {len(chosen)} suitable test images for scenario {scenario}")
    return chosen


def apply_scenario(image, scenario):
    if scenario == "dark":
        return ImageEnhance.Brightness(image).enhance(DARK_FACTOR)
    if scenario == "blur":
        return image.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    return image


def encode_png(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--category", default="cable")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--url", default="https://anomaly-api-sw4p2ayosa-ew.a.run.app")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/mvtec-ad"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = MVTecDataset(args.dataset_root, args.category, "test")
    samples = pick_samples(dataset, args.scenario, args.count, args.seed)
    flagged = 0
    with httpx.Client(base_url=args.url, timeout=180) as client:
        for index, sample in enumerate(samples, start=1):
            with Image.open(dataset.root / sample["filepath"]) as source:
                image = apply_scenario(source.convert("RGB"), args.scenario)
            response = client.post(f"/predict/{args.category}",
                                   files={"upload_file": ("image.png", encode_png(image))})
            response.raise_for_status()
            flagged += bool(response.json()["is_anomaly"])
            if index % 10 == 0:
                print(f"{index}/{len(samples)} sent, {flagged} flagged", flush=True)
    print(f"{args.scenario}: sent {len(samples)} {args.category} images, "
          f"{flagged} flagged as defective")


if __name__ == "__main__":
    main()
