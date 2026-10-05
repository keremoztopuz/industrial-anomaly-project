import argparse
import json
from collections import Counter
from pathlib import Path


def count_samples(samples):
    categories = {}
    for sample in samples:
        category = sample["category"]["label"]
        counts = categories.setdefault(category, Counter())
        counts[(sample["split"], sample["defect"]["label"])] += 1
    return categories


def main():
    parser = argparse.ArgumentParser(
        description="Summarize MVTec images by category, split and defect.")
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/mvtec-ad/samples.json",
    )
    args = parser.parse_args()
    with args.samples.open(encoding="utf-8") as file:
        samples = json.load(file)["samples"]

    categories = count_samples(samples)
    print(f"Total images: {len(samples)} | Categories: {len(categories)}\n")
    headers = ("Train normal", "Train defect", "Test normal", "Test defect")
    print(f"{'Category':<16} {'Total':>6}" + "".join(f"{header:>13}" for header in headers))
    for category, counts in sorted(categories.items()):
        totals = [
            sum(n for (s, d), n in counts.items() if s == split and (d == "good") == normal)
            for split, normal in [("train", True), ("train", False),
                                  ("test", True), ("test", False)]
        ]
        print(f"{category:<16} {sum(counts.values()):>6}" + "".join(f"{n:>13}" for n in totals))

    print("\nCounts by split and defect:")
    for category, counts in sorted(categories.items()):
        print(f"\n{category}:")
        for (split, defect), count in sorted(counts.items()):
            print(f"  {split}/{defect}: {count}")


if __name__ == "__main__":
    main()
