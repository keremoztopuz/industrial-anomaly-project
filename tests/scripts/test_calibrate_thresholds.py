import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.modeling.calibrate_thresholds import (
    confusion,
    merge_categories,
    split_indices,
)


class SplitIndicesTests(unittest.TestCase):
    def test_split_is_disjoint_complete_and_seeded(self):
        fit, holdout = split_indices(10, 0.2, seed=42)
        self.assertEqual(len(holdout), 2)
        self.assertEqual(sorted(fit + holdout), list(range(10)))
        self.assertEqual(split_indices(10, 0.2, seed=42), (fit, holdout))
        self.assertNotEqual(split_indices(10, 0.2, seed=7)[1], holdout)

    def test_rounds_holdout_up_and_keeps_one_fit_image(self):
        self.assertEqual(len(split_indices(11, 0.2, seed=0)[1]), 3)
        self.assertEqual(len(split_indices(2, 0.9, seed=0)[0]), 1)

    def test_rejects_bad_arguments(self):
        for count, fraction in [(10, 0), (10, 1), (1, 0.2)]:
            with self.subTest(count=count, fraction=fraction):
                with self.assertRaises(ValueError):
                    split_indices(count, fraction, seed=0)


class ConfusionTests(unittest.TestCase):
    def test_counts_strictly_above_threshold_as_anomalous(self):
        result = confusion(
            torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([0, 1, 0, 1]), 2.0
        )
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_negative"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["true_negative"], 1)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["false_positive_rate"], 0.5)

    def test_rates_are_none_without_positives_or_negatives(self):
        result = confusion(torch.tensor([1.0]), torch.tensor([0]), 0.0)
        self.assertIsNone(result["recall"])
        self.assertEqual(result["false_positive_rate"], 1.0)


class MergeCategoriesTests(unittest.TestCase):
    def test_updates_only_the_given_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thresholds.json"
            path.write_text(
                json.dumps(
                    {
                        "method": "old",
                        "categories": {
                            "bottle": {"threshold": 12.8},
                            "pill": {"threshold": 17.5},
                        },
                    }
                ),
                encoding="utf-8",
            )
            merged = merge_categories(
                path,
                {"method": "max held-out normal score"},
                {"pill": {"threshold": 15.0, "image_size": 320}},
            )
        self.assertEqual(merged["method"], "max held-out normal score")
        self.assertEqual(merged["categories"]["bottle"], {"threshold": 12.8})
        self.assertEqual(
            merged["categories"]["pill"],
            {"threshold": 15.0, "image_size": 320},
        )

    def test_missing_file_starts_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            merged = merge_categories(
                Path(directory) / "none.json",
                {"seed": 42},
                {"pill": {"threshold": 1.0}},
            )
        self.assertEqual(
            merged, {"seed": 42, "categories": {"pill": {"threshold": 1.0}}}
        )


if __name__ == "__main__":
    unittest.main()
