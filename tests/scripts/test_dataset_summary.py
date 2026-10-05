import unittest

from scripts.data.dataset_summary import count_samples


class DatasetSummaryTest(unittest.TestCase):
    def test_counts_keep_categories_splits_and_defects_separate(self):
        samples = [
            {"category": {"label": category}, "split": split, "defect": {"label": defect}}
            for category, split, defect in [
                ("cable", "train", "good"),
                ("cable", "train", "good"),
                ("cable", "test", "good"),
                ("cable", "test", "cut"),
                ("cable", "test", "bent"),
                ("bottle", "train", "good"),
                ("bottle", "test", "cut"),
            ]
        ]
        self.assertEqual(count_samples(samples), {
            "cable": {("train", "good"): 2, ("test", "good"): 1,
                      ("test", "cut"): 1, ("test", "bent"): 1},
            "bottle": {("train", "good"): 1, ("test", "cut"): 1},
        })


if __name__ == "__main__":
    unittest.main()
