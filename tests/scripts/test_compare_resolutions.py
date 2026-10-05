import unittest

import torch

from scripts.modeling import compare_resolutions


class CompareResolutionsTests(unittest.TestCase):
    def test_highest_validation_auroc_wins(self):
        self.assertEqual(
            compare_resolutions.pick_winner({256: 0.90, 320: 0.95, 384: 0.93}),
            320,
        )

    def test_ties_go_to_the_smaller_size(self):
        self.assertEqual(
            compare_resolutions.pick_winner({256: 0.95, 320: 0.95, 384: 0.95}),
            256,
        )

    def test_auroc_uses_only_the_given_indices(self):
        scores = torch.tensor([0.1, 0.9, 0.8, 0.2])
        labels = torch.tensor([0, 1, 0, 1])
        self.assertEqual(
            compare_resolutions.auroc(scores, labels, torch.tensor([0, 1])),
            1.0,
        )
        self.assertEqual(
            compare_resolutions.auroc(scores, labels, torch.tensor([2, 3])),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
