import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import patchcore
from patchcore import PatchCore


@contextmanager
def no_backbone():
    """Skip loading Wide ResNet weights."""
    with mock.patch.object(patchcore, "wide_resnet50_2"), \
            mock.patch.object(patchcore, "create_feature_extractor",
                              return_value=torch.nn.Identity()):
        yield


def make_model(**kwargs):
    """Build PatchCore whose images are already patch features."""
    with no_backbone():
        model = PatchCore(**kwargs)
    model._embed = lambda images: images
    return model


def loader(features):
    """Yield one batch of shape [1, N, 1, D] so fit sees N patches."""
    return [{"image": features.reshape(1, -1, 1, features.shape[-1])}]


class PatchCoreSelectionTests(unittest.TestCase):
    def test_rejects_unknown_selection(self):
        with self.assertRaises(ValueError):
            make_model(selection="greedy")

    def test_random_bank_is_bounded_subset(self):
        features = torch.randn(50, 4)
        model = make_model(max_patches=8, projection_dim=4).fit(loader(features))
        self.assertEqual(model.memory_bank.shape, (8, 4))
        for row in model.memory_bank:
            self.assertTrue((features == row).all(dim=1).any())

    def test_coreset_bank_is_bounded_unique_subset(self):
        features = torch.randn(50, 4)
        model = make_model(max_patches=8, projection_dim=4,
                           selection="coreset").fit(loader(features))
        self.assertEqual(model.memory_bank.shape, (8, 4))
        self.assertEqual(len(torch.unique(model.memory_bank, dim=0)), 8)
        for row in model.memory_bank:
            self.assertTrue((features == row).all(dim=1).any())

    def test_coreset_covers_separate_clusters(self):
        features = torch.tensor([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])
        model = make_model(max_patches=2, projection_dim=1,
                           selection="coreset").fit(loader(features))
        bank = sorted(model.memory_bank.flatten().tolist())
        self.assertLess(bank[0], 1)
        self.assertGreater(bank[1], 9)

    def test_small_training_set_keeps_all_patches(self):
        features = torch.randn(3, 4)
        model = make_model(max_patches=8, projection_dim=4,
                           selection="coreset").fit(loader(features))
        self.assertEqual(len(model.memory_bank), 3)

    def test_fit_rejects_anomalous_images(self):
        batch = loader(torch.randn(4, 4))[0] | {"label": torch.tensor([1])}
        with self.assertRaises(ValueError):
            make_model(max_patches=2, projection_dim=4).fit([batch])

    def test_save_load_preserves_selection(self):
        model = make_model(max_patches=4, projection_dim=4, selection="coreset")
        model.fit(loader(torch.randn(20, 4)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.pt"
            model.save(path)
            with no_backbone():
                loaded = PatchCore.load(path)
        self.assertEqual(loaded.selection, "coreset")
        self.assertTrue(torch.equal(loaded.memory_bank, model.memory_bank))
        self.assertTrue(torch.equal(loaded.projection, model.projection))

    def test_load_defaults_old_banks_to_random(self):
        model = make_model(max_patches=4, projection_dim=4)
        model.fit(loader(torch.randn(20, 4)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.pt"
            model.save(path)
            state = torch.load(path, weights_only=True)
            del state["selection"]
            torch.save(state, path)
            with no_backbone():
                loaded = PatchCore.load(path)
        self.assertEqual(loaded.selection, "random")


if __name__ == "__main__":
    unittest.main()
