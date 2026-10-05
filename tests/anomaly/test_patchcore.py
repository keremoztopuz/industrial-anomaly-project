import tempfile
import unittest
from pathlib import Path

import torch

from anomaly.model.patchcore import PatchCore
from tests.support.model_fakes import no_backbone


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


class PatchCoreNeighborhoodTests(unittest.TestCase):
    def embed(self, neighborhood, features):
        with no_backbone():
            model = PatchCore(projection_dim=4, neighborhood=neighborhood)
        model.backbone = lambda images: features
        return model._embed(torch.zeros(1, 3, 8, 8))

    def test_rejects_even_or_nonpositive_neighborhood(self):
        for neighborhood in (0, 2, -1):
            with self.assertRaises(ValueError):
                make_model(neighborhood=neighborhood)

    def test_neighborhood_averages_each_feature_map(self):
        shallow, deep = torch.randn(1, 512, 4, 4), torch.randn(1, 1024, 2, 2)
        features = {"layer2": shallow, "layer3": deep}
        averaged = self.embed(3, features)
        expected = self.embed(1, {
            name: torch.nn.functional.avg_pool2d(value, 3, stride=1, padding=1,
                                                 count_include_pad=False)
            for name, value in features.items()
        })
        self.assertEqual(averaged.shape, (1, 4, 4, 4))
        self.assertTrue(torch.allclose(averaged, expected, atol=1e-6))
        self.assertFalse(torch.allclose(averaged, self.embed(1, features)))

    def test_save_load_preserves_neighborhood(self):
        model = make_model(max_patches=4, projection_dim=4, neighborhood=3)
        model.fit(loader(torch.randn(20, 4)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.pt"
            model.save(path)
            state = torch.load(path, weights_only=True)
            del state["neighborhood"]
            old_path = Path(directory) / "old.pt"
            torch.save(state, old_path)
            with no_backbone():
                loaded, old = PatchCore.load(path), PatchCore.load(old_path)
        self.assertEqual(loaded.neighborhood, 3)
        self.assertEqual(old.neighborhood, 1)


class PatchCoreBorderTests(unittest.TestCase):
    def predict(self, border):
        model = make_model(max_patches=1, projection_dim=1, border=border)
        model.memory_bank = torch.zeros(1, 1)
        features = torch.zeros(1, 4, 4, 1)
        features[0, 0, 0] = 9.0
        features[0, 1, 2] = 2.0
        model._embed = lambda images: features
        return model.predict(torch.zeros(1, 3, 8, 8))

    def test_rejects_negative_border(self):
        with self.assertRaises(ValueError):
            make_model(border=-1)

    def test_border_excludes_outer_ring_from_image_score(self):
        scores, maps = self.predict(border=0)
        self.assertAlmostEqual(float(scores[0]), 9.0, places=5)
        scores, border_maps = self.predict(border=1)
        self.assertAlmostEqual(float(scores[0]), 2.0, places=5)
        self.assertTrue(torch.equal(maps, border_maps))

    def test_border_must_leave_patches(self):
        with self.assertRaises(ValueError):
            self.predict(border=2)

    def test_save_load_preserves_border(self):
        model = make_model(max_patches=4, projection_dim=4, border=2)
        model.fit(loader(torch.randn(20, 4)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.pt"
            model.save(path)
            state = torch.load(path, weights_only=True)
            del state["border"]
            old_path = Path(directory) / "old.pt"
            torch.save(state, old_path)
            with no_backbone():
                loaded, old = PatchCore.load(path), PatchCore.load(old_path)
        self.assertEqual(loaded.border, 2)
        self.assertEqual(old.border, 0)


class PatchCoreImageSizeTests(unittest.TestCase):
    def test_save_load_preserves_image_size_and_old_banks_are_256(self):
        model = make_model(max_patches=4, projection_dim=4, image_size=320)
        model.fit(loader(torch.randn(20, 4)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.pt"
            model.save(path)
            state = torch.load(path, weights_only=True)
            del state["image_size"]
            old_path = Path(directory) / "old.pt"
            torch.save(state, old_path)
            with no_backbone():
                loaded, old = PatchCore.load(path), PatchCore.load(old_path)
        self.assertEqual(loaded.image_size, 320)
        self.assertEqual(old.image_size, 256)


if __name__ == "__main__":
    unittest.main()
