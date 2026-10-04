import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anomaly.mvtec_dataset import MVTecDataset


class MVTecDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        Image.new("L", (4, 4), 255).save(self.root / "normal.png")
        image = Image.new("RGB", (4, 4))
        mask = Image.new("L", (4, 4))
        for y in range(4):
            for x in range(2):
                image.putpixel((x, y), (255, 255, 255))
                mask.putpixel((x, y), 255)
        image.save(self.root / "anomaly.png")
        mask.save(self.root / "mask.png")
        self.samples = [
            {"category": {"label": "cable"}, "split": "train",
             "defect": {"label": "good"}, "filepath": "normal.png"},
            {"category": {"label": "cable"}, "split": "test",
             "defect": {"label": "good"}, "filepath": "normal.png"},
            {"category": {"label": "cable"}, "split": "test",
             "defect": {"label": "cut"}, "filepath": "anomaly.png",
             "defect_mask": {"mask_path": "mask.png"}},
        ]
        self.write_samples()

    def write_samples(self):
        (self.root / "samples.json").write_text(json.dumps({"samples": self.samples}))

    def test_normal_rgb_normalization_and_train_filter(self):
        anomaly = {**self.samples[2], "split": "train"}
        other_category = {**self.samples[0], "category": {"label": "bottle"}}
        self.samples.extend([anomaly, other_category])
        self.write_samples()
        dataset = MVTecDataset(self.root, "cable", "train", image_size=8)
        self.assertEqual(len(dataset), 1)
        item = dataset[0]
        self.assertEqual(item["image"].shape, (3, 8, 8))
        self.assertEqual(item["image"].dtype, torch.float32)
        expected = (torch.ones(3) - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
        torch.testing.assert_close(item["image"], expected[:, None, None].expand(3, 8, 8))
        self.assertEqual(item["mask"].dtype, torch.bool)
        self.assertFalse(item["mask"].any())
        self.assertEqual(item["label"], 0)

    def test_anomaly_mask_geometry_labels_and_batching(self):
        dataset = MVTecDataset(self.root, "cable", "test", image_size=8)
        item = dataset[1]
        self.assertEqual(item["label"], 1)
        self.assertEqual(item["path"], str((self.root / "anomaly.png").resolve()))
        expected_mask = torch.zeros((1, 8, 8), dtype=torch.bool)
        expected_mask[:, :, :4] = True
        self.assertTrue(torch.equal(item["mask"], expected_mask))
        # The bright left half stays on the same side as the defect mask.
        self.assertTrue((item["image"][:, :, :3] > 0).all())
        self.assertTrue((item["image"][:, :, 5:] < 0).all())
        batch = next(iter(DataLoader(dataset, batch_size=8, num_workers=0)))
        self.assertEqual(batch["image"].shape, (2, 3, 8, 8))
        self.assertEqual(batch["mask"].shape, (2, 1, 8, 8))
        self.assertEqual(batch["label"].tolist(), [0, 1])

    def test_invalid_configuration_and_paths(self):
        for category, split, size in [("unknown", "train", 8), ("cable", "val", 8), ("cable", "train", 0)]:
            with self.subTest(category=category, split=split, size=size):
                with self.assertRaises(ValueError):
                    MVTecDataset(self.root, category, split, size)
        self.samples[0]["filepath"] = "../outside.png"
        self.write_samples()
        with self.assertRaisesRegex(ValueError, "escapes dataset root"):
            MVTecDataset(self.root, "cable", "train")[0]

    def test_missing_corrupt_files_and_mask_size_mismatch(self):
        self.samples[2].pop("defect_mask")
        self.write_samples()
        with self.assertRaisesRegex(ValueError, "Missing mask reference"):
            MVTecDataset(self.root, "cable", "test")[1]
        self.samples[2]["defect_mask"] = {"mask_path": "missing.png"}
        self.write_samples()
        with self.assertRaises(FileNotFoundError):
            MVTecDataset(self.root, "cable", "test")[1]
        self.samples[2]["defect_mask"] = {"mask_path": "mask.png"}
        self.write_samples()
        Image.new("L", (2, 4)).save(self.root / "mask.png")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            MVTecDataset(self.root, "cable", "test")[1]
        (self.root / "normal.png").write_bytes(b"not an image")
        with self.assertRaises(OSError):
            MVTecDataset(self.root, "cable", "train")[0]
        (self.root / "normal.png").unlink()
        with self.assertRaises(FileNotFoundError):
            MVTecDataset(self.root, "cable", "train")[0]


if __name__ == "__main__":
    unittest.main()
