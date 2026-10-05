import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


from scripts.data.validate_dataset import validate_dataset


class ValidateDatasetTests(unittest.TestCase):
    def test_valid_normal_and_anomaly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            Image.new("RGB", (8, 6)).save(root / "images/normal.png")
            Image.new("RGB", (8, 6)).save(root / "images/anomaly.png")
            Image.new("L", (8, 6)).save(root / "masks/anomaly.png")
            samples = [
                {"filepath": "images/normal.png",
                 "category": {"label": "thing"},
                 "defect": {"label": "good"}},
                {"filepath": "images/anomaly.png", "category": {"label": "thing"},
                    "defect": {"label": "crack"},
                 "defect_mask": {"mask_path": "masks/anomaly.png"}},
            ]
            (root / "samples.json").write_text(json.dumps({"samples": samples}), encoding="utf-8")

            total, categories, errors = validate_dataset(root)
            self.assertEqual((total, categories["thing"]["sizes"][(8, 6)], errors), (2, 2, []))

    def test_reports_corrupt_image_missing_mask_and_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 6)).save(root / "valid.png")
            (root / "corrupt.png").write_bytes(b"not an image")
            Image.new("L", (4, 6)).save(root / "wrong-size.png")
            samples = [
                {"filepath": "corrupt.png", "category": {"label": "a"},
                 "defect": {"label": "good"}},
                {"filepath": "valid.png", "category": {"label": "b"},
                 "defect": {"label": "crack"}},
                {"filepath": "valid.png", "category": {"label": "c"}, "defect": {
                    "label": "crack"}, "defect_mask": {"mask_path": "missing.png"}},
                {"filepath": "valid.png", "category": {"label": "d"}, "defect": {
                    "label": "crack"}, "defect_mask": {"mask_path": "wrong-size.png"}},
            ]
            (root / "samples.json").write_text(json.dumps({"samples": samples}), encoding="utf-8")

            _, _, errors = validate_dataset(root)
            self.assertEqual(len(errors), 4)
            self.assertTrue(any("corrupt image" in error for error in errors))
            self.assertTrue(any("missing mask" in error for error in errors))
            self.assertTrue(any("missing mask reference" in error for error in errors))
            self.assertTrue(any("size mismatch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
