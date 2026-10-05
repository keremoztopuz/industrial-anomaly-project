import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from anomaly.model.patchcore import PatchCore
from scripts.modeling import build_patchcore
from scripts.reporting import visualize_anomalies
from tests.support.model_fakes import no_backbone


class ModelCliTests(unittest.TestCase):
    def test_cli_image_size_reaches_saved_and_reloaded_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "samples.json").write_text(
                json.dumps({"samples": [{"category": {"label": "bottle"}}]})
            )
            argv = [
                "build",
                "--dataset-root",
                str(root),
                "--output-dir",
                str(root),
                "--image-size",
                "320",
                "--device",
                "cpu",
                "--max-patches",
                "4",
            ]

            def fit(model, loader):
                model.memory_bank = torch.zeros(4, model.projection_dim)

            with (
                mock.patch("sys.argv", argv),
                no_backbone(),
                mock.patch.object(build_patchcore, "MVTecDataset") as dataset,
                mock.patch.object(build_patchcore, "DataLoader"),
                mock.patch.object(PatchCore, "fit", fit),
            ):
                build_patchcore.main()
                loaded = PatchCore.load(root / "bottle.pt")
            dataset.assert_called_once_with(root, "bottle", "train", 320)
            self.assertEqual(loaded.image_size, 320)

    def test_visualization_infers_size_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "patchcore").mkdir()
            (root / "patchcore" / "bottle.pt").touch()
            with (
                mock.patch.object(
                    visualize_anomalies.PatchCore, "load"
                ) as load,
                mock.patch.object(
                    visualize_anomalies, "MVTecDataset"
                ) as dataset,
            ):
                load.return_value.image_size = 320
                dataset.return_value.samples = []
                with self.assertRaisesRegex(ValueError, "two good"):
                    visualize_anomalies.render_category(
                        root, root, "bottle", None, "cpu"
                    )
                dataset.assert_called_once_with(root, "bottle", "test", 320)
                dataset.reset_mock()
                with self.assertRaisesRegex(ValueError, "must match"):
                    visualize_anomalies.render_category(
                        root, root, "bottle", 256, "cpu"
                    )
                dataset.assert_not_called()
