import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mlflow.tracking import MlflowClient

import run_pipeline

RESULTS = {
    "bottle": {"image_auroc": 1.0, "pixel_auroc": 0.9},
    "cable": {"image_auroc": 0.8, "pixel_auroc": 0.7},
}


def tracking_uri_in(root):
    """Use a temporary store and artifact root so tests leave nothing in the repo."""
    uri = f"sqlite:///{root / 'mlflow.db'}"
    MlflowClient(uri).create_experiment("patchcore-mvtec",
                                        artifact_location=(root / "artifacts").as_uri())
    return uri


class RunPipelineMlflowTests(unittest.TestCase):
    def test_logs_params_metrics_tags_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            (dataset_root / "samples.json").write_text(json.dumps({"samples": [
                {"category": {"label": category}} for category in RESULTS
            ]}), encoding="utf-8")
            tracking_uri = tracking_uri_in(root)

            with mock.patch.object(run_pipeline, "run_category",
                                   side_effect=lambda category, *args: RESULTS[category]):
                returned = run_pipeline.run_pipeline.fn(
                    dataset_root, root / "smoke-run", max_patches=16384,
                    selection="coreset", neighborhood=3, border=2,
                    tracking_uri=tracking_uri)

            self.assertEqual(returned, RESULTS)
            client = MlflowClient(tracking_uri)
            experiment = client.get_experiment_by_name("patchcore-mvtec")
            [run] = client.search_runs([experiment.experiment_id])
            self.assertEqual(run.info.status, "FINISHED")
            self.assertEqual(run.info.run_name, "smoke-run")
            self.assertEqual(run.data.params["max_patches"], "16384")
            self.assertEqual(run.data.params["selection"], "coreset")
            self.assertEqual(run.data.params["border"], "2")
            self.assertEqual(run.data.metrics["image_auroc/cable"], 0.8)
            self.assertAlmostEqual(run.data.metrics["image_auroc_mean"], 0.9)
            self.assertAlmostEqual(run.data.metrics["pixel_auroc_mean"], 0.8)
            self.assertEqual(
                len(client.get_metric_history(run.info.run_id, "image_auroc_mean")), 1)
            self.assertIn("samples_json_sha256", run.data.tags)
            self.assertEqual(
                sorted(item.path for item in client.list_artifacts(run.info.run_id)),
                ["manifest.json", "metrics.json"])

    def test_failed_run_is_marked_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "samples.json").write_text(json.dumps({"samples": [
                {"category": {"label": "bottle"}}]}), encoding="utf-8")
            tracking_uri = tracking_uri_in(root)
            with mock.patch.object(run_pipeline, "run_category",
                                   side_effect=RuntimeError("out of memory")):
                with self.assertRaises(RuntimeError):
                    run_pipeline.run_pipeline.fn(root, root / "broken",
                                                 tracking_uri=tracking_uri)
            client = MlflowClient(tracking_uri)
            experiment = client.get_experiment_by_name("patchcore-mvtec")
            [run] = client.search_runs([experiment.experiment_id])
            self.assertEqual(run.info.status, "FAILED")


if __name__ == "__main__":
    unittest.main()
