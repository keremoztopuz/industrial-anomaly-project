import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from mlflow.tracking import MlflowClient

from scripts.registry import backfill_mlflow

MANIFEST = {"created_at_utc": "2026-09-30T12:00:00+00:00", "git_commit": "abc123",
            "git_dirty": False, "samples_json_sha256": "f00d", "image_size": 256,
            "batch_size": 8, "device": "mps", "max_patches": 8192, "projection_dim": 256,
            "seed": 42, "selection": "coreset"}
METRICS = {"bottle": {"image_auroc": 1.0, "pixel_auroc": 0.9},
           "grid": {"image_auroc": 0.6, "pixel_auroc": 0.8}}


def write_run(folder, metrics, manifest=None, log=False):
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if manifest is not None:
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if log:
        (folder / "run.log").write_text("done\n", encoding="utf-8")


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "artifacts"
        write_run(self.artifacts / "bank-size-sweep" / "coreset-8192", METRICS, MANIFEST, log=True)
        write_run(self.artifacts / "full", {"bottle": {"image_auroc": 0.8, "pixel_auroc": 0.9}})
        write_run(self.artifacts / "run-manifest-smoke", METRICS, MANIFEST)
        self.uri = f"sqlite:///{self.root / 'mlflow.db'}"
        self.client = MlflowClient(self.uri)
        self.experiment_id = self.client.create_experiment(
            "patchcore-mvtec", artifact_location=(self.root / "store").as_uri())

    def tearDown(self):
        self.temporary.cleanup()

    def backfill(self, *extra):
        argv = ["backfill_mlflow", "--artifacts-root", str(self.artifacts),
                "--tracking-uri", self.uri, *extra]
        output = StringIO()
        with mock.patch("sys.argv", argv), redirect_stdout(output):
            backfill_mlflow.main()
        return output.getvalue()

    def runs(self):
        return {run.info.run_name: run for run in self.client.search_runs([self.experiment_id])}

    def test_old_manifest_gets_defaults_metrics_tags_and_files(self):
        self.backfill()
        run = self.runs()["bank-size-sweep/coreset-8192"]
        self.assertEqual(run.info.status, "FINISHED")
        self.assertEqual(run.info.start_time, 1790769600000)
        self.assertEqual(run.data.params["max_patches"], "8192")
        self.assertEqual(run.data.params["selection"], "coreset")
        self.assertEqual(run.data.params["neighborhood"], "1")
        self.assertEqual(run.data.params["border"], "0")
        self.assertEqual(run.data.tags["defaulted_params"], "neighborhood,border")
        self.assertEqual(run.data.tags["git_commit"], "abc123")
        self.assertEqual(run.data.tags["report"], "reports/bank_size_sweep.md")
        self.assertEqual(run.data.tags["num_categories"], "2")
        self.assertEqual(run.data.metrics["image_auroc/grid"], 0.6)
        self.assertAlmostEqual(run.data.metrics["image_auroc_mean"], 0.8)
        self.assertAlmostEqual(run.data.metrics["pixel_auroc_mean"], 0.85)
        self.assertEqual(
            len(self.client.get_metric_history(run.info.run_id, "image_auroc_mean")), 1)
        self.assertEqual(
            sorted(item.path for item in self.client.list_artifacts(run.info.run_id)),
            ["manifest.json", "metrics.json", "run.log"])

    def test_baseline_without_manifest_uses_report_settings(self):
        self.backfill()
        run = self.runs()["full"]
        self.assertEqual(run.data.params["max_patches"], "2048")
        self.assertEqual(run.data.params["selection"], "random")
        self.assertIn("missing", run.data.tags["manifest"])
        self.assertEqual(run.data.tags["report"], "reports/baseline.md")

    def test_smoke_runs_are_skipped_unless_asked(self):
        self.backfill()
        self.assertNotIn("run-manifest-smoke", self.runs())
        self.backfill("--include-smoke")
        self.assertIn("run-manifest-smoke", self.runs())

    def test_second_run_adds_nothing(self):
        self.assertIn("2 new runs", self.backfill())
        self.assertIn("0 new runs, 2 skipped", self.backfill())
        self.assertEqual(len(self.runs()), 2)

    def test_skips_folder_already_logged_by_run_pipeline(self):
        logged = self.client.create_run(self.experiment_id, run_name="full")
        self.client.log_param(logged.info.run_id, "output_root",
                              str((self.artifacts / "full").resolve()))
        self.backfill()
        self.assertEqual(len(self.client.search_runs([self.experiment_id])), 2)

    def test_dry_run_writes_nothing(self):
        output = self.backfill("--dry-run")
        self.assertIn("2 runs found", output)
        self.assertEqual(self.runs(), {})

    def test_unknown_folder_without_manifest_is_an_error(self):
        write_run(self.artifacts / "mystery", METRICS)
        with self.assertRaises(ValueError):
            self.backfill("--dry-run")


if __name__ == "__main__":
    unittest.main()
