import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from mlflow.tracking import MlflowClient

from scripts import deploy_model


class DeployModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.uri = f"sqlite:///{self.root / 'mlflow.db'}"
        client = MlflowClient(self.uri)
        experiment_id = client.create_experiment(
            "patchcore-mvtec", artifact_location=(self.root / "store").as_uri())
        run_id = client.create_run(experiment_id).info.run_id
        client.create_registered_model("patchcore-mvtec")
        for _ in range(3):
            version = client.create_model_version(
                "patchcore-mvtec", source=f"runs:/{run_id}/model", run_id=run_id)
        client.set_registered_model_alias("patchcore-mvtec", "production", "2")
        self.downloaded = self.root / "downloaded" / "model"
        self.downloaded.mkdir(parents=True)
        for name in ("cable.pt", "bottle.pt", "thresholds.json", "test_scores.json"):
            (self.downloaded / name).write_text("x", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def deploy(self, bucket_has_version, *extra):
        calls = []

        def fake_gcloud(*args, check=True):
            calls.append(args)
            code = 0 if args[:2] != ("storage", "ls") or bucket_has_version else 1
            stdout = "registry/anomaly-api:abc123\n" if args[:3] == ("run", "services", "describe") else ""
            return subprocess.CompletedProcess(args, code, stdout=stdout)

        argv = ["deploy", "--tracking-uri", self.uri, *extra]
        with mock.patch("sys.argv", argv), \
                mock.patch.object(deploy_model, "gcloud", side_effect=fake_gcloud), \
                mock.patch.object(deploy_model.mlflow.artifacts, "download_artifacts",
                                  return_value=str(self.downloaded)) as download, \
                redirect_stdout(StringIO()):
            deploy_model.main()
        return calls, download

    def test_uploads_missing_version_then_points_cloud_run_at_it(self):
        calls, download = self.deploy(bucket_has_version=False)
        self.assertEqual(download.call_args.args[0], "models:/patchcore-mvtec@production")
        ls, cp, update, describe, job = calls
        self.assertEqual(ls[:3], ("storage", "ls", "gs://anomaly-api/patchcore-mvtec/v2/"))
        self.assertEqual([Path(path).name for path in cp[2:5]],
                         ["bottle.pt", "cable.pt", "thresholds.json"])
        self.assertEqual(cp[5], "gs://anomaly-api/patchcore-mvtec/v2/")
        self.assertEqual(update[:4], ("run", "services", "update", "anomaly-api"))
        self.assertIn("--update-env-vars", update)
        self.assertEqual(
            update[update.index("--update-env-vars") + 1],
            "MODEL_DIR=/models/patchcore-mvtec/v2,MODEL_NAME=patchcore-mvtec,MODEL_VERSION=2")
        self.assertNotIn("--image", update)
        self.assertEqual(describe[:4], ("run", "services", "describe", "anomaly-api"))
        self.assertEqual(job[:4], ("run", "jobs", "update", "drift-check"))
        self.assertEqual(job[job.index("--image") + 1], "registry/anomaly-api:abc123")
        self.assertEqual(job[job.index("--update-env-vars") + 1],
                         "DRIFT_REFERENCE=/models/patchcore-mvtec/v2/drift_reference.json")

    def test_existing_version_is_not_uploaded_again(self):
        calls, download = self.deploy(bucket_has_version=True)
        download.assert_not_called()
        self.assertEqual([call[:3] for call in calls],
                         [("storage", "ls", "gs://anomaly-api/patchcore-mvtec/v2/"),
                          ("run", "services", "update"), ("run", "services", "describe"),
                          ("run", "jobs", "update")])

    def test_empty_drift_job_skips_the_job_update(self):
        calls, _ = self.deploy(True, "--drift-job", "")
        self.assertEqual([call[:2] for call in calls], [("storage", "ls"), ("run", "services")])


if __name__ == "__main__":
    unittest.main()
