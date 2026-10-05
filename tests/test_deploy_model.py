import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from scripts import deploy_model


class DeployModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.local = Path(self.temporary.name)
        for name in ("bottle.pt", "cable.pt"):
            (self.local / name).write_bytes(b"bank")
        (self.local / "thresholds.json").write_text(json.dumps({"categories": {
            "bottle": {"threshold": 1}, "cable": {"threshold": 2}}}))
        (self.local / "drift_reference.json").write_text(json.dumps({
            "model_version": "2", "window": 50, "categories": {"bottle": {}, "cable": {}}}))
        (self.local / "test_scores.json").write_text("not a serving artifact")
        self.expected = {p.name: deploy_model.file_identity(p) for p in self.local.iterdir()
                         if p.name != "test_scores.json"}
        self.remote = {}
        self.calls = []
        self.list_error = False
        self.corrupt_upload = False
        self.client = mock.Mock()
        self.client.get_model_version_by_alias.return_value.version = "2"

    def cloud(self, *args, check=True):
        self.calls.append(args)
        if args[:3] == ("storage", "objects", "list"):
            if self.list_error:
                raise subprocess.CalledProcessError(1, args, stderr="Permission denied")
            data = [{"name": f"patchcore-mvtec/v2/{name}"} for name in self.remote]
            output = json.dumps(data)
        elif args[:3] == ("storage", "objects", "describe"):
            size, digest = self.remote[args[3].split("/v2/")[1]]
            output = json.dumps({"size": str(size), "md5Hash": digest})
        elif args[:2] == ("storage", "cp"):
            self.assertIn("--if-generation-match=0", args)
            self.remote = dict(self.expected)
            if self.corrupt_upload:
                self.remote.pop("cable.pt")
            output = ""
        elif args[:3] == ("run", "services", "describe"):
            output = "registry/anomaly-api:abc123\n"
        else:
            output = ""
        return subprocess.CompletedProcess(args, 0, stdout=output)

    def deploy(self, *extra):
        def download(uri, dst_path):
            self.assertEqual(uri, "models:/patchcore-mvtec/2")
            # The alias moves after resolution; downloading must stay pinned to v2.
            self.client.get_model_version_by_alias.return_value.version = "3"
            return str(self.local)

        # Use an immutable snapshot like the real MLflow response.
        self.client.get_model_version_by_alias.return_value = type(
            "Version", (), {"version": "2"})()
        with mock.patch("sys.argv", ["deploy", *extra]), \
                mock.patch.object(deploy_model, "MlflowClient", return_value=self.client), \
                mock.patch.object(deploy_model.mlflow, "set_tracking_uri"), \
                mock.patch.object(deploy_model, "gcloud", side_effect=self.cloud), \
                mock.patch.object(deploy_model.mlflow.artifacts, "download_artifacts",
                                  side_effect=download), redirect_stdout(StringIO()):
            deploy_model.main()

    def assert_no_update(self):
        self.assertFalse(any(call[0] == "run" for call in self.calls))

    def test_uploads_missing_version_and_pins_numeric_version(self):
        self.deploy()
        self.client.get_model_version_by_alias.assert_called_once()
        updates = [call for call in self.calls if call[:3] == ("run", "services", "update")]
        self.assertEqual(len(updates), 1)
        self.assertIn("MODEL_DIR=/models/patchcore-mvtec/v2,"
                      "MODEL_NAME=patchcore-mvtec,MODEL_VERSION=2",
                      updates[0])
        job = self.calls[-1]
        self.assertEqual(job[:4], ("run", "jobs", "update", "drift-check"))
        self.assertIn("registry/anomaly-api:abc123", job)
        self.assertIn("DRIFT_REFERENCE=/models/patchcore-mvtec/v2/drift_reference.json", job)

    def test_existing_version_is_verified_without_upload(self):
        self.remote = self.expected
        self.deploy()
        self.assertFalse(any(call[:2] == ("storage", "cp") for call in self.calls))

    def test_empty_drift_job_skips_reference_and_job_update(self):
        (self.local / "drift_reference.json").unlink()
        self.expected.pop("drift_reference.json")
        self.remote = self.expected
        self.deploy("--drift-job", "")
        self.assertFalse(any(call[:2] == ("run", "jobs") for call in self.calls))

    def test_partial_remote_version_is_rejected_without_upload(self):
        self.remote = {"bottle.pt": self.expected["bottle.pt"]}
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.deploy()
        self.assert_no_update()
        self.assertFalse(any(call[:2] == ("storage", "cp") for call in self.calls))

    def test_wrong_size_hash_or_extra_name_is_rejected(self):
        for name, identity in (("bottle.pt", (0, "bad")),
                               ("bottle.pt", (4, "bad")), ("extra.pt", (4, "bad"))):
            with self.subTest(name=name, identity=identity):
                self.remote = {**self.expected, name: identity}
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    self.deploy()
                self.assert_no_update()

    def test_failed_listing_is_not_treated_as_absent(self):
        self.list_error = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.deploy()
        self.assert_no_update()
        self.assertEqual(len(self.calls), 1)

    def test_partial_upload_is_rejected(self):
        self.corrupt_upload = True
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.deploy()
        self.assert_no_update()

    def test_missing_reference_is_rejected_before_cloud_calls(self):
        (self.local / "drift_reference.json").unlink()
        with self.assertRaisesRegex(ValueError, "requires drift_reference"):
            self.deploy()
        self.assertEqual(self.calls, [])

    def test_reference_version_mismatch_is_rejected(self):
        (self.local / "drift_reference.json").write_text('{"model_version": "1"}')
        with self.assertRaisesRegex(ValueError, "model_version"):
            self.deploy()
        self.assertEqual(self.calls, [])

    def test_invalid_local_bundle_is_rejected_before_cloud_calls(self):
        cases = [
            ("thresholds.json", None),
            ("thresholds.json", b"not json"),
            ("thresholds.json", b'{"categories": {}}'),
            ("bottle.pt", b""),
            ("drift_reference.json", b'{"model_version": "2", "categories": {}}'),
            ("drift_reference.json", json.dumps({
                "model_version": "2", "window": 0,
                "categories": {"bottle": {}, "cable": {}}}).encode()),
        ]
        for name, contents in cases:
            path = self.local / name
            original = path.read_bytes()
            try:
                if contents is None:
                    path.unlink()
                else:
                    path.write_bytes(contents)
                with self.subTest(name=name, contents=contents), self.assertRaises(ValueError):
                    self.deploy()
                self.assertEqual(self.calls, [])
            finally:
                path.write_bytes(original)
        for path in self.local.glob("*.pt"):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "nonempty category banks"):
            self.deploy()
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
