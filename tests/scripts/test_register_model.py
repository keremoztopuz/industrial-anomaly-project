import hashlib
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from mlflow.tracking import MlflowClient

from scripts.registry import register_model


class RegisterModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.uri = f"sqlite:///{root / 'mlflow.db'}"
        self.client = MlflowClient(self.uri)
        experiment_id = self.client.create_experiment(
            "patchcore-mvtec", artifact_location=(root / "store").as_uri())
        self.run_id = self.client.create_run(
            experiment_id, tags={"source_dir": "border-exclusion/best"}).info.run_id
        self.model_dir = root / "banks"
        self.model_dir.mkdir()
        (self.model_dir / "bottle.pt").write_bytes(b"bank")
        (self.model_dir / "thresholds.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def register(self, source_dir="border-exclusion/best"):
        argv = ["register", "--source-dir", source_dir, "--model-dir", str(self.model_dir),
                "--tracking-uri", self.uri]
        with mock.patch("sys.argv", argv), redirect_stdout(StringIO()):
            register_model.main()

    def test_registers_banks_as_a_version_with_the_alias(self):
        self.register()
        version = self.client.get_model_version_by_alias("patchcore-mvtec", "production")
        self.assertEqual(str(version.version), "1")
        self.assertEqual(version.run_id, self.run_id)
        path = version.source.split(f"runs:/{self.run_id}/")[1]
        self.assertTrue(path.startswith("model/"))
        self.assertEqual(
            sorted(item.path for item in self.client.list_artifacts(self.run_id, path)),
            [f"{path}/bottle.pt", f"{path}/thresholds.json"])

    def test_registering_again_adds_a_version_and_moves_the_alias(self):
        self.register()
        first = self.client.get_model_version("patchcore-mvtec", "1")
        artifact = first.source.split(f"runs:/{self.run_id}/")[1] + "/bottle.pt"
        path = Path(self.client.download_artifacts(self.run_id, artifact))
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.model_dir / "bottle.pt").write_bytes(b"different bank")
        self.register()
        path = Path(self.client.download_artifacts(self.run_id, artifact))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        second = self.client.get_model_version("patchcore-mvtec", "2")
        self.assertNotEqual(first.source, second.source)
        self.assertEqual(
            str(self.client.get_model_version_by_alias(
                "patchcore-mvtec", "production").version), "2")

    def test_run_id_tags_and_no_alias(self):
        argv = ["register", "--run-id", self.run_id, "--model-dir", str(self.model_dir),
                "--tracking-uri", self.uri, "--alias", "", "--tag", "base_version=1"]
        with mock.patch("sys.argv", argv), redirect_stdout(StringIO()):
            register_model.main()
        [version] = self.client.search_model_versions("name = 'patchcore-mvtec'")
        self.assertEqual(version.tags, {"base_version": "1"})
        self.assertEqual(self.client.get_registered_model("patchcore-mvtec").aliases, {})

    def test_unknown_source_dir_is_an_error(self):
        with self.assertRaises(ValueError):
            self.register("does/not/exist")


if __name__ == "__main__":
    unittest.main()
