import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from anomaly.settings import Settings
from api.config import ApiSettings

EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def clean_environment(**values):
    return mock.patch.dict(os.environ, values, clear=True)


class SettingsTests(unittest.TestCase):
    def test_defaults_without_environment_or_env_file(self):
        with clean_environment():
            settings = ApiSettings(_env_file=None)
        self.assertEqual(settings.model_name, "patchcore-mvtec")
        self.assertEqual(settings.model_version, "local")
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, 8000)
        self.assertEqual(
            settings.max_request_bytes, settings.max_upload_bytes + 1024 * 1024
        )

    def test_environment_variables_override_defaults_case_insensitively(self):
        with clean_environment(
            MODEL_DIR="/models/patchcore-mvtec/v2",
            MODEL_VERSION="2",
            port="8080",
            MAX_UPLOAD_BYTES="5",
        ):
            settings = ApiSettings(_env_file=None)
        self.assertEqual(
            settings.model_dir, Path("/models/patchcore-mvtec/v2")
        )
        self.assertEqual(settings.model_version, "2")
        self.assertEqual(settings.port, 8080)
        self.assertEqual(settings.max_request_bytes, 5 + 1024 * 1024)

    def test_explicit_request_limit_wins_over_the_derived_one(self):
        with clean_environment(MAX_UPLOAD_BYTES="5", MAX_REQUEST_BYTES="7"):
            self.assertEqual(ApiSettings(_env_file=None).max_request_bytes, 7)

    def test_env_file_is_read_and_unknown_keys_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "MODEL_VERSION=7\nSOMETHING_ELSE=1\n", encoding="utf-8"
            )
            with clean_environment():
                settings = Settings(_env_file=env_file)
        self.assertEqual(settings.model_version, "7")

    def test_real_environment_wins_over_the_env_file(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("MODEL_VERSION=7\n", encoding="utf-8")
            with clean_environment(MODEL_VERSION="2"):
                self.assertEqual(
                    Settings(_env_file=env_file).model_version, "2"
                )

    def test_nonpositive_limits_are_rejected(self):
        for name in (
            "MAX_UPLOAD_BYTES",
            "MAX_IMAGE_PIXELS",
            "MAX_REQUEST_BYTES",
            "PORT",
        ):
            with self.subTest(name=name), clean_environment(**{name: "0"}):
                with self.assertRaises(ValidationError):
                    ApiSettings(_env_file=None)

    def test_env_example_matches_the_defaults(self):
        with clean_environment():
            defaults = ApiSettings(_env_file=None)
            example = ApiSettings(_env_file=EXAMPLE)
        self.assertEqual(example.model_dump(), defaults.model_dump())


if __name__ == "__main__":
    unittest.main()
