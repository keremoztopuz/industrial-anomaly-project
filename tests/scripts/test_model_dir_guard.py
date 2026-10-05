import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

from scripts.modeling import calibrate_thresholds, compare_threshold_rules
from scripts.monitoring import build_drift_reference
from scripts.reporting import plot_threshold_results


class ModelDirGuardTests(unittest.TestCase):
    """The scripts that write next to the banks only accept a directory below
    the cwd."""

    def run_main(self, module, model_dir):
        argv = ["script", "--model-dir", str(model_dir)]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("sys.stderr", StringIO()),
            self.assertRaises(SystemExit) as exit_info,
        ):
            module.main()
        return exit_info.exception.code

    def test_directory_outside_the_working_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory)
            if outside.resolve().is_relative_to(Path.cwd().resolve()):
                self.skipTest("temp directory is below the working directory")
            for module in (
                calibrate_thresholds,
                compare_threshold_rules,
                build_drift_reference,
                plot_threshold_results,
            ):
                with self.subTest(script=module.__name__):
                    self.assertEqual(self.run_main(module, outside), 2)


if __name__ == "__main__":
    unittest.main()
