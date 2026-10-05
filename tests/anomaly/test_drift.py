import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np
from mlflow.tracking import MlflowClient

from anomaly.monitoring import drift
from scripts.monitoring import check_drift
from tests.support.drift_data import NORMAL, REFERENCE, RNG, records


class PsiTests(unittest.TestCase):
    def test_identical_distributions_score_zero(self):
        self.assertAlmostEqual(drift.psi(NORMAL, NORMAL), 0.0)

    def test_shift_scores_higher_than_noise(self):
        same = drift.psi(NORMAL, RNG.normal(10, 1, 50))
        shifted = drift.psi(NORMAL, RNG.normal(12, 1, 50))
        self.assertGreater(shifted, 5 * same)

    def test_empty_bins_stay_finite(self):
        self.assertTrue(np.isfinite(drift.psi(NORMAL, [100.0] * 50)))

    def test_status_cut_offs(self):
        self.assertEqual(drift.psi_status(0.05), "stable")
        self.assertEqual(drift.psi_status(0.2), "warning")
        self.assertEqual(drift.psi_status(0.3), "alarm")
        self.assertEqual(drift.psi_status(0.3, warn=0.4, alarm=0.6), "stable")

    def test_alarm_rate_rule_is_three_times_expected(self):
        self.assertEqual(drift.alarm_rate_status([True] * 10 + [False] * 40), "stable")
        self.assertEqual(drift.alarm_rate_status([True] * 11 + [False] * 39), "alarm")


class CalibrationTests(unittest.TestCase):
    def test_no_drift_windows_rarely_cross_the_alarm_cut_off(self):
        warn, alarm = drift.calibrate_psi_thresholds(NORMAL)
        self.assertLess(warn, alarm)
        rng = np.random.default_rng(7)
        crossed = np.mean([drift.psi(NORMAL, rng.choice(NORMAL, 50)) > alarm
                           for _ in range(1000)])
        self.assertLess(crossed, 0.03)

    def test_is_reproducible(self):
        first = drift.calibrate_psi_thresholds(NORMAL, trials=200)
        second = drift.calibrate_psi_thresholds(NORMAL, trials=200)
        self.assertEqual(first, second)


class CheckCategoryTests(unittest.TestCase):
    def test_too_few_predictions_is_insufficient(self):
        result = drift.check_category(records(49), REFERENCE)
        self.assertEqual(result, {"status": "insufficient_data", "predictions": 49})

    def test_matching_window_is_stable(self):
        result = drift.check_category(records(60), REFERENCE)
        self.assertEqual(result["status"], "stable")
        self.assertEqual(result["predictions"], 50)

    def test_score_shift_raises_alarm(self):
        result = drift.check_category(records(50, shift=2), REFERENCE)
        self.assertEqual(result["anomaly_score_status"], "alarm")
        self.assertEqual(result["status"], "alarm")

    def test_flagged_share_raises_alarm(self):
        result = drift.check_category(records(50, flagged=20), REFERENCE)
        self.assertEqual(result["alarm_rate_status"], "alarm")
        self.assertAlmostEqual(result["alarm_rate"], 0.4)

    def test_input_change_alone_does_not_change_status(self):
        result = drift.check_category(records(50, brightness_shift=3), REFERENCE)
        self.assertEqual(result["status"], "stable")
        self.assertAlmostEqual(result["brightness_change"], 0.3, delta=0.05)
        self.assertAlmostEqual(result["contrast_change"], 0.0, delta=0.05)

    def test_missing_decisions_are_not_normal(self):
        for decision in (None, 0, "false"):
            recent = [{**r, "is_anomaly": decision} for r in records(50)]
            result = drift.check_category(recent, REFERENCE)
            self.assertIsNone(result["alarm_rate"])
            self.assertEqual(result["alarm_rate_status"], "insufficient_data")
            self.assertEqual(result["status"], "insufficient_data")
            self.assertEqual(result["decisions"], 0)

    def test_partial_decisions_and_zero_reference(self):
        recent = records(50, shift=2)
        recent[0]["is_anomaly"] = None
        reference = {**REFERENCE, "brightness": {"reference": [0]},
                     "contrast": {"reference": [0]}}
        result = drift.check_category(recent, reference)
        self.assertEqual(result["decisions"], 49)
        self.assertIsNone(result["alarm_rate"])
        self.assertEqual(result["status"], "alarm")
        for signal in ("brightness", "contrast"):
            self.assertIsNone(result[f"{signal}_change"])
            self.assertEqual(result[f"{signal}_change_status"], "zero_reference")
        self.assertEqual(check_drift.percent(None), "n/a")
        with mock.patch.object(check_drift, "MlflowClient") as client:
            check_drift.log_to_mlflow("unused", {"model_version": "2", "window": 50},
                                      {"bottle": result}, 50)
            self.assertTrue(client.return_value.log_metric.called)
            for call in client.return_value.log_metric.call_args_list:
                self.assertIsNotNone(call.args[2])


class CheckDriftScriptTests(unittest.TestCase):
    def test_reads_logs_reports_and_logs_to_mlflow(self):
        # The script only reads a reference inside the current directory.
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            reference_path = root / "drift_reference.json"
            reference_path.write_text(json.dumps({
                "model_version": "1", "window": 50,
                "categories": {"bottle": REFERENCE, "pill": REFERENCE}}), encoding="utf-8")
            payloads = records(50, shift=2) + records(10, category="pill")
            logs = json.dumps([{"jsonPayload": payload} for payload in payloads])
            uri = f"sqlite:///{root / 'mlflow.db'}"
            client = MlflowClient(uri)
            experiment_id = client.create_experiment(
                "monitoring", artifact_location=(root / "store").as_uri())

            argv = ["check", "--reference", str(reference_path), "--tracking-uri", uri]
            output = StringIO()
            with mock.patch("sys.argv", argv), \
                    mock.patch.object(check_drift, "gcloud",
                                      return_value=subprocess.CompletedProcess(
                                          [], 0, stdout=logs)) as gcloud, \
                    redirect_stdout(output), self.assertRaises(SystemExit) as exit_info:
                check_drift.main()

            self.assertEqual(exit_info.exception.code, 1)
            query = gcloud.call_args.args[2]
            self.assertIn('jsonPayload.event="prediction"', query)
            self.assertIn('jsonPayload.model_version="1"', query)
            self.assertIn("overall: alarm", output.getvalue())
            [run] = client.search_runs([experiment_id])
            self.assertEqual(run.data.tags["status"], "alarm")
            self.assertEqual(run.data.tags["status/bottle"], "alarm")
            self.assertEqual(run.data.tags["status/pill"], "insufficient_data")
            self.assertIn("psi_anomaly_score/bottle", run.data.metrics)
            self.assertIn("brightness_change/bottle", run.data.metrics)


class ReferencePathTests(unittest.TestCase):
    def test_reference_outside_the_repository_is_rejected(self):
        argv = ["check", "--reference", "/etc/drift_reference.json", "--no-mlflow"]
        with mock.patch("sys.argv", argv), redirect_stdout(StringIO()), \
                mock.patch("sys.stderr", StringIO()), self.assertRaises(SystemExit) as exit_info:
            check_drift.main()
        self.assertEqual(exit_info.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
