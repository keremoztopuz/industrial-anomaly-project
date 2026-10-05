import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from unittest import mock

from anomaly.monitoring import drift, drift_job
from anomaly.settings import settings
from tests.support.drift_data import REFERENCE, records


class FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class DriftJobTests(unittest.TestCase):
    def test_filter_selects_service_event_version_and_time(self):
        since = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
        log_filter = drift_job.build_filter("anomaly-api", "1", since)
        self.assertIn('resource.labels.service_name="anomaly-api"', log_filter)
        self.assertIn('jsonPayload.event="prediction"', log_filter)
        self.assertIn('jsonPayload.model_version="1"', log_filter)
        self.assertIn('timestamp>="2026-10-01T08:00:00Z"', log_filter)

    def test_fetch_follows_pages_newest_first(self):
        pages = [{"entries": [{"jsonPayload": {"n": 1}}, {"jsonPayload": {"n": 2}}],
                  "nextPageToken": "next"},
                 {"entries": [{"jsonPayload": {"n": 3}}]}]
        sent = []

        def fake_urlopen(request, timeout):
            sent.append(json.loads(request.data))
            return FakeResponse(json.dumps(pages[len(sent) - 1]).encode())

        with mock.patch.object(drift_job.urllib.request, "urlopen", side_effect=fake_urlopen):
            payloads = drift_job.fetch_predictions("proj", "token", "filter")
        self.assertEqual([p["n"] for p in payloads], [1, 2, 3])
        self.assertEqual(sent[0]["orderBy"], "timestamp desc")
        self.assertEqual(sent[0]["resourceNames"], ["projects/proj"])
        self.assertEqual(sent[1]["pageToken"], "next")

    def test_alarm_report_is_an_error_line_naming_the_category(self):
        reference = {"model_version": "1", "window": 50, "categories": {"bottle": REFERENCE}}
        predictions = records(50, shift=2)
        line = drift_job.report(reference, predictions, drift.check_all(predictions, reference))
        self.assertEqual(line["event"], "drift_check")
        self.assertEqual(line["severity"], "ERROR")
        self.assertEqual(line["status"], "alarm")
        self.assertIn("bottle", line["message"])
        json.dumps(line)

    def test_quiet_report_is_info(self):
        reference = {"model_version": "1", "window": 50, "categories": {"bottle": REFERENCE}}
        line = drift_job.report(reference, [], drift.check_all([], reference))
        self.assertEqual(line["severity"], "INFO")
        self.assertEqual(line["status"], "insufficient_data")

    def test_main_reads_reference_and_prints_one_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "drift_reference.json"
            path.write_text(json.dumps({"model_version": "1", "window": 50,
                                        "categories": {"bottle": REFERENCE}}), encoding="utf-8")
            output = StringIO()
            with mock.patch.object(settings, "drift_reference", path), \
                    mock.patch.object(drift_job, "metadata",
                                      side_effect=["proj", json.dumps({"access_token": "t"})]), \
                    mock.patch.object(drift_job, "fetch_predictions",
                                      return_value=records(50)) as fetch, \
                    redirect_stdout(output):
                drift_job.main()
        line = json.loads(output.getvalue())
        self.assertEqual(line["status"], "stable")
        self.assertEqual(fetch.call_args.args[:2], ("proj", "t"))


if __name__ == "__main__":
    unittest.main()
