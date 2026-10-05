"""Scheduled drift check that runs as a Cloud Run Job next to the service.

Reads the service's recent prediction lines from the Cloud Logging API, runs
the same checks as scripts/monitoring/check_drift.py, and writes one JSON line
with event "drift_check". The line has severity ERROR when a category is in
alarm, so a log-based alert policy can email on it. Uses the metadata server's
token, so it runs in the serving image with no keys.

Settings (environment variables, see anomaly/settings.py):
  DRIFT_REFERENCE  path to drift_reference.json (the bucket is mounted at
                   /models)
  SERVICE_NAME     Cloud Run service to read predictions from (default
                   anomaly-api)
  LOOKBACK_DAYS    how far back to read logs (default 30)

Run: python -m anomaly.monitoring.drift_job
"""

import json
import urllib.request
from datetime import datetime, timedelta, timezone

from anomaly.monitoring.drift import check_all, overall_status
from anomaly.settings import settings

METADATA = "http://metadata.google.internal/computeMetadata/v1"
LOGGING_API = "https://logging.googleapis.com/v2/entries:list"
MAX_ENTRIES = 5000


def metadata(path):
    request = urllib.request.Request(
        f"{METADATA}/{path}", headers={"Metadata-Flavor": "Google"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read().decode()


def build_filter(service, model_version, since):
    return (
        f'resource.type="cloud_run_revision" AND '
        f'resource.labels.service_name="{service}" '
        f'AND jsonPayload.event="prediction" '
        f'AND jsonPayload.model_version="{model_version}" '
        f'AND timestamp>="{since.strftime("%Y-%m-%dT%H:%M:%SZ")}"'
    )


def fetch_predictions(project, token, log_filter):
    """Return prediction payloads, newest first, following pages up to
    MAX_ENTRIES."""
    predictions, page_token = [], None
    while len(predictions) < MAX_ENTRIES:
        body = {
            "resourceNames": [f"projects/{project}"],
            "filter": log_filter,
            "orderBy": "timestamp desc",
            "pageSize": 1000,
        }
        if page_token:
            body["pageToken"] = page_token
        request = urllib.request.Request(
            LOGGING_API,
            data=json.dumps(body).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            page = json.load(response)
        predictions += [
            entry["jsonPayload"] for entry in page.get("entries", [])
        ]
        page_token = page.get("nextPageToken")
        if not page_token:
            break
    return predictions[:MAX_ENTRIES]


def report(reference, predictions, results):
    status = overall_status(results)
    flagged = sorted(
        category
        for category, result in results.items()
        if result["status"] in ("warning", "alarm")
    )
    summary = (
        f"drift check: {status}"
        + (f" ({', '.join(flagged)})" if flagged else "")
        + f", {len(predictions)} predictions for model "
          f"v{reference['model_version']}"
    )
    return {
        "event": "drift_check",
        "severity": "ERROR" if status == "alarm" else "INFO",
        "message": summary,
        "status": status,
        "model_version": reference["model_version"],
        "predictions": len(predictions),
        "categories": results,
    }


def main():
    if settings.drift_reference is None:
        raise SystemExit("DRIFT_REFERENCE must point to drift_reference.json")
    reference = json.loads(
        settings.drift_reference.read_text(encoding="utf-8")
    )
    since = datetime.now(timezone.utc) - timedelta(days=settings.lookback_days)
    project = metadata("project/project-id")
    token = json.loads(metadata("instance/service-accounts/default/token"))[
        "access_token"
    ]
    predictions = fetch_predictions(
        project,
        token,
        build_filter(settings.service_name, reference["model_version"], since),
    )
    results = check_all(predictions, reference)
    print(json.dumps(report(reference, predictions, results)), flush=True)


if __name__ == "__main__":
    main()
