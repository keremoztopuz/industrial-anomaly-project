"""Check the live service's recent predictions for drift and log the result to MLflow.

Reads prediction lines from Cloud Logging, keeps the newest `window` per category for
the model version the reference was built for, and compares scores, brightness,
contrast and the alarm rate with drift_reference.json. Exits with status 1 when any
category is in alarm, so a scheduler can act on it.

Run from the repository root: python -m scripts.check_drift
"""

import argparse
import json
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient

from anomaly.drift import INPUT_SIGNALS, check_all, overall_status
from scripts.deploy_model import gcloud

EXPERIMENT = "monitoring"


def fetch_predictions(project, service, model_version, freshness, limit):
    """Return prediction log payloads, newest first."""
    query = (f'resource.type="cloud_run_revision" AND resource.labels.service_name="{service}" '
             f'AND jsonPayload.event="prediction" '
             f'AND jsonPayload.model_version="{model_version}"')
    result = gcloud("logging", "read", query, "--project", project, "--freshness", freshness,
                    "--limit", str(limit), "--order", "desc", "--format", "json")
    return [entry["jsonPayload"] for entry in json.loads(result.stdout or "[]")]


def log_to_mlflow(tracking_uri, reference, results, fetched, label=None):
    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = (experiment.experiment_id if experiment
                     else client.create_experiment(EXPERIMENT))
    overall = overall_status(results)
    tags = {"status": overall, **({"label": label} if label else {})}
    run = client.create_run(
        experiment_id, run_name=label or f"drift-check-v{reference['model_version']}",
        tags=tags)
    run_id = run.info.run_id
    for key, value in {"model_version": reference["model_version"],
                       "window": reference["window"], "predictions_fetched": fetched}.items():
        client.log_param(run_id, key, value)
    for category, result in results.items():
        client.set_tag(run_id, f"status/{category}", result["status"])
        for key in ["psi_anomaly_score", "alarm_rate"] + [f"{s}_change" for s in INPUT_SIGNALS]:
            if result.get(key) is not None:
                client.log_metric(run_id, f"{key}/{category}", result[key])
    client.set_terminated(run_id)
    return overall


def percent(value):
    return "n/a" if value is None else f"{value:+.1%}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=Path("artifacts/border-exclusion/coreset-16384-n3-b2/patchcore/"
                                     "drift_reference.json"))
    parser.add_argument("--project", default="industrial-anomaly-510523")
    parser.add_argument("--service", default="anomaly-api")
    parser.add_argument("--freshness", default="30d", help="How far back to read logs")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--label", help="Name for this check in MLflow, e.g. a simulation scenario")
    args = parser.parse_args()

    reference_path = args.reference.resolve()
    if not reference_path.is_relative_to(Path.cwd().resolve()):
        parser.error("--reference must be inside the current directory")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    predictions = fetch_predictions(args.project, args.service, reference["model_version"],
                                    args.freshness, args.limit)
    results = check_all(predictions, reference)
    for category, result in results.items():
        details = "" if result["status"] == "insufficient_data" else (
            f"  score {result['anomaly_score_status']}, alarm rate {percent(result['alarm_rate'])}"
            f" | brightness {percent(result['brightness_change'])}, "
            f"contrast {percent(result['contrast_change'])}")
        print(
            f"{category:11} {result['status']:17} {result['predictions']:3d} predictions{details}")
    overall = overall_status(results)
    if not args.no_mlflow:
        log_to_mlflow(args.tracking_uri, reference, results, len(predictions), args.label)
    print(f"overall: {overall} ({len(predictions)} predictions for model "
          f"v{reference['model_version']})")
    if overall == "alarm":
        sys.exit(1)


if __name__ == "__main__":
    main()
