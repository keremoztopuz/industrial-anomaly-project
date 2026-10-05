"""Deploy the patchcore-mvtec version behind an MLflow alias to Cloud Run.

Run from the repository root: python -m scripts.registry.deploy_model
"""

import argparse
import base64
import hashlib
import json
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from anomaly.model.thresholds import load_thresholds
from scripts.cloud import gcloud

MODEL_NAME = "patchcore-mvtec"
# The drift job runs this module from the serving image; the command is set on every
# deployment so the job and the code layout cannot drift apart.
DRIFT_JOB_COMMAND = "python"
DRIFT_JOB_ARGS = "-m,anomaly.monitoring.drift_job"


def deployment_files(local, version, drift_enabled):
    """Validate the serving bundle before any upload or service mutation."""
    banks = sorted(local.glob("*.pt"))
    if not banks or any(not path.is_file() or path.stat().st_size == 0 for path in banks):
        raise ValueError("Model bundle must contain nonempty category banks (*.pt)")
    threshold_path = local / "thresholds.json"
    if not threshold_path.is_file():
        raise ValueError("Model bundle is missing thresholds.json")
    if set(load_thresholds(threshold_path)) != {path.stem for path in banks}:
        raise ValueError("Threshold categories must match the model banks")
    files = banks + [threshold_path]
    reference_path = local / "drift_reference.json"
    if drift_enabled or reference_path.exists():
        if not reference_path.is_file():
            raise ValueError("Drift job requires drift_reference.json")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if str(reference.get("model_version")) != str(version):
            raise ValueError(
                "drift_reference.json model_version does not match the resolved version")
        if set(reference.get("categories", {})) != {path.stem for path in banks}:
            raise ValueError("Drift reference categories must match the model banks")
        if type(reference.get("window")) is not int or reference["window"] <= 0:
            raise ValueError("Drift reference window must be a positive integer")
        files.append(reference_path)
    return files


def file_identity(path):
    """Match GCS's size and base64 MD5 metadata without loading a bank into RAM."""
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "md5").digest()
    return path.stat().st_size, base64.b64encode(digest).decode("ascii")


def remote_files(bucket_path, project):
    """Empty successful listing means absent; command/auth/network errors propagate."""
    listing = gcloud("storage", "objects", "list", bucket_path + "**",
                     "--raw", "--format=json", "--project", project)
    objects = json.loads(listing.stdout)
    prefix = bucket_path.split("/", 3)[3]
    result = {}
    for name in sorted({item["name"] for item in objects}):
        # Describe the live object, even when a versioned bucket lists older generations.
        metadata = json.loads(gcloud(
            "storage", "objects", "describe", bucket_path + name.removeprefix(prefix),
            "--raw", "--format=json", "--project", project).stdout)
        result[name.removeprefix(prefix)] = (int(metadata["size"]), metadata.get("md5Hash"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", default="production")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--bucket", default="anomaly-api")
    parser.add_argument("--project", default="industrial-anomaly-510523")
    parser.add_argument("--region", default="europe-west1")
    parser.add_argument("--service", default="anomaly-api")
    parser.add_argument("--drift-job", default="drift-check",
                        help="Cloud Run Job that checks drift; empty to skip")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(args.tracking_uri)

    version = str(client.get_model_version_by_alias(MODEL_NAME, args.alias).version)
    bucket_path = f"gs://{args.bucket}/{MODEL_NAME}/v{version}/"
    container_path = f"/models/{MODEL_NAME}/v{version}"

    with tempfile.TemporaryDirectory() as directory:
        local = Path(mlflow.artifacts.download_artifacts(
            f"models:/{MODEL_NAME}/{version}", dst_path=directory))
        files = deployment_files(local, version, bool(args.drift_job))
        expected = {path.name: file_identity(path) for path in files}
        remote = remote_files(bucket_path, args.project)
        if not remote:
            # Creation precondition prevents a concurrent deployment overwriting files.
            gcloud("storage", "cp", *(str(path) for path in files), bucket_path,
                   "--if-generation-match=0", "--project", args.project)
            remote = remote_files(bucket_path, args.project)
        if remote != expected:
            raise ValueError(f"Incomplete or different artifacts at {bucket_path}; "
                             "refusing to update the service or drift job")

    gcloud("run",
           "services",
           "update", args.service,
           "--project", args.project,
           "--region", args.region,
           "--update-env-vars",
           f"MODEL_DIR={container_path},MODEL_NAME={MODEL_NAME},MODEL_VERSION={version}",
           )

    print(f"{args.service} now serves {MODEL_NAME} v{version} (@{args.alias})")

    if args.drift_job:
        # Keep the drift check on the same image and the new version's reference.
        image = gcloud("run", "services", "describe", args.service, "--project", args.project,
                       "--region", args.region,
                       "--format", "value(spec.template.spec.containers[0].image)").stdout.strip()
        gcloud("run", "jobs", "update", args.drift_job, "--project", args.project,
               "--region", args.region, "--image", image,
               "--command", DRIFT_JOB_COMMAND, f"--args={DRIFT_JOB_ARGS}",
               "--update-env-vars", f"DRIFT_REFERENCE={container_path}/drift_reference.json")
        print(f"{args.drift_job} now checks v{version} with {image}")


if __name__ == "__main__":
    main()
