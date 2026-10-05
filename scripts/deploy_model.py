"""Deploy the patchcore-mvtec version behind an MLflow alias to Cloud Run.

Run from the repository root: python -m scripts.deploy_model
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "patchcore-mvtec"

def gcloud(*args, check=True):
    """Run a gcloud command; the full path keeps SonarCloud happy."""
    executable = shutil.which("gcloud")
    if executable is None:
        raise RuntimeError("gcloud not found on PATH")
    return subprocess.run([executable, *args], check=check, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", default="production")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--bucket", default="anomaly-api")
    parser.add_argument("--project", default="industrial-anomaly-510523")
    parser.add_argument("--region", default="europe-west1")
    parser.add_argument("--service", default="anomaly-api")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(args.tracking_uri)

    version = client.get_model_version_by_alias(MODEL_NAME, args.alias)
    bucket_path = f"gs://{args.bucket}/{MODEL_NAME}/v{version.version}/"
    container_path = f"/models/{MODEL_NAME}/v{version.version}"

    listing = gcloud("storage", "ls", bucket_path, "--project", args.project, check=False)
    if listing.returncode != 0:
        with tempfile.TemporaryDirectory() as directory:
            local = Path(mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}@{args.alias}", dst_path=directory))
            files = [str(p) for p in sorted(local.glob("*.pt"))] + [str(local / "thresholds.json")]
            if (local / "drift_reference.json").is_file():
                files.append(str(local / "drift_reference.json"))
            gcloud("storage", "cp", *files, bucket_path, "--project", args.project)

    gcloud("run",
           "services",
           "update", args.service,
           "--project", args.project,
           "--region", args.region,
           "--update-env-vars", f"MODEL_DIR={container_path},MODEL_NAME={MODEL_NAME},MODEL_VERSION={version.version}",
           )

    print(f"{args.service} now serves {MODEL_NAME} v{version.version} (@{args.alias})")


if __name__ == "__main__":
    main()
