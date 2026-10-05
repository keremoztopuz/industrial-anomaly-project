"""Settings shared by the API, the pipeline and the command line tools.

Values come from environment variables and, when the file exists, from a `.env` file in
the working directory (copy `.env.example` to start). A real environment variable wins
over `.env`, and a command line flag wins over both. None of these values is a secret:
the project has no keys or passwords, and Google Cloud access goes through IAM.
"""

from pathlib import Path

from pydantic import PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict

# MLflow experiment names are part of the project, not of one deployment.
TRAINING_EXPERIMENT = "patchcore-mvtec"
THRESHOLD_RULES_EXPERIMENT = "threshold-rules"
MONITORING_EXPERIMENT = "monitoring"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    # Where the banks and the data live.
    model_dir: Path = Path("artifacts/border-exclusion/coreset-16384-n3-b2/patchcore")
    dataset_root: Path = Path("data/mvtec-ad")
    artifacts_root: Path = Path("artifacts")

    # Which model is meant, and the registry it is tracked in.
    model_name: str = "patchcore-mvtec"
    model_version: str = "local"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    # Where the service is deployed.
    gcp_project: str = "industrial-anomaly-510523"
    gcp_region: str = "europe-west1"
    model_bucket: str = "anomaly-api"
    service_name: str = "anomaly-api"
    drift_job_name: str = "drift-check"
    drift_reference: Path | None = None  # DRIFT_REFERENCE, read by the drift job
    lookback_days: PositiveInt = 30  # LOOKBACK_DAYS, how far back the drift job reads logs


settings = Settings()
