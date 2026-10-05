"""Register a run's PatchCore banks as a new patchcore-mvtec version.

Run from the repository root: python -m scripts.registry.register_model
"""

import argparse
from pathlib import Path
from uuid import uuid4

from mlflow.tracking import MlflowClient

from anomaly.settings import TRAINING_EXPERIMENT, settings

MODEL_NAME = settings.model_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="border-exclusion/coreset-16384-n3-b2",
        help="source_dir tag of the run to register",
    )
    parser.add_argument("--model-dir", type=Path, default=settings.model_dir)
    parser.add_argument(
        "--run-id",
        help="Register this run directly instead of looking up --source-dir",
    )
    parser.add_argument(
        "--alias",
        default="production",
        help="Alias to move; empty to leave aliases",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Tag for the new version, e.g. base_version=1; repeatable",
    )
    parser.add_argument("--tracking-uri", default=settings.mlflow_tracking_uri)
    args = parser.parse_args()
    client = MlflowClient(args.tracking_uri)

    tags = dict(tag.split("=", 1) for tag in args.tag)
    if args.run_id:
        run_id = client.get_run(args.run_id).info.run_id
    else:
        experiment = client.get_experiment_by_name(TRAINING_EXPERIMENT)
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.source_dir = '{args.source_dir}'",
        )
        if len(runs) != 1:
            raise ValueError(
                f"Expected one run with source_dir={args.source_dir}, found "
                f"{len(runs)}"
            )
        run_id = runs[0].info.run_id

    artifact_path = f"model/{uuid4().hex}"
    client.log_artifacts(
        run_id, str(args.model_dir), artifact_path=artifact_path
    )
    if not client.search_registered_models(
        filter_string=f"name = '{MODEL_NAME}'"
    ):
        client.create_registered_model(MODEL_NAME)

    version = client.create_model_version(
        MODEL_NAME,
        source=f"runs:/{run_id}/{artifact_path}",
        run_id=run_id,
        tags=tags,
    )

    if args.alias:
        client.set_registered_model_alias(
            MODEL_NAME, args.alias, version.version
        )
        print(f"{MODEL_NAME} v{version.version} -> @{args.alias}")
    else:
        print(f"{MODEL_NAME} v{version.version} registered, aliases unchanged")


if __name__ == "__main__":
    main()
