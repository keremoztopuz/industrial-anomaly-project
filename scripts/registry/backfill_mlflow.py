"""Record runs made before MLflow tracking in the patchcore-mvtec experiment.

Each artifacts/**/metrics.json becomes one FINISHED run with the same params, metrics
and tags that run_pipeline.py logs, plus tags pointing back to its folder and report.
Settings missing from older manifests take the defaults those banks load with.
Running it again skips folders that already have a run.

Run from the repository root: python -m scripts.registry.backfill_mlflow
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

EXPERIMENT = "patchcore-mvtec"
PARAM_DEFAULTS = {"selection": "random", "neighborhood": 1, "border": 0}
PARAM_KEYS = ("image_size", "batch_size", "device", "max_patches", "projection_dim",
              "seed", "selection", "neighborhood", "border")
# Settings of artifacts/full, the baseline run that predates manifest.json.
BASELINE = {"image_size": 256, "batch_size": 8, "device": "mps", "max_patches": 2048,
            "projection_dim": 256, "seed": 42, "selection": "random",
            "neighborhood": 1, "border": 0}
REPORTS = {
    "full": "reports/baseline.md",
    "coreset-experiment": "reports/coreset_comparison.md",
    "bank-size-sweep": "reports/bank_size_sweep.md",
    "local-aggregation": "reports/local_aggregation.md",
    "local-aggregation-16384": "reports/local_aggregation.md",
    "border-exclusion": "reports/border_exclusion.md",
}


def is_smoke(source_dir):
    return any(word in source_dir for word in ("smoke", "recheck"))


def describe_run(run_dir, artifacts_root):
    """Return the name, start time, params, metrics, tags and files for one run folder."""
    source_dir = run_dir.relative_to(artifacts_root).as_posix()
    metrics_path = run_dir / "metrics.json"
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest_path = run_dir / "manifest.json"
    study = source_dir.split("/")[0]
    tags = {"backfilled": "true", "source_dir": source_dir, "study": study,
            "num_categories": str(len(results))}
    if study in REPORTS:
        tags["report"] = REPORTS[study]

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        defaulted = [key for key in PARAM_DEFAULTS if manifest.get(key) is None]
        params = {key: manifest.get(key) for key in PARAM_KEYS}
        params.update({key: PARAM_DEFAULTS[key] for key in defaulted})
        if defaulted:
            tags["defaulted_params"] = ",".join(defaulted)
        for key in ("git_commit", "git_dirty", "samples_json_sha256"):
            if manifest.get(key) is not None:
                tags[key] = str(manifest[key])
        start = datetime.fromisoformat(manifest["created_at_utc"]).timestamp()
    elif study == "full":
        params = dict(BASELINE)
        tags["manifest"] = "missing (settings from reports/baseline.md)"
        start = metrics_path.stat().st_mtime
    else:
        raise ValueError(f"{source_dir}: no manifest.json and no known settings")
    params["output_root"] = str(run_dir.resolve())

    metrics = {}
    for category, values in sorted(results.items()):
        metrics[f"image_auroc/{category}"] = values["image_auroc"]
        metrics[f"pixel_auroc/{category}"] = values["pixel_auroc"]
    metrics["image_auroc_mean"] = sum(v["image_auroc"] for v in results.values()) / len(results)
    metrics["pixel_auroc_mean"] = sum(v["pixel_auroc"] for v in results.values()) / len(results)

    files = [path for path in (manifest_path, metrics_path, run_dir / "run.log") if path.is_file()]
    return {"name": source_dir, "start_ms": int(start * 1000), "params": params,
            "metrics": metrics, "tags": tags, "files": files}


def find_runs(artifacts_root, include_smoke=False):
    runs = []
    for metrics_path in sorted(artifacts_root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        source_dir = run_dir.relative_to(artifacts_root).as_posix()
        if include_smoke or not is_smoke(source_dir):
            runs.append(describe_run(run_dir, artifacts_root))
    return runs


def existing_sources(client, experiment_id):
    """Folders already recorded, by backfill tag or by run_pipeline's output_root param."""
    sources, roots = set(), set()
    for run in client.search_runs([experiment_id], max_results=10_000):
        sources.add(run.data.tags.get("source_dir"))
        roots.add(run.data.params.get("output_root"))
    return sources, roots


def log_run(client, experiment_id, run):
    created = client.create_run(experiment_id, start_time=run["start_ms"],
                                tags=run["tags"], run_name=run["name"])
    run_id = created.info.run_id
    client.log_batch(
        run_id,
        metrics=[Metric(key, value, run["start_ms"], 0) for key, value in run["metrics"].items()],
        params=[Param(key, str(value)) for key, value in run["params"].items()],
        tags=[RunTag("mlflow.runName", run["name"])],
    )
    for path in run["files"]:
        client.log_artifact(run_id, str(path))
    client.set_terminated(run_id, "FINISHED", end_time=run["start_ms"])
    return run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--include-smoke", action="store_true",
                        help="Also record smoke and recheck runs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be recorded without writing to MLflow")
    args = parser.parse_args()
    runs = find_runs(args.artifacts_root, args.include_smoke)

    if args.dry_run:
        for run in runs:
            print(f"{run['name']}: image_auroc_mean {run['metrics']['image_auroc_mean']:.4f}, "
                  f"{len(run['metrics']) // 2 - 1} categories")
        print(f"{len(runs)} runs found")
        return

    client = MlflowClient(args.tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = (experiment.experiment_id if experiment
                     else client.create_experiment(EXPERIMENT))
    sources, roots = existing_sources(client, experiment_id)
    created = 0
    for run in runs:
        if run["name"] in sources or run["params"]["output_root"] in roots:
            print(f"skip {run['name']}: already recorded")
            continue
        log_run(client, experiment_id, run)
        created += 1
        print(f"{run['name']}: image_auroc_mean {run['metrics']['image_auroc_mean']:.4f}")
    print(f"{created} new runs, {len(runs) - created} skipped")


if __name__ == "__main__":
    main()
