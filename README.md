# Industrial Anomaly Detection

Unsupervised defect detection on [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) with a PatchCore-style model, run as a reproducible Prefect pipeline.

The model sees only defect-free images during fitting. It flags a test image as anomalous when parts of it do not look like anything in its memory of normal images. It outputs an image-level score ("is this part defective?") and a pixel-level heat map ("where is the defect?").

**Live API:** [anomaly-api-sw4p2ayosa-ew.a.run.app/docs](https://anomaly-api-sw4p2ayosa-ew.a.run.app/docs). Upload an image from the interactive docs page and pick a category. The service scales to zero when idle, so the first request after a quiet period takes about 30 seconds while the models load.

## Results

All 15 MVTec AD categories, single seed (42), 256 × 256 images, Apple M4 (MPS). AUROC is the probability that a random defective sample scores higher than a random normal one: 1.0 is perfect, 0.5 is chance.

| Configuration | Image AUROC | Pixel AUROC |
| --- | ---: | ---: |
| Baseline: 2,048 random patches | 0.8364 | 0.9439 |
| 16,384 coreset patches | 0.9317 | 0.9676 |
| + 3 × 3 neighborhood aggregation | 0.9299 | 0.9767 |
| + exclude 2 border patch rings from image scores | **0.9770** | **0.9767** |

### Defective or not: the deployed thresholds

Each category's `is_anomaly` threshold is chosen from held-out normal training images, without looking at the test set. On all 1,725 test images it catches 84.6% of defective parts and flags 6.9% of normal ones, but the balance varies a lot by category ([`threshold_calibration.md`](reports/threshold_calibration.md)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/threshold_confusion_dark.svg">
  <img alt="Confusion matrix over all test images: 435 normal and 1,064 defective parts classified correctly, 32 false alarms and 194 missed defects" src="reports/figures/threshold_confusion_light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/threshold_rates_dark.svg">
  <img alt="Recall and false positive rate per category" src="reports/figures/threshold_rates_light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/threshold_scores_dark.svg">
  <img alt="Test score distributions per category with each category's threshold" src="reports/figures/threshold_scores_light.svg">
</picture>

Per-category tables, run manifests and caveats are in [`reports/`](reports/) (written in Turkish):

| Report | Question |
| --- | --- |
| [`baseline.md`](reports/baseline.md) | How well does the simplest version work? |
| [`coreset_comparison.md`](reports/coreset_comparison.md) | Does coreset selection beat random sampling? |
| [`bank_size_sweep.md`](reports/bank_size_sweep.md) | How much does memory bank size matter? |
| [`local_aggregation.md`](reports/local_aggregation.md) | Does averaging neighboring features help? |
| [`border_exclusion.md`](reports/border_exclusion.md) | Why was `grid` failing, and how was it fixed? |
| [`threshold_calibration.md`](reports/threshold_calibration.md) | Where should the defective/normal cut-off be? |
| [`drift_simulation.md`](reports/drift_simulation.md) | Does the drift check catch lighting, focus and defect changes? |
| [`pill_resolution.md`](reports/pill_resolution.md) | Does a higher resolution help `pill`, and how fast is a rollback? |

## How it works

1. **Features.** An ImageNet-pretrained Wide ResNet-50-2 is used as a frozen feature extractor; it is never trained. Features from `layer2` and `layer3` are combined, giving a 32 × 32 grid of patch features per image. A fixed random projection reduces each one to 256 dimensions. With `--neighborhood 3`, each feature is first averaged with its 3 × 3 neighbors.
2. **Memory bank.** Patch features from a category's normal training images (about 270k per category) are reduced to a fixed-size bank. `random` keeps a uniform sample. `coreset` draws a candidate pool four times the bank size, then greedily adds whichever candidate is farthest from those already chosen.
3. **Scoring.** Each test patch is scored by its distance to the nearest patch in the bank. The patch scores, upsampled to image size, form the anomaly map. The image score is the highest patch score, ignoring the outer `--border` rings. Border patches are distorted by the network's zero padding and score high even on normal images.
4. **Evaluation.** Image AUROC compares image scores with normal/defective labels. Pixel AUROC compares the anomaly maps with ground-truth defect masks.

There is no loss function and no gradient training: everything the model "learns" is in its memory bank. Each bank is saved as a small `.pt` file per category (about 18 MB at 16,384 patches).

## Repository layout

```
anomaly/                      Core package: everything the model and the pipeline need
  data/mvtec_dataset.py       PyTorch dataset for the FiftyOne export of MVTec AD
  model/patchcore.py          PatchCore: feature extraction, bank selection, scoring, save/load
  model/thresholds.py         Read the per-category is_anomaly thresholds
  evaluation/metrics.py       Exact image and pixel AUROC
  evaluation/scoring.py       Score a dataset with a model and count decisions
  evaluation/splits.py        Seeded splits of training and test images
  pipeline/run_pipeline.py    Prefect flow: fit, save, evaluate, write a manifest, log to MLflow
  monitoring/drift.py         PSI drift checks with calibrated cut-offs
  monitoring/drift_job.py     Daily drift check for Cloud Run Jobs (reads the Logging API)
  settings.py                 Settings from environment variables and .env
  utils.py                    write_json and default_device
api/                          FastAPI service, started with python -m api.main
  main.py                     App entry point and startup
  config.py                   API settings: server and upload limits on top of the shared ones
  middleware.py               Request size limit
  routes/                     health.py, models.py, predictions.py: HTTP only
  schemas/                    Response contracts
  services/                   model_loading.py, image_decoding.py, prediction.py
scripts/                      Command line tools, grouped by what they work on
  cloud.py                    Run gcloud commands
  data/                       validate_dataset, dataset_summary
  modeling/                   build_patchcore, calibrate_thresholds, compare_threshold_rules,
                              compare_resolutions
  registry/                   register_model, deploy_model, backfill_mlflow
  monitoring/                 build_drift_reference, check_drift, simulate_drift
  reporting/                  plot_threshold_results, visualize_anomalies
tests/                        Mirrors the packages: api/, anomaly/, scripts/; support/ holds shared fakes
reports/                      Experiment write-ups
Dockerfile, requirements-*.in / .lock, .flake8, .env.example
```

Each layer has one job. Routes only deal with HTTP and delegate to services; services do the work; `anomaly` knows nothing about HTTP; scripts are thin command line wrappers that call `anomaly` and never import each other (shared code lives in `anomaly/` or `scripts/cloud.py`). The serving image copies only `anomaly/` and `api/`.

## Setup

Requires Python 3.12. With [uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
```

### Data

The code reads the [FiftyOne export of MVTec AD](https://huggingface.co/datasets/Voxel51/mvtec-ad) from `data/mvtec-ad/`, which must contain `samples.json` and the image folders:

```sh
uv pip install huggingface_hub
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Voxel51/mvtec-ad', repo_type='dataset', local_dir='data/mvtec-ad')"
```

Check the download:

```sh
.venv/bin/python -m scripts.data.validate_dataset
.venv/bin/python -m scripts.data.dataset_summary
```

MVTec AD is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), which allows non-commercial use only. The dataset is not stored in this repository.

## Usage

Run the best configuration on every category:

```sh
.venv/bin/python -m anomaly.pipeline.run_pipeline \
  --dataset-root data/mvtec-ad --output-root artifacts/best \
  --device mps --max-patches 16384 --selection coreset \
  --neighborhood 3 --border 2
```

Use `--device cuda` on an NVIDIA GPU, or `cpu`. Add `--category grid` to run a single category. On an M4, the full run takes about 55 minutes, almost all of it in coreset selection. `--selection random` finishes in about 4 minutes, with lower scores.

The pipeline writes these files to `--output-root`:

- `patchcore/<category>.pt`: one memory bank per category
- `metrics.json`: image and pixel AUROC per category
- `manifest.json`: settings, git commit, dirty-worktree flag, and the dataset's `samples.json` SHA-256, so every number can be traced to the code and data that produced it

Render anomaly overlays from the saved banks into `<output-root>/visualizations/`:

```sh
.venv/bin/python -m scripts.reporting.visualize_anomalies \
  --dataset-root data/mvtec-ad --output-root artifacts/best --device mps
```

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--max-patches` | 2048 | Memory bank size per category |
| `--selection` | `random` | `random` or `coreset` bank selection |
| `--neighborhood` | 1 | Odd feature-averaging window (PatchCore uses 3) |
| `--border` | 0 | Outer patch rings ignored by image scores |
| `--projection-dim` | 256 | Random projection size |
| `--image-size` | 256 | Square resize applied to every image |
| `--seed` | 42 | Seed for projection and sampling |

The defaults reproduce the original baseline. Banks saved before an option existed load with that option's default.

## Serving API

`api/main.py` loads every `<category>.pt` bank from `MODEL_DIR` at startup and shares one backbone between them (about 1 GB of RAM for all 15 categories).

| Endpoint | Returns |
| --- | --- |
| `GET /health` | `{"status": "healthy"}` |
| `GET /categories` | Sorted list of loaded categories |
| `GET /model` | Model name, registry version, model directory and number of loaded categories |
| `POST /predict/{category}` | Image score, threshold and `is_anomaly` for an uploaded image (`upload_file` form field) |

Only PNG and JPEG are accepted (including grayscale PNG/JPEG and RGBA PNG). Unknown categories return 404, invalid or unsupported images return 400, exceeded byte/pixel limits return 413, and a missing upload returns 422. `/docs` and `/openapi.json` describe the typed responses, including nullable `threshold` and `is_anomaly` fields. Existing URLs and JSON field names are unchanged; `/model`'s `categories` remains a count.

Startup fails if `MODEL_DIR` contains no category banks. `/health` reports HTTP process liveness after startup; it does not probe external services. Missing thresholds preserve the null decision behavior; a present but malformed file or a nonnumeric, negative, boolean or non-finite threshold prevents startup.

| Setting | Default | Purpose |
| --- | --- | --- |
| `MAX_UPLOAD_BYTES` | 10485760 (10 MiB) | Maximum encoded image size |
| `MAX_IMAGE_PIXELS` | 16000000 | Maximum width × height before RGB decoding |
| `MAX_REQUEST_BYTES` | `MAX_UPLOAD_BYTES + 1048576` | Complete request body, including multipart overhead |

All limits must be positive. The ASGI middleware bounds the whole body **before multipart parsing**, even without `Content-Length`. The image decoder separately reads at most `MAX_UPLOAD_BYTES + 1` bytes. The middleware buffers one bounded request body in RAM; account for concurrent requests when setting limits.

### Configuration

Every value that used to sit at the top of a script (model directory, model and bucket names, project, region, MLflow location, server port, upload limits) is a setting in `anomaly/settings.py` or `api/config.py`. Settings come from environment variables and, when the file exists, from a `.env` file in the working directory. A real environment variable wins over `.env`, and a command line flag wins over both. The environment variable names are the upper-case field names, for example `MODEL_DIR`, `MODEL_VERSION`, `PORT` or `GCP_PROJECT`.

```sh
cp .env.example .env   # every default is listed there with a comment
```

`.env` is ignored by Git and by the Docker build context, so a local file never ships. This project has no secrets (no API keys or passwords; Google Cloud access goes through IAM), so `.env` only holds ordinary configuration. A secret would belong in the platform's secret store (for Cloud Run, Secret Manager), never in the code or in Git. In the cloud, the values are set on the Cloud Run service and job: `deploy_model.py` sets `MODEL_DIR`, `MODEL_NAME` and `MODEL_VERSION`, and Cloud Run provides `PORT`.

```sh
curl -X POST -F "upload_file=@bottle.png" \
  https://anomaly-api-sw4p2ayosa-ew.a.run.app/predict/bottle
# {"category": "bottle", "filename": "bottle.png", "anomaly_score": 24.67,
#  "threshold": 12.82, "is_anomaly": true}
```

`is_anomaly` is `anomaly_score > threshold`. Each category's threshold is the highest score among 20% of its normal training images held out from a bank fit on the rest, so the test set plays no part in choosing it. The thresholds are read from `thresholds.json` next to the banks; without that file `threshold` and `is_anomaly` are `null`. Over all test images this catches 84.6% of defective parts with a 6.9% false alarm rate, but the balance differs a lot by category (see [`threshold_calibration.md`](reports/threshold_calibration.md)).

Run it locally against saved banks:

```sh
MODEL_DIR=artifacts/best/patchcore .venv/bin/python -m api.main
```

### Docker

```sh
docker build -t anomaly-api .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts/best/patchcore:/models:ro" anomaly-api
```

Then open http://127.0.0.1:8000/docs. The image installs CPU-only PyTorch from the hash-pinned `requirements-api.lock`, downloads the Wide ResNet weights at build time so containers start without network access, and runs as a non-root user. The banks are not part of the image: they are mounted at `/models`, so new banks need no rebuild. `HOST`, `PORT` and `MODEL_DIR` can be overridden with environment variables.

### Deployment

```
GitHub main ──▶ Cloud Build (Dockerfile) ──▶ Artifact Registry ──▶ Cloud Run ──▶ public HTTPS URL
                                                                       ▲
     MLflow registry ──deploy_model.py──▶ Cloud Storage bucket ────────┘ mounted read-only at /models
     patchcore-mvtec@production           patchcore-mvtec/v<N>/
```

The live service runs on Google Cloud Run in `europe-west1`. A Cloud Build trigger builds the Dockerfile and deploys a new revision on every push to `main`, and `main` only accepts pull requests whose tests and SonarCloud checks pass. Code and models ship separately: the banks live in a Cloud Storage bucket that Cloud Run mounts read-only at `/models`, the same way `-v` works locally.

The service has 2 GiB of memory and 1 vCPU. It scales between 0 and 1 instances, which keeps the cost close to zero and caps it under load, and the project has a budget alert.

## Experiment tracking and model registry

Every `anomaly.pipeline.run_pipeline` run is logged to MLflow in the `patchcore-mvtec` experiment: settings as params, image and pixel AUROC per category and their means as metrics, git commit and dataset hash as tags, and `metrics.json` and `manifest.json` as artifacts. `--tracking-uri` defaults to a local `sqlite:///mlflow.db`. Runs made before tracking existed were added with `scripts/registry/backfill_mlflow.py`, so every report table can be compared in one place (filter on `tags.num_categories = "15"` to compare full runs). The threshold rule comparison is logged in a separate `threshold-rules` experiment.

```sh
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # macOS uses port 5000 for AirPlay
```

The deployed banks are registered as the `patchcore-mvtec` model. Each version points to the run it came from, and the `production` alias marks the version that should be live:

```sh
.venv/bin/python -m scripts.registry.register_model    # log a run's banks + thresholds.json, add a version, set @production
.venv/bin/python -m scripts.registry.deploy_model      # upload @production to gs://<bucket>/patchcore-mvtec/v<N>/ if needed,
                                              # then set MODEL_DIR and MODEL_VERSION on the Cloud Run service
```

Each registration writes to a fresh `model/<uuid>` MLflow artifact directory, including registrations against the same run. Existing registry sources are not rewritten. Deployment resolves the alias once, downloads `models:/patchcore-mvtec/<numeric-version>`, and checks the local bank list and calibration coverage. It compares the remote file names, byte sizes and GCS MD5 hashes with that bundle before updating Cloud Run. An empty successful listing allows a new upload with a create-only generation precondition; listing errors, incomplete existing versions, extra files and mismatched hashes stop deployment. Incomplete versions are not silently repaired or overwritten.

With the default `--drift-job drift-check`, the bundle must include `drift_reference.json` with the same numeric `model_version`, matching categories and a positive window. Prepare the reference for the intended registry version **before registration**; do not modify a registered bundle. Use `--drift-job ''` to deploy without updating monitoring. To roll back, move the alias and deploy again; the older bundle must pass the same checks. `GET /model` shows which version is live.

The service and drift job are separate Cloud Run updates, not a transaction. If a later cloud command fails, inspect both revisions before retrying. The script checks integrity but does not run a full inference on every bank or enforce bucket IAM immutability. No live deployment is performed by the tests.

Each bank stores the input size it was fit at, and the service resizes each category's images to its own bank's size. **Version 2**, live now, is version 1 with `pill` refit at 320 × 320, chosen on a validation half of the pill test set. Pill image AUROC on the final half went from 0.953 to 0.963, and pill recall at the deployed threshold from 49% to 65% with no false alarms. A rollback drill took 97 s to promote v2 (including the upload), 49 s to roll back to v1 and 70 s to promote v2 again ([`pill_resolution.md`](reports/pill_resolution.md)).

## Monitoring

Every prediction is logged as one JSON line (category, model version, score, threshold, `is_anomaly`, latency, image size, brightness and contrast; never the image or filename). Cloud Run sends it to Cloud Logging.

`scripts/monitoring/check_drift.py` takes each category's last 50 predictions for the deployed model version and compares them with `drift_reference.json`: the held-out normal scores, and the brightness and contrast of the normal training images. Only the model's behavior raises an alarm: the score distribution (PSI) and the share of flagged predictions (above three times the expected false alarm rate). Brightness and contrast are shown as the percentage change of their mean, for a person to read: a failing lamp shows up as roughly −40% brightness. They don't raise alarms or drive an automatic diagnosis, because MVTec's test images already differ from the training images by up to 30% in contrast in some categories, more than a simulated out-of-focus camera (6%). A run is logged to the MLflow `monitoring` experiment, and the script exits with status 1 on an alarm.

The textbook PSI cut-offs (0.1 warning, 0.25 alarm) assume thousands of samples. On 50 predictions with no drift at all they would warn 86% and alarm 28% of the time. So each cut-off is measured instead: `build_drift_reference.py` resamples windows from the reference thousands of times and puts the warning and alarm cut-offs where no-drift windows land only 5% and 1% of the time. Bin shares use Laplace smoothing, because with ~50 reference values a single empty bin otherwise dominates PSI.

```sh
.venv/bin/python -m scripts.monitoring.build_drift_reference   # once per deployed model version
.venv/bin/python -m scripts.monitoring.check_drift
```

`anomaly/monitoring/drift_job.py` runs the same check as the `drift-check` Cloud Run Job, from the serving image, every day at 08:00 Istanbul time (Cloud Scheduler). It reads the prediction logs through the Cloud Logging API with the job's own credentials and writes one `drift_check` log line, with severity `ERROR` when a category is in alarm. `drift_reference.json` lives in the model's version folder in the bucket next to the banks. `deploy_model.py` uploads it with new versions and points the job at the new version's reference and the service's current image.

Cloud Monitoring emails on five alert policies:

| Alert | Fires when |
| --- | --- |
| drift alarm | the daily `drift_check` line has severity `ERROR` (the message names the categories) |
| drift job failed | the job logs any other error, so a broken monitor doesn't stay silent |
| server errors | more than 3 responses with a 5xx code in 5 minutes |
| slow responses | 95th percentile latency above 5 s for 15 minutes (a single ~30 s cold start doesn't count) |
| memory | container memory above 90% of 2 GiB for 5 minutes |

A simulation against the live service ([`drift_simulation.md`](reports/drift_simulation.md)) checked four scenarios on `cable`: a normal day only warns, while a failing lamp, an out-of-focus camera and a defect wave all raise an alarm.

## Tests

```sh
.venv/bin/python -m unittest discover -t . -s tests
```

The tests mock the backbone, so they need neither the dataset nor the pretrained weights. The API tests in `tests/test_service.py` replace the banks with fake models and skip the startup that loads them.

## Limitations

- Every result comes from a single seed, so there are no confidence intervals. Differences of about ±0.01 may be noise.
- MVTec AD has no validation split. The `--border` value was chosen by looking at test results, so the best row above may slightly overstate performance on unseen data.
- One threshold rule does not fit every category: it flags over 40% of normal `carpet` and `toothbrush` test images, and still misses about a third of `pill` defects in v2 (half in v1). The held-out sets are small (12–79 images), so the highest held-out score is a noisy estimate.
- A defect that lies only within the excluded border strip (about 16 pixels at 256 × 256) does not affect the image score. It still shows up in the anomaly map.
- Images are resized without the center crop used in the PatchCore paper, which reports about 0.99 image AUROC.

## Future work

- **Hosted MLflow.** Experiment tracking and the model registry live in a local SQLite file (`mlflow.db`). The banks are safe in Cloud Storage, but run history and version lineage would be lost with that file. A tracking server with a managed database and a bucket artifact store would make them shared and durable, and `deploy_model.py` could run from CI instead of a laptop.
- **Infrastructure as code.** The Cloud Run service is deployed by Cloud Build, but the drift job, the scheduler, the bucket mount, the IAM role and the five alert policies were created once from the command line. Describing them in Terraform would make the whole setup reviewable and reproducible.
- **Drift reference from the line itself.** The drift reference comes from training images, which miss normal day-to-day lighting variation ([`drift_simulation.md`](reports/drift_simulation.md)). A reference built from the first weeks of real traffic would allow alarms on brightness and contrast and an automatic camera-versus-defect diagnosis.
- **Retraining loop.** A drift alarm currently ends with an email. The next step is a pipeline that refits the affected category on recent normal images, compares it with the live version under the same protocol, and registers it for review.
- **Per-category tuning.** Only `pill` was tried at a higher resolution. Other weak categories (`screw`, `capsule`, `toothbrush`) could get the same comparison, and threshold calibration could use k-fold held-out scores instead of one 20% split.
- **API hardening.** The demo endpoint is public with one instance and no authentication. A production service would add an API key or IAM authentication and request rate limits.

## Development checks

Use Python 3.12 and run commands from the repository root. `python -m pytest tests` needs no test-specific `sys.path` changes. CI lives in `.github/workflows/ci.yml`, uses `contents: read`, and installs the hash-locked Linux CPU dependencies before testing, linting and auditing.

```sh
.venv/bin/python -m pytest tests
.venv/bin/python -m flake8 anomaly api scripts tests
uv pip check
.venv/bin/python -m pip_audit --no-deps --disable-pip -r requirements-api.lock
.venv/bin/python -m pip_audit --no-deps --disable-pip -r requirements-ci.lock
.venv/bin/python -m pip_audit --no-deps --disable-pip -r requirements.txt
```

Flake8 uses a 99-character line limit (`.flake8`); no error categories are suppressed. Install `flake8==7.4.1` and `pip-audit==2.9.0` in a local development environment when needed; the CI lock already includes them. Lock regeneration commands are at the top of `requirements-api.in` and `requirements-ci.in`. NumPy, HTTPX (development/tests) and Pydantic are explicit direct dependencies. The extra direct-dependency audit checks the public Torch/Torchvision release versions because PyPI does not resolve their `+cpu` wheel versions during auditing; this is an advisory lookup, not a wheel binary scan.

`build_patchcore --image-size` is persisted in each bank. `visualize_anomalies` defaults to the loaded bank's size and rejects a conflicting explicit `--image-size`. Legacy banks without metadata still load as 256; do not rewrite them without training evidence.

Drift alarm rates use only actual boolean decisions from the newest window. Fewer than `window` decisions yields `alarm_rate: null`, `alarm_rate_status: insufficient_data`, and a `decisions` count. A score PSI warning/alarm remains visible even when decisions are missing; otherwise the category is insufficient. Zero brightness/contrast reference means yield a null change and `*_change_status: zero_reference`. The CLI prints `n/a`, and MLflow skips null metrics. The existing overall severity ordering is retained; read individual category/signal statuses for incomplete data.
