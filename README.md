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

## How it works

1. **Features.** An ImageNet-pretrained Wide ResNet-50-2 is used as a frozen feature extractor; it is never trained. Features from `layer2` and `layer3` are combined, giving a 32 × 32 grid of patch features per image. A fixed random projection reduces each one to 256 dimensions. With `--neighborhood 3`, each feature is first averaged with its 3 × 3 neighbors.
2. **Memory bank.** Patch features from a category's normal training images (about 270k per category) are reduced to a fixed-size bank. `random` keeps a uniform sample. `coreset` draws a candidate pool four times the bank size, then greedily adds whichever candidate is farthest from those already chosen.
3. **Scoring.** Each test patch is scored by its distance to the nearest patch in the bank. The patch scores, upsampled to image size, form the anomaly map. The image score is the highest patch score, ignoring the outer `--border` rings. Border patches are distorted by the network's zero padding and score high even on normal images.
4. **Evaluation.** Image AUROC compares image scores with normal/defective labels. Pixel AUROC compares the anomaly maps with ground-truth defect masks.

There is no loss function and no gradient training: everything the model "learns" is in its memory bank. Each bank is saved as a small `.pt` file per category (about 18 MB at 16,384 patches).

## Repository layout

```
anomaly/
  patchcore.py            PatchCore model: feature extraction, bank selection, scoring, save/load
  mvtec_dataset.py        PyTorch dataset for the FiftyOne export of MVTec AD
  metrics.py              Exact image and pixel AUROC
  drift.py                PSI drift checks with calibrated cut-offs
  drift_job.py            Scheduled drift check for Cloud Run Jobs (reads the Logging API)
  run_pipeline.py         Prefect flow: fit, save, evaluate, write a run manifest and log to MLflow
api/service.py            FastAPI service that serves the saved banks
Dockerfile                CPU-only image for the service
requirements-api.in       Serving dependencies; compiled to the hashed requirements-api.lock
scripts/
  build_patchcore.py      Build memory banks without evaluating
  calibrate_thresholds.py Pick each category's is_anomaly threshold from held-out normal images
  compare_threshold_rules.py  Compare threshold rules on a validation half of the test set
  plot_threshold_results.py   Draw the threshold figures in reports/figures/
  backfill_mlflow.py      Record runs made before MLflow tracking
  register_model.py       Register a run's banks as a new patchcore-mvtec version
  deploy_model.py         Point Cloud Run at the version behind an MLflow alias
  build_drift_reference.py  Reference scores, brightness and contrast for drift checks
  check_drift.py          Compare recent live predictions with the reference
  simulate_drift.py       Send drift scenarios (dark, blur, defect wave) to the live service
  visualize_anomalies.py  Render example images with masks and anomaly overlays
  dataset_summary.py      Count images per category, split and defect
  validate_dataset.py     Decode every image and mask and check their sizes
tests/                    Unit and API tests (no dataset or model weights needed)
reports/                  Experiment write-ups
```

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
.venv/bin/python -m scripts.validate_dataset
.venv/bin/python -m scripts.dataset_summary
```

MVTec AD is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), which allows non-commercial use only. The dataset is not stored in this repository.

## Usage

Run the best configuration on every category:

```sh
.venv/bin/python -m anomaly.run_pipeline \
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
.venv/bin/python -m scripts.visualize_anomalies \
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

`api/service.py` loads every `<category>.pt` bank from `MODEL_DIR` at startup and shares one backbone between them (about 1 GB of RAM for all 15 categories).

| Endpoint | Returns |
| --- | --- |
| `GET /health` | `{"status": "healthy"}` |
| `GET /categories` | Sorted list of loaded categories |
| `GET /model` | Model name, registry version, model directory and number of loaded categories |
| `POST /predict/{category}` | Image score, threshold and `is_anomaly` for an uploaded image (`upload_file` form field) |

Unknown categories return 404 and files that are not readable images return 400.

```sh
curl -X POST -F "upload_file=@bottle.png" \
  https://anomaly-api-sw4p2ayosa-ew.a.run.app/predict/bottle
# {"category": "bottle", "filename": "bottle.png", "anomaly_score": 24.67,
#  "threshold": 12.82, "is_anomaly": true}
```

`is_anomaly` is `anomaly_score > threshold`. Each category's threshold is the highest score among 20% of its normal training images held out from a bank fit on the rest, so the test set plays no part in choosing it. The thresholds are read from `thresholds.json` next to the banks; without that file `threshold` and `is_anomaly` are `null`. Over all test images this catches 84.6% of defective parts with a 6.9% false alarm rate, but the balance differs a lot by category (see [`threshold_calibration.md`](reports/threshold_calibration.md)).

Run it locally against saved banks:

```sh
PYTHONPATH=. MODEL_DIR=artifacts/best/patchcore .venv/bin/python api/service.py
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

Every `anomaly.run_pipeline` run is logged to MLflow in the `patchcore-mvtec` experiment: settings as params, image and pixel AUROC per category and their means as metrics, git commit and dataset hash as tags, and `metrics.json` and `manifest.json` as artifacts. `--tracking-uri` defaults to a local `sqlite:///mlflow.db`. Runs made before tracking existed were added with `scripts/backfill_mlflow.py`, so every report table can be compared in one place (filter on `tags.num_categories = "15"` to compare full runs). The threshold rule comparison is logged in a separate `threshold-rules` experiment.

```sh
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # macOS uses port 5000 for AirPlay
```

The deployed banks are registered as the `patchcore-mvtec` model. Each version points to the run it came from, and the `production` alias marks the version that should be live:

```sh
.venv/bin/python -m scripts.register_model    # log a run's banks + thresholds.json, add a version, set @production
.venv/bin/python -m scripts.deploy_model      # upload @production to gs://<bucket>/patchcore-mvtec/v<N>/ if needed,
                                              # then set MODEL_DIR and MODEL_VERSION on the Cloud Run service
```

Each version gets its own folder in the bucket and is never overwritten. To roll back, move the `production` alias to an older version and run `deploy_model.py` again: the files are already there, so only the service's environment changes. `GET /model` shows which version is live.

## Monitoring

Every prediction is logged as one JSON line (category, model version, score, threshold, `is_anomaly`, latency, image size, brightness and contrast; never the image or filename). Cloud Run sends it to Cloud Logging.

`scripts/check_drift.py` takes each category's last 50 predictions for the deployed model version and compares them with `drift_reference.json`: the held-out normal scores, and the brightness and contrast of the normal training images. Only the model's behavior raises an alarm: the score distribution (PSI) and the share of flagged predictions (above three times the expected false alarm rate). Brightness and contrast are shown as the percentage change of their mean, for a person to read: a failing lamp shows up as roughly −40% brightness. They don't raise alarms or drive an automatic diagnosis, because MVTec's test images already differ from the training images by up to 30% in contrast in some categories, more than a simulated out-of-focus camera (6%). A run is logged to the MLflow `monitoring` experiment, and the script exits with status 1 on an alarm.

The textbook PSI cut-offs (0.1 warning, 0.25 alarm) assume thousands of samples. On 50 predictions with no drift at all they would warn 86% and alarm 28% of the time. So each cut-off is measured instead: `build_drift_reference.py` resamples windows from the reference thousands of times and puts the warning and alarm cut-offs where no-drift windows land only 5% and 1% of the time. Bin shares use Laplace smoothing, because with ~50 reference values a single empty bin otherwise dominates PSI.

```sh
.venv/bin/python -m scripts.build_drift_reference   # once per deployed model version
.venv/bin/python -m scripts.check_drift
```

`anomaly/drift_job.py` runs the same check as the `drift-check` Cloud Run Job, from the serving image, every day at 08:00 Istanbul time (Cloud Scheduler). It reads the prediction logs through the Cloud Logging API with the job's own credentials and writes one `drift_check` log line, with severity `ERROR` when a category is in alarm. `drift_reference.json` lives in the model's version folder in the bucket next to the banks. `deploy_model.py` uploads it with new versions and points the job at the new version's reference and the service's current image.

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
.venv/bin/python -m unittest discover -s tests
```

The tests mock the backbone, so they need neither the dataset nor the pretrained weights. The API tests in `tests/test_service.py` replace the banks with fake models and skip the startup that loads them.

## Limitations

- Every result comes from a single seed, so there are no confidence intervals. Differences of about ±0.01 may be noise.
- MVTec AD has no validation split. The `--border` value was chosen by looking at test results, so the best row above may slightly overstate performance on unseen data.
- One threshold rule does not fit every category: it misses about half of the `pill` defects and flags over 40% of normal `carpet` and `toothbrush` test images. The held-out sets are small (12–79 images), so the highest held-out score is a noisy estimate.
- A defect that lies only within the excluded border strip (about 16 pixels at 256 × 256) does not affect the image score. It still shows up in the anomaly map.
- Images are resized without the center crop used in the PatchCore paper, which reports about 0.99 image AUROC.
