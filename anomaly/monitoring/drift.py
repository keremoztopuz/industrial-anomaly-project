"""Drift checks that compare recent predictions with a reference distribution.

Decisions (2026-10-05):
- Distributions are compared with PSI over 10 bins cut at the reference deciles.
- The textbook PSI cut-offs (0.1 warning, 0.25 alarm) assume thousands of samples. With
  50 predictions and no drift at all they warn 86% and alarm 28% of the time, so each
  category gets its own cut-offs, measured by simulation (see calibrate_psi_thresholds):
  the PSI that no-drift windows exceed 5% (warning) and 1% (alarm) of the time.
- Bin shares use Laplace smoothing (half a count per bin) instead of a tiny epsilon. With
  ~50 reference values an empty bin otherwise adds ~1.15 to PSI, and the cut-offs end up
  measuring chance empty bins. At a 1% false alarm rate on pill scores, smoothing raised
  detection of a +0.5 standard deviation shift from 57% to 77%.
- Each category is checked on its last 50 predictions; fewer means "insufficient data".
- The alarm rate raises an alarm above 3 times the expected false positive rate.
- Only the model's behavior (score PSI and alarm rate) decides the status. Brightness and
  contrast are reported as the relative change of their mean, for a person to read. MVTec's
  test images already differ from the training images by up to 30% in contrast in some
  categories (leather), more than a simulated out-of-focus camera (6%), so no cut-off on
  them separates normal days from camera faults, and an automatic diagnosis was dropped.
"""

from collections import defaultdict

import numpy as np

BINS = 10
SMOOTHING = 0.5  # Laplace: half a count per bin, so an empty bin can't blow PSI up
WARN_PSI = 0.1
ALARM_PSI = 0.25
WINDOW = 50
EXPECTED_FALSE_POSITIVE_RATE = 0.069  # test set, reports/threshold_calibration.md
ALARM_RATE_FACTOR = 3


def psi(reference, current, bins=BINS):
    """Population Stability Index of `current` against `reference`."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    edges = np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1])
    edges = np.concatenate([[-np.inf], edges, [np.inf]])

    reference_share = ((np.histogram(reference, bins=edges)[0] + SMOOTHING)
                       / (len(reference) + SMOOTHING * bins))
    current_share = ((np.histogram(current, bins=edges)[0] + SMOOTHING)
                     / (len(current) + SMOOTHING * bins))

    return float(np.sum((current_share - reference_share)
                        * np.log(current_share / reference_share)))


def psi_status(value, warn=WARN_PSI, alarm=ALARM_PSI):
    """Return "stable", "warning" or "alarm" for a PSI value."""
    if value > alarm:
        return "alarm"
    if value > warn:
        return "warning"
    return "stable"


def alarm_rate_status(is_anomaly_flags, window=WINDOW):
    """Check the boolean decision rate, or report insufficient_data for a short window."""
    flags = [flag for flag in is_anomaly_flags if type(flag) is bool]
    if len(flags) < window:
        return "insufficient_data"
    rate = sum(flags) / len(flags)
    return "alarm" if rate > EXPECTED_FALSE_POSITIVE_RATE * ALARM_RATE_FACTOR else "stable"


def calibrate_psi_thresholds(reference, window=WINDOW, bins=BINS, trials=2000, seed=42):
    """Measure how high PSI goes on `window` predictions when nothing has drifted.

    Returns (warn, alarm): the PSI values that no-drift windows exceed only 5% and 1%
    of the time, so the monitor's own false alarm rate is a number we chose.
    """
    reference = np.asarray(reference, dtype=float)
    rng = np.random.default_rng(seed)

    values = [psi(reference, rng.choice(reference, size=window, replace=True), bins)
              for _ in range(trials)]
    return float(np.quantile(values, 0.95)), float(np.quantile(values, 0.99))


INPUT_SIGNALS = ("brightness", "contrast")
SEVERITY = {"insufficient_data": 0, "stable": 1, "warning": 2, "alarm": 3}


def check_category(records, reference, window=WINDOW):
    """Compare a category's newest `window` prediction records with its reference.

    `records` are prediction log dicts, newest first. `reference` maps "anomaly_score" to
    {"reference": [...], "warn": float, "alarm": float} and each input signal to at least
    {"reference": [...]}. Returns one result dict.
    """
    recent = records[:window]
    if len(recent) < window:
        return {"status": "insufficient_data", "predictions": len(recent)}
    score = reference["anomaly_score"]
    result = {"predictions": len(recent),
              "psi_anomaly_score": psi(score["reference"],
                                       [record["anomaly_score"] for record in recent])}
    result["anomaly_score_status"] = psi_status(
        result["psi_anomaly_score"], score["warn"], score["alarm"])
    flags = [record["is_anomaly"] for record in recent
             if type(record.get("is_anomaly")) is bool]
    result["decisions"] = len(flags)
    result["alarm_rate"] = sum(flags) / len(flags) if len(flags) >= window else None
    result["alarm_rate_status"] = alarm_rate_status(flags, window)
    result["status"] = max((result["anomaly_score_status"], result["alarm_rate_status"]),
                           key=SEVERITY.get)
    if result["status"] == "stable" and result["alarm_rate"] is None:
        result["status"] = "insufficient_data"
    for signal in INPUT_SIGNALS:
        baseline = float(np.mean(reference[signal]["reference"]))
        result[f"{signal}_change"] = (
            (float(np.mean([record[signal] for record in recent])) - baseline) / baseline
            if baseline != 0 else None)
        result[f"{signal}_change_status"] = "ok" if baseline != 0 else "zero_reference"
    return result


def check_all(predictions, reference):
    """Run check_category for every category in the reference; predictions newest first."""
    by_category = defaultdict(list)
    for record in predictions:
        by_category[record["category"]].append(record)
    return {category: check_category(by_category[category], values, reference["window"])
            for category, values in reference["categories"].items()}


def overall_status(results):
    return max((result["status"] for result in results.values()), key=SEVERITY.get)
