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
"""

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


def alarm_rate_status(is_anomaly_flags):
    """Return "stable" or "alarm" for the share of flagged predictions in a window."""
    rate = sum(is_anomaly_flags) / len(is_anomaly_flags)
    return "alarm" if rate > EXPECTED_FALSE_POSITIVE_RATE * ALARM_RATE_FACTOR else "stable"


def calibrate_psi_thresholds(reference, window=WINDOW, bins=BINS, trials=2000, seed=42):
    """Measure how high PSI goes on `window` predictions when nothing has drifted.

    Returns (warn, alarm): the PSI values that no-drift windows exceed only 5% and 1%
    of the time, so the monitor's own false alarm rate is a number we chose.
    """
    reference = np.asarray(reference, dtype=float)
    rng = np.random.default_rng(seed)

    sample = rng.choice(reference, size=window, replace=True)
    psi_values = [psi(reference, sample, bins)]
    psi_values += np.quantile([psi(reference, rng.choice(reference, size=window, replace=True), bins)
                                   for _ in range(trials)], [0.05, 0.01]).tolist()

    values = [psi(reference, rng.choice(reference, size=window, replace=True), bins)
              for _ in range(trials)]
    return float(np.quantile(values, 0.95)), float(np.quantile(values, 0.99))


SIGNALS = ("anomaly_score", "brightness", "contrast")
SEVERITY = {"insufficient_data": 0, "stable": 1, "warning": 2, "alarm": 3}


def check_category(records, reference, window=WINDOW):
    """Compare a category's newest `window` prediction records with its reference.

    `records` are prediction log dicts, newest first. `reference` maps each signal to
    {"reference": [...], "warn": float, "alarm": float}. Returns one result dict.
    """
    recent = records[:window]
    if len(recent) < window:
        return {"status": "insufficient_data", "predictions": len(recent)}
    result = {"predictions": len(recent)}
    statuses = []
    for signal in SIGNALS:
        value = psi(reference[signal]["reference"], [record[signal] for record in recent])
        status = psi_status(value, reference[signal]["warn"], reference[signal]["alarm"])
        result[f"psi_{signal}"] = value
        result[f"{signal}_status"] = status
        statuses.append(status)
    flags = [bool(record["is_anomaly"]) for record in recent]
    result["alarm_rate"] = sum(flags) / len(flags)
    result["alarm_rate_status"] = alarm_rate_status(flags)
    statuses.append(result["alarm_rate_status"])
    result["status"] = max(statuses, key=SEVERITY.get)
    return result
