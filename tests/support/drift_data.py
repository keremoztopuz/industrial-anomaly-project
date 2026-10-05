"""A normal reference and prediction records for drift tests."""

import numpy as np

from anomaly.monitoring import drift

RNG = np.random.default_rng(0)
NORMAL = RNG.normal(10, 1, 300).tolist()


def reference_for(values):
    warn, alarm = drift.calibrate_psi_thresholds(values, trials=500)
    return {"reference": values, "warn": warn, "alarm": alarm}


REFERENCE = {"anomaly_score": reference_for(NORMAL),
             "brightness": {"reference": NORMAL}, "contrast": {"reference": NORMAL}}


def records(n, shift=0.0, flagged=0, category="bottle", brightness_shift=0.0):
    rng = np.random.default_rng(1)
    return [{"category": category, "anomaly_score": float(rng.normal(10 + shift, 1)),
             "brightness": float(rng.normal(10 + brightness_shift, 1)),
             "contrast": float(rng.normal(10, 1)),
             "is_anomaly": i < flagged} for i in range(n)]
