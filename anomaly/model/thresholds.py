"""Read the per-category is_anomaly thresholds written by the calibration script."""

import json
import math


def load_thresholds(path):
    """Allow missing calibration; reject malformed or non-finite thresholds."""
    if not path.exists():
        print(f"No thresholds at {path}; is_anomaly will be null")
        return {}
    try:
        categories = json.loads(path.read_text(encoding="utf-8"))["categories"]
        if not isinstance(categories, dict):
            raise ValueError("categories must be an object")
        thresholds = {}
        for category, values in categories.items():
            value = values["threshold"]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{category}: threshold must be a finite nonnegative number")
            thresholds[category] = value
        return thresholds
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid thresholds at {path}: {error}") from error
