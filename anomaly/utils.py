"""Small helpers shared by the pipeline and the scripts."""

import json

import torch


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_json(path, data):
    """Write JSON through a temporary file so a crash never leaves a half-written file."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
