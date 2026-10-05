"""Small helpers shared by the pipeline and the scripts."""

import json
import tempfile
from pathlib import Path

import torch


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_json(path, data):
    """Write JSON through a temporary file, only below the working or the temp directory."""
    target = Path(path).resolve()
    allowed = (Path.cwd().resolve(), Path(tempfile.gettempdir()).resolve())
    if not any(target.is_relative_to(root) for root in allowed):
        raise ValueError(f"Refusing to write {target} outside the working and temp directories")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
