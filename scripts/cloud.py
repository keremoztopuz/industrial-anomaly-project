"""Run gcloud commands from the scripts."""

import shutil
import subprocess


def gcloud(*args, check=True):
    """Run a gcloud command; the full path keeps SonarCloud happy."""
    executable = shutil.which("gcloud")
    if executable is None:
        raise RuntimeError("gcloud not found on PATH")
    return subprocess.run(
        [executable, *args], check=check, capture_output=True, text=True
    )
