"""Build PatchCore without downloading the Wide ResNet weights."""

from contextlib import contextmanager
from unittest import mock

import torch

from anomaly.model import patchcore


@contextmanager
def no_backbone():
    """Skip loading Wide ResNet weights."""
    with mock.patch.object(patchcore, "wide_resnet50_2"), \
            mock.patch.object(patchcore, "create_feature_extractor",
                              return_value=torch.nn.Identity()):
        yield
