"""Seeded splits of normal training images and of test images."""

import math

import torch


def split_indices(count, holdout_fraction, seed):
    """Return (fit, holdout) index lists from a seeded shuffle; both are nonempty."""
    if count < 2:
        raise ValueError("need at least two images to hold some out")
    if not 0.0 < float(holdout_fraction) < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    holdout_size = min(count - 1, max(1, math.ceil(count * holdout_fraction)))
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
    return sorted(order[holdout_size:]), sorted(order[:holdout_size])


def split_halves(labels, seed):
    """Split indices into validation and final halves, separately per label."""
    generator = torch.Generator().manual_seed(seed)
    validation, final = [], []
    for label in (0, 1):
        indices = [i for i, value in enumerate(labels) if value == label]
        order = torch.randperm(len(indices), generator=generator).tolist()
        half = (len(indices) + 1) // 2
        validation += [indices[i] for i in order[:half]]
        final += [indices[i] for i in order[half:]]
    return sorted(validation), sorted(final)
