"""Score a dataset with a fitted model and count image-level decisions."""

import torch
from torch.utils.data import DataLoader


def score(model, dataset, batch_size):
    scores, labels = [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        batch_scores, _ = model.predict(batch["image"])
        scores.append(batch_scores)
        labels.append(torch.as_tensor(batch["label"]))
    return torch.cat(scores), torch.cat(labels)


def confusion(scores, labels, threshold):
    """Count image decisions where score > threshold means anomalous."""
    predicted = torch.as_tensor(scores) > threshold
    labels = torch.as_tensor(labels).bool()
    counts = {
        "true_positive": int((predicted & labels).sum()),
        "false_positive": int((predicted & ~labels).sum()),
        "true_negative": int((~predicted & ~labels).sum()),
        "false_negative": int((~predicted & labels).sum()),
    }
    positives = counts["true_positive"] + counts["false_negative"]
    negatives = counts["false_positive"] + counts["true_negative"]
    counts["recall"] = counts["true_positive"] / positives if positives else None
    counts["false_positive_rate"] = counts["false_positive"] / negatives if negatives else None
    return counts


def balanced_accuracy(result):
    return (result["recall"] + 1 - result["false_positive_rate"]) / 2
