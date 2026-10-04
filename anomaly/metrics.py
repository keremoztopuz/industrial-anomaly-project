"""Image and pixel AUROC for one MVTec category."""

import torch


def binary_auroc(scores, labels):
    """Return exact AUROC, averaging tied scores through the ROC curve."""
    scores = torch.as_tensor(scores, dtype=torch.float64).detach().cpu().flatten()
    labels = torch.as_tensor(labels).detach().cpu().flatten()
    if scores.numel() == 0 or scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same nonempty shape")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must contain only 0 and 1")

    positives = int(labels.sum())
    negatives = labels.numel() - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires both normal and anomalous labels")

    sorted_scores, order = torch.sort(scores, descending=True)
    true_positive = labels[order].to(torch.int64).cumsum(0)
    ends = torch.cat((
        torch.nonzero(sorted_scores[1:] != sorted_scores[:-1]).flatten(),
        torch.tensor([scores.numel() - 1]),
    ))
    true_positive = true_positive[ends]
    false_positive = ends + 1 - true_positive
    tpr = torch.cat((torch.zeros(1, dtype=torch.float64), true_positive.to(torch.float64) / positives))
    fpr = torch.cat((torch.zeros(1, dtype=torch.float64), false_positive.to(torch.float64) / negatives))
    return float(torch.trapezoid(tpr, fpr))


def evaluate_category(image_scores, anomaly_maps, labels, masks):
    """Evaluate [N] image scores and matching [N,H,W] maps/masks."""
    image_scores = torch.as_tensor(image_scores)
    labels = torch.as_tensor(labels)
    anomaly_maps = torch.as_tensor(anomaly_maps)
    masks = torch.as_tensor(masks)
    if anomaly_maps.ndim == 4 and anomaly_maps.shape[1] == 1:
        anomaly_maps = anomaly_maps[:, 0]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if (image_scores.ndim != 1 or labels.shape != image_scores.shape
            or anomaly_maps.ndim != 3 or masks.shape != anomaly_maps.shape
            or anomaly_maps.shape[0] != labels.numel()):
        raise ValueError("Expected scores/labels [N] and matching maps/masks [N,H,W]")
    return {
        "image_auroc": binary_auroc(image_scores, labels),
        "pixel_auroc": binary_auroc(anomaly_maps, masks),
    }


if __name__ == "__main__":
    assert binary_auroc([0.1, 0.9], [0, 1]) == 1.0
    assert binary_auroc([0.5, 0.5], [0, 1]) == 0.5
