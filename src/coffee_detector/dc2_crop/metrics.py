from __future__ import annotations

import torch


def per_class_f1(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int) -> list[float]:
    predictions = predictions.reshape(-1).long()
    labels = labels.reshape(-1).long()
    if predictions.shape != labels.shape:
        raise ValueError("Predictions dan labels tidak sejajar")
    values: list[float] = []
    for class_id in range(int(num_classes)):
        tp = int(((predictions == class_id) & (labels == class_id)).sum())
        fp = int(((predictions == class_id) & (labels != class_id)).sum())
        fn = int(((predictions != class_id) & (labels == class_id)).sum())
        denom = 2 * tp + fp + fn
        values.append((2.0 * tp / denom) if denom else 0.0)
    return values


def classification_summary(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    predictions = logits.argmax(dim=1)
    class_f1 = per_class_f1(predictions, labels, num_classes)
    ordered = sorted(class_f1)
    return {
        "accuracy": float((predictions == labels).float().mean()),
        "macro_f1": float(sum(class_f1) / len(class_f1)),
        "bottom3_f1": float(sum(ordered[:3]) / min(3, len(ordered))),
        "worst_f1": float(ordered[0]),
        "per_class_f1": class_f1,
    }
