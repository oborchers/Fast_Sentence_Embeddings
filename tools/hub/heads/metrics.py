"""Shared per-label quality summaries for frozen classifier scores."""

from __future__ import annotations

import numpy as np
from inputs import coverage
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def predicted_targets(scores, thresholds, predictions, binary):
    if binary:
        if predictions is None:
            raise ValueError("binary metrics require actual predicted class indices")
        index = np.asarray(predictions)
        if index.shape != (len(scores),) or not np.isin(index, (0, 1)).all():
            raise ValueError("binary predictions must contain one class index (0 or 1) per row")
        return np.column_stack([index == 0, index == 1])
    if predictions is not None:
        raise ValueError("multilabel metrics derive predictions from the stored thresholds")
    cutoffs = np.asarray(thresholds, dtype=np.float32)
    if cutoffs.shape not in ((), (scores.shape[1],)) or not np.isfinite(cutoffs).all():
        raise ValueError("thresholds must be finite, scalar or one per label")
    if ((cutoffs < 0) | (cutoffs > 1)).any():
        raise ValueError("thresholds must lie in [0, 1]")
    return scores > cutoffs


def metrics(y, scores, labels, *, thresholds=0.5, predictions=None):
    binary = y.ndim == 1
    if binary:
        if set(np.unique(y)) != {0, 1} or scores.shape != (len(y), 2):
            raise ValueError("binary targets need both classes and two probability columns")
        y = np.column_stack([y == 0, y == 1])
    if y.ndim != 2 or y.shape[1] != len(labels):
        raise ValueError("one label is required per target column")
    coverage(y)
    if scores.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError("invalid classifier scores")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("classifier scores must lie in [0, 1]")
    predicted = predicted_targets(scores, thresholds, predictions, binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        predicted,
        average=None,
        zero_division=0,
    )
    auc = roc_auc_score(y, scores, average=None)
    ap = average_precision_score(y, scores, average=None)
    names = ("roc_auc", "average_precision", "precision", "recall", "f1")
    values = np.stack([auc, ap, precision, recall, f1], axis=1)
    return {
        "rows": len(y),
        "per_label": {
            label: {
                "positive": int(y[:, i].sum()),
                **dict(zip(names, values[i].tolist(), strict=True)),
            }
            for i, label in enumerate(labels)
        },
        "macro": dict(zip(names, values.mean(axis=0).tolist(), strict=True)),
    }
