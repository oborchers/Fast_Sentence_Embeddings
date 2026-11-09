"""Shared per-label quality summaries for frozen classifier scores."""

from __future__ import annotations

import numpy as np
from inputs import coverage
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def metrics(y, scores, labels):
    if y.ndim == 1:
        if set(np.unique(y)) != {0, 1} or scores.shape != (len(y), 2):
            raise ValueError("binary targets need both classes and two probability columns")
        y = np.column_stack([y == 0, y == 1])
    if y.ndim != 2 or y.shape[1] != len(labels):
        raise ValueError("one label is required per target column")
    coverage(y)
    if scores.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError("invalid classifier scores")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        scores > 0.5,
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
