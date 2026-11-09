"""Frozen-score summaries with group-resampled AP intervals and explicit subsets."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def intervals(y, scores, rows, config, budget):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["normalized"]].append(i)
    members = list(groups.values())
    rng = np.random.default_rng(config["seed"])
    samples = [[] for _ in range(y.shape[1])]
    missing = [0] * y.shape[1]
    for _ in range(config["bootstrap_draws"]):
        budget.check()
        indices = np.concatenate(
            [members[i] for i in rng.integers(len(members), size=len(members))]
        )
        yy, ss = y[indices], scores[indices]
        for j in range(y.shape[1]):
            if not yy[:, j].any():
                missing[j] += 1
            else:
                samples[j].append(float(average_precision_score(yy[:, j], ss[:, j])))
    return [
        {
            "ap_percentile_95": np.percentile(values, [2.5, 97.5]).tolist() if values else None,
            "draws_without_positive": missing[j],
            "accepted_draws": len(values),
            "groups": len(members),
        }
        for j, values in enumerate(samples)
    ]


def summarize(y, scores, predicted, rows, labels, config, budget, *, bootstrap):
    if y.ndim == 1:
        y = np.column_stack([y == 0, y == 1])
        predicted = np.column_stack([predicted == 0, predicted == 1])
    if y.shape != scores.shape or predicted.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError("target, prediction and finite score shapes must agree")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average=None, zero_division=0
    )
    uncertainty = intervals(y, scores, rows, config, budget) if bootstrap else [{} for _ in labels]
    values = {}
    for j, label in enumerate(labels):
        positives = int(y[:, j].sum())
        negatives = len(y) - positives
        values[label] = {
            "positive": positives,
            "negative": negatives,
            "precision": float(precision[j]),
            "recall": float(recall[j]),
            "f1": float(f1[j]),
            "average_precision": float(average_precision_score(y[:, j], scores[:, j]))
            if positives
            else None,
            "roc_auc": float(roc_auc_score(y[:, j], scores[:, j]))
            if positives and negatives
            else None,
            "exploratory": positives < config["minimum_positives"],
            **uncertainty[j],
        }
    return {
        "rows": len(rows),
        "groups": len({r["normalized"] for r in rows}),
        "unsafe_fraction": sum(r["unsafe"] for r in rows) / len(rows),
        "per_label": values,
    }


def evaluate(targets, scores, binary_predictions, rows, config, budget, *, bootstrap):
    subsets = {
        "all": np.ones(len(rows), dtype=bool),
        "prompt_only": np.array([not r["has_response"] for r in rows]),
        "without_exact_aart": np.array([not r["source_aart"] for r in rows]),
    }
    result = {}
    for subset, keep in subsets.items():
        if not keep.any():
            raise ValueError("empty evaluation subset")
        selected = [r for r, include in zip(rows, keep, strict=True) if include]
        result[subset] = {}
        for name, labels in config["labels"].items():
            predicted = (
                binary_predictions if name == "safety" else scores[name] > config["threshold"]
            )
            result[subset][name] = summarize(
                targets[name][keep],
                scores[name][keep],
                predicted[keep],
                selected,
                labels,
                config,
                budget,
                bootstrap=bootstrap,
            )
    return result
