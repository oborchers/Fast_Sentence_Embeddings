"""Quality metrics shared by preparation verification and evidence identities."""

from __future__ import annotations

import numpy as np


def normalized(rows):
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms != 0)


def score(labels, rows):
    from scipy.stats import pearsonr

    rows = normalized(rows)
    half = len(labels)
    similarities = (rows[:half] * rows[half:]).sum(axis=1)
    return float(pearsonr(labels, similarities).statistic * 100)


def neighbors(rows):
    rows = normalized(rows)
    result = []
    for start in range(0, len(rows), 256):
        sims = rows[start : start + 256] @ rows.T
        sims[np.arange(len(sims)), start + np.arange(len(sims))] = -np.inf
        # Stable sorting fixes ties by row id, including duplicate STS sentences.
        result.append(np.argsort(-sims, axis=1, kind="stable")[:, :10])
    return np.concatenate(result)


def recall(reference, actual):
    return float(
        np.mean([len(set(a) & set(b)) / 10 for a, b in zip(reference, actual, strict=True)])
    )
