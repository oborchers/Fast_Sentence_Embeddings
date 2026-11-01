"""Offline SIF/uSIF equations, following the references in recipe 01."""

import numpy as np
from collections import Counter
from scipy.linalg import eigh


def probabilities(words, corpus):
    counts = Counter(token for tokens in corpus for token in tokens)
    values = np.array([counts[word] for word in words], dtype=np.float64)
    if values.sum() == 0:
        raise ValueError("corpus has no vocabulary overlap")
    return values / values.sum()


def word_weights(p, method, length):
    if method == "mean":
        return np.ones(len(p), dtype=np.float32)
    if method == "sif":
        a = 1e-3
        return (a / (a + p)).astype(np.float32)
    if method != "usif":
        raise ValueError(method)
    v = len(p)
    threshold = 1 - (1 - 1 / v) ** length
    alpha = np.count_nonzero(p > threshold) / v
    a = (1 - alpha) / (alpha * (v / 2) + np.finfo(np.float32).eps)
    weights = a / (a / 2 + p)
    if not np.isfinite(weights).all():
        raise ValueError("uSIF weights are not finite")
    return weights.astype(np.float32)


def means(table, words, corpus):
    lookup = {word: i for i, word in enumerate(words)}
    rows = np.zeros((len(corpus), table.shape[1]), dtype=np.float32)
    for i, tokens in enumerate(corpus):
        ids = [lookup[t] for t in tokens if t in lookup]
        if ids:
            # Like 1.1.0, the offline mean counts unknown tokens in its denominator.
            rows[i] = table[ids].sum(axis=0, dtype=np.float32) / len(tokens)
    return rows


def components(rows, k, weighted):
    gram = np.zeros((rows.shape[1], rows.shape[1]), dtype=np.float64)
    for start in range(0, len(rows), 16384):
        block = rows[start : start + 16384]
        gram += block.T @ block  # float32 block product, float64 accumulation
    values, basis = eigh(gram, driver="evd")
    values = np.maximum(values[-k:][::-1], 0)
    basis = basis[:, -k:][:, ::-1].T.astype(np.float32)
    factors = (
        (values / values.sum()).astype(np.float32) if weighted else np.ones(k, dtype=np.float32)
    )
    return basis, factors


def remove(rows, basis, factors):
    return rows - ((rows @ basis.T) * factors) @ basis


def unit(rows):
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms != 0)
