"""The recipe's SIF weights, uncentered ten components and bounded table folding."""

from __future__ import annotations

import numpy as np


def sif_weights(words, counts) -> np.ndarray:
    probabilities = np.asarray([counts[word] for word in words], dtype=np.float64)
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("fit corpus has no vocabulary overlap")
    probabilities /= total
    return (1e-3 / (1e-3 + probabilities)).astype(np.float32)


def remove(rows, basis):
    return rows - (rows @ basis.T) @ basis


def components(gram: np.ndarray) -> np.ndarray:
    from scipy.linalg import eigh

    _, basis = eigh(gram, driver="evd")
    basis = basis[:, -10:][:, ::-1].T.astype(np.float32)
    # Eigenvector sign has no effect on the projection. Canonicalize it for the fit receipt.
    for row in basis:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return basis


def gram_of_batches(batches) -> np.ndarray:
    gram = None
    for rows in batches:
        if gram is None:
            gram = np.zeros((rows.shape[1], rows.shape[1]), dtype=np.float64)
        gram += rows.T @ rows  # Recipe: float32 block product, float64 accumulation.
    if gram is None:
        raise ValueError("fit corpus is empty")
    return gram


def reference_means(table, lookup, tokens):
    rows = np.zeros((len(tokens), table.shape[1]), dtype=np.float32)
    for index, words in enumerate(tokens):
        ids = [lookup[word] for word in words if word in lookup]
        if ids:
            rows[index] = table[ids].sum(axis=0, dtype=np.float32) / len(words)
    return rows


def fold_table(table, basis, target):
    for start in range(0, len(table), 16384):
        target[start : start + 16384] = remove(table[start : start + 16384], basis)
