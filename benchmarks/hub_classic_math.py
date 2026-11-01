"""The recipe's SIF weights, uncentered ten components and bounded table folding."""

from __future__ import annotations

import numpy as np


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


def fold_table(table, basis, target):
    for start in range(0, len(table), 16384):
        target[start : start + 16384] = remove(table[start : start + 16384], basis)


def wordfreq_weights(words):
    """Exactly the frequency quantization and float32 weights of fse 1.x lang_freq=en."""
    from wordfreq import get_frequency_dict

    frequencies = get_frequency_dict("en", wordlist="best")
    domain = 2**31 - 1
    counts = np.asarray(
        [
            int(frequencies[word] * domain) if word in frequencies else int(1e-8 * domain)
            for word in words
        ],
        dtype=np.int64,
    )
    probabilities = counts.astype(np.float32)
    probabilities /= counts.sum()
    return (0.001 / (0.001 + probabilities)).astype(np.float32)
