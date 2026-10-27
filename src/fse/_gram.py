#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Principal components from the Gram matrix, exactly, for ``fit`` (numpy only).

The Gram matrix ``X^T X`` is summed in float64 over blocks of ``REMOVAL_BLOCK_ROWS``
rows in row order, each block product computed in the precision of the rows, and
decomposed with ``eigh``. Blocks are fixed by absolute row number (ARCHITECTURE.md,
"Numerical contract"), so the result does not depend on how the rows were produced.
"""

from __future__ import annotations

import logging
from time import time
from typing import Any

import numpy as np
from numpy import ndarray

logger = logging.getLogger(__name__)

# Rows per block of the Gram matrix: about 20 MB at 300 dimensions.
REMOVAL_BLOCK_ROWS = 16_384


def compute_principal_components(vectors: ndarray, components: int = 1) -> tuple[ndarray, ndarray]:
    """The first singular values and right singular vectors of a matrix, exactly.

    The vectors are not centered, as for a truncated SVD. Every row is read once, front to
    back, so a memmap streams from disk; besides the dim x dim matrices, memory stays at
    about one block. Each block product runs in the precision of the vectors (float32 for
    float16), so directions whose variance is below that rounding are not determined; a
    warning is logged when a requested component falls below a heuristic bound for it.

    Returns the singular values, largest first, and the singular vectors as rows, both
    float32.
    """
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be a 2-dimensional array, got {vectors.ndim} dimensions")
    if vectors.dtype.kind not in "iuf":
        raise TypeError(f"vectors must hold real numbers, got dtype {vectors.dtype}")
    if isinstance(components, bool) or not isinstance(components, int | np.integer):
        raise TypeError(f"components must be an int, got {type(components)}")
    num_vectors, dim = vectors.shape
    if not 1 <= components <= min(num_vectors, dim):
        raise ValueError(
            f"components must be between 1 and the smaller of the number of vectors "
            f"({num_vectors}) and their dimension ({dim}), got {components}"
        )
    start = time()
    block_dtype = gram_block_dtype(vectors.dtype)
    gram = gram_matrix(vectors, block_dtype)
    if not np.isfinite(gram).all():
        raise ValueError("the Gram matrix of the vectors overflows; scale the vectors down")
    result = components_from_gram(gram, num_vectors, components, block_dtype)
    logger.info(f"computing {components} principal components took {int(time() - start)}s")
    return result


def gram_block_dtype(vectors_dtype: np.dtype) -> Any:
    """The type of the block products: float32 for float32 and float16, else float64.

    Integers are converted to float64, so large values are not rounded.
    """
    return np.float32 if vectors_dtype.kind == "f" and vectors_dtype.itemsize <= 4 else np.float64


def gram_block_product(block: ndarray) -> ndarray:
    """``block.T @ block`` without numpy floating point warnings.

    An overflow is reported by the caller as an error, not as a warning of the product;
    some numpy builds also warn spuriously about finite BLAS results (numpy#28687).
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return block.T @ block


def add_gram_block(gram: ndarray, product: ndarray):
    """Add a block product into the float64 sum; inf - inf of an overflow does not warn."""
    with np.errstate(over="ignore", invalid="ignore"):
        gram += product


def gram_matrix(vectors: ndarray, block_dtype: Any) -> ndarray:
    """``X^T X`` summed in float64 over blocks of ``REMOVAL_BLOCK_ROWS`` rows, in order."""
    gram = np.zeros((vectors.shape[1], vectors.shape[1]), dtype=np.float64)
    for begin in range(0, vectors.shape[0], REMOVAL_BLOCK_ROWS):
        block = np.asarray(vectors[begin : begin + REMOVAL_BLOCK_ROWS], dtype=block_dtype)
        if not np.isfinite(block).all():
            raise ValueError(f"the vectors contain NaN or infinite values in rows from {begin}")
        add_gram_block(gram, gram_block_product(block))
        del block  # release the block before the next one is converted
    return gram


def components_from_gram(
    gram: ndarray, num_vectors: int, components: int, block_dtype: Any
) -> tuple[ndarray, ndarray]:
    """Singular values and right singular vectors from the Gram matrix of the vectors."""
    if not np.isfinite(gram).all():
        raise ValueError(
            "the Gram matrix of the sentence vectors is not finite: they contain NaN or "
            "infinite values, or values too large to square"
        )
    dim = gram.shape[0]
    # eigh returns the eigenvalues in ascending order; take the largest ones first.
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = eigenvalues[::-1][:components]
    basis = eigenvectors[:, ::-1][:, :components].T
    # The sign of a singular vector is arbitrary; make the largest entry positive, as
    # scikit-learn's svd_flip does, so v and -v from different LAPACK builds agree.
    largest = np.absolute(basis).argmax(axis=1)
    basis *= np.sign(basis[np.arange(components), largest])[:, None]
    # A heuristic bound for the rounding of the block products: it grows with the length
    # of the dot products (the block rows) and with the dimension.
    block_rows = min(num_vectors, REMOVAL_BLOCK_ROWS)
    resolution = eigenvalues[0] * np.finfo(block_dtype).eps * max(dim, block_rows**0.5)
    if eigenvalues[-1] <= resolution:
        logger.warning(
            f"{int((eigenvalues <= resolution).sum())} of {components} principal components "
            "have a variance below a heuristic bound for the rounding of the vectors; their "
            "directions may be affected by rounding"
        )
    # Rounding can make eigenvalues of a rank-deficient matrix slightly negative.
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))
    # C order, so the basis can be a step of an encoder as it is.
    return singular_values.astype(np.float32), np.ascontiguousarray(basis, dtype=np.float32)
