#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

import ctypes
import logging
import sys
import warnings
from sys import platform
from time import time
from typing import Any

from numpy import (
    absolute,
    arange,
    asarray,
    empty,
    errstate,
    finfo,
    float64,
    integer,
    isfinite,
    maximum,
    ndarray,
    result_type,
    sign,
    sqrt,
    subtract,
    zeros,
)
from numpy import float32 as REAL
from numpy.linalg import eigh

logger = logging.getLogger(__name__)

EPS = finfo(REAL).eps

# Rows per block when removing principal components: about 20 MB at 300 dimensions.
REMOVAL_BLOCK_ROWS = 16_384


def set_madvise_for_mmap(return_madvise: bool = False) -> Any:
    """Method used to set madvise parameters.
    This problem adresses the memmap issue raised in https://github.com/numpy/numpy/issues/13172
    The issue is not applicable for windows

    Parameters
    ----------
    return_madvise : bool
        Returns the madvise object for unittests, se test_utils.py

    Returns
    -------
    object
        madvise object

    """

    if platform in ["linux", "linux2", "darwin", "aix"]:
        if platform == "darwin":
            # Path different for Macos
            madvise = ctypes.CDLL("libc.dylib").madvise
        if platform in ["linux", "linux2", "aix"]:
            madvise = ctypes.CDLL("libc.so.6").madvise
        madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        madvise.restype = ctypes.c_int

        if return_madvise:
            return madvise


def compute_principal_components(
    vectors: ndarray, components: int = 1, cache_size_gb: float | None = None
) -> tuple[ndarray, ndarray]:
    """Compute the first singular values and right singular vectors of a matrix, exactly.

    The vectors are not centered, as for a truncated SVD. The Gram matrix ``X^T X`` is
    computed over blocks of rows, summed in float64 and decomposed; its eigenvalues are the
    squared singular values and its eigenvectors the right singular vectors. There is no
    random sample: every row is read once, front to back, so a memmap streams from disk.
    Besides the dim x dim matrices, memory stays at about one block: float32 and float64
    rows are used as they are, other types are converted per block.

    Each block product runs in the precision of the vectors (float32 for float16), so
    directions whose variance is below the rounding of that precision are not determined;
    their removal hardly changes the vectors. A warning is logged when a requested
    component falls below a heuristic bound for that rounding. Where singular values are equal, the basis inside their
    subspace depends on LAPACK.

    Parameters
    ----------
    vectors : ndarray
        (Sentence) vectors, one per row
    components : int, optional
        Number of singular values/vectors to compute
    cache_size_gb : float, optional
        Deprecated and without effect since 1.1.0: all vectors are used. Removed in 2.0.

    Returns
    -------
    ndarray, ndarray
        Singular values, largest first, and the singular vectors as rows, both float32
    """
    if cache_size_gb is not None:
        warn_cache_size_gb_deprecated()
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be a 2-dimensional array, got {vectors.ndim} dimensions")
    if vectors.dtype.kind not in "iuf":
        raise TypeError(f"vectors must hold real numbers, got dtype {vectors.dtype}")
    if isinstance(components, bool) or not isinstance(components, int | integer):
        raise TypeError(f"components must be an int, got {type(components)}")
    num_vectors, dim = vectors.shape
    if not 1 <= components <= min(num_vectors, dim):
        raise ValueError(
            f"components must be between 1 and the smaller of the number of vectors "
            f"({num_vectors}) and their dimension ({dim}), got {components}"
        )

    start = time()
    # Integers are converted to float64, so large values are not rounded.
    block_dtype = REAL if vectors.dtype.kind == "f" and vectors.dtype.itemsize <= 4 else float64
    gram = zeros((dim, dim), dtype=float64)
    for begin in range(0, num_vectors, REMOVAL_BLOCK_ROWS):
        block = asarray(vectors[begin : begin + REMOVAL_BLOCK_ROWS], dtype=block_dtype)
        if not isfinite(block).all():
            raise ValueError(f"the vectors contain NaN or infinite values in rows from {begin}")
        # An overflow is reported below as an error, not as a warning of the product; some
        # numpy builds also warn spuriously about finite BLAS results (numpy#28687).
        with errstate(over="ignore", invalid="ignore", divide="ignore"):
            gram += block.T @ block
        del block  # release the block before the next one is converted
    if not isfinite(gram).all():
        raise ValueError("the Gram matrix of the vectors overflows; scale the vectors down")

    # eigh returns the eigenvalues in ascending order; take the largest ones first.
    eigenvalues, eigenvectors = eigh(gram)
    eigenvalues = eigenvalues[::-1][:components]
    basis = eigenvectors[:, ::-1][:, :components].T
    # The sign of a singular vector is arbitrary; make the largest entry positive, as
    # scikit-learn's svd_flip does, so v and -v from different LAPACK builds agree.
    largest = absolute(basis).argmax(axis=1)
    basis *= sign(basis[arange(components), largest])[:, None]
    # A heuristic bound for the rounding of the block products: it grows with the length
    # of the dot products (the block rows) and with the dimension.
    resolution = eigenvalues[0] * finfo(block_dtype).eps * max(dim, REMOVAL_BLOCK_ROWS**0.5)
    if eigenvalues[-1] <= resolution:
        logger.warning(
            f"{int((eigenvalues <= resolution).sum())} of {components} principal components "
            "have a variance below the rounding of the vectors; their directions are not "
            "determined by the data"
        )
    # Rounding can make eigenvalues of a rank-deficient matrix slightly negative.
    singular_values = sqrt(maximum(eigenvalues, 0.0))

    elapsed = time()
    logger.info(f"computing {components} principal components took {int(elapsed - start)}s")
    return singular_values.astype(REAL), basis.astype(REAL)


def check_components_fit(
    components: int, svd_res: Any, update: bool, rows_before: int, new_rows: int, dim: int
):
    """Raise before training if the principal components cannot be computed afterwards.

    Called before any state changes, so a failing train leaves the model as it was. An
    update with learned components reuses them and needs no check.
    """
    if components == 0 or (update and isinstance(svd_res, tuple)):
        return
    rows = (rows_before if update else 0) + new_rows
    if components > min(rows, dim):
        raise ValueError(
            f"components={components} needs at least as many sentence vectors and dimensions; "
            f"training would give {rows} vectors of dimension {dim}"
        )


def warn_cache_size_gb_deprecated():
    """Warn that cache_size_gb has no effect, pointing at the caller outside fse."""
    frame, level = sys._getframe(1), 2
    while frame.f_back is not None and frame.f_globals.get("__name__", "").startswith(
        ("fse.", "pydantic")
    ):
        frame, level = frame.f_back, level + 1
    warnings.warn(
        "cache_size_gb has no effect since fse 1.1.0: principal components are computed "
        "exactly over all vectors. It will be removed in fse 2.0.",
        DeprecationWarning,
        stacklevel=level,
    )


def remove_principal_components(
    vectors: ndarray,
    svd_res: tuple[ndarray, ndarray],
    weights: ndarray | None = None,
    inplace: bool = True,
) -> ndarray | None:
    """Method used to remove the first singular vectors of a given matrix

    Parameters
    ----------
    vectors : ndarray
        (Sentence) vectors to remove components fromm
    svd_res : (ndarray, ndarray)
        Tuple consisting of the singular values and components to remove from the vectors
    weights : ndarray, optional
        Weights to be used to weigh the components which are removed from the vectors
    inplace : bool, optional
        If true, removes the components from the vectors inplace (memory efficient)

    Returns
    -------
    ndarray or None
        The vectors without the components if inplace is False, otherwise None.
    """
    components = svd_res[1].astype(REAL)
    scale = None if weights is None else weights.astype(REAL)[None, :]

    start = time()
    # The result type of vectors - projection, as for the whole matrix at once: an integer
    # input gives a float result instead of being truncated.
    output = vectors if inplace else empty(vectors.shape, dtype=result_type(vectors, components))
    # Work in blocks of rows, so the temporary projection never exceeds one block, also for
    # memory-mapped vectors that do not fit into RAM.
    for begin in range(0, len(vectors), REMOVAL_BLOCK_ROWS):
        block = vectors[begin : begin + REMOVAL_BLOCK_ROWS]
        # Coefficients of each vector on each component, scaled by the component weight.
        # The weight is applied once, so each component is removed as weight * projection.
        coefficients = block.dot(components.transpose())
        if scale is not None:
            coefficients *= scale
        # subtract with out raises for an unsafe cast, as vectors -= projection did.
        subtract(
            block, coefficients.dot(components), out=output[begin : begin + REMOVAL_BLOCK_ROWS]
        )
    elapsed = time()

    logger.info(f"removing {len(components)} principal components took {int(elapsed - start)}s")
    if not inplace:
        return output
    return None
