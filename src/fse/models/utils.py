#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

import ctypes
import logging
from sys import platform
from time import time
from typing import Any

from numpy import argsort, dtype, empty, finfo, ndarray, result_type, subtract
from numpy import float32 as REAL
from numpy.random import default_rng
from sklearn.decomposition import TruncatedSVD

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
    vectors: ndarray, components: int = 1, cache_size_gb: float = 1.0
) -> tuple[ndarray, ndarray]:
    """Method used to compute the first singular vectors of a given (sub)matrix

    Parameters
    ----------
    vectors : ndarray
        (Sentence) vectors to compute the truncated SVD on
    components : int, optional
        Number of singular values/vectors to compute
    cache_size_gb : float, optional
            Cache size for computing the principal components in GB

    Returns
    -------
    ndarray, ndarray
        Singular values and singular vectors
    """
    start = time()
    num_vectors = vectors.shape[0]
    svd = TruncatedSVD(n_components=components, n_iter=7, random_state=42, algorithm="randomized")

    sample_size = int(1024**3 * cache_size_gb / (vectors.shape[1] * dtype(REAL).itemsize))

    if sample_size > num_vectors:
        svd.fit(vectors)
    else:
        logger.info(f"sampling {sample_size} vectors to compute principal components")
        # Seeded like the SVD itself, so repeated runs pick the same sample.
        sample_indices = default_rng(42).choice(num_vectors, size=sample_size, replace=False)
        svd.fit(_take_rows_in_file_order(vectors, sample_indices))

    elapsed = time()
    logger.info(f"computing {components} principal components took {int(elapsed - start)}s")
    return svd.singular_values_.astype(REAL), svd.components_.astype(REAL)


def _take_rows_in_file_order(vectors: ndarray, indices: ndarray) -> ndarray:
    """Return vectors[indices], reading the rows in ascending order.

    The result is identical to fancy indexing, but a memmap is read front to back instead
    of at random positions. Rows are gathered in blocks, so the only large allocation is
    the result itself.
    """
    order = argsort(indices, kind="stable")
    sample = empty((len(indices), vectors.shape[1]), dtype=vectors.dtype)
    for begin in range(0, len(order), REMOVAL_BLOCK_ROWS):
        positions = order[begin : begin + REMOVAL_BLOCK_ROWS]
        sample[positions] = vectors[indices[positions]]
    return sample


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
