#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2020 Oliver Borchers

from sklearn.decomposition import TruncatedSVD

from numpy import ndarray, float32 as REAL, ones, vstack, inf as INF, dtype
from numpy.random import choice

from time import time

import logging

from sys import platform

import ctypes

logger = logging.getLogger(__name__)


def set_madvise_for_mmap(return_madvise: bool = False) -> object:
    """ Method used to set madvise parameters.
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
) -> [ndarray, ndarray]:
    """ Method used to compute the first singular vectors of a given (sub)matrix

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
    svd = TruncatedSVD(
        n_components=components, n_iter=7, random_state=42, algorithm="randomized"
    )

    current_mem = INF
    sample_size = len(vectors)
    while current_mem >= cache_size_gb:
        current_mem = sample_size * vectors.shape[1] * dtype(REAL).itemsize / 1024 ** 3
        if current_mem < cache_size_gb:
            # Skip if full dataset is already smaller than cache_size_gb
            break
        sample_size *= 0.995
    sample_size = int(sample_size)

    if sample_size < len(vectors):
        logger.info(f"sampling {sample_size} vectors to compute principal components")
        sample_indices = choice(range(vectors.shape[0]), replace=False, size=int(1e6))
        svd.fit(vstack([vectors[i] for i in sample_indices]))
    else:
        svd.fit(vectors)

    elapsed = time()
    logger.info(
        f"computing {components} principal components took {int(elapsed-start)}s"
    )
    return svd.singular_values_.astype(REAL), svd.components_.astype(REAL)


def remove_principal_components(
    vectors: ndarray,
    svd_res: [ndarray, ndarray],
    weights: ndarray = None,
    inplace: bool = True,
) -> ndarray:
    """ Method used to remove the first singular vectors of a given matrix

    Parameters
    ----------
    vectors : ndarray
        (Sentence) vectors to remove components fromm
    svd_res : (ndarray, ndarray)
        Tuple consisting of the singular values and components to remove from the vectors
    weights : ndarray, optional
        Weights to be used to weigh the components which are removed from the vectors
    inplace : bool, optional
        If true, removes the componentens from the vectors inplace (memory efficient)

    Returns
    -------
    ndarray, ndarray
        Singular values and singular vectors
    """
    components = svd_res[1].astype(REAL)

    start = time()
    if weights is None:
        w_comp = components * ones(len(components), dtype=REAL)[:, None]
    else:
        w_comp = components * (weights[:, None].astype(REAL))

    output = None
    if len(components) == 1:
        if not inplace:
            output = vectors.dot(w_comp.transpose()) * w_comp
        else:
            vectors -= vectors.dot(w_comp.transpose()) * w_comp
    else:
        if not inplace:
            output = vectors.dot(w_comp.transpose()).dot(w_comp)
        else:
            vectors -= vectors.dot(w_comp.transpose()).dot(w_comp)
    elapsed = time()
    logger.info(
        f"removing {len(components)} principal components took {int(elapsed-start)}s"
    )
    if not inplace:
        return output
