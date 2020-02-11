#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

from sklearn.decomposition import TruncatedSVD

from numpy import ndarray, float32 as REAL, ones, vstack, inf as INF, dtype
from numpy.random import choice

from time import time

import logging

from sys import platform

import ctypes

logger = logging.getLogger(__name__)


def set_madvise_for_mmap(return_madvise: bool = False) -> bool:
    # See memmap issue (https://github.com/numpy/numpy/issues/13172)
    if platform in ["linux", "darwin", "aix"]:
        if platform == "darwin":
            madvise = ctypes.CDLL("libc.dylib").madvise
        if platform in ["linux", "aix"]:
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
    while 1:
        current_mem = sample_size * vectors.shape[1] * dtype(REAL).itemsize / 1024 ** 3
        if current_mem < cache_size_gb:
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
    singular_values = svd_res[0].astype(REAL)
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
