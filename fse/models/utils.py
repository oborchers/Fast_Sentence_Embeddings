#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

from sklearn.decomposition import TruncatedSVD

from numpy import ndarray, float32 as REAL, ones

from time import time

import logging

logger = logging.getLogger(__name__)

def compute_principal_components(vectors:ndarray, components:int=1) -> ndarray:
    """ Method used to compute the first singular vectors of a given matrix

    Parameters
    ----------
    vectors : ndarray
        (Sentence) vectors to compute the truncated SVD on
    components : int, optional
        Number of singular values/vectors to compute

    Returns
    -------
    ndarray, ndarray
        Singular values and singular vectors
    """
    start = time()
    svd = TruncatedSVD(n_components=components, n_iter=7, random_state=42, algorithm="randomized")
    svd.fit(vectors)
    elapsed = time()
    logger.info(f"computing {components} principal components took {int(elapsed-start)}s")
    return svd.singular_values_.astype(REAL), svd.components_.astype(REAL)

def remove_principal_components(vectors:ndarray, svd_res:[ndarray, ndarray], weights:ndarray=None, inplace:bool=True) -> ndarray:
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
    if len(components)==1:
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
    logger.info(f"removing {len(components)} principal components took {int(elapsed-start)}s")
    if not inplace:
        return output