#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

import logging
from typing import Any, cast

from numpy import float32 as REAL
from numpy import isfinite, ndarray

from fse._validation import (
    NonNegativeInt,
    PathLike,
    PositiveFloat,
    PositiveInt,
    WordVectorModel,
    validate_arguments,
)
from fse.models.average import Average
from fse.models.utils import (
    compute_principal_components,
    remove_principal_components,
    warn_cache_size_gb_deprecated,
)

logger = logging.getLogger(__name__)


class SIF(Average):
    # svd_res is None before training and 0 when no components are removed.
    svd_res: Any

    @validate_arguments
    def __init__(
        self,
        model: WordVectorModel,
        alpha: PositiveFloat = 1e-3,
        components: NonNegativeInt = 1,
        cache_size_gb: PositiveFloat | None = None,
        sv_mapfile_path: PathLike | None = None,
        wv_mapfile_path: PathLike | None = None,
        workers: PositiveInt = 1,
        lang_freq: str | None = None,
    ):
        """Smooth-inverse frequency (SIF) weighted sentence embeddings model. Performs a weighted averaging operation over all
        words in a sentences. After training, the model removes a number of singular vectors.

        The implementation is based on Arora et al. (2017): A Simple but Tough-to-Beat Baseline for Sentence Embeddings.
        For more information, see <https://openreview.net/pdf?id=SyK00v5xx> and <https://github.com/PrincetonML/SIF>

        Parameters
        ----------
        model : :class:`~gensim.models.keyedvectors.KeyedVectors` or a gensim model with a ``wv`` attribute, such as :class:`~gensim.models.Word2Vec` or :class:`~gensim.models.FastText`
            This object essentially contains the mapping between words and embeddings. To compute the sentence embeddings
            its ``key_to_index`` and ``vectors`` are required.
        alpha : float, optional
            Alpha is the weighting factor used to downweigh each individual word.
        components : int, optional
            Corresponds to the number of singular vectors to remove from the sentence embeddings.
        cache_size_gb : float, optional
            Deprecated and without effect since 1.1.0: the singular vectors are computed
            exactly over all sentence vectors. Removed in 2.0.
        sv_mapfile_path : str, optional
            Optional path to store the sentence-vectors in for very large datasets. Used for memmap.
        wv_mapfile_path : str, optional
            Optional path to store the word-vectors in for very large datasets. Used for memmap.
            Use sv_mapfile_path and wv_mapfile_path to train disk-to-disk without needing much ram.
        workers : int, optional
            Number of working threads, used for multithreading. For most tasks (few words in a sentence)
            a value of 1 should be more than enough.
        lang_freq : str, optional
            Some pre-trained embeddings, i.e. "GoogleNews-vectors-negative300.bin", do not contain information about
            the frequency of a word. As the frequency is required for estimating the word weights, we induce
            frequencies into the ``count`` attribute of each word based on :class:`~wordfreq`
            If no frequency information is available, you can choose the language to estimate the frequency.
            See https://github.com/LuminosoInsight/wordfreq

        """

        self.alpha = float(alpha)
        self.components = int(components)
        if cache_size_gb is not None:
            warn_cache_size_gb_deprecated()
        self.svd_res = None

        if lang_freq is None:
            logger.info(
                "make sure you are using a model with valid word-frequency information. Otherwise use lang_freq argument."
            )

        super().__init__(
            model=model,
            sv_mapfile_path=sv_mapfile_path,
            wv_mapfile_path=wv_mapfile_path,
            workers=workers,
            lang_freq=lang_freq,
        )

    def _check_parameter_sanity(self, **kwargs):
        """Check the sanity of all paramters"""
        if not all(self.word_weights <= 1.0) or not all(self.word_weights >= 0.0):
            raise ValueError("For SIF, all word weights must be 0 <= w_weight <= 1")
        if self.alpha <= 0.0:
            raise ValueError("Alpha must be greater than zero.")
        if self.components < 0.0:
            raise ValueError("Components must be greater or equal zero")

    def _pre_train_calls(self, **kwargs):
        """Function calls to perform before training"""
        self._compute_sif_weights()

    def _post_train_calls(self, update: bool = False, offset: int = 0, **kwargs):
        """Function calls to perform after training, such as computing eigenvectors"""
        if self.components > 0 and update and isinstance(self.svd_res, tuple):
            # Appended rows are treated like inference: reuse the learned components.
            remove_principal_components(self.sv.vectors[offset:], svd_res=self.svd_res)
        elif self.components > 0:
            self.svd_res = compute_principal_components(
                self.sv.vectors,
                components=self.components,
            )
            remove_principal_components(self.sv.vectors, svd_res=self.svd_res, inplace=True)
        else:
            self.svd_res = 0
            logger.info("no removal of principal components")

    def _post_inference_calls(self, output: ndarray | None = None, **kwargs):
        """Function calls to perform after training & inference"""
        if self.svd_res is None:
            raise RuntimeError("You must first train the model to obtain SVD components")
        elif self.components > 0:
            remove_principal_components(cast(ndarray, output), svd_res=self.svd_res, inplace=True)
        else:
            logger.info("no removal of principal components")

    def _check_dtype_santiy(self, **kwargs):
        """Check the dtypes of all attributes"""
        if self.word_weights.dtype != REAL:
            raise TypeError(f"type of word_weights is wrong: {self.word_weights.dtype}")
        if isinstance(self.svd_res, tuple):
            if self.svd_res[0].dtype != REAL:
                raise TypeError(f"type of svd values is wrong: {self.svd_res[0].dtype}")
            if self.svd_res[1].dtype != REAL:
                raise TypeError(f"type of svd components is wrong: {self.svd_res[1].dtype}")

    def _compute_sif_weights(self):
        """Precomputes the SIF weights for all words in the vocabulary"""
        logger.info(f"pre-computing SIF weights for {len(self.wv)} words")
        pw = self._word_probabilities()

        self.word_weights = (self.alpha / (self.alpha + pw)).astype(REAL)

        if not all(isfinite(self.word_weights)) or any(self.word_weights < 0):
            raise RuntimeError(
                "Encountered nan values. "
                "This likely happens because the word frequency information is wrong/missing. "
                "Consider restarting using lang_freq argument to infer frequency. "
            )
