#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

from fse.models.average import Average
from fse.models.utils import compute_principal_components, remove_principal_components

from gensim.models.keyedvectors import KeyedVectors

from numpy import ndarray, float32 as REAL, zeros, isfinite

import logging

logger = logging.getLogger(__name__)


class SIF(Average):
    def __init__(
        self,
        model: KeyedVectors,
        alpha: float = 1e-3,
        components: int = 1,
        cache_size_gb: float = 1.0,
        sv_mapfile_path: str = None,
        wv_mapfile_path: str = None,
        workers: int = 1,
        lang_freq: str = None,
    ):
        """Smooth-inverse frequency (SIF) weighted sentence embeddings model. Performs a weighted averaging operation over all
        words in a sentences. After training, the model removes a number of singular vectors.

        The implementation is based on Arora et al. (2017): A Simple but Tough-to-Beat Baseline for Sentence Embeddings.
        For more information, see <https://openreview.net/pdf?id=SyK00v5xx> and <https://github.com/PrincetonML/SIF>

        Parameters
        ----------
        model : :class:`~gensim.models.keyedvectors.KeyedVectors` or :class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            This object essentially contains the mapping between words and embeddings. To compute the sentence embeddings
            the wv.vocab and wv.vector elements are required.
        alpha : float, optional
            Alpha is the weighting factor used to downweigh each individual word.
        components : int, optional
            Corresponds to the number of singular vectors to remove from the sentence embeddings.
        cache_size_gb : float, optional
            Cache size for computing the singular vectors in GB.
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
            frequencies into the wv.vocab.count based on :class:`~wordfreq`
            If no frequency information is available, you can choose the language to estimate the frequency.
            See https://github.com/LuminosoInsight/wordfreq

        """

        self.alpha = float(alpha)
        self.components = int(components)
        self.cache_size_gb = float(cache_size_gb)
        self.svd_res = None

        if lang_freq is None:
            logger.info(
                "make sure you are using a model with valid word-frequency information. Otherwise use lang_freq argument."
            )

        super(SIF, self).__init__(
            model=model,
            sv_mapfile_path=sv_mapfile_path,
            wv_mapfile_path=wv_mapfile_path,
            workers=workers,
            lang_freq=lang_freq,
        )

    def _check_parameter_sanity(self):
        """ Check the sanity of all paramters """
        if not all(self.word_weights <= 1.0) or not all(self.word_weights >= 0.0):
            raise ValueError("For SIF, all word weights must be 0 <= w_weight <= 1")
        if self.alpha <= 0.0:
            raise ValueError("Alpha must be greater than zero.")
        if self.components < 0.0:
            raise ValueError("Components must be greater or equal zero")

    def _pre_train_calls(self, **kwargs):
        """Function calls to perform before training """
        self._compute_sif_weights()

    def _post_train_calls(self):
        """ Function calls to perform after training, such as computing eigenvectors """
        if self.components > 0:
            self.svd_res = compute_principal_components(
                self.sv.vectors,
                components=self.components,
                cache_size_gb=self.cache_size_gb,
            )
            remove_principal_components(
                self.sv.vectors, svd_res=self.svd_res, inplace=True
            )
        else:
            self.svd_res = 0
            logger.info(f"no removal of principal components")

    def _post_inference_calls(self, output: ndarray, **kwargs):
        """ Function calls to perform after training & inference """
        if self.svd_res is None:
            raise RuntimeError(
                "You must first train the model to obtain SVD components"
            )
        elif self.components > 0:
            remove_principal_components(output, svd_res=self.svd_res, inplace=True)
        else:
            logger.info(f"no removal of principal components")

    def _check_dtype_santiy(self):
        """ Check the dtypes of all attributes """
        if self.word_weights.dtype != REAL:
            raise TypeError(f"type of word_weights is wrong: {self.word_weights.dtype}")
        if self.svd_res is not None:
            if self.svd_res[0].dtype != REAL:
                raise TypeError(f"type of svd values is wrong: {self.svd_res[0].dtype}")
            if self.svd_res[1].dtype != REAL:
                raise TypeError(
                    f"type of svd components is wrong: {self.svd_res[1].dtype}"
                )

    def _compute_sif_weights(self):
        """ Precomputes the SIF weights for all words in the vocabulary """
        logger.info(f"pre-computing SIF weights for {len(self.wv)} words")
        v = len(self.wv)
        corpus_size = 0

        pw = zeros(v, dtype=REAL)
        for word in self.wv.key_to_index:
            c = self.wv.get_vecattr(word, "count")
            corpus_size += c
            pw[self.wv.key_to_index[word]] = c
        pw /= corpus_size

        self.word_weights = (self.alpha / (self.alpha + pw)).astype(REAL)

        if not all(isfinite(self.word_weights)) or any(self.word_weights < 0):
            raise RuntimeError(
                "Encountered nan values. "
                "This likely happens because the word frequency information is wrong/missing. "
                "Consider restarting using lang_freq argument to infer frequency. "
            )
