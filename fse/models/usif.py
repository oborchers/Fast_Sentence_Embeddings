#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

import logging

from gensim.models.keyedvectors import KeyedVectors
from numpy import float32 as REAL
from numpy import isfinite, ndarray, zeros

from fse.models.average import Average
from fse.models.utils import (
    EPS,
    compute_principal_components,
    remove_principal_components,
)

logger = logging.getLogger(__name__)


class uSIF(Average):
    def __init__(
        self,
        model: KeyedVectors,
        length: int = None,
        components: int = 5,
        cache_size_gb: float = 1.0,
        sv_mapfile_path: str = None,
        wv_mapfile_path: str = None,
        workers: int = 1,
        lang_freq: str = None,
    ):
        """Unsupervised smooth-inverse frequency (uSIF) weighted sentence embeddings
        model. Performs a weighted averaging operation over all words in a sentences.
        After training, the model removes a number of weighted singular vectors.

        The implementation is based on Ethayarajh (2018): Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline.
        For more information, see <https://www.aclweb.org/anthology/W18-3012> and <https://github.com/kawine/usif>

        Parameters
        ----------
        model : :class:`~gensim.models.keyedvectors.KeyedVectors` or :class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            This object essentially contains the mapping between words and embeddings. To compute the sentence embeddings
            the wv.vocab and wv.vector elements are required.
        length : int, optional
            Corresponds to the average number of words in a sentence in the training corpus.
            If length is None, then the model takes the average number of words from
            :meth: `~fse.models.base_s2v.BaseSentence2VecModel.scan_sentences`
            Is equivalent to n in the paper.
        components : int, optional
            Corresponds to the number of singular vectors to remove from the sentence embeddings.
            Is equivalent to m in the paper.
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

        self.length = length
        self.components = int(components)
        self.cache_size_gb = float(cache_size_gb)
        self.svd_res = None
        self.svd_weights = None

        if lang_freq is None:
            logger.info(
                "make sure you are using a model with valid word-frequency information. Otherwise use lang_freq argument."
            )

        super(uSIF, self).__init__(
            model=model,
            sv_mapfile_path=sv_mapfile_path,
            wv_mapfile_path=wv_mapfile_path,
            workers=workers,
            lang_freq=lang_freq,
        )

    def _check_parameter_sanity(self):
        """Check the sanity of all paramters."""
        if self.length <= 0.0:
            raise ValueError("Length must be greater than zero.")
        if self.components < 0.0:
            raise ValueError("Components must be greater or equal zero")

    def _pre_train_calls(self, **kwargs):
        """Function calls to perform before training."""
        self.length = kwargs["average_length"] if self.length is None else self.length
        self._compute_usif_weights()

    def _post_train_calls(self):
        """Function calls to perform after training, such as computing eigenvectors."""
        if self.components > 0:
            self.svd_res = compute_principal_components(
                self.sv.vectors,
                components=self.components,
                cache_size_gb=self.cache_size_gb,
            )
            self.svd_weights = (self.svd_res[0] ** 2) / (
                self.svd_res[0] ** 2
            ).sum().astype(REAL)
            remove_principal_components(
                self.sv.vectors,
                svd_res=self.svd_res,
                weights=self.svd_weights,
                inplace=True,
            )
        else:
            self.svd_res = 0
            logger.info(f"no removal of principal components")

    def _post_inference_calls(self, output: ndarray, **kwargs):
        """Function calls to perform after training & inference."""
        if self.svd_res is None:
            raise RuntimeError(
                "You must first train the model to obtain SVD components"
            )
        elif self.components > 0:
            remove_principal_components(
                output, svd_res=self.svd_res, weights=self.svd_weights, inplace=True
            )
        else:
            logger.info(f"no removal of principal components")

    def _check_dtype_santiy(self):
        """Check the dtypes of all attributes."""
        if self.word_weights.dtype != REAL:
            raise TypeError(f"type of word_weights is wrong: {self.word_weights.dtype}")
        if self.svd_res is not None:
            if self.svd_res[0].dtype != REAL:
                raise TypeError(f"type of svd values is wrong: {self.svd_res[0].dtype}")
            if self.svd_res[1].dtype != REAL:
                raise TypeError(
                    f"type of svd components is wrong: {self.svd_res[1].dtype}"
                )
            if self.svd_weights.dtype != REAL:
                raise TypeError(
                    f"type of svd weights is wrong: {self.svd_weights.dtype}"
                )

    def _compute_usif_weights(self):
        """Precomputes the uSIF weights."""
        logger.info(f"pre-computing uSIF weights for {len(self.wv)} words")
        v = len(self.wv)
        corpus_size = 0

        pw = zeros(v, dtype=REAL)
        for word in self.wv.key_to_index.__dict__.keys():
            c = self.wv.get_vecattr(word, "count")
            corpus_size += c
            pw[self.wv.key_to_index[word]] = c
        pw /= corpus_size

        threshold = 1 - (1 - (1 / v)) ** self.length
        alpha = sum(pw > threshold) / v
        z = v / 2
        a = (1 - alpha) / ((alpha * z) + EPS)

        self.word_weights = (a / ((a / 2) + pw)).astype(REAL)

        if not all(isfinite(self.word_weights)):
            raise RuntimeError(
                "Encountered nan values. "
                "This likely happens because the word frequency information is wrong/missing. "
                "Consider restarting using lang_freq argument to infer frequency. "
            )
