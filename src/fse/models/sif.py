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
    validate_arguments,
)
from fse.models._gensim_types import WordVectorModel
from fse.models._quantize import StorageDtype
from fse.models.average import Average
from fse.models.utils import (
    check_components_fit,
    components_from_gram,
    compute_principal_components,
    gram_block_dtype,
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
        sv_dtype: StorageDtype = "float32",
        wv_dtype: StorageDtype = "float32",
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
        sv_dtype : {"float32", "float16", "int8"}, optional
            Storage format of the sentence vectors: float16 halves and int8 quarters their
            memory. They are computed in float32 and every read returns float32.
        wv_dtype : {"float32", "float16", "int8"}, optional
            Format in which the model keeps the word vectors it averages: float16 halves
            their memory and int8 quarters it. With AVX2 or on arm64, averaging also gets
            somewhat faster; without AVX2, including on Windows, it gets slower. The word
            vectors passed in are not changed.

        """

        self.alpha = float(alpha)
        self.components = int(components)
        # Without effect since 1.1.0; kept with its old default until 2.0 for code that reads it.
        self.cache_size_gb = 1.0 if cache_size_gb is None else float(cache_size_gb)
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
            sv_dtype=sv_dtype,
            wv_dtype=wv_dtype,
        )

    def _check_pre_training_sanity(
        self,
        total_sentences: int,
        total_words: int,
        average_length: int,
        update: bool = False,
        **kwargs,
    ):
        """Also check that the principal components can be computed after training."""
        super()._check_pre_training_sanity(total_sentences, total_words, average_length, **kwargs)
        check_components_fit(
            self.components,
            self.svd_res,
            update,
            len(self.sv.vectors),
            kwargs["max_index"],
            self.wv.vector_size,
            quantized=self.sv.quantized,
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

    def _fuses_gram(self, update: bool) -> bool:
        """New components are computed from all rows, which training writes unless updating.

        A replaced _post_train_calls might change the vectors before computing them, so it
        gets no Gram matrix summed during training.
        """
        own_hook = "_post_train_calls" in self.__dict__ or (
            type(self)._post_train_calls is not SIF._post_train_calls
        )
        return self.components > 0 and not update and not own_hook

    def _post_train_calls(self, update: bool = False, offset: int = 0, gram: Any = None, **kwargs):
        """Function calls to perform after training, such as computing eigenvectors"""
        if self.components > 0 and update and isinstance(self.svd_res, tuple):
            # Appended rows are treated like inference: reuse the learned components.
            self._remove_components(self.sv.vectors[offset:], self.workers)
        elif self.components > 0:
            self._set_components(
                compute_principal_components(self.sv.vectors, components=self.components)
                if gram is None
                # Summed during training, bit-identical to compute_principal_components.
                else components_from_gram(
                    gram,
                    len(self.sv.vectors),
                    self.components,
                    gram_block_dtype(self.sv.vectors.dtype),
                )
            )
            self._remove_components(self.sv.vectors, self.workers)
        else:
            self._set_components(None)

    def _component_plan(self, update: bool) -> str:
        """How training into windows treats the components: reuse, learn, or none."""
        if self.components == 0:
            return "none"
        return "reuse" if update and isinstance(self.svd_res, tuple) else "learn"

    def _set_components(self, svd_res: Any):
        """Keep the learned components, or none (svd_res 0) without components."""
        if svd_res is None:
            self.svd_res = 0
            logger.info("no removal of principal components")
        else:
            self.svd_res = svd_res

    def _remove_components(self, vectors: ndarray, workers: int = 1):
        """Remove the learned components from vectors in place, in blocks of rows."""
        remove_principal_components(vectors, svd_res=self.svd_res, inplace=True, workers=workers)

    def _post_inference_calls(self, output: ndarray | None = None, **kwargs):
        """Function calls to perform after training & inference"""
        if self.svd_res is None:
            raise RuntimeError("You must first train the model to obtain SVD components")
        elif self.components > 0:
            self._remove_components(cast(ndarray, output), self.workers)
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
