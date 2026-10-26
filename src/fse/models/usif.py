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
from fse.models._quantize import StorageDtype
from fse.models.average import Average
from fse.models.utils import (
    EPS,
    check_components_fit,
    components_from_gram,
    compute_principal_components,
    gram_block_dtype,
    remove_principal_components,
    warn_cache_size_gb_deprecated,
)

logger = logging.getLogger(__name__)


class uSIF(Average):
    # svd_res is None before training and 0 when no components are removed.
    svd_res: Any
    svd_weights: Any
    component_weighting: str
    length: Any

    @validate_arguments
    def __init__(
        self,
        model: WordVectorModel,
        length: PositiveInt | None = None,
        components: NonNegativeInt = 5,
        cache_size_gb: PositiveFloat | None = None,
        sv_mapfile_path: PathLike | None = None,
        wv_mapfile_path: PathLike | None = None,
        workers: PositiveInt = 1,
        lang_freq: str | None = None,
        sv_dtype: StorageDtype = "float32",
        wv_dtype: StorageDtype = "float32",
    ):
        """Unsupervised smooth-inverse frequency (uSIF) weighted sentence embeddings
        model. Performs a weighted averaging operation over all words in a sentences.
        After training, the model removes a number of weighted singular vectors.

        The implementation is based on Ethayarajh (2018): Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline.
        For more information, see <https://www.aclweb.org/anthology/W18-3012> and <https://github.com/kawine/usif>

        Parameters
        ----------
        model : :class:`~gensim.models.keyedvectors.KeyedVectors` or a gensim model with a ``wv`` attribute, such as :class:`~gensim.models.Word2Vec` or :class:`~gensim.models.FastText`
            This object essentially contains the mapping between words and embeddings. To compute the sentence embeddings
            its ``key_to_index`` and ``vectors`` are required.
        length : int, optional
            Corresponds to the average number of words in a sentence in the training corpus.
            If length is None, then the model takes the average number of words from
            :meth: `~fse.models.base_s2v.BaseSentence2VecModel.scan_sentences`
            Is equivalent to n in the paper.
        components : int, optional
            Corresponds to the number of singular vectors to remove from the sentence embeddings.
            Is equivalent to m in the paper.
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

        self.length = length
        self.components = int(components)
        # Without effect since 1.1.0; kept with its old default until 2.0 for code that reads it.
        self.cache_size_gb = 1.0 if cache_size_gb is None else float(cache_size_gb)
        if cache_size_gb is not None:
            warn_cache_size_gb_deprecated()
        self.svd_res = None
        self.svd_weights = None
        # How svd_weights scale the removed components; models saved before 1.1.0 lack it.
        self.component_weighting = "lambda"

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
        """Check the sanity of all paramters."""
        if self.length <= 0.0:
            raise ValueError("Length must be greater than zero.")
        if self.components < 0.0:
            raise ValueError("Components must be greater or equal zero")

    def _pre_train_calls(self, **kwargs):
        """Function calls to perform before training."""
        self.length = kwargs["average_length"] if self.length is None else self.length
        self._compute_usif_weights()

    def _fuses_gram(self, update: bool) -> bool:
        """New components are computed from all rows, which training writes unless updating.

        A replaced _post_train_calls might change the vectors before computing them, so it
        gets no Gram matrix summed during training.
        """
        own_hook = "_post_train_calls" in self.__dict__ or (
            type(self)._post_train_calls is not uSIF._post_train_calls
        )
        return self.components > 0 and not update and not own_hook

    def _post_train_calls(self, update: bool = False, offset: int = 0, gram: Any = None, **kwargs):
        """Function calls to perform after training, such as computing eigenvectors."""
        if self.components > 0 and update and isinstance(self.svd_res, tuple):
            # Appended rows are treated like inference: reuse the learned components.
            self._remove_components(self.sv.vectors[offset:])
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
            self._remove_components(self.sv.vectors)
        else:
            self._set_components(None)

    def _component_plan(self, update: bool) -> str:
        """How training into windows treats the components: reuse, learn, or none."""
        if self.components == 0:
            return "none"
        return "reuse" if update and isinstance(self.svd_res, tuple) else "learn"

    def _set_components(self, svd_res: Any):
        """Keep the learned components and their weights, or none (svd_res 0)."""
        if svd_res is None:
            self.svd_res = 0
            logger.info("no removal of principal components")
            return
        self.svd_res = svd_res
        self.svd_weights = (svd_res[0] ** 2) / (svd_res[0] ** 2).sum().astype(REAL)
        self.component_weighting = "lambda"

    def _remove_components(self, vectors: ndarray):
        """Remove the learned components, weighted, from vectors in place, in blocks of rows."""
        remove_principal_components(
            vectors, svd_res=self.svd_res, weights=self.svd_weights, inplace=True
        )

    def _post_load_calls(self):
        """Keep models saved before 1.1.0 consistent with their stored sentence vectors.

        Before 1.1.0, uSIF removed each component weighted by lambda squared. The stored
        vectors of such a model carry that removal, so infer and train(update=True) keep
        using it: with svd_weights squared, the current formula reproduces the old one.
        """
        if hasattr(self, "component_weighting"):
            return
        self.component_weighting = "lambda_squared"
        if isinstance(self.svd_res, tuple) and self.svd_weights is not None:
            self.svd_weights = (self.svd_weights**2).astype(REAL)
            logger.warning(
                "this uSIF model was saved before fse 1.1.0 and removes principal components "
                "weighted by lambda squared; infer and train(update=True) keep that weighting "
                "so new vectors match the stored ones. Retrain the model to use the corrected "
                "weighting."
            )

    def _post_inference_calls(self, output: ndarray | None = None, **kwargs):
        """Function calls to perform after training & inference."""
        if self.svd_res is None:
            raise RuntimeError("You must first train the model to obtain SVD components")
        elif self.components > 0:
            self._remove_components(cast(ndarray, output))
        else:
            logger.info("no removal of principal components")

    def _check_dtype_santiy(self, **kwargs):
        """Check the dtypes of all attributes."""
        if self.word_weights.dtype != REAL:
            raise TypeError(f"type of word_weights is wrong: {self.word_weights.dtype}")
        if isinstance(self.svd_res, tuple):
            if self.svd_res[0].dtype != REAL:
                raise TypeError(f"type of svd values is wrong: {self.svd_res[0].dtype}")
            if self.svd_res[1].dtype != REAL:
                raise TypeError(f"type of svd components is wrong: {self.svd_res[1].dtype}")
            if self.svd_weights.dtype != REAL:
                raise TypeError(f"type of svd weights is wrong: {self.svd_weights.dtype}")

    def _compute_usif_weights(self):
        """Precomputes the uSIF weights."""
        logger.info(f"pre-computing uSIF weights for {len(self.wv)} words")
        v = len(self.wv)
        pw = self._word_probabilities()

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
