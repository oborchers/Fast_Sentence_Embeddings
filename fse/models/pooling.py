#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Oliver Borchers <oliver-borchers@outlook.de>
# Copyright (C) 2020 Oliver Borchers
# For License information, see corresponding LICENSE file.

"""This module implements the base class to compute Max Pooling representations for sentences, using highly optimized C routines,
data streaming and Pythonic interfaces.

The implementation is based on Shen et al. (2018): Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms.
For more information, see <https://arxiv.org/pdf/1805.09843.pdf>.

The training algorithms is based on the Gensim implementation of Word2Vec, FastText, and Doc2Vec. 
For more information, see: :class:`~gensim.models.word2vec.Word2Vec`, :class:`~gensim.models.fasttext.FastText`, or
:class:`~gensim.models.doc2vec.Doc2Vec`.

Initialize and train a :class:`~fse.models.pooling.MaxPooling` model

.. sourcecode:: pycon

        >>> from gensim.models.word2vec import Word2Vec
        >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        >>> model = Word2Vec(sentences, min_count=1, size=20)

        >>> from fse.models.pooling import MaxPooling        
        >>> avg = MaxPooling(model)
        >>> avg.train([(s, i) for i, s in enumerate(sentences)])
        >>> avg.sv.vectors.shape
        (2, 20)

"""

from __future__ import division

from fse.models.base_s2v import BaseSentence2VecModel
from fse.models.utils import get_ft_word_vector
from fse.models.base_iterator import base_iterator, sentence_length

from gensim.models.keyedvectors import BaseKeyedVectors

from numpy import (
    ndarray,
    float32 as REAL,
    sum as np_sum,
    multiply as np_mult,
    zeros,
    amax as np_amax,
    maximum as np_maximum,
    max as np_max,
)

from typing import List

import logging

logger = logging.getLogger(__name__)

def average_window_kernel(
        model,
        word : str,
        mem : ndarray,
        mem2 : ndarray,
    ) -> int:
    """ Window kernel implements aggregation function for window.
    Does the vector conversion.
    All results will be stored in mem.
    """
    if word in model.wv.vocab:
        word_index = model.wv.vocab[word].index
        mem += model.wv.vectors[word_index] * model.word_weights[word_index]
        return 1
    else:
        if model.is_ft:
            mem += get_ft_word_vector(word, model) * np_max(model.word_weights)
            return 1
        else:
            return 0 # Word not taken into account
    return 1

def average_window_scaler(
        model,
        window_length : int,
        mem : ndarray,
        mem2 : ndarray,
    ) -> None:
    """ Window scaler implements scaling function for window result.
    All results will be stored in mem2.
    """
    mem /= window_length

def max_sentence_kernel(
        sent_length : int,
        sent_index : int,
        target : ndarray,
        mem : ndarray,
        mem2 : ndarray,
    ) -> int:
    """ Sentence kernel implements aggregation function for all windows.
    All results will be stored in target.
    """
    target[sent_index] = np_maximum(mem, target[sent_index],)

def max_sentence_scaler(
        win_count : int,
        sent_adr : int,
        target : ndarray,
        mem : ndarray,
        mem2 : ndarray,
    ) -> None:
    """ Sentence scaler implements scaling function for accumulated window result.
    All results will be stored in target.
    """
    # Does nothing, because max pooling
    pass

def train_pooling_np(
    model: BaseSentence2VecModel,
    indexed_sentences: List[tuple],
    target: ndarray,
    memory: tuple,
) -> [int, int]:
    return base_iterator(
        model=model,
        indexed_sentences=indexed_sentences,
        target=target,
        memory=memory,
        sentence_length=sentence_length,
        window_kernel=average_window_kernel,
        window_scaler=average_window_scaler,
        sentence_kernel=max_sentence_kernel,
        sentence_scaler=max_sentence_scaler,
    )

# try:
#     from fse.models.pooling_inner import train_pooling_cy
#     from fse.models.pooling_inner import (
#         FAST_VERSION,
#         MAX_WORDS_IN_BATCH,
#         MAX_NGRAMS_IN_BATCH,
#     )
#     train_pooling = train_pooling_cy
# except ImportError:
FAST_VERSION = -1
MAX_WORDS_IN_BATCH = 10000
MAX_NGRAMS_IN_BATCH = 40
train_pooling = train_pooling_np


class MaxPooling(BaseSentence2VecModel):
    """ Train, use and evaluate max pooling sentence vectors.

    The model can be stored/loaded via its :meth:`~fse.models.pooling.MaxPooling.save` and
    :meth:`~fse.models.pooling.MaxPooling.load` methods.

    Some important attributes are the following:

    Attributes
    ----------
    wv : :class:`~gensim.models.keyedvectors.BaseKeyedVectors`
        This object essentially contains the mapping between words and embeddings. After training, it can be used
        directly to query those embeddings in various ways. See the module level docstring for examples.
    
    sv : :class:`~fse.models.sentencevectors.SentenceVectors`
        This object contains the sentence vectors inferred from the training data. There will be one such vector
        for each unique docusentence supplied during training. They may be individually accessed using the index.
    
    prep : :class:`~fse.models.base_s2v.BaseSentence2VecPreparer`
        The prep object is used to transform and initialize the sv.vectors. Aditionally, it can be used
        to move the vectors to disk for training with memmap.
    
    """

    def __init__(
        self,
        model: BaseKeyedVectors,
        window_size: int = 1,
        window_stride: int = 1,
        sv_mapfile_path: str = None,
        wv_mapfile_path: str = None,
        workers: int = 1,
    ):
        """ Max pooling sentence embeddings model. Performs a simple maximum pooling operation over all
        words in a sentences without further transformations.

        The implementation is based on Shen et al. (2018): Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms.
        For more information, see <https://arxiv.org/pdf/1805.09843.pdf>.

        Parameters
        ----------
        model : :class:`~gensim.models.keyedvectors.BaseKeyedVectors` or :class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            This object essentially contains the mapping between words and embeddings. To compute the sentence embeddings
            the wv.vocab and wv.vector elements are required.
        hierarchical : bool
            If true, then perform hierarchical pooling operation
        window_size : int
            Set the size of the window used for hierarchical pooling operation
        window_stride: int
            Set adjacency of the window used for hierarchical pooling operation
        sv_mapfile_path : str, optional
            Optional path to store the sentence-vectors in for very large datasets. Used for memmap.
        wv_mapfile_path : str, optional
            Optional path to store the word-vectors in for very large datasets. Used for memmap.
            Use sv_mapfile_path and wv_mapfile_path to train disk-to-disk without needing much ram.
        workers : int, optional
            Number of working threads, used for multithreading. For most tasks (few words in a sentence)
            a value of 1 should be more than enough. 
        """
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)

        super(MaxPooling, self).__init__(
            model=model,
            sv_mapfile_path=sv_mapfile_path,
            wv_mapfile_path=wv_mapfile_path,
            workers=workers,
            lang_freq=None,
            batch_words=MAX_WORDS_IN_BATCH,
            batch_ngrams=MAX_NGRAMS_IN_BATCH,
            fast_version=FAST_VERSION,
        )

    def _do_train_job(
        self, data_iterable: List[tuple], target: ndarray, memory: tuple
    ) -> [int, int]:
        """ Internal routine which is called on training and performs pooling for all entries in the iterable """
        eff_sentences, eff_words = train_pooling(
            model=self, indexed_sentences=data_iterable, target=target, memory=memory,
        )
        return eff_sentences, eff_words

    def _check_parameter_sanity(self, **kwargs):
        """ Check the sanity of all child paramters """
        if not all(self.word_weights == 1.0):
            raise ValueError("All word weights must equal one for pool")
        if self.window_size < 1:
            raise ValueError("Window size must be greater than 1")
        if not 1 <= self.window_stride <= self.window_size:
            raise ValueError(f"Window stride must be 1 <= stride <= {self.window_size}")

    def _pre_train_calls(self, **kwargs):
        """Function calls to perform before training """
        pass

    def _post_train_calls(self, **kwargs):
        """ Function calls to perform after training, such as computing eigenvectors """
        pass

    def _post_inference_calls(self, **kwargs):
        """ Function calls to perform after training & inference
        Examples include the removal of components
        """
        pass

    def _check_dtype_santiy(self, **kwargs):
        """ Check the dtypes of all child attributes"""
        pass
