#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2020 Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Optimized cython functions for computing sentence embeddings"""

import cython
import numpy as np

cimport numpy as np

from libc.string cimport memset

import scipy.linalg.blas as fblas

from average_inner cimport (
    REAL_t,
    uINT_t,
    ONE,
    ZERO,
    ONEF,
    ZEROF,
    saxpy, 
    sscal, 
    BaseSentenceVecsConfig,
    FTSentenceVecsConfig,
    init_base_s2v_config,
    init_ft_s2v_config,
    populate_base_s2v_config,
    populate_ft_s2v_config,
)

DEF MAX_WORDS = 10000
DEF MAX_NGRAMS = 40

cdef void sl_max_pool(const int *N, float *X, const float *Y) nogil:
    """ Performs single left max pooling op

    Parameters
    ----------
    N : int *
        Vector size.
    X : float *
        Left vector.
    Y : float *
        Right vector.

    """
    cdef int i
    for i from 0 <= i < N[0] by 1:
        if X[i] < Y[i]:
            X[i] = Y[i]

cdef void compute_base_sentence_pooling(BaseSentenceVecsConfig *c, uINT_t num_sentences) nogil:
    """Perform optimized sentence-level max pooling for BaseAny2Vec model.

    Parameters
    ----------
    c : BaseSentenceVecsConfig *
        A pointer to a fully initialized and populated struct.
    num_sentences : uINT_t
        The number of sentences used to train the model.
    
    Notes
    -----
    This routine does not provide oov support.

    """
    cdef:
        int size = c.size

        uINT_t sent_idx, sent_start, sent_end, sent_row

        uINT_t i, word_idx, word_row

        REAL_t sent_len, inv_count

    for sent_idx in range(num_sentences):
        memset(c.mem, 0, size * cython.sizeof(REAL_t))
        memset(c.mem2, 0, size * cython.sizeof(REAL_t))

        sent_start = c.sentence_boundary[sent_idx]
        sent_end = c.sentence_boundary[sent_idx + 1]
        sent_len = ZEROF

        for i in range(sent_start, sent_end):
            sent_len += ONEF
            sent_row = c.sent_adresses[i] * size
            word_row = c.word_indices[i] * size
            
            sl_max_pool(
                &size, 
                &c.sentence_vectors[sent_row],
                &c.word_vectors[word_row],
            )
        # There's nothing to do here for many-to-one mappings


cdef void compute_ft_sentence_pooling(FTSentenceVecsConfig *c, uINT_t num_sentences) nogil:
    """Perform optimized sentence-level max pooling for FastText model.

    Parameters
    ----------
    c : FTSentenceVecsConfig *
        A pointer to a fully initialized and populated struct.
    num_sentences : uINT_t
        The number of sentences used to train the model.
    
    Notes
    -----
    This routine DOES provide oov support.

    """
    cdef:
        int size = c.size

        uINT_t sent_idx, sent_start, sent_end, sent_row

        uINT_t ngram_row, ngrams

        uINT_t i, j, word_idx, word_row

        REAL_t sent_len
        REAL_t inv_count, inv_ngram
        REAL_t oov_weight = c.oov_weight

    for sent_idx in range(num_sentences):
        sent_start = c.sentence_boundary[sent_idx]
        sent_end = c.sentence_boundary[sent_idx + 1]
        sent_len = ZEROF

        for i in range(sent_start, sent_end):
            sent_len += ONEF
            sent_row = c.sent_adresses[i] * size

            word_idx = c.word_indices[i]
            ngrams = c.subwords_idx_len[i]

            if ngrams == 0:
                word_row = c.word_indices[i] * size

                sl_max_pool(
                    &size, 
                    &c.sentence_vectors[sent_row],
                    &c.word_vectors[word_row],
                )

            else:
                memset(c.mem, 0, size * cython.sizeof(REAL_t))
                inv_ngram = (ONEF / <REAL_t>ngrams) * c.oov_weight
                for j in range(ngrams):
                    ngram_row = c.subwords_idx[(i * MAX_NGRAMS)+j] * size
                    saxpy(
                        &size, 
                        &inv_ngram, 
                        &c.ngram_vectors[ngram_row], 
                        &ONE, 
                        c.mem, 
                        &ONE
                    )

                sl_max_pool(
                    &size, 
                    &c.sentence_vectors[sent_row],
                    c.mem,
                )
        # There's nothing to do here for many-to-one mappings


def train_pooling_cy(model, indexed_sentences, target, memory):
    """Training on a sequence of sentences and update the target ndarray.

    Called internally from :meth:`~fse.models.pooling.MaxPooling._do_train_job`.

    Parameters
    ----------
    model : :class:`~fse.models.base_s2v.BaseSentence2VecModel`
        The BaseSentence2VecModel model instance.
    indexed_sentences : iterable of tuple
        The sentences used to train the model.
    target : ndarray
        The target ndarray. We use the index from indexed_sentences
        to write into the corresponding row of target.
    memory : ndarray
        Private memory for each working thread.

    Returns
    -------
    int, int
        Number of effective sentences (non-zero) and effective words in the vocabulary used 
        during training the sentence embedding.
    """

    cdef uINT_t eff_sentences = 0
    cdef uINT_t eff_words = 0
    cdef BaseSentenceVecsConfig w2v
    cdef FTSentenceVecsConfig ft

    if not model.is_ft:
        init_base_s2v_config(&w2v, model, target, memory)

        eff_sentences, eff_words = populate_base_s2v_config(&w2v, model.wv.vocab, indexed_sentences)

        if not model.hierarchical:
            with nogil:
                compute_base_sentence_pooling(&w2v, eff_sentences)
    else:        
        init_ft_s2v_config(&ft, model, target, memory)

        eff_sentences, eff_words = populate_ft_s2v_config(&ft, model.wv.vocab, indexed_sentences)

        if not model.hierarchical:
            with nogil:
                compute_ft_sentence_pooling(&ft, eff_sentences) 
    
    return eff_sentences, eff_words

def init():
    return 1

MAX_WORDS_IN_BATCH = MAX_WORDS
MAX_NGRAMS_IN_BATCH = MAX_NGRAMS
FAST_VERSION = init()