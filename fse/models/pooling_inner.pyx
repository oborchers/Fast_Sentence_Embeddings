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
from libc.stdio cimport printf

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

cdef void swrmax_pool(
    const int *N, 
    const float *alpha,
    const float *X,
    float *Y,
) nogil:
    """ Performs single right weighted max pooling op

    Parameters
    ----------
    N : int *
        Vector size.
    alpha : float *
        Weighting applied to X.
    X : float *
        Left vector.
    Y : float *
        Right vector.

    """
    cdef int i
    for i from 0 <= i < N[0] by 1:
        if (alpha[0] * X[i]) > Y[i]:
            Y[i] = alpha[0] * X[i]

cdef void compute_base_sentence_pooling(
    BaseSentenceVecsConfig *c, 
    uINT_t num_sentences,
) nogil:
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

        uINT_t sent_pos, word_idx, word_row

        REAL_t sent_len, inv_count

    for sent_idx in range(num_sentences):
        sent_start = c.sentence_boundary[sent_idx]
        sent_end = c.sentence_boundary[sent_idx + 1]
        sent_len = ZEROF

        for sent_pos in range(sent_start, sent_end):
            sent_len += ONEF
            sent_row = c.sent_adresses[sent_pos] * size
            word_row = c.word_indices[sent_pos] * size
            word_idx = c.word_indices[sent_pos]
            
            swrmax_pool(
                &size, 
                &c.word_weights[word_idx], 
                &c.word_vectors[word_row], 
                &c.sentence_vectors[sent_row],
            )
        # There's nothing to do here for many-to-one mappings


cdef void compute_base_sentence_hier_pooling(
    BaseSentenceVecsConfig *c, 
    uINT_t num_sentences,
    uINT_t window_size,
    REAL_t window_stride,
) nogil:
    """Perform optimized sentence-level hierarchical max pooling for BaseAny2Vec model.

    Parameters
    ----------
    c : BaseSentenceVecsConfig *
        A pointer to a fully initialized and populated struct.
    num_sentences : uINT_t
        The number of sentences used to train the model.
    window_size : uINT_t
        The local window size.
    window_stride : REAL_t
        The local window stride.

    Notes
    -----
    This routine does not provide oov support.

    """
    cdef:
        int size = c.size

        uINT_t sent_idx, sent_start, sent_end, sent_row, window_end

        uINT_t sent_pos, window_pos, word_idx, word_row

        REAL_t sent_len, win_len, inv_count

    for sent_idx in range(num_sentences):
        sent_start = c.sentence_boundary[sent_idx]
        sent_end = c.sentence_boundary[sent_idx + 1]
        sent_len = ZEROF

        for sent_pos in range(sent_start, sent_end):
            sent_len += ONEF

            if (sent_len-ONEF) % window_stride != ZEROF:
                continue

            sent_row = c.sent_adresses[sent_pos] * size    

            if sent_pos + window_size > sent_end:
                window_end = sent_end
            else:
                window_end = sent_pos + window_size
            
            # Compute the locally averaged window
            win_len = ZEROF
            memset(c.mem, 0, size * cython.sizeof(REAL_t))
            memset(c.mem2, 0, size * cython.sizeof(REAL_t))
            for window_pos in range(sent_pos, window_end):
                win_len += ONEF
                
                word_row = c.word_indices[window_pos] * size
                word_idx = c.word_indices[window_pos]

                saxpy(
                    &size, 
                    &c.word_weights[word_idx], 
                    &c.word_vectors[word_row], 
                    &ONE, 
                    c.mem, 
                    &ONE
                )

            # Rescale for dynamic window size
            if win_len > ZEROF:
                inv_count = ONEF / win_len
                saxpy(
                    &size, 
                    &inv_count, 
                    c.mem, 
                    &ONE, 
                    c.mem2,
                    &ONE
                )

            swrmax_pool(
                &size, 
                &ONEF, 
                c.mem2,
                &c.sentence_vectors[sent_row],
            )

cdef void compute_ft_sentence_pooling(
    FTSentenceVecsConfig *c, 
    uINT_t num_sentences,
) nogil:
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

        uINT_t sent_pos, ngram_pos, word_idx, word_row

        REAL_t sent_len
        REAL_t inv_count, inv_ngram
        REAL_t oov_weight = c.oov_weight

    for sent_idx in range(num_sentences):
        sent_start = c.sentence_boundary[sent_idx]
        sent_end = c.sentence_boundary[sent_idx + 1]
        sent_len = ZEROF

        for sent_pos in range(sent_start, sent_end):
            sent_len += ONEF
            sent_row = c.sent_adresses[sent_pos] * size

            word_idx = c.word_indices[sent_pos]
            ngrams = c.subwords_idx_len[sent_pos]

            if ngrams == 0:
                word_row = c.word_indices[sent_pos] * size

                swrmax_pool(
                    &size, 
                    &c.word_weights[word_idx], 
                    &c.word_vectors[word_row], 
                    &c.sentence_vectors[sent_row],
                )

            else:
                memset(c.mem, 0, size * cython.sizeof(REAL_t))
                inv_ngram = (ONEF / <REAL_t>ngrams) * c.oov_weight
                for ngram_pos in range(ngrams):
                    ngram_row = c.subwords_idx[(sent_pos * MAX_NGRAMS)+ngram_pos] * size
                    saxpy(
                        &size, 
                        &inv_ngram, 
                        &c.ngram_vectors[ngram_row], 
                        &ONE, 
                        c.mem, 
                        &ONE
                    )

                swrmax_pool(
                    &size, 
                    &oov_weight, 
                    c.mem,
                    &c.sentence_vectors[sent_row],
                )
        # There's nothing to do here for many-to-one mappings

cdef void compute_ft_sentence_hier_pooling(
    FTSentenceVecsConfig *c, 
    uINT_t num_sentences,
    uINT_t window_size,
    REAL_t window_stride,
) nogil:
    """Perform optimized sentence-level hierarchical max pooling for FastText model.

    Parameters
    ----------
    c : FTSentenceVecsConfig *
        A pointer to a fully initialized and populated struct.
    num_sentences : uINT_t
        The number of sentences used to train the model.
    window_size : uINT_t
        The local window size.
    window_stride : REAL_t
        The local window stride.

    Notes
    -----
    This routine DOES provide oov support.

    """
    # The naming of the i,j,k vars is a bit different here

    cdef:
        int size = c.size

        uINT_t sent_idx, sent_start, sent_end, sent_row

        uINT_t ngram_row, ngrams

        uINT_t sent_pos, window_pos, ngram_pos, word_idx, word_row

        REAL_t sent_len, win_len
        REAL_t inv_count, inv_ngram
        REAL_t oov_weight = c.oov_weight

    for sent_idx in range(num_sentences):
        sent_start = c.sentence_boundary[sent_idx]
        sent_end = c.sentence_boundary[sent_idx + 1]
        sent_len = ZEROF

        for sent_pos in range(sent_start, sent_end):
            sent_len += ONEF

            if (sent_len-ONEF) % window_stride != ZEROF:
                continue

            sent_row = c.sent_adresses[sent_pos] * size    

            if sent_pos + window_size > sent_end:
                window_end = sent_end
            else:
                window_end = sent_pos + window_size
            
            # Compute the locally averaged window
            win_len = ZEROF
            memset(c.mem, 0, size * cython.sizeof(REAL_t))
            memset(c.mem2, 0, size * cython.sizeof(REAL_t))
            for window_pos in range(sent_pos, window_end):
                win_len += ONEF                
                ngrams = c.subwords_idx_len[window_pos]

                if ngrams == 0:
                    word_row = c.word_indices[window_pos] * size
                    word_idx = c.word_indices[window_pos]

                    saxpy(
                        &size, 
                        &c.word_weights[word_idx], 
                        &c.word_vectors[word_row], 
                        &ONE, 
                        c.mem, 
                        &ONE
                    )
                    
                else:
                    memset(c.mem2, 0, size * cython.sizeof(REAL_t))
                    inv_ngram = (ONEF / <REAL_t>ngrams) * c.oov_weight
                    for ngram_pos in range(ngrams):
                        ngram_row = c.subwords_idx[(window_pos * MAX_NGRAMS)+ngram_pos] * size
                        saxpy(
                            &size, 
                            &inv_ngram, 
                            &c.ngram_vectors[ngram_row], 
                            &ONE, 
                            c.mem2, 
                            &ONE
                        )

                    saxpy(
                        &size, 
                        &oov_weight, 
                        c.mem2, 
                        &ONE, 
                        c.mem,
                        &ONE
                    )

            memset(c.mem2, 0, size * cython.sizeof(REAL_t)) 
            # Rescale for dynamic window size
            if win_len > ZEROF:
                inv_count = ONEF / win_len
                saxpy(
                    &size, 
                    &inv_count, 
                    c.mem, 
                    &ONE, 
                    c.mem2,
                    &ONE
                )

            swrmax_pool(
                &size, 
                &ONEF, 
                c.mem2,
                &c.sentence_vectors[sent_row],
            )
        # There's nothing to do here for many-to-one mappings

def train_pooling_cy(
    model, 
    indexed_sentences, 
    target, 
    memory
):
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
    cdef uINT_t window_size = <uINT_t> model.window_size
    cdef REAL_t window_stride = <REAL_t> model.window_stride
    cdef BaseSentenceVecsConfig w2v
    cdef FTSentenceVecsConfig ft

    if not model.is_ft:
        init_base_s2v_config(&w2v, model, target, memory)

        eff_sentences, eff_words = populate_base_s2v_config(
            &w2v, 
            model.wv.vocab, 
            indexed_sentences
        )

        if not model.hierarchical:
            with nogil: 
                compute_base_sentence_pooling(
                    &w2v, 
                    eff_sentences
                )
        else:
            with nogil: 
                compute_base_sentence_hier_pooling(
                    &w2v, 
                    eff_sentences, 
                    window_size,
                    window_stride,
                )
    else:        
        init_ft_s2v_config(&ft, model, target, memory)

        eff_sentences, eff_words = populate_ft_s2v_config(
            &ft, 
            model.wv.vocab, 
            indexed_sentences
        )

        if not model.hierarchical:
            with nogil:
                compute_ft_sentence_pooling(&ft, eff_sentences) 
        else:
            with nogil: 
                compute_ft_sentence_hier_pooling(
                    &ft, 
                    eff_sentences, 
                    window_size,
                    window_stride,
                )
    
    return eff_sentences, eff_words

def init():
    return 1

MAX_WORDS_IN_BATCH = MAX_WORDS
MAX_NGRAMS_IN_BATCH = MAX_NGRAMS
FAST_VERSION = init()