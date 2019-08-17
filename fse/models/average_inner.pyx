#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

"""Optimized cython functions for computing sentence embeddings"""

import cython
import numpy as np

cimport numpy as np

from libc.string cimport memset

import scipy.linalg.blas as fblas

cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x

cdef int ONE = <int>1
cdef int ZERO = <int>0

cdef REAL_t ONEF = <REAL_t>1.0
cdef REAL_t ZEROF = <REAL_t>0.0

DEF MAX_WORDS = 10000

from libc.stdio cimport printf
cdef void fprint(const int size, REAL_t *in_vec) nogil:
    for d in range(size):
        printf("%+4.4f ", in_vec[d])
    printf("\n")

cdef void iprint(const int size, uINT_t *in_vec) nogil:
    for d in range(size):
        printf("%d ", in_vec[d])
    printf("\n")

cdef init_base_s2v_config(BaseSentenceVecsConfig *c, model):
    c[0].workers = model.workers
    c[0].size = model.sv.vector_size

    c[0].word_vectors = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    c[0].word_weights = <REAL_t *>(np.PyArray_DATA(model.word_weights))
    c[0].sentence_vectors = <REAL_t *>(np.PyArray_DATA(model.sv.vectors))

cdef object populate_base_s2v_config(BaseSentenceVecsConfig *c, vocab, indexed_sentences):

    cdef uINT_t eff_words = ZERO    # Effective words encountered in a sentence
    cdef uINT_t eff_sents = ZERO    # Effective sentences encountered

    c.sentence_boundary[0] = ZERO

    for obj in indexed_sentences:
        if not obj.words:
            continue
        for token in obj.words:
            word = vocab[token] if token in vocab else None # Vocab obj
            if word is None:
                continue
            c.word_indices[eff_words] = <uINT_t>word.index
            c.sent_adresses[eff_words] = <uINT_t>obj.index

            eff_words += ONE

            if eff_words == MAX_WORDS:
                break
        
        eff_sents += 1
        c.sentence_boundary[eff_sents] = eff_words

        if eff_words == MAX_WORDS:
            break   

    return eff_sents, eff_words

cdef void compute_base_sentence_averages(BaseSentenceVecsConfig *c, uINT_t num_sentences) nogil:
    cdef:
        int size = c.size

        uINT_t sent_idx
        uINT_t sent_start
        uINT_t sent_end 
        uINT_t sent_row

        uINT_t i
        uINT_t word_idx
        uINT_t word_row

        uINT_t *word_ind = c.word_indices
        uINT_t *sent_adr = c.sent_adresses

        REAL_t sent_len
        REAL_t inv_count

        REAL_t *word_vectors = c.word_vectors
        REAL_t *word_weights = c.word_weights
        REAL_t *sent_vectors = c.sentence_vectors

    for sent_idx in range(num_sentences):
        sent_start = c.sentence_boundary[sent_idx]
        sent_end = c.sentence_boundary[sent_idx + 1]
        sent_len = ZEROF

        for i in range(sent_start, sent_end):
            sent_len += ONEF
            sent_row = sent_adr[i] * size
            word_row = word_ind[i] * size
            word_idx = word_ind[i]

            saxpy(&size, &word_weights[word_idx], &word_vectors[word_row], &ONE, &sent_vectors[sent_row], &ONE)

        if sent_len > ZEROF:
            inv_count = ONEF / sent_len
            sscal(&size, &inv_count, &sent_vectors[sent_row], &ONE)

def train_average_cy(model, indexed_sentences):

    cdef:
        BaseSentenceVecsConfig c
        uINT_t eff_sentences = 0
        uINT_t eff_words = 0

    init_base_s2v_config(&c, model)

    eff_sentences, eff_words = populate_base_s2v_config(&c, model.wv.vocab, indexed_sentences)

    with nogil:
        compute_base_sentence_averages(&c, eff_sentences)

    return eff_sentences, eff_words

def init():
    return 1

FAST_VERSION = init()
MAX_WORDS_IN_BATCH = MAX_WORDS