# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

cimport numpy as np

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

ctypedef np.float32_t REAL_t
ctypedef np.uint32_t uINT_t

# BLAS routine signatures
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil

cdef saxpy_ptr saxpy
cdef sscal_ptr sscal

DEF MAX_WORDS = 10000
DEF MAX_NGRAMS = 40

cdef struct BaseSentenceVecsConfig:
    int size, workers

    # Vectors
    REAL_t *mem
    REAL_t *word_vectors
    REAL_t *word_weights
    REAL_t *sentence_vectors

    uINT_t word_indices[MAX_WORDS]
    uINT_t sent_adresses[MAX_WORDS]
    uINT_t sentence_boundary[MAX_WORDS + 1]

cdef struct FTSentenceVecsConfig:
    int size, workers, min_n, max_n, bucket

    REAL_t oov_weight

    # Vectors
    REAL_t *mem
    REAL_t *word_vectors # Note: these will be the vocab vectors, not wv.vectors
    REAL_t *ngram_vectors
    REAL_t *word_weights

    REAL_t *sentence_vectors

    # REAL_t *work memory for summation?
    uINT_t word_indices[MAX_WORDS]
    uINT_t sent_adresses[MAX_WORDS]
    uINT_t sentence_boundary[MAX_WORDS + 1]

    # For storing the oov items
    uINT_t subwords_idx_len[MAX_WORDS]
    uINT_t *subwords_idx
    
cdef init_base_s2v_config(BaseSentenceVecsConfig *c, model, target, memory)

cdef init_ft_s2v_config(FTSentenceVecsConfig *c, model, target, memory)