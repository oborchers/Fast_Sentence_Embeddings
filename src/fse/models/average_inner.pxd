# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

cimport numpy as np
from cpython.pythread cimport PyThread_type_lock

ctypedef np.float32_t REAL_t
ctypedef np.uint32_t uINT_t

# BLAS routine signatures
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) noexcept nogil

cdef saxpy_ptr saxpy

# Defaults for batch_words (words per job) and batch_ngrams (ngrams per unknown FastText
# word). They are no limits of the kernel, whose buffers fit every job.
cdef enum:
    MAX_WORDS = 10000
    MAX_NGRAMS = 40

# Read-only parameters of one train or infer call, passed to the kernel without the GIL.
cdef struct KernelParams:
    int size
    bint is_ft
    unsigned int min_n, max_n
    uINT_t bucket
    size_t max_ngrams
    REAL_t oov_weight

    REAL_t *word_vectors  # wv.vectors, or wv.vectors_vocab for FastText
    REAL_t *ngram_vectors  # wv.vectors_ngrams for FastText, else NULL
    REAL_t *word_weights
    REAL_t *target
    # 0, or the rows of a ring of windows in target: row r is then stored at r % ring_rows.
    size_t ring_rows
    # Striped row locks for many-to-one index mappings, or NULL for one row per sentence.
    PyThread_type_lock *locks

cdef enum:
    N_ROW_LOCKS = 256

cdef class RowLocks:
    cdef PyThread_type_lock locks[N_ROW_LOCKS]
