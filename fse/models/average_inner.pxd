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
ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil

cdef saxpy_ptr saxpy
cdef scopy_ptr scopy
cdef sscal_ptr sscal

DEF MAX_WORDS = 10000

cdef struct BaseSentenceVecsConfig:
    int size, workers

    # Vectors
    REAL_t *word_vectors
    REAL_t *word_weights
    REAL_t *sentence_vectors

    # REAL_t *work memory for summation?
    uINT_t word_indices[MAX_WORDS]
    uINT_t sent_adresses[MAX_WORDS]
    uINT_t sentence_boundary[MAX_WORDS + 1]
    
cdef init_base_s2v_config(BaseSentenceVecsConfig *c, model)