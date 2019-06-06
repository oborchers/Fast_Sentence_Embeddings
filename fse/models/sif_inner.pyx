#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

"""Optimized cython functions for computing SIF embeddings"""

import numpy as np
cimport numpy as np
import cython

import scipy.linalg.blas as fblas

cdef extern from "../voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

REAL = np.float32 
INT = np.intc

ctypedef np.float32_t REAL_t
ctypedef np.int32_t INT_t

ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)

ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer)

cdef REAL_t ONEF = <REAL_t>1.0
cdef int ONE = 1

cdef REAL_t EPS = <REAL_t>1e-8

zeros = np.zeros

def sif_embeddings(sentences, wv, sif_vectors):

    cdef int i, sentence_len, size = wv.vector_size
    cdef REAL_t *vectors = <REAL_t *>(np.PyArray_DATA(sif_vectors))
    
    output = zeros((len(sentences), size), dtype=REAL)   
    cdef REAL_t *sv = <REAL_t *>(np.PyArray_DATA(output))
    cdef INT_t *sentence_view

    cdef str w

    vlookup = wv.vocab
    as_array = np.asarray
    
    for i in xrange(len(sentences)):
        sentence_idx = as_array([vlookup[w].index for w in sentences[i] if w in vlookup], dtype=INT)
        sentence_len = len(sentence_idx)
        if sentence_len:
            sentence_view = <INT_t *>(np.PyArray_DATA(sentence_idx))
            sif_embeddings_blas_inner(size, sentence_view, sentence_len, i, vectors, sv)
    return output + EPS

cdef void sif_embeddings_blas_inner(const int size, const INT_t *sentence_view, const int sentence_len, 
                                   const int sentence_idx, const REAL_t *vectors, REAL_t *summary_vectors) nogil:
    
    cdef int i,d, word_index
    cdef REAL_t inv = ONEF, count = <REAL_t> 0.
    
    for i in xrange(sentence_len):
        count += ONEF
        word_index = sentence_view[i]
        saxpy(&size, &ONEF, &vectors[word_index * size], &ONE, &summary_vectors[sentence_idx * size], &ONE)
        
    inv = ONEF / count
    sscal(&size, &inv, &summary_vectors[sentence_idx * size], &ONE)