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

ctypedef np.float32_t REAL_t
ctypedef np.int32_t INT_t

ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)

ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer)

cdef REAL_t ONEF = <REAL_t>1.0
cdef int ONE = 1


def sif_embeddings_6(sentences, model):
    cdef int size = model.vector_size
    cdef float[:,:] vectors = model.wv.sif_vectors

    cdef int sentence_index, word_index, d, count = 0
    cdef float inv = 1.
    np_sum = np.sum
    
    output = np.zeros((len(sentences), size), dtype=np.float32)   
    cdef float[:,:] sv = output
    
    for sentence_index, sentence in enumerate(sentences):
        if len(sentence) > 0:
            count = 0
            for word_index in sentence:
                count += 1
                for d in range(size):
                    sv[sentence_index, d] += vectors[word_index, d]

            inv = (1./ <float>count)
            for d in range(size):
                sv[sentence_index, d] *= inv
    return output

def sif_embeddings_7(sentences, model):
    cdef int size = model.vector_size
    cdef float[:,:] vectors = model.wv.sif_vectors

    np_sum = np.sum
    np_asarray = np.asarray
    
    output = np.zeros((len(sentences), size), dtype=np.float32)   
    cdef float[:,:] sv = output
    
    cdef int[:] sentence_view
    cdef int sentence_len
    
    
    for i in xrange(len(sentences)):
        if len(sentences[i]) > 0:
            sentence_view = sentences[i]
            sentence_len = len(sentences[i])
            sif_embeddings_7_cloop(size, sentence_view, sentence_len, i, vectors, sv)
        
    return output

cdef void sif_embeddings_7_cloop(int size, int[:] sentence_view, int sentence_len, int sentence_idx, float[:,:] vectors, float[:,:] summary_vectors) nogil:
    cdef int i,d, word_index, count = 0
    cdef float inv = 1.
    
    for i in xrange(sentence_len):
        word_index = sentence_view[i]
        count += 1
        for d in xrange(size):
            summary_vectors[sentence_idx, d] += vectors[word_index, d]
    
    inv = (1./ <float>count)
    for d in xrange(size):
        summary_vectors[sentence_idx, d] *= inv


def sif_embeddings_8(sentences, model):
    cdef int size = model.vector_size
    cdef REAL_t *vectors = <REAL_t *>(np.PyArray_DATA(model.wv.sif_vectors))
    
    output = np.zeros((len(sentences), size), dtype=np.float32)   
    cdef REAL_t *sv = <REAL_t *>(np.PyArray_DATA(output))
    
    cdef INT_t *sentence_view
    
    for i in xrange(len(sentences)):
        if len(sentences[i]):
            sentence_view = <INT_t *>(np.PyArray_DATA(sentences[i]))
            sentence_len = len(sentences[i])
            sif_embeddings_8_inner(size, sentence_view, sentence_len, i, vectors, sv)
    return output

cdef void sif_embeddings_8_inner(const int size, const INT_t *sentence_view, const int sentence_len, 
                                   const int sentence_idx, const REAL_t *vectors, REAL_t *summary_vectors) nogil:
    
    cdef int i,d, word_index
    cdef REAL_t inv = ONEF, count = <REAL_t> 0.
    
    for i in xrange(sentence_len):
        count += ONEF
        word_index = sentence_view[i]
        saxpy(&size, &ONEF, &vectors[word_index * size], &ONE, &summary_vectors[sentence_idx * size], &ONE)
        
    inv = ONEF / count
    sscal(&size, &inv, &summary_vectors[sentence_idx * size], &ONE)