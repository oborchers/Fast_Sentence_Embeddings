#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

# Author: Oliver Borchers <borchers@bwl.uni-mannheim.de>
# Copyright (C) 2019 Oliver Borchers

"""Optimized cython functions for computing sentence embeddings"""

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

# Type definitions and pointers for BLAS routines
# See http://www.netlib.org/blas/#_level_1 for more information
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)

ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer)

cdef REAL_t ONEF = <REAL_t>1.0
cdef int ONE = 1

cdef REAL_t EPS = <REAL_t>1e-8

full = np.full
empty = np.empty

def s2v_train(sentences, len_sentences, wv, weights):
    """Train sentence embedding on a list of sentences

    Called internally from :meth:`~fse.models.sentence2vec.Sentence2Vec.train`.

    Parameters
    ----------
    sentences : iterable of list of str
        The corpus used to train the model.
    len_sentences : int
        Length of the sentence iterable
    wv : :class:`~gensim.models.keyedvectors.BaseKeyedVectors`
        The BaseKeyedVectors instance containing the vectors used for training
    weights : np.ndarray
        Weights used in the summation of the vectors

    Returns
    -------
    np.ndarray
        The sentence embedding matrix of dim len(sentences) * vector_size
    int
        Number of words in the vocabulary actually used for training.
    int 
        Number of sentences used for training.
    """

    # Setup variables
    cdef int len_sen = len_sentences, effective_words = 0, effective_sentences = 0
    cdef int i, sentence_len, size = wv.vector_size
    cdef REAL_t *vectors = <REAL_t *>(np.PyArray_DATA(wv.vectors))
    cdef REAL_t *vec_weights  = <REAL_t *>(np.PyArray_DATA(weights))

    # Materialize output array iteratively
    # We do start from a matrix with EPS values to prohibit divisions by zero in subsequent applications
    output = empty((len_sen, size), dtype=REAL)
    for i in xrange(len_sen):
        output[i] = full(size, EPS, dtype=REAL)

    cdef REAL_t *output_view = <REAL_t *>(np.PyArray_DATA(output))
    cdef INT_t *sentence_view

    cdef str w

    vlookup = wv.vocab
    as_array = np.asarray
    
    for i, s in enumerate(sentences):
        sentence_idx = as_array([vlookup[w].index for w in s if w in vlookup], dtype=INT)
        sentence_len = len(sentence_idx)
        if sentence_len:
            effective_words += sentence_len
            effective_sentences += ONE
            sentence_view = <INT_t *>(np.PyArray_DATA(sentence_idx))
            s2v_train_core(size, sentence_view, sentence_len, i, vectors, output_view, vec_weights)
    return output, effective_words, effective_sentences

cdef void s2v_train_core(const int size, const INT_t *sentence_view, const int sentence_len, 
                                   const int sentence_idx, const REAL_t *vectors, REAL_t *sent_vecs,
                                   const REAL_t *vec_weights) nogil:
    
    cdef int i,d, word_index
    cdef REAL_t inv = ONEF, count = <REAL_t> 0.
    
    for i in xrange(sentence_len):
        count += ONEF
        word_index = sentence_view[i]
        saxpy(&size, &vec_weights[word_index], &vectors[word_index * size], &ONE, &sent_vecs[sentence_idx * size], &ONE)

    inv = ONEF / count
    sscal(&size, &inv, &sent_vecs[sentence_idx * size], &ONE)