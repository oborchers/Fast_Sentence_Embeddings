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

from gensim.models._utils_any2vec import compute_ngrams_bytes, ft_hash_bytes

from libc.string cimport memset
from libc.stdio cimport printf

import scipy.linalg.blas as fblas

cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x

cdef int ONE = <int>1
cdef int ZERO = <int>0

cdef REAL_t ONEF = <REAL_t>1.0
cdef REAL_t ZEROF = <REAL_t>0.0

DEF MAX_WORDS = 10000
DEF MAX_NGRAMS = 40

cdef init_base_s2v_config(BaseSentenceVecsConfig *c, model, target, memory):
    """Load BaseAny2Vec parameters into a BaseSentenceVecsConfig struct.

    Parameters
    ----------
    c : FTSentenceVecsConfig *
        A pointer to the struct to initialize.
    model : fse.models.base_s2v.BaseSentence2VecModel
        The model to load.
    target : np.ndarray
        The target array to write the averages to.
    memory : np.ndarray
        Private working memory for each worker.
        Consists of 2 nd.arrays.

    """
    c[0].workers = model.workers
    c[0].size = model.sv.vector_size

    c[0].mem = <REAL_t *>(np.PyArray_DATA(memory[0]))

    c[0].word_vectors = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    c[0].word_weights = <REAL_t *>(np.PyArray_DATA(model.word_weights))

    c[0].sentence_vectors = <REAL_t *>(np.PyArray_DATA(target))

cdef init_ft_s2v_config(FTSentenceVecsConfig *c, model, target, memory):
    """Load Fasttext parameters into a FTSentenceVecsConfig struct.

    Parameters
    ----------
    c : FTSentenceVecsConfig *
        A pointer to the struct to initialize.
    model : fse.models.base_s2v.BaseSentence2VecModel
        The model to load.
    target : np.ndarray
        The target array to write the averages to.
    memory : np.ndarray
        Private working memory for each worker.
        Consists of 2 nd.arrays.

    """

    c[0].workers = model.workers
    c[0].size = model.sv.vector_size
    c[0].min_n = model.wv.min_n
    c[0].max_n = model.wv.max_n
    c[0].bucket = model.wv.bucket

    c[0].oov_weight = <REAL_t>np.max(model.word_weights)

    c[0].mem = <REAL_t *>(np.PyArray_DATA(memory[0]))

    memory[1].fill(ZERO)    # Reset the ngram storage before filling the struct
    c[0].subwords_idx = <uINT_t *>(np.PyArray_DATA(memory[1]))

    c[0].word_vectors = <REAL_t *>(np.PyArray_DATA(model.wv.vectors_vocab))
    c[0].ngram_vectors = <REAL_t *>(np.PyArray_DATA(model.wv.vectors_ngrams))
    c[0].word_weights = <REAL_t *>(np.PyArray_DATA(model.word_weights))

    c[0].sentence_vectors = <REAL_t *>(np.PyArray_DATA(target))

cdef object populate_base_s2v_config(BaseSentenceVecsConfig *c, vocab, indexed_sentences):
    """Prepare C structures for BaseAny2VecModel so we can go "full C" and release the Python GIL.

    We create indices over the sentences.  We also perform some calculations for
    each token/ngram and store the result up front to save time.

    Parameters
    ----------
    c : BaseSentenceVecsConfig*
        A pointer to the struct that will contain the populated indices.
    vocab : dict
        The vocabulary
    indexed_sentences : iterable of tuple
        The sentences to read

    Returns
    -------
    eff_words : int
        The number of in-vocabulary tokens.
    eff_sents : int
        The number of non-empty sentences.

    """

    cdef uINT_t eff_words = ZERO    # Effective words encountered in a sentence
    cdef uINT_t eff_sents = ZERO    # Effective sentences encountered

    c.sentence_boundary[0] = ZERO

    for obj in indexed_sentences:
        if not obj[0]:
            continue
        for token in obj[0]:
            word = vocab[token] if token in vocab else None # Vocab obj
            if word is None:
                continue
            c.word_indices[eff_words] = <uINT_t>word.index
            c.sent_adresses[eff_words] = <uINT_t>obj[1]

            eff_words += ONE
            if eff_words == MAX_WORDS:
                break
        eff_sents += 1
        c.sentence_boundary[eff_sents] = eff_words

        if eff_words == MAX_WORDS:
            break   

    return eff_sents, eff_words

cdef object populate_ft_s2v_config(FTSentenceVecsConfig *c, vocab, indexed_sentences):
    """Prepare C structures for FastText so we can go "full C" and release the Python GIL.

    We create indices over the sentences.  We also perform some calculations for
    each token/ngram and store the result up front to save time.

    Parameters
    ----------
    c : FTSentenceVecsConfig*
        A pointer to the struct that will contain the populated indices.
    vocab : dict
        The vocabulary
    indexed_sentences : iterable of tuples
        The sentences to read

    Returns
    -------
    eff_words : int
        The number of in-vocabulary tokens.
    eff_sents : int
        The number of non-empty sentences.

    """

    cdef uINT_t eff_words = ZERO    # Effective words encountered in a sentence
    cdef uINT_t eff_sents = ZERO    # Effective sentences encountered

    c.sentence_boundary[0] = ZERO

    for obj in indexed_sentences:
        if not obj[0]:
            continue
        for token in obj[0]:
            c.sent_adresses[eff_words] = <uINT_t>obj[1]
            if token in vocab:
                # In Vocabulary
                word = vocab[token]
                c.word_indices[eff_words] = <uINT_t>word.index    
                c.subwords_idx_len[eff_words] = ZERO
            else:
                # OOV words --> write ngram indices to memory
                c.word_indices[eff_words] = ZERO

                encoded_ngrams = compute_ngrams_bytes(token, c.min_n, c.max_n)
                hashes = [ft_hash_bytes(n) % c.bucket for n in encoded_ngrams]

                c.subwords_idx_len[eff_words] = <uINT_t>min(len(encoded_ngrams), MAX_NGRAMS)
                for i, h in enumerate(hashes[:MAX_NGRAMS]):
                    c.subwords_idx[(eff_words * MAX_NGRAMS) + i] = <uINT_t>h
            
            eff_words += ONE

            if eff_words == MAX_WORDS:
                break
                
        eff_sents += 1
        c.sentence_boundary[eff_sents] = eff_words

        if eff_words == MAX_WORDS:
            break   

    return eff_sents, eff_words

cdef void compute_base_sentence_averages(BaseSentenceVecsConfig *c, uINT_t num_sentences) nogil:
    """Perform optimized sentence-level averaging for BaseAny2Vec model.

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

        sent_start = c.sentence_boundary[sent_idx]
        sent_end = c.sentence_boundary[sent_idx + 1]
        sent_len = ZEROF

        for i in range(sent_start, sent_end):
            sent_len += ONEF
            sent_row = c.sent_adresses[i] * size
            word_row = c.word_indices[i] * size
            word_idx = c.word_indices[i]

            saxpy(&size, &c.word_weights[word_idx], &c.word_vectors[word_row], &ONE, c.mem, &ONE)

        if sent_len > ZEROF:
            inv_count = ONEF / sent_len
            # If we perform the a*x on memory, the computation is compatible with many-to-one mappings
            # because it doesn't rescale the overall result
            saxpy(&size, &inv_count, c.mem, &ONE, &c.sentence_vectors[sent_row], &ONE)

cdef void compute_ft_sentence_averages(FTSentenceVecsConfig *c, uINT_t num_sentences) nogil:
    """Perform optimized sentence-level averaging for FastText model.

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
        memset(c.mem, 0, size * cython.sizeof(REAL_t))
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
                saxpy(&size, &c.word_weights[word_idx], &c.word_vectors[word_row], &ONE, c.mem, &ONE)
            else:
                inv_ngram = (ONEF / <REAL_t>ngrams) * c.oov_weight
                for j in range(ngrams):
                    ngram_row = c.subwords_idx[(i * MAX_NGRAMS)+j] * size
                    saxpy(&size, &inv_ngram, &c.ngram_vectors[ngram_row], &ONE, c.mem, &ONE)
                
        if sent_len > ZEROF:
            inv_count = ONEF / sent_len
            saxpy(&size, &inv_count, c.mem, &ONE, &c.sentence_vectors[sent_row], &ONE)

def train_average_cy(model, indexed_sentences, target, memory):
    """Training on a sequence of sentences and update the target ndarray.

    Called internally from :meth:`~fse.models.average.Average._do_train_job`.

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

        with nogil:
            compute_base_sentence_averages(&w2v, eff_sentences)
    else:        
        init_ft_s2v_config(&ft, model, target, memory)

        eff_sentences, eff_words = populate_ft_s2v_config(&ft, model.wv.vocab, indexed_sentences)

        with nogil:
            compute_ft_sentence_averages(&ft, eff_sentences) 
    
    return eff_sentences, eff_words

def init():
    return 1

MAX_WORDS_IN_BATCH = MAX_WORDS
MAX_NGRAMS_IN_BATCH = MAX_NGRAMS
FAST_VERSION = init()