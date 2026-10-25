#!/usr/bin/env cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""Optimized cython functions for computing sentence embeddings"""

import cython
import numpy as np

cimport numpy as np

from cpython.pycapsule cimport PyCapsule_GetPointer
from cpython.pythread cimport (
    WAIT_LOCK,
    PyThread_acquire_lock,
    PyThread_allocate_lock,
    PyThread_free_lock,
    PyThread_release_lock,
)
from libc.string cimport memset
from libc.stdio cimport printf

import scipy.linalg.blas as fblas

# y += alpha * x, taken from scipy's BLAS. Raises at import if scipy exposes no pointer.
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCapsule_GetPointer(fblas.saxpy._cpointer, NULL)

cdef int ONE = <int>1
cdef int ZERO = <int>0

# Marks an out-of-vocabulary token that has no ngrams: it counts for the sentence length
# but adds no vector, as in the numpy fallback.
cdef uINT_t NO_VECTOR = <uINT_t>0xFFFFFFFF

cdef REAL_t ONEF = <REAL_t>1.0
cdef REAL_t ZEROF = <REAL_t>0.0

cdef class RowLocks:
    """Striped locks that serialize adding sentence averages into shared target rows.

    Only needed when several sentences map to the same row and more than one worker
    trains at the same time. Row ``r`` uses lock ``r % N_ROW_LOCKS``.
    """

    def __cinit__(self):
        cdef int i
        for i in range(N_ROW_LOCKS):
            self.locks[i] = NULL
        for i in range(N_ROW_LOCKS):
            self.locks[i] = PyThread_allocate_lock()
            if self.locks[i] == NULL:
                raise MemoryError("could not allocate row locks")

    def __dealloc__(self):
        cdef int i
        for i in range(N_ROW_LOCKS):
            if self.locks[i] != NULL:
                PyThread_free_lock(self.locks[i])

    def __reduce__(self):
        raise TypeError("RowLocks only exist during training and cannot be pickled")


# FastText ngram hashing, reimplemented after gensim's compute_ngrams_bytes and
# ft_hash_bytes (gensim/models/fasttext_inner.pyx), which port Facebook's fastText
# (src/dictionary.cc). Ngrams are taken over the UTF-8 characters of "<word>" in the
# same order: by start character, then by length. UTF-8 continuation bytes start with
# the bits 10.
cdef unsigned char _MB_MASK = 0xC0
cdef unsigned char _MB_START = 0x80


cdef inline uINT_t _ft_hash(const unsigned char *bytez, size_t start, size_t stop) noexcept nogil:
    """FNV-1a over bytez[start:stop], with each byte sign-extended as in fastText."""
    cdef uINT_t h = 2166136261
    cdef size_t k
    for k in range(start, stop):
        h = h ^ <uINT_t>(<signed char>bytez[k])
        h = h * 16777619
    return h


cdef size_t ft_ngram_hashes_into(
    const unsigned char *bytez,
    size_t num_bytes,
    unsigned int min_n,
    unsigned int max_n,
    uINT_t bucket,
    uINT_t *out,
    size_t max_out,
) noexcept nogil:
    """Write the bucket of the first max_out ngrams of the UTF-8 word "<word>" into out.

    Returns the number written, min(number of ngrams, max_out). A bucket of 0 means the
    model has no ngram vectors, so no ngram is written.
    """
    cdef size_t i, j, n
    cdef size_t written = 0
    if bucket == 0:
        return 0
    for i in range(num_bytes):
        if bytez[i] & _MB_MASK == _MB_START:
            continue
        j = i
        n = 1
        while j < num_bytes and n <= max_n:
            j += 1
            while j < num_bytes and (bytez[j] & _MB_MASK) == _MB_START:
                j += 1
            if n >= min_n and not (n == 1 and (i == 0 or j == num_bytes)):
                if written == max_out:
                    return written
                out[written] = _ft_hash(bytez, i, j) % bucket
                written += 1
            n += 1
    return written


def ft_ngram_hashes(word, unsigned int min_n, unsigned int max_n, uINT_t bucket, size_t max_ngrams):
    """Return the ngram buckets of word as gensim's ft_ngram_hashes(...)[:max_ngrams] does."""
    cdef bytes utf8_word = ("<%s>" % word).encode("utf-8")
    out = np.empty(max(len(utf8_word) * max(max_n, 1), 1), dtype=np.uint32)
    cdef uINT_t[::1] view = out
    cdef size_t n = ft_ngram_hashes_into(
        utf8_word, len(utf8_word), min_n, max_n, bucket, &view[0], min(max_ngrams, view.shape[0])
    )
    return out[:n].tolist()


cdef PyThread_type_lock *_row_locks_of(model):
    """Return the row locks set on the model for the current training, or NULL."""
    locks = getattr(model, "_row_locks", None)
    if locks is None:
        return NULL
    return (<RowLocks?>locks).locks


cdef init_base_s2v_config(BaseSentenceVecsConfig *c, model, target, memory):
    """Load the model parameters into a BaseSentenceVecsConfig struct.

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
    c[0].locks = _row_locks_of(model)

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
    c[0].locks = _row_locks_of(model)

cdef object populate_base_s2v_config(BaseSentenceVecsConfig *c, wv, indexed_sentences, n_rows):
    """Prepare C structures for Word2Vec-style vectors so the kernel can release the Python GIL.

    We create indices over the sentences.  We also perform some calculations for
    each token/ngram and store the result up front to save time.

    Parameters
    ----------
    c : BaseSentenceVecsConfig*
        A pointer to the struct that will contain the populated indices.
    wv : obj
        The word vector object
    indexed_sentences : iterable of tuple
        The sentences to read

    Returns
    -------
    eff_sents : int
        The number of non-empty sentences.
    eff_words : int
        The number of in-vocabulary tokens.

    """

    cdef uINT_t eff_words = ZERO    # Effective words encountered in a sentence
    cdef uINT_t eff_sents = ZERO    # Effective sentences encountered

    c.sentence_boundary[0] = ZERO

    for obj in indexed_sentences:
        # The kernel writes into target[obj[1]] through a raw pointer.
        if not 0 <= obj[1] < n_rows:
            raise IndexError(f"sentence index {obj[1]} is out of range for {n_rows} target rows")
        if not obj[0]:
            continue
        for token in obj[0]:
            word = token if token in wv.key_to_index else None
            if word is None:
                continue
            c.word_indices[eff_words] = <uINT_t>wv.key_to_index[token]
            c.sent_adresses[eff_words] = <uINT_t>obj[1]

            eff_words += ONE
            if eff_words == MAX_WORDS:
                break
        eff_sents += 1
        c.sentence_boundary[eff_sents] = eff_words

        if eff_words == MAX_WORDS:
            break

    return eff_sents, eff_words

cdef object populate_ft_s2v_config(FTSentenceVecsConfig *c, wv, indexed_sentences, n_rows):
    """Prepare C structures for FastText so we can go "full C" and release the Python GIL.

    We create indices over the sentences.  We also perform some calculations for
    each token/ngram and store the result up front to save time.

    Parameters
    ----------
    c : FTSentenceVecsConfig*
        A pointer to the struct that will contain the populated indices.
    wv : obj
        The word vector object
    indexed_sentences : iterable of tuples
        The sentences to read

    Returns
    -------
    eff_sents : int
        The number of non-empty sentences.
    eff_words : int
        The number of in-vocabulary tokens.

    """

    cdef uINT_t eff_words = ZERO    # Effective words encountered in a sentence
    cdef uINT_t eff_sents = ZERO    # Effective sentences encountered
    cdef bytes utf8_word
    cdef size_t ngrams

    c.sentence_boundary[0] = ZERO

    for obj in indexed_sentences:
        # The kernel writes into target[obj[1]] through a raw pointer.
        if not 0 <= obj[1] < n_rows:
            raise IndexError(f"sentence index {obj[1]} is out of range for {n_rows} target rows")
        if not obj[0]:
            continue
        for token in obj[0]:
            c.sent_adresses[eff_words] = <uINT_t>obj[1]
            if token in wv.key_to_index:
                # In Vocabulary
                c.word_indices[eff_words] = <uINT_t>wv.key_to_index[token]
                c.subwords_idx_len[eff_words] = ZERO
            else:
                # OOV words --> write ngram indices to memory
                utf8_word = ("<%s>" % token).encode("utf-8")
                ngrams = ft_ngram_hashes_into(
                    utf8_word, len(utf8_word), c.min_n, c.max_n, c.bucket,
                    &c.subwords_idx[eff_words * MAX_NGRAMS], MAX_NGRAMS,
                )
                c.word_indices[eff_words] = ZERO if ngrams else NO_VECTOR
                c.subwords_idx_len[eff_words] = <uINT_t>ngrams

            eff_words += ONE

            if eff_words == MAX_WORDS:
                break

        eff_sents += 1
        c.sentence_boundary[eff_sents] = eff_words

        if eff_words == MAX_WORDS:
            break

    return eff_sents, eff_words

cdef void compute_base_sentence_averages(BaseSentenceVecsConfig *c, uINT_t num_sentences) noexcept nogil:
    """Perform optimized sentence-level averaging for Word2Vec-style vectors.

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

        uINT_t sent_idx, sent_start, sent_end
        # Flat offsets row * size overflow 32 bits from 2**32 / size rows on.
        size_t sent_row, word_row

        uINT_t i, word_idx

        REAL_t sent_len, inv_count
        PyThread_type_lock lock

    for sent_idx in range(num_sentences):
        memset(c.mem, 0, size * cython.sizeof(REAL_t))

        sent_start = c.sentence_boundary[sent_idx]
        sent_end = c.sentence_boundary[sent_idx + 1]
        sent_len = ZEROF

        for i in range(sent_start, sent_end):
            sent_len += ONEF
            sent_row = <size_t>c.sent_adresses[i] * <size_t>size
            word_row = <size_t>c.word_indices[i] * <size_t>size
            word_idx = c.word_indices[i]

            saxpy(&size, &c.word_weights[word_idx], &c.word_vectors[word_row], &ONE, c.mem, &ONE)

        if sent_len > ZEROF:
            inv_count = ONEF / sent_len
            # If we perform the a*x on memory, the computation is compatible with many-to-one mappings
            # because it doesn't rescale the overall result
            if c.locks != NULL:
                lock = c.locks[c.sent_adresses[sent_start] % N_ROW_LOCKS]
                PyThread_acquire_lock(lock, WAIT_LOCK)
                saxpy(&size, &inv_count, c.mem, &ONE, &c.sentence_vectors[sent_row], &ONE)
                PyThread_release_lock(lock)
            else:
                saxpy(&size, &inv_count, c.mem, &ONE, &c.sentence_vectors[sent_row], &ONE)

cdef void compute_ft_sentence_averages(FTSentenceVecsConfig *c, uINT_t num_sentences) noexcept nogil:
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

        uINT_t sent_idx, sent_start, sent_end
        # Flat offsets row * size overflow 32 bits from 2**32 / size rows on.
        size_t sent_row, word_row, ngram_row

        uINT_t ngrams

        uINT_t i, j, word_idx

        REAL_t sent_len
        REAL_t inv_count, inv_ngram
        PyThread_type_lock lock
        REAL_t oov_weight = c.oov_weight


    for sent_idx in range(num_sentences):
        memset(c.mem, 0, size * cython.sizeof(REAL_t))
        sent_start = c.sentence_boundary[sent_idx]
        sent_end = c.sentence_boundary[sent_idx + 1]
        sent_len = ZEROF

        for i in range(sent_start, sent_end):
            sent_len += ONEF
            sent_row = <size_t>c.sent_adresses[i] * <size_t>size

            word_idx = c.word_indices[i]
            ngrams = c.subwords_idx_len[i]

            if ngrams == 0:
                if word_idx == NO_VECTOR:
                    continue
                word_row = <size_t>c.word_indices[i] * <size_t>size
                saxpy(&size, &c.word_weights[word_idx], &c.word_vectors[word_row], &ONE, c.mem, &ONE)
            else:
                inv_ngram = (ONEF / <REAL_t>ngrams) * c.oov_weight
                for j in range(ngrams):
                    ngram_row = <size_t>c.subwords_idx[(i * MAX_NGRAMS)+j] * <size_t>size
                    saxpy(&size, &inv_ngram, &c.ngram_vectors[ngram_row], &ONE, c.mem, &ONE)

        if sent_len > ZEROF:
            inv_count = ONEF / sent_len
            if c.locks != NULL:
                lock = c.locks[c.sent_adresses[sent_start] % N_ROW_LOCKS]
                PyThread_acquire_lock(lock, WAIT_LOCK)
                saxpy(&size, &inv_count, c.mem, &ONE, &c.sentence_vectors[sent_row], &ONE)
                PyThread_release_lock(lock)
            else:
                saxpy(&size, &inv_count, c.mem, &ONE, &c.sentence_vectors[sent_row], &ONE)

def _check_array(array, name, dtype):
    """Raise unless array is a C-contiguous ndarray of dtype, because the kernel takes raw pointers."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {np.dtype(dtype).name}, got {array.dtype}")
    if not array.flags.c_contiguous:
        raise TypeError(f"{name} must be C-contiguous")


def _check_inputs(model, target, memory):
    """Validate every array the kernel reads or writes through a raw pointer."""
    _check_array(target, "target", np.float32)
    _check_array(memory[0], "memory[0]", np.float32)
    _check_array(model.word_weights, "word_weights", np.float32)
    if not target.flags.writeable:
        raise ValueError("target must be writeable")
    # The kernel strides through all matrices by the sentence vector size.
    size = model.sv.vector_size
    if model.is_ft:
        word_vectors = model.wv.vectors_vocab
        _check_array(word_vectors, "wv.vectors_vocab", np.float32)
        _check_array(model.wv.vectors_ngrams, "wv.vectors_ngrams", np.float32)
        if model.wv.vectors_ngrams.shape != (model.wv.bucket, size):
            raise ValueError(
                f"wv.vectors_ngrams must have shape ({model.wv.bucket}, {size}), "
                f"got {model.wv.vectors_ngrams.shape}"
            )
        _check_array(memory[1], "memory[1]", np.uint32)
        if memory[1].size < MAX_WORDS * MAX_NGRAMS:
            raise ValueError(
                f"memory[1] must hold at least {MAX_WORDS} x {MAX_NGRAMS} entries, "
                f"got shape {memory[1].shape}"
            )
    else:
        word_vectors = model.wv.vectors
        _check_array(word_vectors, "wv.vectors", np.float32)
    if word_vectors.ndim != 2 or word_vectors.shape[1] != size:
        raise ValueError(f"word vectors must have shape (n, {size}), got {word_vectors.shape}")
    if model.word_weights.shape != (word_vectors.shape[0],):
        raise ValueError(
            f"word_weights must have one weight per word vector ({word_vectors.shape[0]}), "
            f"got shape {model.word_weights.shape}"
        )
    if target.ndim != 2 or target.shape[1] != size:
        raise ValueError(f"target must have shape (n, {size}), got {target.shape}")
    # Row and word indices are stored as uint32, and 0xFFFFFFFF marks a missing vector.
    if target.shape[0] >= NO_VECTOR or word_vectors.shape[0] >= NO_VECTOR:
        raise ValueError(f"the kernel supports fewer than {NO_VECTOR} rows and words")
    if memory[0].size < size:
        raise ValueError(f"memory[0] must hold at least {size} entries, got {memory[0].size}")


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

    _check_inputs(model, target, memory)

    if not model.is_ft:
        init_base_s2v_config(&w2v, model, target, memory)

        eff_sentences, eff_words = populate_base_s2v_config(&w2v, model.wv, indexed_sentences, target.shape[0])

        with nogil:
            compute_base_sentence_averages(&w2v, eff_sentences)
    else:
        init_ft_s2v_config(&ft, model, target, memory)

        eff_sentences, eff_words = populate_ft_s2v_config(&ft, model.wv, indexed_sentences, target.shape[0])

        with nogil:
            compute_ft_sentence_averages(&ft, eff_sentences)

    return eff_sentences, eff_words

def init():
    return 1

MAX_WORDS_IN_BATCH = MAX_WORDS
MAX_NGRAMS_IN_BATCH = MAX_NGRAMS
FAST_VERSION = init()
