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

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.object cimport PyObject
from cpython.pycapsule cimport PyCapsule_GetPointer
from cpython.pythread cimport (
    WAIT_LOCK,
    PyThread_acquire_lock,
    PyThread_allocate_lock,
    PyThread_free_lock,
    PyThread_release_lock,
)
from libc.stdlib cimport free, malloc, realloc
from libc.string cimport memcpy, memset

from time import time

import scipy.linalg.blas as fblas

cdef extern from "Python.h":
    # Raises (TypeError, UnicodeEncodeError) instead of returning NULL silently.
    const char *PyUnicode_AsUTF8AndSize(object unicode, Py_ssize_t *size) except NULL
    # NULL with an exception set on errors in __hash__ or __eq__, NULL without one if the
    # key is missing. Unlike PyDict_GetItem it does not swallow the errors.
    PyObject *PyDict_GetItemWithError(object d, object key) except? NULL

# y += alpha * x, taken from scipy's BLAS. Raises at import if scipy exposes no pointer.
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCapsule_GetPointer(fblas.saxpy._cpointer, NULL)

cdef int ONE = <int>1
cdef int ZERO = <int>0

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


# A token of a FastText job that is not in the vocabulary. Its UTF-8 bytes "<token>" are
# stored with the job and hashed into ngram buckets by the worker, without the GIL.
cdef uINT_t OOV_TOKEN = <uINT_t>0xFFFFFFFE


cdef class KernelContext:
    """Read-only state of one train or infer call, shared by all workers.

    Built once per call instead of once per job. It keeps references to every array it
    holds a raw pointer into, so they stay alive while workers use the pointers.
    """

    cdef KernelParams params
    cdef dict vocab
    cdef Py_ssize_t n_rows
    cdef object _keep
    cdef object _locks

    def __init__(self, model, target):
        _check_inputs(model, target)
        wv = model.wv
        self.vocab = wv.key_to_index
        self.n_rows = target.shape[0]
        self.params.size = model.sv.vector_size
        self.params.is_ft = model.is_ft
        self.params.word_weights = <REAL_t *>(np.PyArray_DATA(model.word_weights))
        self.params.target = <REAL_t *>(np.PyArray_DATA(target))
        self.params.locks = _row_locks_of(model)
        if model.is_ft:
            self.params.word_vectors = <REAL_t *>(np.PyArray_DATA(wv.vectors_vocab))
            self.params.ngram_vectors = <REAL_t *>(np.PyArray_DATA(wv.vectors_ngrams))
            self.params.min_n = wv.min_n
            self.params.max_n = wv.max_n
            self.params.bucket = wv.bucket
            self.params.max_ngrams = model.batch_ngrams
            # Unknown words are weighted like the most heavily weighted known word.
            self.params.oov_weight = <REAL_t>np.max(model.word_weights)
            self._keep = (model.word_weights, target, wv.vectors_vocab, wv.vectors_ngrams)
        else:
            self.params.word_vectors = <REAL_t *>(np.PyArray_DATA(wv.vectors))
            self.params.ngram_vectors = NULL
            self.params.max_ngrams = 0
            self._keep = (model.word_weights, target, wv.vectors)
        self._locks = getattr(model, "_row_locks", None)

    def __reduce__(self):
        raise TypeError("KernelContext only exists during a train or infer call")


cdef class EncodedJob:
    """A batch of sentences encoded for the kernel.

    ``words[offsets[s]:offsets[s + 1]]`` are the tokens of sentence ``s``, which is added
    into row ``targets[s]``: vocabulary indices, or ``OOV_TOKEN`` for an unknown FastText
    word whose bytes are ``oov_bytes[oov_offsets[k]:oov_offsets[k + 1]]``, in order.
    ``n_items`` counts every input item of the job, including empty sentences.
    """

    cdef readonly Py_ssize_t n_items
    cdef readonly Py_ssize_t n_sentences
    cdef readonly Py_ssize_t eff_words
    cdef object words
    cdef object offsets
    cdef object targets
    cdef bytes oov_bytes
    cdef object oov_offsets


cdef struct _Buffer:
    char *data
    Py_ssize_t size  # bytes used
    Py_ssize_t capacity  # bytes allocated


cdef int _reserve(_Buffer *b, Py_ssize_t extra) except -1:
    """Grow b so that extra more bytes fit, doubling the capacity."""
    cdef Py_ssize_t needed = b.size + extra
    cdef Py_ssize_t capacity
    cdef char *data
    if needed <= b.capacity:
        return 0
    capacity = max(needed, 2 * b.capacity, 4096)
    data = <char *>realloc(b.data, capacity)
    if data == NULL:
        raise MemoryError("could not grow the job buffer")
    b.data = data
    b.capacity = capacity
    return 0


cdef inline int _push_u32(_Buffer *b, uINT_t value) except -1:
    _reserve(b, sizeof(uINT_t))
    (<uINT_t *>(b.data + b.size))[0] = value
    b.size += sizeof(uINT_t)
    return 0


cdef inline int _push_i64(_Buffer *b, np.int64_t value) except -1:
    _reserve(b, sizeof(np.int64_t))
    (<np.int64_t *>(b.data + b.size))[0] = value
    b.size += sizeof(np.int64_t)
    return 0


cdef inline int _push_bytes(_Buffer *b, const char *data, Py_ssize_t n) except -1:
    _reserve(b, n)
    memcpy(b.data + b.size, data, n)
    b.size += n
    return 0


cdef object _to_array(_Buffer *b, dtype):
    """Copy the used part of b into a new numpy array of dtype."""
    out = np.empty(b.size // np.dtype(dtype).itemsize, dtype=dtype)
    if b.size:
        memcpy(np.PyArray_DATA(out), b.data, b.size)
    return out


cdef class JobEncoder:
    """Read (tokens, index) tuples from an iterable and encode them into jobs.

    Batching and encoding happen in one pass, holding the GIL: a job takes items while
    it has fewer than ``batch_words`` items and its words stay within ``batch_words``; a
    longer item starts a job of its own. Empty sentences are skipped, as is every
    unknown word of a Word2Vec-style model. The number of non-empty sentences and of
    used tokens are the counts train reports.
    """

    cdef KernelContext ctx
    cdef object iterator
    cdef Py_ssize_t batch_words
    cdef object pending
    cdef bint has_pending
    cdef _Buffer words, offsets, targets, oov_bytes, oov_offsets

    def __cinit__(self):
        memset(&self.words, 0, sizeof(_Buffer))
        memset(&self.offsets, 0, sizeof(_Buffer))
        memset(&self.targets, 0, sizeof(_Buffer))
        memset(&self.oov_bytes, 0, sizeof(_Buffer))
        memset(&self.oov_offsets, 0, sizeof(_Buffer))

    def __init__(self, KernelContext ctx, iterable, Py_ssize_t batch_words):
        if batch_words < 1:
            raise ValueError(f"batch_words must be positive, got {batch_words}")
        self.ctx = ctx
        self.iterator = iter(iterable)
        self.batch_words = batch_words
        self.has_pending = False

    def __dealloc__(self):
        free(self.words.data)
        free(self.offsets.data)
        free(self.targets.data)
        free(self.oov_bytes.data)
        free(self.oov_offsets.data)

    def __reduce__(self):
        raise TypeError("JobEncoder only exists during a train or infer call")

    def next_job(self):
        """Return the next EncodedJob, or None once the input is exhausted."""
        cdef dict vocab = self.ctx.vocab
        cdef bint is_ft = self.ctx.params.is_ft
        cdef Py_ssize_t n_rows = self.ctx.n_rows
        cdef Py_ssize_t n_items = 0
        cdef Py_ssize_t n_words = 0
        cdef Py_ssize_t n_sentences = 0
        cdef Py_ssize_t length
        cdef Py_ssize_t utf8_size
        cdef const char *utf8
        cdef PyObject *hit

        self.words.size = 0
        self.offsets.size = 0
        self.targets.size = 0
        self.oov_bytes.size = 0
        self.oov_offsets.size = 0
        _push_i64(&self.offsets, 0)
        _push_i64(&self.oov_offsets, 0)

        while True:
            if self.has_pending:
                obj = self.pending
                self.pending = None
                self.has_pending = False
            else:
                try:
                    obj = next(self.iterator)
                except StopIteration:
                    break
            sentence = obj[0]
            length = len(sentence)
            if n_items and (n_words + length > self.batch_words or n_items >= self.batch_words):
                self.pending = obj
                self.has_pending = True
                break
            n_items += 1
            n_words += length

            index = obj[1]
            # The kernel writes into target[index] through a raw pointer.
            if not 0 <= index < n_rows:
                raise IndexError(f"sentence index {index} is out of range for {n_rows} target rows")
            if not length:
                continue
            for token in sentence:
                hit = PyDict_GetItemWithError(vocab, token)
                if hit != NULL:
                    _push_u32(&self.words, <uINT_t>(<object>hit))
                elif is_ft:
                    # The UTF-8 bytes of "<token>", as fastText hashes them.
                    utf8 = PyUnicode_AsUTF8AndSize(token, &utf8_size)
                    _push_bytes(&self.oov_bytes, "<", 1)
                    _push_bytes(&self.oov_bytes, utf8, utf8_size)
                    _push_bytes(&self.oov_bytes, ">", 1)
                    _push_i64(&self.oov_offsets, self.oov_bytes.size)
                    _push_u32(&self.words, OOV_TOKEN)
            _push_u32(&self.targets, <uINT_t>index)
            n_sentences += 1
            _push_i64(&self.offsets, self.words.size // sizeof(uINT_t))

        if not n_items:
            return None
        cdef EncodedJob job = EncodedJob.__new__(EncodedJob)
        job.n_items = n_items
        job.n_sentences = n_sentences
        job.eff_words = self.words.size // sizeof(uINT_t)
        job.words = _to_array(&self.words, np.uint32)
        job.offsets = _to_array(&self.offsets, np.int64)
        job.targets = _to_array(&self.targets, np.uint32)
        job.oov_bytes = PyBytes_FromStringAndSize(self.oov_bytes.data, self.oov_bytes.size)
        job.oov_offsets = _to_array(&self.oov_offsets, np.int64)
        return job


def encode_job(KernelContext ctx, batch):
    """Encode all (tokens, index) tuples of batch into one EncodedJob."""
    items = list(batch)
    job = JobEncoder(ctx, items, max(len(items), 1) + sum(len(obj[0]) for obj in items)).next_job()
    if job is None:
        job = EncodedJob.__new__(EncodedJob)
        job.words = np.empty(0, dtype=np.uint32)
        job.offsets = np.zeros(1, dtype=np.int64)
        job.targets = np.empty(0, dtype=np.uint32)
        job.oov_bytes = b""
        job.oov_offsets = np.zeros(1, dtype=np.int64)
    return job


cdef void _average_job(
    const KernelParams *p,
    const uINT_t *words,
    const np.int64_t *offsets,
    const uINT_t *targets,
    Py_ssize_t n_sentences,
    const unsigned char *oov_bytes,
    const np.int64_t *oov_offsets,
    REAL_t *mem,
    uINT_t *ngram_buffer,
    size_t ngram_capacity,
) noexcept nogil:
    """Add the average of every sentence of a job into its target row.

    The weighted word vectors of a sentence are summed into mem, then mem / length is
    added into the row, as in the numpy fallback. For an unknown FastText word, each
    ngram vector is added scaled by oov_weight / ngrams, where numpy sums the ngram
    vectors first and scales the sum: the same value within float32 rounding.
    """
    cdef int size = p.size
    cdef Py_ssize_t s, k = 0
    cdef np.int64_t t, sent_len
    cdef size_t j, ngrams, row
    cdef uINT_t word
    cdef REAL_t inv_count, inv_ngram
    cdef PyThread_type_lock lock

    for s in range(n_sentences):
        memset(mem, 0, size * cython.sizeof(REAL_t))
        # Every stored token counts for the length, also unknown words without ngrams.
        sent_len = offsets[s + 1] - offsets[s]
        for t in range(offsets[s], offsets[s + 1]):
            word = words[t]
            if word == OOV_TOKEN:
                ngrams = ft_ngram_hashes_into(
                    &oov_bytes[oov_offsets[k]],
                    <size_t>(oov_offsets[k + 1] - oov_offsets[k]),
                    p.min_n, p.max_n, p.bucket, ngram_buffer, ngram_capacity,
                )
                k += 1
                if ngrams == 0:
                    # No ngrams: the word counts for the length but adds no vector.
                    continue
                inv_ngram = (ONEF / <REAL_t>ngrams) * p.oov_weight
                for j in range(ngrams):
                    row = <size_t>ngram_buffer[j] * <size_t>size
                    saxpy(&size, &inv_ngram, &p.ngram_vectors[row], &ONE, mem, &ONE)
            else:
                row = <size_t>word * <size_t>size
                saxpy(&size, &p.word_weights[word], &p.word_vectors[row], &ONE, mem, &ONE)

        if sent_len > 0:
            inv_count = ONEF / <REAL_t>sent_len
            row = <size_t>targets[s] * <size_t>size
            # Adding mem / length keeps many-to-one mappings correct: rows accumulate.
            if p.locks != NULL:
                lock = p.locks[targets[s] % N_ROW_LOCKS]
                PyThread_acquire_lock(lock, WAIT_LOCK)
                saxpy(&size, &inv_count, mem, &ONE, &p.target[row], &ONE)
                PyThread_release_lock(lock)
            else:
                saxpy(&size, &inv_count, mem, &ONE, &p.target[row], &ONE)


def run_job(KernelContext ctx, EncodedJob job, scratch):
    """Average an encoded job into the target of ctx without holding the GIL.

    scratch is the worker's private float32 buffer of at least the vector size. Returns
    the number of sentences and of tokens the job used.
    """
    _check_array(scratch, "scratch", np.float32)
    if scratch.size < ctx.params.size:
        raise ValueError(f"scratch must hold at least {ctx.params.size} entries")
    cdef uINT_t[::1] words = job.words
    cdef np.int64_t[::1] offsets = job.offsets
    cdef uINT_t[::1] targets = job.targets
    cdef np.int64_t[::1] oov_offsets = job.oov_offsets
    cdef const unsigned char *oov_bytes = job.oov_bytes
    cdef REAL_t *mem = <REAL_t *>(np.PyArray_DATA(scratch))
    cdef uINT_t *ngram_buffer = NULL
    cdef Py_ssize_t n_sentences = job.n_sentences
    # Typed memoryviews of length 0 have no element to take the address of.
    cdef const uINT_t *words_ptr = &words[0] if words.shape[0] else NULL
    cdef const uINT_t *targets_ptr = &targets[0] if targets.shape[0] else NULL

    # A word of b UTF-8 bytes has fewer than b * max_n ngrams, so the buffer needs at most
    # that many entries for the longest unknown word, even if batch_ngrams is larger.
    cdef Py_ssize_t longest = 0
    cdef Py_ssize_t i
    for i in range(oov_offsets.shape[0] - 1):
        longest = max(longest, oov_offsets[i + 1] - oov_offsets[i])
    capacity = min(int(ctx.params.max_ngrams), int(longest) * max(int(ctx.params.max_n), 1))
    cdef size_t ngram_capacity = capacity
    if ngram_capacity:
        ngram_buffer = <uINT_t *>malloc(ngram_capacity * sizeof(uINT_t))
        if ngram_buffer == NULL:
            raise MemoryError("could not allocate the ngram buffer")
    try:
        with nogil:
            _average_job(
                &ctx.params, words_ptr, &offsets[0], targets_ptr, n_sentences,
                oov_bytes, &oov_offsets[0], mem, ngram_buffer, ngram_capacity,
            )
    finally:
        free(ngram_buffer)
    return job.n_sentences, job.eff_words


def scan_indexed_sentences(sentences, double progress_per, progress):
    """Validate every (tokens, index) tuple and count them, like the Python scan loop.

    The checks and their messages are those of
    ``BaseSentence2VecModel._check_indexed_sent_valid``. ``progress(sentences, words)`` is
    called about every ``progress_per`` seconds. Returns the number of sentences and
    words, the number of empty sentences, the largest index and whether every index
    occurs at most once.
    """
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t total_words = 0
    cdef Py_ssize_t empty_sentences = 0
    cdef object max_index = 0
    cdef bint unique_indices = True
    cdef bint identity = True
    cdef set seen = set()
    cdef double last = time()

    for obj in sentences:
        if not isinstance(obj, tuple):
            raise TypeError(f"Passed {type(obj)}: {obj}. Iterable must contain tuple.")
        if len(<tuple>obj) != 2:
            raise ValueError(
                f"At {i}: Passed a tuple of length {len(obj)}. "
                "Each item must be a (words, index) tuple."
            )
        words = (<tuple>obj)[0]
        index = (<tuple>obj)[1]
        if not isinstance(words, list):
            raise TypeError(
                f"At {i}: Passed {type(words)}: {words}. tuple.words must contain list of str."
            )
        # Subclasses of list may override __iter__, which the Python check uses.
        for word in (<list>words if type(words) is list else words):
            if not isinstance(word, str):
                raise TypeError(
                    f"At {i}: Passed {type(words)}: {words}. tuple.words must contain list of str."
                )
        if type(index) is not int:
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
                raise TypeError(
                    f"At {i}: Passed {type(index)}: {index}. tuple.index must contain index"
                )
            index = int(index)
        if index < 0:
            raise ValueError(f"At {i}: Passed negative {index}")

        if (i & 4095) == 0 and time() - last > progress_per:
            last = time()
            progress(i, total_words)

        if index > max_index:
            max_index = index
        if identity and index != i:
            identity = False
            seen = set(range(i))
        if not identity and unique_indices:
            if index in seen:
                unique_indices = False
                seen.clear()
            else:
                seen.add(index)
        total_words += len(words)
        if not len(words):
            empty_sentences += 1
        i += 1

    return i, total_words, empty_sentences, max_index, unique_indices


def _check_array(array, name, dtype):
    """Raise unless array is a C-contiguous ndarray of dtype, because the kernel takes raw pointers."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {np.dtype(dtype).name}, got {array.dtype}")
    if not array.flags.c_contiguous:
        raise TypeError(f"{name} must be C-contiguous")


def _check_inputs(model, target, memory=None):
    """Validate every array the kernel reads or writes through a raw pointer."""
    _check_array(target, "target", np.float32)
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
        if model.batch_ngrams < 1:
            raise ValueError(f"batch_ngrams must be positive, got {model.batch_ngrams}")
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
    # Row and word indices are stored as uint32; the two largest values are markers.
    if target.shape[0] >= OOV_TOKEN or word_vectors.shape[0] >= OOV_TOKEN:
        raise ValueError(f"the kernel supports fewer than {OOV_TOKEN} rows and words")
    if memory is not None:
        _check_array(memory[0], "memory[0]", np.float32)
        if memory[0].size < size:
            raise ValueError(f"memory[0] must hold at least {size} entries, got {memory[0].size}")


def train_average_cy(model, indexed_sentences, target, memory):
    """Training on a sequence of sentences and update the target ndarray.

    Called internally from :meth:`~fse.models.average.Average._do_train_job`. Encodes the
    sentences and averages them like the training workers do.

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
        Private memory for each working thread; ``memory[0]`` is the float32 buffer
        the sentence sums are built in.

    Returns
    -------
    int, int
        Number of effective sentences (non-zero) and effective words in the vocabulary used
        during training the sentence embedding.
    """
    _check_inputs(model, target, memory)
    ctx = KernelContext(model, target)
    return run_job(ctx, encode_job(ctx, indexed_sentences), memory[0])


def init():
    return 1

MAX_WORDS_IN_BATCH = MAX_WORDS
MAX_NGRAMS_IN_BATCH = MAX_NGRAMS
FAST_VERSION = init()
