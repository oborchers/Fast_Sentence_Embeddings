#!/usr/bin/env cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Parts adapted from gensim (LGPL 2.1, used under the GPL 3); see NOTICE.

"""Optimized cython functions for computing sentence embeddings"""

import cython
import numpy as np

cimport numpy as np

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.pyport cimport PY_SSIZE_T_MAX
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
from libc.string cimport memchr, memcmp, memcpy, memset

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
    cdef Py_ssize_t needed
    cdef Py_ssize_t capacity
    cdef char *data
    if extra > PY_SSIZE_T_MAX - b.size:
        raise MemoryError("the job buffer would exceed the address space")
    needed = b.size + extra
    if needed <= b.capacity:
        return 0
    capacity = b.capacity * 2 if b.capacity <= PY_SSIZE_T_MAX // 2 else PY_SSIZE_T_MAX
    capacity = max(needed, capacity, 4096)
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
    cdef bint failed
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
        self.failed = False

    def __dealloc__(self):
        free(self.words.data)
        free(self.offsets.data)
        free(self.targets.data)
        free(self.oov_bytes.data)
        free(self.oov_offsets.data)

    def __reduce__(self):
        raise TypeError("JobEncoder only exists during a train or infer call")

    def next_job(self):
        """Return the next EncodedJob, or None once the input is exhausted.

        After an exception the encoder refuses further calls: the items it had already
        read for the failed job are lost, so continuing would silently skip them.
        """
        if self.failed:
            raise RuntimeError("the JobEncoder failed before and cannot continue")
        try:
            return self._next_job()
        except BaseException:
            self.failed = True
            raise

    cdef object _next_job(self):
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
            # Compared without n_words + length, which could overflow for a huge length.
            if n_items and (
                n_items >= self.batch_words
                or n_words > self.batch_words
                or length > self.batch_words - n_words
            ):
                self.pending = obj
                self.has_pending = True
                break
            n_items += 1
            n_words = n_words + length if length <= PY_SSIZE_T_MAX - n_words else PY_SSIZE_T_MAX

            index = obj[1]
            # The kernel writes into target[index] through a raw pointer.
            if not 0 <= index < n_rows:
                raise IndexError(f"sentence index {index} is out of range for {n_rows} target rows")
            # Truthiness, as in the numpy fallback.
            if not sentence:
                continue
            for token in sentence:
                hit = PyDict_GetItemWithError(vocab, token)
                if hit != NULL:
                    _push_u32(&self.words, <uINT_t>(<object>hit))
                elif is_ft:
                    # The UTF-8 bytes of "<token>", as fastText hashes them. A subclass of
                    # str is formatted like gensim does, which may call its __str__.
                    if type(token) is str:
                        utf8 = PyUnicode_AsUTF8AndSize(token, &utf8_size)
                        _push_bytes(&self.oov_bytes, "<", 1)
                        _push_bytes(&self.oov_bytes, utf8, utf8_size)
                        _push_bytes(&self.oov_bytes, ">", 1)
                    else:
                        formatted = ("<%s>" % token).encode("utf-8")
                        _push_bytes(&self.oov_bytes, formatted, len(formatted))
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


cdef void _average_sentence(
    const KernelParams *p,
    const uINT_t *words,
    Py_ssize_t n_tokens,
    const unsigned char *oov_bytes,
    const np.int64_t *oov_offsets,
    Py_ssize_t *k,
    uINT_t target,
    REAL_t *mem,
    uINT_t *ngram_buffer,
    size_t ngram_capacity,
) noexcept nogil:
    """Add the average of one sentence into row target.

    The weighted word vectors of the sentence are summed into mem, then mem / length is
    added into the row, as in the numpy fallback. For an unknown FastText word, each
    ngram vector is added scaled by oov_weight / ngrams, where numpy sums the ngram
    vectors first and scales the sum: the same value within float32 rounding. k is the
    index of the next unknown word in oov_offsets and is advanced past this sentence.
    """
    cdef int size = p.size
    cdef Py_ssize_t t
    cdef size_t j, ngrams, row
    cdef uINT_t word
    cdef REAL_t inv_count, inv_ngram
    cdef PyThread_type_lock lock

    memset(mem, 0, size * cython.sizeof(REAL_t))
    for t in range(n_tokens):
        word = words[t]
        if word == OOV_TOKEN:
            ngrams = ft_ngram_hashes_into(
                &oov_bytes[oov_offsets[k[0]]],
                <size_t>(oov_offsets[k[0] + 1] - oov_offsets[k[0]]),
                p.min_n, p.max_n, p.bucket, ngram_buffer, ngram_capacity,
            )
            k[0] += 1
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

    # Every stored token counts for the length, also unknown words without ngrams.
    if n_tokens > 0:
        inv_count = ONEF / <REAL_t>n_tokens
        row = <size_t>target * <size_t>size
        # Adding mem / length keeps many-to-one mappings correct: rows accumulate.
        if p.locks != NULL:
            lock = p.locks[target % N_ROW_LOCKS]
            PyThread_acquire_lock(lock, WAIT_LOCK)
            saxpy(&size, &inv_count, mem, &ONE, &p.target[row], &ONE)
            PyThread_release_lock(lock)
        else:
            saxpy(&size, &inv_count, mem, &ONE, &p.target[row], &ONE)


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
    """Add the average of every sentence of an encoded job into its target row."""
    cdef Py_ssize_t s, k = 0
    for s in range(n_sentences):
        _average_sentence(
            p, &words[offsets[s]] if words != NULL else NULL, offsets[s + 1] - offsets[s],
            oov_bytes, oov_offsets, &k, targets[s], mem, ngram_buffer, ngram_capacity,
        )


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
    if capacity > PY_SSIZE_T_MAX // sizeof(uINT_t):
        raise OverflowError(f"cannot allocate an ngram buffer for {capacity} ngrams")
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


# ---------------------------------------------------------------------------------------
# Native text input (CorpusFile, TextList): tokenize UTF-8 bytes without the GIL.
#
# A sentence is a span of bytes: a line of a file without its "\n", or one string of a
# TextList. It is decoded strictly as UTF-8 and split like str.split(): at every code
# point for which str.isspace() is true. Tests compare the result with Python.
# ---------------------------------------------------------------------------------------

cdef uINT_t NOT_FOUND = <uINT_t>0xFFFFFFFF

# Return codes of the native routines, translated into exceptions with the GIL.
cdef enum:
    NATIVE_OK = 0
    NATIVE_NO_MEMORY = -1
    NATIVE_BAD_UTF8 = -2
    NATIVE_BAD_ROW = -3


cdef inline int _utf8_len(const unsigned char *p, Py_ssize_t i, Py_ssize_t end) noexcept nogil:
    """Length of the valid UTF-8 sequence at p[i], or 0 where CPython's decoder fails."""
    cdef unsigned char b = p[i]
    cdef unsigned char c
    if b < 0x80:
        return 1
    if b < 0xC2:
        return 0
    if b < 0xE0:
        if i + 1 < end and (p[i + 1] & 0xC0) == 0x80:
            return 2
        return 0
    if b < 0xF0:
        if i + 2 >= end:
            return 0
        c = p[i + 1]
        if (b == 0xE0 and c < 0xA0) or (b == 0xED and c > 0x9F):
            return 0
        if (c & 0xC0) != 0x80 or (p[i + 2] & 0xC0) != 0x80:
            return 0
        return 3
    if b < 0xF5:
        if i + 3 >= end:
            return 0
        c = p[i + 1]
        if (b == 0xF0 and c < 0x90) or (b == 0xF4 and c > 0x8F):
            return 0
        if (c & 0xC0) != 0x80 or (p[i + 2] & 0xC0) != 0x80 or (p[i + 3] & 0xC0) != 0x80:
            return 0
        return 4
    return 0


cdef inline bint _is_space(const unsigned char *p, Py_ssize_t i, int n) noexcept nogil:
    """Whether the valid UTF-8 sequence of length n at p[i] satisfies str.isspace()."""
    cdef unsigned char b = p[i]
    cdef unsigned char c, d
    if n == 1:
        return (9 <= b <= 13) or (28 <= b <= 32)
    if n == 2:
        return b == 0xC2 and (p[i + 1] == 0x85 or p[i + 1] == 0xA0)
    if n == 3:
        c = p[i + 1]
        d = p[i + 2]
        if b == 0xE1:
            return c == 0x9A and d == 0x80  # U+1680
        if b == 0xE2:
            if c == 0x80:  # U+2000 to U+200A, U+2028, U+2029, U+202F
                return d <= 0x8A or d == 0xA8 or d == 0xA9 or d == 0xAF
            return c == 0x81 and d == 0x9F  # U+205F
        if b == 0xE3:
            return c == 0x80 and d == 0x80  # U+3000
    return False


cdef inline np.uint64_t _fnv1a64(const unsigned char *p, Py_ssize_t n) noexcept nogil:
    cdef np.uint64_t h = 14695981039346656037ULL
    cdef Py_ssize_t i
    for i in range(n):
        h ^= p[i]
        h *= 1099511628211ULL
    return h


cdef struct VocabView:
    const unsigned char *keys
    const np.int64_t *key_offsets
    const np.uint32_t *key_lengths
    const uINT_t *values
    const uINT_t *slots
    const np.uint64_t *slot_hashes
    np.uint64_t mask


cdef inline uINT_t _vocab_lookup(const VocabView *v, const unsigned char *p, Py_ssize_t n) noexcept nogil:
    """Vocabulary index of the n bytes at p, or NOT_FOUND."""
    cdef np.uint64_t h = _fnv1a64(p, n)
    cdef np.uint64_t j = h & v.mask
    cdef uINT_t k
    while True:
        k = v.slots[j]
        if k == NOT_FOUND:
            return NOT_FOUND
        if (
            v.slot_hashes[j] == h
            and v.key_lengths[k] == n
            and memcmp(v.keys + v.key_offsets[k], p, n) == 0
        ):
            return v.values[k]
        j = (j + 1) & v.mask


cdef class VocabTable:
    """Immutable map from the UTF-8 bytes of a word to its vocabulary index.

    Built once per train or infer call from key_to_index while holding the GIL, then
    read without it: open addressing with linear probing, at most half full.
    """

    cdef bytes arena
    cdef object _arrays
    cdef VocabView view

    def __init__(self, dict key_to_index):
        cdef Py_ssize_t n = len(key_to_index)
        offsets_arr = np.empty(max(n, 1), dtype=np.int64)
        lengths_arr = np.empty(max(n, 1), dtype=np.uint32)
        values_arr = np.empty(max(n, 1), dtype=np.uint32)
        cdef np.int64_t[::1] offsets = offsets_arr
        cdef np.uint32_t[::1] lengths = lengths_arr
        cdef uINT_t[::1] values = values_arr
        cdef list parts = []
        cdef np.int64_t position = 0
        cdef Py_ssize_t k = 0
        for word, index in key_to_index.items():
            encoded = (<str?>word).encode("utf-8")
            parts.append(encoded)
            offsets[k] = position
            lengths[k] = len(encoded)
            values[k] = <uINT_t>index
            position += len(encoded)
            k += 1
        self.arena = b"".join(parts)

        cdef np.uint64_t size = 2
        while size < 2 * <np.uint64_t>n:
            size <<= 1
        self.view.mask = size - 1
        slots_arr = np.full(size, NOT_FOUND, dtype=np.uint32)
        hashes_arr = np.zeros(size, dtype=np.uint64)
        cdef uINT_t[::1] slots = slots_arr
        cdef np.uint64_t[::1] hashes = hashes_arr
        self._arrays = (offsets_arr, lengths_arr, values_arr, slots_arr, hashes_arr)
        self.view.keys = self.arena
        self.view.key_offsets = &offsets[0]
        self.view.key_lengths = &lengths[0]
        self.view.values = &values[0]
        self.view.slots = &slots[0]
        self.view.slot_hashes = &hashes[0]

        cdef np.uint64_t h, j
        for k in range(n):
            h = _fnv1a64(self.view.keys + offsets[k], lengths[k])
            j = h & self.view.mask
            while slots[j] != NOT_FOUND:
                j = (j + 1) & self.view.mask
            slots[j] = <uINT_t>k
            hashes[j] = h

    def __reduce__(self):
        raise TypeError("VocabTable only exists during a train or infer call")

    def get(self, word):
        """Vocabulary index of word (str), or None; for tests."""
        encoded = (<str?>word).encode("utf-8")
        cdef uINT_t index = _vocab_lookup(&self.view, encoded, len(encoded))
        return None if index == NOT_FOUND else index


cdef Py_ssize_t _count_tokens(
    const unsigned char *p, Py_ssize_t start, Py_ssize_t end, Py_ssize_t *error
) noexcept nogil:
    """Number of str.split() tokens in p[start:end]; sets error to the first bad byte."""
    cdef Py_ssize_t i = start
    cdef Py_ssize_t tokens = 0
    cdef bint inside = False
    cdef int m
    while i < end:
        m = _utf8_len(p, i, end)
        if m == 0:
            error[0] = i
            return tokens
        if _is_space(p, i, m):
            inside = False
        elif not inside:
            inside = True
            tokens += 1
        i += m
    return tokens


def scan_spans(const unsigned char[::1] data, offsets, Py_ssize_t first, Py_ssize_t last,
               Py_ssize_t chunk_bytes):
    """Validate and count the sentences of one scan range, without the GIL.

    Line mode (offsets is None): the range holds the lines that start at byte positions
    in [first, last); the caller aligns first and last to line starts. Item mode: the
    range holds the strings first to last - 1 of a TextList, delimited by offsets.

    Returns (sentences, tokens, empty, chunks, error_byte, error_sentence): chunks is a
    list of (start, end, first_sentence) with positions in the same unit as the range and
    sentence numbers relative to the range; error_byte is -1 if the range is valid.
    """
    cdef Py_ssize_t n = data.shape[0]
    cdef const unsigned char *p = &data[0] if n else NULL
    cdef bint lines = offsets is None
    cdef np.int64_t[::1] item_offsets
    cdef const np.int64_t *item_ptr = NULL
    if not lines:
        item_offsets = offsets
        item_ptr = &item_offsets[0]
    cdef Py_ssize_t sentences = 0, tokens = 0, empty = 0, count
    cdef Py_ssize_t error = -1
    cdef Py_ssize_t pos = first, span_end, next_pos, chunk_start = first, chunk_first = 0
    cdef const unsigned char *newline
    chunk_list = []
    cdef np.int64_t[::1] chunk_buffer
    cdef Py_ssize_t n_chunks = 0

    # Chunk boundaries are collected in C and converted to a list at the end. Every chunk
    # but the last holds at least chunk_bytes bytes, which bounds their number.
    if chunk_bytes < 1:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
    cdef Py_ssize_t span_bytes = (last - first) if lines else (item_ptr[last] - item_ptr[first])
    cdef Py_ssize_t max_chunks = span_bytes // chunk_bytes + 2
    chunk_arr = np.empty(3 * max_chunks, dtype=np.int64)
    chunk_buffer = chunk_arr
    with nogil:
        while pos < last:
            if lines:
                newline = <const unsigned char *>memchr(p + pos, 10, n - pos)
                span_end = (newline - p) if newline != NULL else n
                next_pos = span_end + 1 if newline != NULL else n
            else:
                span_end = item_ptr[pos + 1]
                next_pos = pos + 1
            count = _count_tokens(p, item_ptr[pos] if not lines else pos, span_end, &error)
            if error >= 0:
                break
            tokens += count
            if count == 0:
                empty += 1
            sentences += 1
            pos = next_pos
            if (
                (lines and pos - chunk_start >= chunk_bytes)
                or (not lines and (item_ptr[pos] - item_ptr[chunk_start]) >= chunk_bytes)
            ) and pos < last and n_chunks + 1 < max_chunks:
                chunk_buffer[3 * n_chunks] = chunk_start
                chunk_buffer[3 * n_chunks + 1] = pos
                chunk_buffer[3 * n_chunks + 2] = chunk_first
                n_chunks += 1
                chunk_start = pos
                chunk_first = sentences
    if error >= 0:
        return sentences, tokens, empty, [], error, sentences
    if chunk_start < last:
        chunk_buffer[3 * n_chunks] = chunk_start
        chunk_buffer[3 * n_chunks + 1] = last
        chunk_buffer[3 * n_chunks + 2] = chunk_first
        n_chunks += 1
    chunk_list = [tuple(chunk_arr[3 * c : 3 * c + 3].tolist()) for c in range(n_chunks)]
    return sentences, tokens, empty, chunk_list, -1, -1


cdef int _push_u32_nogil(_Buffer *b, uINT_t value) noexcept nogil:
    if b.size + <Py_ssize_t>sizeof(uINT_t) > b.capacity:
        if _grow_nogil(b, sizeof(uINT_t)) != 0:
            return -1
    (<uINT_t *>(b.data + b.size))[0] = value
    b.size += sizeof(uINT_t)
    return 0


cdef int _push_i64_nogil(_Buffer *b, np.int64_t value) noexcept nogil:
    if b.size + <Py_ssize_t>sizeof(np.int64_t) > b.capacity:
        if _grow_nogil(b, sizeof(np.int64_t)) != 0:
            return -1
    (<np.int64_t *>(b.data + b.size))[0] = value
    b.size += sizeof(np.int64_t)
    return 0


cdef int _push_bytes_nogil(_Buffer *b, const unsigned char *data, Py_ssize_t n) noexcept nogil:
    if b.size + n > b.capacity:
        if _grow_nogil(b, n) != 0:
            return -1
    memcpy(b.data + b.size, data, n)
    b.size += n
    return 0


cdef int _grow_nogil(_Buffer *b, Py_ssize_t extra) noexcept nogil:
    """Like _reserve, without the GIL: returns -1 instead of raising."""
    cdef Py_ssize_t capacity
    cdef char *data
    if extra > PY_SSIZE_T_MAX - b.size:
        return -1
    capacity = b.capacity * 2 if b.capacity <= PY_SSIZE_T_MAX // 2 else PY_SSIZE_T_MAX
    capacity = max(b.size + extra, capacity, 4096)
    data = <char *>realloc(b.data, capacity)
    if data == NULL:
        return -1
    b.data = data
    b.capacity = capacity
    return 0


cdef int _average_spans(
    const KernelParams *p,
    const VocabView *table,
    const unsigned char *data,
    Py_ssize_t data_size,
    const np.int64_t *item_offsets,
    Py_ssize_t first,
    Py_ssize_t last,
    np.int64_t first_row,
    Py_ssize_t n_rows,
    REAL_t *mem,
    _Buffer *words,
    _Buffer *oov_bytes,
    _Buffer *oov_offsets,
    _Buffer *ngrams,
    Py_ssize_t *eff_sentences,
    Py_ssize_t *eff_words,
    Py_ssize_t *error,
) noexcept nogil:
    """Tokenize, look up and average every sentence of a chunk without the GIL."""
    cdef bint lines = item_offsets == NULL
    cdef Py_ssize_t pos = first, span_start, span_end, next_pos, i, token_start, k
    cdef Py_ssize_t longest, capacity
    cdef np.int64_t row = first_row
    cdef const unsigned char *newline
    cdef bint inside
    cdef int m
    cdef Py_ssize_t seen
    cdef uINT_t index
    cdef unsigned char open_mark = 60, close_mark = 62  # "<" and ">"

    while pos < last:
        if lines:
            newline = <const unsigned char *>memchr(data + pos, 10, data_size - pos)
            span_start = pos
            span_end = (newline - data) if newline != NULL else data_size
            next_pos = span_end + 1 if newline != NULL else data_size
        else:
            span_start = item_offsets[pos]
            span_end = item_offsets[pos + 1]
            next_pos = pos + 1
        if row < 0 or row >= n_rows:
            return NATIVE_BAD_ROW

        words.size = 0
        oov_bytes.size = 0
        oov_offsets.size = 0
        if _push_i64_nogil(oov_offsets, 0) != 0:
            return NATIVE_NO_MEMORY
        longest = 0
        seen = 0
        inside = False
        token_start = span_start
        i = span_start
        while i <= span_end:
            if i < span_end:
                m = _utf8_len(data, i, span_end)
                if m == 0:
                    error[0] = i
                    return NATIVE_BAD_UTF8
            else:
                m = 0
            if i == span_end or _is_space(data, i, m):
                if inside:
                    seen += 1
                    index = _vocab_lookup(table, data + token_start, i - token_start)
                    if index != NOT_FOUND:
                        if _push_u32_nogil(words, index) != 0:
                            return NATIVE_NO_MEMORY
                    elif p.is_ft:
                        # The UTF-8 bytes of "<token>", as fastText hashes them.
                        if (
                            _push_bytes_nogil(oov_bytes, &open_mark, 1) != 0
                            or _push_bytes_nogil(oov_bytes, data + token_start, i - token_start) != 0
                            or _push_bytes_nogil(oov_bytes, &close_mark, 1) != 0
                            or _push_i64_nogil(oov_offsets, oov_bytes.size) != 0
                            or _push_u32_nogil(words, OOV_TOKEN) != 0
                        ):
                            return NATIVE_NO_MEMORY
                        longest = max(longest, i - token_start + 2)
                    inside = False
                if i == span_end:
                    break
            elif not inside:
                inside = True
                token_start = i
            i += m

        # A sentence with tokens counts even if none is known, as in the job path.
        if seen:
            eff_sentences[0] += 1
        if words.size:
            capacity = min(<Py_ssize_t>p.max_ngrams, longest * max(<Py_ssize_t>p.max_n, 1))
            if capacity * <Py_ssize_t>sizeof(uINT_t) > ngrams.capacity:
                if _grow_nogil(ngrams, capacity * sizeof(uINT_t) - ngrams.size) != 0:
                    return NATIVE_NO_MEMORY
            k = 0
            eff_words[0] += words.size // sizeof(uINT_t)
            _average_sentence(
                p, <const uINT_t *>words.data, words.size // sizeof(uINT_t),
                <const unsigned char *>oov_bytes.data, <const np.int64_t *>oov_offsets.data,
                &k, <uINT_t>row, mem, <uINT_t *>ngrams.data, <size_t>capacity,
            )
        row += 1
        pos = next_pos
    return NATIVE_OK


def run_chunk(KernelContext ctx, VocabTable table, const unsigned char[::1] data, offsets,
              Py_ssize_t first, Py_ssize_t last, np.int64_t first_row, scratch):
    """Average the sentences first to last - 1 of a native source without the GIL.

    Positions are bytes in line mode (offsets is None) and string numbers otherwise; the
    first sentence goes into row first_row. Returns the number of non-empty sentences and
    of used tokens, like run_job.
    """
    _check_array(scratch, "scratch", np.float32)
    if scratch.size < ctx.params.size:
        raise ValueError(f"scratch must hold at least {ctx.params.size} entries")
    cdef Py_ssize_t n = data.shape[0]
    cdef const unsigned char *p = &data[0] if n else NULL
    cdef np.int64_t[::1] item_offsets
    cdef const np.int64_t *item_ptr = NULL
    if offsets is not None:
        item_offsets = offsets
        item_ptr = &item_offsets[0]
    cdef REAL_t *mem = <REAL_t *>(np.PyArray_DATA(scratch))
    cdef _Buffer words, oov_bytes, oov_offsets, ngrams
    memset(&words, 0, sizeof(_Buffer))
    memset(&oov_bytes, 0, sizeof(_Buffer))
    memset(&oov_offsets, 0, sizeof(_Buffer))
    memset(&ngrams, 0, sizeof(_Buffer))
    cdef Py_ssize_t eff_sentences = 0, eff_words = 0, error = -1
    cdef Py_ssize_t n_rows = ctx.n_rows
    cdef int status
    try:
        with nogil:
            status = _average_spans(
                &ctx.params, &table.view, p, n, item_ptr, first, last, first_row, n_rows, mem,
                &words, &oov_bytes, &oov_offsets, &ngrams, &eff_sentences, &eff_words, &error,
            )
    finally:
        free(words.data)
        free(oov_bytes.data)
        free(oov_offsets.data)
        free(ngrams.data)
    if status == NATIVE_NO_MEMORY:
        raise MemoryError("could not grow the buffers of a native chunk")
    if status == NATIVE_BAD_UTF8:
        raise UnicodeDecodeError(
            "utf-8", bytes(data[error : error + 1]), 0, 1,
            f"invalid UTF-8 at byte {error}; the input changed after the scan",
        )
    if status == NATIVE_BAD_ROW:
        raise IndexError(f"a sentence of the chunk maps past the {n_rows} target rows")
    return eff_sentences, eff_words


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
