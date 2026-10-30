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

"""Native routines of fse 2.0 without the GIL: ngram hashing, vocabulary table, UTF-8 scan,
token lists joined, the weighted mean kernel, dot16, steps, score and storage, the search, linear
heads, the tokenizers (wordpiece@1 and unigram@1 among them), the text lookup and Arrow string
columns."""

import cython
import numpy as np

cimport numpy as np

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.pyport cimport PY_SSIZE_T_MAX
from libc.stdint cimport SIZE_MAX
from libc.stdlib cimport calloc, free, malloc, realloc
from libc.string cimport memchr, memcmp, memcpy, memset

np.import_array()

# The parts, in this order; each may use what an earlier one defines.
include "_native_hash.pxi"
include "_native_buffer.pxi"
include "_native_vocab.pxi"
include "_native_scan.pxi"
include "_native_lists.pxi"
include "_native_kernel.pxi"
include "_native_steps.pxi"
include "_native_store.pxi"
include "_native_search.pxi"
include "_native_head.pxi"
include "_native_text.pxi"
include "_native_wordpiece.pxi"
include "_native_unigram.pxi"
include "_native_unigram_steps.pxi"
include "_native_arrow.pxi"
include "_native_arrow_column.pxi"
