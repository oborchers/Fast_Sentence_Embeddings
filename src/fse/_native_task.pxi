# Included by _native.pyx after _native_arrow_column.pxi: stages 1 to 4 of one task in one
# section without the GIL (ARCHITECTURE.md, "Runtime", Tasks).

# ---------------------------------------------------------------------------------------
# pool_text: tokenize, check the batch, pool and normalize, without the GIL in between.
# Its errors and their order are those of TextLookup.encode, Batch, the source's check of
# the chunk, check_out, WeightedMean.pool and postprocess_pooled, called one after the
# other: the section runs a stage only if every check before it passed, and the code after
# it raises in that order.
# ---------------------------------------------------------------------------------------

cdef enum:
    # The checks of the batch in the buffers of a workspace. Those of Batch.__post_init__
    # first, then the sentence count of the chunk, then those of WeightedMean._check_batch
    # (where Batch rejects a defect first, _check_batch's message cannot occur).
    CHECK_OK = 0
    CHECK_OFFSETS_START = 1
    CHECK_OFFSETS_DECREASE = 2
    CHECK_OFFSETS_END = 3
    CHECK_UNKNOWN_WITHOUT_NGRAMS = 4
    CHECK_NGRAM_COUNT = 5
    CHECK_NGRAM_OFFSETS_START = 6
    CHECK_NGRAM_OFFSETS_DECREASE = 7
    CHECK_NGRAM_OFFSETS_END = 8
    CHECK_ROWS = 9
    CHECK_NO_NGRAM_VECTORS = 10
    CHECK_NO_OOV_WEIGHT = 11
    CHECK_BUCKET = 12
    CHECK_TOKEN = 13

# For the tests: break one buffer of the workspace after stage 2 (see _test_corrupt).
cdef int _test_corrupt_kind = 0


def _test_corrupt(int kind):
    """For the tests: from now on, break the batch of every pool_text after stage 2 (1:
    the first offset, 2: an offset that decreases, 3: the last offset, 4: a token id at
    the number of words, 5: a bucket at the number of buckets, 6: one ngram offset
    less, 7: an unknown word, 8: one sentence more, 9: the first ngram offset, 10: the
    last ngram offset); 0 stops. Returns the previous kind."""
    global _test_corrupt_kind
    previous = _test_corrupt_kind
    _test_corrupt_kind = kind
    return previous


cdef void _corrupt(Workspace ws, WeightedMean kernel) noexcept nogil:
    cdef int64_t *offsets = <int64_t *>ws.offsets.data
    cdef Py_ssize_t n_offsets = ws.offsets.size // <Py_ssize_t>sizeof(int64_t)
    cdef uINT_t *tokens = <uINT_t *>ws.tokens.data
    cdef Py_ssize_t n_tokens = ws.tokens.size // <Py_ssize_t>sizeof(uINT_t)
    cdef int64_t *ngram_offsets = <int64_t *>ws.ngram_offsets.data
    cdef Py_ssize_t n_ngram_offsets = ws.ngram_offsets.size // <Py_ssize_t>sizeof(int64_t)
    cdef int kind = _test_corrupt_kind
    if kind == 1:
        offsets[0] = 1
    elif kind == 2 and n_offsets > 2:
        offsets[1] = offsets[n_offsets - 1] + 1
    elif kind == 3:
        offsets[n_offsets - 1] += 1
    elif kind == 4 and n_tokens:
        tokens[0] = <uINT_t>kernel.n_words
    elif kind == 5 and ws.ngrams.size:
        (<uINT_t *>ws.ngrams.data)[0] = <uINT_t>kernel.n_buckets
    elif kind == 6 and n_ngram_offsets:
        ws.ngram_offsets.size -= sizeof(int64_t)
    elif kind == 7 and n_tokens:
        tokens[0] = OOV_TOKEN
    elif kind == 8:
        _push_i64_nogil(&ws.offsets, offsets[n_offsets - 1])
    elif kind == 9 and n_ngram_offsets:
        ngram_offsets[0] = 1
    elif kind == 10 and n_ngram_offsets:
        ngram_offsets[n_ngram_offsets - 1] += 1


cdef int _check_offsets_nogil(const int64_t *offsets, Py_ssize_t n, int64_t total,
                              int start, Py_ssize_t *at) noexcept nogil:
    """The rule of fse._contract._check_offsets: from 0, never decreasing, to total;
    ``start`` is the status of the first of its three messages."""
    cdef Py_ssize_t i
    if n == 0 or offsets[0] != 0:
        return start
    for i in range(1, n):
        if offsets[i] < offsets[i - 1]:
            return start + 1
    if offsets[n - 1] != total:
        at[0] = offsets[n - 1]
        return start + 2
    return CHECK_OK


cdef struct _Counts:
    # The entries of the batch buffers of a workspace, and its unknown words and largest id.
    Py_ssize_t offsets, tokens, ngram_offsets, ngrams, unknown
    uINT_t largest


cdef void _count(Workspace ws, _Counts *c) noexcept nogil:
    cdef const uINT_t *tokens = <const uINT_t *>ws.tokens.data
    cdef Py_ssize_t i
    c.offsets = ws.offsets.size // <Py_ssize_t>sizeof(int64_t)
    c.tokens = ws.tokens.size // <Py_ssize_t>sizeof(uINT_t)
    c.ngram_offsets = ws.ngram_offsets.size // <Py_ssize_t>sizeof(int64_t)
    c.ngrams = ws.ngrams.size // <Py_ssize_t>sizeof(uINT_t)
    c.unknown = 0
    c.largest = 0
    for i in range(c.tokens):
        if tokens[i] == OOV_TOKEN:
            c.unknown += 1
        elif tokens[i] > c.largest:
            c.largest = tokens[i]


cdef int _check_built(Workspace ws, _Counts *c, bint ngrams, Py_ssize_t rows,
                      Py_ssize_t *at) noexcept nogil:
    """The checks of a Batch made from the buffers of ws (``ngrams``: the lookup gives
    FastText buckets), then the sentence count of the chunk, in their order."""
    cdef int status = _check_offsets_nogil(<const int64_t *>ws.offsets.data, c.offsets,
                                           c.tokens, CHECK_OFFSETS_START, at)
    if status != CHECK_OK:
        return status
    if not ngrams:
        if c.unknown:
            at[0] = c.unknown
            return CHECK_UNKNOWN_WITHOUT_NGRAMS
    else:
        if c.ngram_offsets != c.unknown + 1:
            at[0] = c.unknown
            return CHECK_NGRAM_COUNT
        status = _check_offsets_nogil(<const int64_t *>ws.ngram_offsets.data, c.ngram_offsets,
                                      c.ngrams, CHECK_NGRAM_OFFSETS_START, at)
        if status != CHECK_OK:
            return status
    if c.offsets - 1 != rows:
        return CHECK_ROWS
    return CHECK_OK


cdef int _check_kernel(Workspace ws, _Counts *c, WeightedMean kernel,
                       bint oov_weight) noexcept nogil:
    """The checks of WeightedMean._check_batch that Batch does not make, in their order."""
    cdef const uINT_t *buckets = <const uINT_t *>ws.ngrams.data
    cdef Py_ssize_t i
    if c.unknown:
        if not kernel.fasttext:
            return CHECK_NO_NGRAM_VECTORS
        if not oov_weight:
            return CHECK_NO_OOV_WEIGHT
        for i in range(c.ngrams):
            if buckets[i] >= kernel.n_buckets:
                return CHECK_BUCKET
    if c.tokens > c.unknown and c.largest >= kernel.n_words:
        return CHECK_TOKEN
    return CHECK_OK


cdef struct _Pooled:
    # What the section without the GIL found: the status of stage 1, of the Batch checks
    # (with the number their message names), of the kernel checks, and the first bad row
    # of pooling and of normalize (-1 if none).
    int encode
    int built
    int kernel
    Py_ssize_t at
    _Counts counts
    int64_t bad
    int64_t unnormalizable
    int64_t not_finite


cdef void _start_result(_Pooled *result) noexcept nogil:
    result.encode = ENCODE_OK
    result.built = CHECK_OK
    result.kernel = CHECK_OK
    result.at = 0
    result.bad = -1
    result.unnormalizable = -1
    result.not_finite = -1


cdef struct _Stages:
    # What the stages after stage 1 need, read with the GIL before the section.
    float *out
    Py_ssize_t rows
    bint out_ok  # out passed its checks; stages 3 and 4 run only then
    bint ngrams
    bint oov_weight
    bint normalize
    int isa


cdef void _pool_stages(WeightedMean kernel, Workspace ws, _Stages *s,
                       _Pooled *result) noexcept nogil:
    """Stages 2 to 4 on the batch stage 1 left in ws: the checks, pooling and normalize,
    each only if everything before it passed."""
    cdef fse_batch b
    if _test_corrupt_kind:
        _corrupt(ws, kernel)
    _count(ws, &result.counts)
    result.built = _check_built(ws, &result.counts, s.ngrams, s.rows, &result.at)
    if result.built != CHECK_OK or not s.out_ok:
        return
    result.kernel = _check_kernel(ws, &result.counts, kernel, s.oov_weight)
    if result.kernel != CHECK_OK or s.rows == 0:
        return
    b.n_sentences = s.rows
    b.offsets = <const int64_t *>ws.offsets.data
    b.tokens = <const uint32_t *>ws.tokens.data if ws.tokens.size else NULL
    b.ngram_offsets = <const int64_t *>ws.ngram_offsets.data if s.ngrams else NULL
    b.ngrams = <const uint32_t *>ws.ngrams.data if s.ngrams and ws.ngrams.size else NULL
    result.bad = fse_run(s.isa, &kernel.params, &b, s.out, NULL, 1)
    if result.bad < 0 and s.normalize:
        result.not_finite = fse_normalize(s.isa, s.out, s.rows, kernel.dim, &result.unnormalizable)


cdef bint _overlaps(out, kept):
    """Whether the bytes of two C-contiguous arrays overlap: the bounds test of
    np.may_share_memory, by pointer compares (no numpy call)."""
    cdef char *a = <char *>np.PyArray_DATA(out)
    cdef char *b = <char *>np.PyArray_DATA(kept)
    cdef Py_ssize_t na = np.PyArray_NBYTES(out), nb = np.PyArray_NBYTES(kept)
    if na == 0 or nb == 0:
        return False  # an empty array shares no memory, as np.may_share_memory says
    return a < b + nb and b < a + na


cdef object _out_error(WeightedMean kernel, out, Py_ssize_t rows):
    """The error of check_out and WeightedMean._out for ``out``, or None; made before the
    section (nothing may be written into a wrong out), raised at their place after it."""
    shape = (rows, kernel.dim)
    if (
        not isinstance(out, np.ndarray)
        or out.dtype != np.float32
        or out.shape != shape
        or not out.flags.c_contiguous
        or not out.flags.writeable
    ):
        return ValueError(f"out must be a writable C-contiguous float32 array of shape {shape}")
    for kept in kernel._keep[:5]:
        if kept is not None and _overlaps(out, kept):
            return ValueError("out must not share memory with the vectors or weights")
    return None


cdef int _stages_of(_Stages *s, WeightedMean kernel, TextLookup lookup, out, Py_ssize_t rows,
                    bint normalize, int isa, error) except -1:
    cdef float[:, ::1] view
    s.out = NULL
    s.rows = rows
    s.out_ok = error is None
    if s.out_ok and rows:
        view = out
        s.out = &view[0, 0]  # out outlives the call: the caller holds it
    s.ngrams = lookup.fasttext
    s.oov_weight = kernel._keep[5] is not None
    s.normalize = normalize
    s.isa = isa
    return 0


def pool_text(WeightedMean kernel not None, TextLookup lookup not None, Workspace ws not None,
              text, Py_ssize_t first, Py_ssize_t last, Py_ssize_t rows, first_row, out,
              bint normalize, after_encode=None, *, isa=None):
    """Stages 1 to 4 of one range of sentences into ``out`` in one section without the
    GIL: tokenize (as ``TextLookup.encode``), the checks of the batch, ``weighted_mean@1``
    and, with ``normalize``, ``normalize@1`` in place. ``text`` is ``(data, offsets)``
    as ``TextLookup.encode`` takes them, or an ArrowColumn; the range holds ``rows``
    sentences (the chunk's), the first of them row ``first_row``. ``after_encode`` (the
    source's own check after encoding, such as a file unchanged) runs after the errors of
    stage 1 and of the batch, before those of ``out``, pooling and normalize. Raises what
    encode, the source, ``CpuBackend.rows`` and ``postprocess_pooled`` raise on the same
    range one after the other; the rows and their bits are theirs. Returns ``rows``."""
    cdef int code = _check_isa(isa)
    cdef _Stages s
    cdef _Pooled result
    error = _out_error(kernel, out, rows)
    _stages_of(&s, kernel, lookup, out, rows, normalize, code, error)
    if isinstance(text, ArrowColumn):
        _pool_arrow(kernel, lookup, ws, <ArrowColumn>text, first, last, &s, &result, first_row)
    elif isinstance(text, tuple) and len(text) == 2:
        _pool_buffer(lookup, kernel, ws, text, first, last, &s, &result, first_row)
    else:
        raise TypeError("text must be (data, offsets) or an ArrowColumn")
    _raise_pooled(&result, kernel, first_row, after_encode, error)
    return rows


cdef int _pool_buffer(TextLookup lookup, WeightedMean kernel, Workspace ws, text,
                      Py_ssize_t first, Py_ssize_t last, _Stages *s, _Pooled *result,
                      first_row) except -1:
    """pool_text of a range of a buffer: stage 1 raised here, the rest by the caller."""
    global _nogil_sections
    data, offsets = text
    cdef const unsigned char[::1] view = data
    offsets, first, last = lookup._check_encode(view, offsets, first, last, ws)
    cdef _Span span
    _span_of(&span, view, offsets, first, last)
    cdef uint64_t saved
    _start_result(result)
    ws.busy = True
    try:
        saved = fse_fpenv_push()
        _nogil_sections += 1
        with nogil:
            result.encode = _encode_span(lookup, &span, ws)
            if result.encode == ENCODE_OK:
                _pool_stages(kernel, ws, s, result)
        fse_fpenv_pop(saved)
        _raise_encode_status(result.encode, view, &span, first_row)
    finally:
        ws.busy = False
    return 0


cdef int _pool_arrow(WeightedMean kernel, TextLookup lookup, Workspace ws, ArrowColumn column,
                     Py_ssize_t first, Py_ssize_t last, _Stages *s, _Pooled *result,
                     first_row) except -1:
    """pool_text of the rows first to last - 1 of an Arrow column: stage 1 raised here."""
    global _nogil_sections
    column._check_encode(lookup, first, last, ws)
    cdef _ArrowRow row
    row.p, row.n, row.bad, row.sentence = NULL, 0, -1, 0
    cdef uint64_t saved
    _start_result(result)
    ws.busy = True
    column._readers += 1
    try:
        saved = fse_fpenv_push()
        _nogil_sections += 1
        with nogil:
            result.encode = column._encode_rows(lookup, first, last, ws, &row)
            if result.encode == ENCODE_OK:
                _pool_stages(kernel, ws, s, result)
        fse_fpenv_pop(saved)
        _arrow_encode_status(result.encode, row.p, row.n, row.bad, row.sentence,
                             first_row + row.sentence)
    finally:
        ws.busy = False
        column._readers -= 1
    return 0


_ROW_MAX = 2**63 - 1  # fse._contract._INT64_MAX: the extension imports nothing at load
_OFFSETS_CHECKS = (CHECK_OFFSETS_START, CHECK_OFFSETS_DECREASE, CHECK_OFFSETS_END)


cdef int _raise_pooled(_Pooled *r, WeightedMean kernel, first_row, after_encode,
                       error) except -1:
    """Raise what follows stage 1, in the order of the calls it replaces: the checks of
    Batch (first_row, offsets, the rows' range, ngrams), the chunk's count, the source's
    own check, out, the checks of the kernel, pooling and normalize."""
    if isinstance(first_row, bool) or not isinstance(first_row, (int, np.integer)):
        raise TypeError(f"first_row must be an int, not {type(first_row).__name__}")
    if not 0 <= first_row <= _ROW_MAX:
        raise ValueError(f"first_row must be between 0 and 2**63 - 1, not {first_row}")
    if r.built in _OFFSETS_CHECKS:
        _raise_check(r.built, kernel, r)
    if int(first_row) + r.counts.offsets - 2 > _ROW_MAX:
        raise ValueError(f"the rows of a batch from {first_row} exceed the int64 range")
    _raise_check(r.built, kernel, r)
    if after_encode is not None:
        after_encode()
    if error is not None:
        raise error
    _raise_check(r.kernel, kernel, r)
    if r.bad >= 0:
        raise ValueError(f"row {first_row + r.bad} contains NaN or infinite values")
    if r.unnormalizable >= 0:
        raise ValueError(
            f"row {first_row + r.unnormalizable} cannot be normalized: the sum of its "
            "squares underflows or overflows float32"
        )
    if r.not_finite >= 0:
        raise ValueError(f"row {first_row + r.not_finite} contains NaN or infinite values")
    return 0


cdef int _raise_check(int status, WeightedMean kernel, _Pooled *r) except -1:
    """Raise the error of a failed check with the message of Batch or _check_batch."""
    if status == CHECK_OK:
        return 0
    if status == CHECK_ROWS:
        raise RuntimeError("the source changed since it was scanned")
    name = "ngram_offsets" if CHECK_NGRAM_OFFSETS_START <= status <= CHECK_NGRAM_OFFSETS_END else "offsets"
    if status in (CHECK_OFFSETS_START, CHECK_NGRAM_OFFSETS_START):
        raise ValueError(f"{name} must start with 0")
    if status in (CHECK_OFFSETS_DECREASE, CHECK_NGRAM_OFFSETS_DECREASE):
        raise ValueError(f"{name} must not decrease")
    if status in (CHECK_OFFSETS_END, CHECK_NGRAM_OFFSETS_END):
        total = r.counts.tokens if status == CHECK_OFFSETS_END else r.counts.ngrams
        raise ValueError(f"{name} must end with {total}, the number of entries, not {r.at}")
    if status == CHECK_UNKNOWN_WITHOUT_NGRAMS:
        raise ValueError(f"{r.at} unknown words in a batch without ngrams")
    if status == CHECK_NGRAM_COUNT:
        raise ValueError(
            f"ngram_offsets must have {r.at + 1} entries for {r.at} unknown words, "
            f"not {r.counts.ngram_offsets}"
        )
    if status == CHECK_NO_NGRAM_VECTORS:
        raise ValueError("an unknown FastText word needs ngram vectors")
    if status == CHECK_NO_OOV_WEIGHT:
        raise ValueError("an unknown FastText word needs oov_weight")
    if status == CHECK_BUCKET:
        raise ValueError(f"ngram buckets must be below {kernel.n_buckets}")
    raise ValueError(f"token ids must be below {kernel.n_words}, the number of words")
