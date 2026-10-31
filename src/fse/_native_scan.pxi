# Included by _native.pyx: UTF-8 classes, the scan of sentence spans and the buffer pushes
# without the GIL.

cdef inline int _utf8_len(const unsigned char *p, Py_ssize_t i, Py_ssize_t end) noexcept nogil:
    """Length of the valid UTF-8 sequence at p[i], or 0 where CPython's decoder fails."""
    cdef unsigned char b = p[i]
    cdef unsigned char c
    if b < 0x80:
        return 1
    if b < 0xC2:
        return 0
    if b < 0xE0:
        if end - i >= 2 and (p[i + 1] & 0xC0) == 0x80:
            return 2
        return 0
    if b < 0xF0:
        if end - i < 3:
            return 0
        c = p[i + 1]
        if (b == 0xE0 and c < 0xA0) or (b == 0xED and c > 0x9F):
            return 0
        if (c & 0xC0) != 0x80 or (p[i + 2] & 0xC0) != 0x80:
            return 0
        return 3
    if b < 0xF5:
        if end - i < 4:
            return 0
        c = p[i + 1]
        if (b == 0xF0 and c < 0x90) or (b == 0xF4 and c > 0x8F):
            return 0
        if (c & 0xC0) != 0x80 or (p[i + 2] & 0xC0) != 0x80 or (p[i + 3] & 0xC0) != 0x80:
            return 0
        return 4
    return 0


cdef inline Py_ssize_t _utf8_error(
    const unsigned char *p, Py_ssize_t start, Py_ssize_t end
) noexcept nogil:
    """The first byte of p[start:end] where CPython's decoder fails, or -1 if it is valid."""
    cdef Py_ssize_t i = start
    cdef int m
    while i < end:
        m = _utf8_len(p, i, end)
        if m == 0:
            return i
        i += m
    return -1


def _check_span(const unsigned char[::1] data, offsets, Py_ssize_t first, Py_ssize_t last):
    """Raise unless [first, last) is a valid range of whole sentences of data.

    The native routines read through raw pointers without bounds checks, so every range
    and offset they get is checked here: lines must start at first and at last; offsets
    must cover the range, stay inside data and never decrease.
    """
    cdef Py_ssize_t n = data.shape[0]
    cdef const np.int64_t[::1] item_offsets
    cdef Py_ssize_t i
    if offsets is None:
        if not 0 <= first <= last <= n:
            raise ValueError(f"range [{first}, {last}) is outside the {n} bytes of the source")
        if first > 0 and data[first - 1] != 10:
            raise ValueError(f"range start {first} is not the start of a line")
        if 0 < last < n and data[last - 1] != 10:
            raise ValueError(f"range end {last} is not the start of a line")
        return
    item_offsets = offsets
    if not 0 <= first <= last <= item_offsets.shape[0] - 1:
        raise ValueError(f"range [{first}, {last}) is outside the {item_offsets.shape[0] - 1} strings")
    if item_offsets[first] < 0 or item_offsets[last] > n:
        raise ValueError("string offsets point outside the source")
    for i in range(first, last):
        if item_offsets[i + 1] < item_offsets[i]:
            raise ValueError(f"string offsets decrease at {i}")


def scan_spans(const unsigned char[::1] data, offsets, Py_ssize_t first, Py_ssize_t last,
               Py_ssize_t chunk_bytes, Py_ssize_t chunk_rows=CHUNK_MAX_ROWS):
    """Validate and count the sentences of one scan range, without the GIL.

    Line mode (offsets is None): the range holds the lines that start at byte positions
    in [first, last); the caller aligns first and last to line starts. Item mode: the
    range holds the strings first to last - 1 of a TextList, delimited by offsets.

    Returns (sentences, chunks, error_byte, error_sentence): chunks is a list of
    (start, end, first_sentence) with positions in the same unit as the range and sentence
    numbers relative to the range; error_byte is -1 if the range is valid.
    """
    _check_span(data, offsets, first, last)
    cdef Py_ssize_t n = data.shape[0]
    cdef const unsigned char *p = &data[0] if n else NULL
    cdef bint lines = offsets is None
    cdef const np.int64_t[::1] item_offsets
    cdef const np.int64_t *item_ptr = NULL
    if not lines:
        item_offsets = offsets
        item_ptr = &item_offsets[0]
    cdef Py_ssize_t sentences = 0
    cdef Py_ssize_t error = -1
    cdef Py_ssize_t pos = first, span_end, next_pos, chunk_start = first, chunk_first = 0
    cdef const unsigned char *newline
    # Chunk boundaries (start, end, first sentence) are collected in a buffer that grows
    # as needed and converted to a list at the end. Every chunk but the last holds at least
    # chunk_bytes bytes or chunk_rows sentences. The row limit keeps a chunk of empty
    # lines, which take no bytes, from holding all rows, and bounds the rows a worker
    # holds at once.
    if chunk_bytes < 1:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
    if not 1 <= chunk_rows <= CHUNK_MAX_ROWS:
        raise ValueError(f"chunk_rows must be between 1 and {CHUNK_MAX_ROWS}, got {chunk_rows}")
    cdef _Buffer bounds
    bounds.data = NULL
    bounds.size = 0
    bounds.capacity = 0
    cdef int failed = 0
    try:
        with nogil:
            while pos < last:
                if lines:
                    newline = <const unsigned char *>memchr(p + pos, 10, n - pos)
                    span_end = (newline - p) if newline != NULL else n
                    next_pos = span_end + 1 if newline != NULL else n
                else:
                    span_end = item_ptr[pos + 1]
                    next_pos = pos + 1
                error = _utf8_error(p, item_ptr[pos] if not lines else pos, span_end)
                if error >= 0:
                    break
                sentences += 1
                pos = next_pos
                if (
                    (lines and pos - chunk_start >= chunk_bytes)
                    or (not lines and (item_ptr[pos] - item_ptr[chunk_start]) >= chunk_bytes)
                    or sentences - chunk_first >= chunk_rows
                ) and pos < last:
                    if (
                        _push_i64_nogil(&bounds, chunk_start) != 0
                        or _push_i64_nogil(&bounds, pos) != 0
                        or _push_i64_nogil(&bounds, chunk_first) != 0
                    ):
                        failed = 1
                        break
                    chunk_start = pos
                    chunk_first = sentences
        if failed:
            raise MemoryError("could not grow the list of chunks")
        if error >= 0:
            return sentences, [], error, sentences
        if chunk_start < last:
            _push_i64(&bounds, chunk_start)
            _push_i64(&bounds, last)
            _push_i64(&bounds, chunk_first)
        flat = _to_array(&bounds, np.int64).tolist()
    finally:
        free(bounds.data)
    chunk_list = [tuple(flat[c : c + 3]) for c in range(0, len(flat), 3)]
    return sentences, chunk_list, -1, -1


cdef int _push_u32_nogil(_Buffer *b, uINT_t value) noexcept nogil:
    if <Py_ssize_t>sizeof(uINT_t) > b.capacity - b.size:
        if _grow_nogil(b, sizeof(uINT_t)) != 0:
            return -1
    (<uINT_t *>(b.data + b.size))[0] = value
    b.size += sizeof(uINT_t)
    return 0


cdef int _push_i64_nogil(_Buffer *b, np.int64_t value) noexcept nogil:
    if <Py_ssize_t>sizeof(np.int64_t) > b.capacity - b.size:
        if _grow_nogil(b, sizeof(np.int64_t)) != 0:
            return -1
    (<np.int64_t *>(b.data + b.size))[0] = value
    b.size += sizeof(np.int64_t)
    return 0


cdef int _push_bytes_nogil(_Buffer *b, const unsigned char *data, Py_ssize_t n) noexcept nogil:
    if n == 0:
        return 0  # b.data may still be NULL, and memcpy needs valid pointers
    if n > b.capacity - b.size:
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
