# Included by _native.pyx after _native_arrow.pxi: the ArrowColumn, the chunks of one Arrow
# string column taken over from their capsules, checked, scanned and encoded without the
# GIL, and released once (ARCHITECTURE.md, "Text sources").



cdef class ArrowColumn:
    """The chunks of one Arrow string column, owned until release().

    Make one with from_stream(capsule) or from_array(schema_capsule, array_capsule).
    """

    cdef ArrowArray *_arrays  # every array taken over, released once
    cdef Py_ssize_t _n_arrays, _arrays_capacity
    cdef fse_arrow_chunk *_chunks  # the non-empty ones
    cdef int64_t *_starts  # n + 1: the row of each chunk, then rows
    cdef int64_t *_byte_starts  # n + 1: the text bytes before each chunk, then nbytes
    cdef Py_ssize_t _n, _capacity
    cdef void **_owned  # own copies: aligned offsets or views, the sizes of view buffers
    cdef Py_ssize_t _n_owned, _owned_capacity
    cdef int _kind
    cdef int _readers
    cdef readonly bint released  # by release(), or when freed
    cdef readonly int64_t rows
    cdef readonly int64_t nbytes
    # The row of the first null, or -1: the rows before it are the column, nothing after
    # it is read, and the source raises it after the rows before it (ARCHITECTURE.md,
    # "Runtime", Errors).
    cdef readonly int64_t null_row
    # Bytes of offsets or views copied because they were not aligned; text is never copied.
    cdef readonly int64_t copied_bytes

    def __cinit__(self):
        if _test_fail_new:
            raise MemoryError("could not allocate an ArrowColumn (test)")
        self._starts = <int64_t *>malloc(sizeof(int64_t))
        self._byte_starts = <int64_t *>malloc(sizeof(int64_t))
        if self._starts == NULL or self._byte_starts == NULL:
            raise MemoryError("could not allocate an ArrowColumn")
        self._starts[0] = 0
        self._byte_starts[0] = 0
        self._kind = -1
        self.null_row = -1

    def __dealloc__(self):
        self._free()

    def __reduce__(self):
        raise TypeError("an ArrowColumn cannot be pickled")

    cdef void _free(self) noexcept:
        """Close and detach everything, then release: a callback may drop the GIL, and
        another thread finds an empty, released column with nothing left to free."""
        cdef ArrowArray *arrays = self._arrays
        cdef void **owned = self._owned
        cdef Py_ssize_t k, n_arrays = self._n_arrays, n_owned = self._n_owned
        free(self._chunks)
        free(self._starts)
        free(self._byte_starts)
        self._arrays = NULL
        self._chunks = NULL
        self._owned = NULL
        self._starts = NULL
        self._byte_starts = NULL
        self._n_arrays = self._n = self._n_owned = 0
        self._arrays_capacity = self._capacity = self._owned_capacity = 0
        self.released = True
        self.rows = self.nbytes = 0
        for k in range(n_arrays):
            if arrays[k].release != NULL:
                arrays[k].release(&arrays[k])
        for k in range(n_owned):
            free(owned[k])
        free(arrays)
        free(owned)

    @staticmethod
    def from_stream(capsule):
        """The column of an ``arrow_array_stream`` capsule (``__arrow_c_stream__``)."""
        cdef ArrowArrayStream *source = <ArrowArrayStream *>_capsule_pointer(capsule, b"arrow_array_stream")
        if source.release == NULL:
            raise ValueError("the Arrow stream was released already")
        cdef ArrowColumn column = ArrowColumn()  # before the move: it may fail
        cdef ArrowArrayStream stream
        memcpy(&stream, source, sizeof(ArrowArrayStream))
        source.release = NULL
        cdef ArrowSchema schema
        cdef ArrowArray array
        cdef int code
        cdef bint have_schema = False  # a failed get_schema may leave anything in schema
        try:
            if stream.get_schema == NULL or stream.get_next == NULL:
                raise ValueError("the Arrow stream has no get_schema or get_next")
            code = stream.get_schema(&stream, &schema)
            if code != 0:
                _stream_error(&stream, code, "get_schema")
            have_schema = True
            column._set_kind(_arrow_kind(&schema))
            while True:
                array.release = NULL
                code = stream.get_next(&stream, &array)
                if code != 0:
                    _stream_error(&stream, code, "get_next")
                if array.release == NULL:
                    break  # the end of the stream
                try:
                    column._take(&array)
                finally:
                    if array.release != NULL:  # not taken over (no memory): ours to release
                        array.release(&array)
                if column.null_row >= 0:
                    break  # nothing after a null is read, so no later error displaces it
        finally:
            if have_schema and schema.release != NULL:
                schema.release(&schema)
            if stream.release != NULL:
                stream.release(&stream)
        return column

    @staticmethod
    def from_array(schema_capsule, array_capsule):
        """The column of one array (``__arrow_c_array__``): the schema is read where it is,
        the array is taken over."""
        cdef ArrowSchema *schema = <ArrowSchema *>_capsule_pointer(schema_capsule, b"arrow_schema")
        cdef ArrowArray *array = <ArrowArray *>_capsule_pointer(array_capsule, b"arrow_array")
        cdef ArrowColumn column = ArrowColumn()
        column._set_kind(_arrow_kind(schema))
        if array.release == NULL:
            raise ValueError("the Arrow array was released already")
        column._take(array)
        return column

    cdef int _set_kind(self, int kind) except -1:
        self._kind = kind
        return 0

    cdef void *_own(self, size_t size) except NULL:
        """A new block of size bytes that release() frees."""
        cdef void **grown
        cdef void *block
        if self._n_owned == self._owned_capacity:
            grown = <void **>realloc(self._owned, (2 * self._owned_capacity + 4) * sizeof(void *))
            if grown == NULL:
                raise MemoryError("could not grow the ArrowColumn")
            self._owned = grown
            self._owned_capacity = 2 * self._owned_capacity + 4
        block = malloc(size if size else 1)
        if block == NULL:
            raise MemoryError("could not copy the Arrow offsets")
        self._owned[self._n_owned] = block
        self._n_owned += 1
        return block

    cdef int _grow(self) except -1:
        cdef Py_ssize_t capacity = 2 * self._capacity + 4
        cdef fse_arrow_chunk *chunks = <fse_arrow_chunk *>realloc(self._chunks, capacity * sizeof(fse_arrow_chunk))
        if chunks == NULL:
            raise MemoryError("could not grow the ArrowColumn")
        self._chunks = chunks
        cdef int64_t *starts = <int64_t *>realloc(self._starts, (capacity + 1) * sizeof(int64_t))
        if starts == NULL:
            raise MemoryError("could not grow the ArrowColumn")
        self._starts = starts
        starts = <int64_t *>realloc(self._byte_starts, (capacity + 1) * sizeof(int64_t))
        if starts == NULL:
            raise MemoryError("could not grow the ArrowColumn")
        self._byte_starts = starts
        self._capacity = capacity
        return 0

    cdef int _take(self, ArrowArray *source) except -1:
        """Take over one array (its release moves here at once), check it and add its rows."""
        cdef ArrowArray *grown
        if self._n_arrays >= _test_take_limit:
            raise MemoryError("could not grow the ArrowColumn (test)")
        if self._n_arrays == self._arrays_capacity:
            grown = <ArrowArray *>realloc(self._arrays, (2 * self._arrays_capacity + 4) * sizeof(ArrowArray))
            if grown == NULL:
                raise MemoryError("could not grow the ArrowColumn")  # the capsule still owns it
            self._arrays = grown
            self._arrays_capacity = 2 * self._arrays_capacity + 4
        memcpy(&self._arrays[self._n_arrays], source, sizeof(ArrowArray))
        source.release = NULL
        self._n_arrays += 1
        cdef ArrowArray *a = &self._arrays[self._n_arrays - 1]
        cdef ArrowArray head  # the rows of a before its first null
        cdef int64_t null
        if self.null_row >= 0:
            return 0  # after a null nothing is read; the array is released with the rest
        null = _arrow_check_array(a, self.rows, self._kind)
        if null >= 0:
            self.null_row = self.rows + null
            memcpy(&head, a, sizeof(ArrowArray))
            head.length = null
            a = &head
        if a.length == 0:
            return 0
        if self._n == self._capacity:
            self._grow()
        cdef fse_arrow_chunk *c = &self._chunks[self._n]
        cdef int64_t text = self._fill(a, c)
        if text > INT64_MAX - self.nbytes:
            raise OverflowError("the Arrow column holds more than 2**63 bytes")
        self._n += 1
        self.rows += a.length
        self.nbytes += text
        self._starts[self._n] = self.rows
        self._byte_starts[self._n] = self.nbytes
        return 0

    cdef const void *_index(self, ArrowArray *a, int64_t count, int64_t width, int64_t *first) except NULL:
        """The offsets or views from a.offset on (count entries of width bytes), copied when
        they are not aligned to their type."""
        cdef const char *index = <const char *>a.buffers[1]
        cdef void *copy
        if index == NULL:
            raise ValueError("the Arrow array has no offsets or views")
        if count > PY_SSIZE_T_MAX // width or a.offset + count > PY_SSIZE_T_MAX // width:
            raise ValueError("the offsets or views of an Arrow array overflow")
        if (<uintptr_t>index) % (4 if width == 16 else width) == 0:
            first[0] = a.offset
            return index
        copy = self._own(count * width)
        memcpy(copy, index + a.offset * width, count * width)
        self.copied_bytes += count * width
        first[0] = 0
        return copy

    cdef int64_t _fill(self, ArrowArray *a, fse_arrow_chunk *c) except -1:
        """Set c from a and give its text bytes."""
        c.kind = self._kind
        c.length = a.length
        c.data = NULL
        c.data_len = 0
        c.n_data = 0
        c.data_ptrs = NULL
        c.sizes = NULL
        if self._kind == FSE_ARROW_UTF8_VIEW:
            c.index = self._index(a, a.length, 16, &c.first)
            return self._fill_views(a, c)
        c.index = self._index(a, a.length + 1, 4 if self._kind == FSE_ARROW_UTF8 else 8, &c.first)
        cdef int64_t base, last
        if self._kind == FSE_ARROW_UTF8:
            base = (<const int32_t *>c.index)[c.first]
            last = (<const int32_t *>c.index)[c.first + a.length]
        else:
            base = (<const int64_t *>c.index)[c.first]
            last = (<const int64_t *>c.index)[c.first + a.length]
        if base < 0 or last < base:
            raise ValueError(f"invalid Arrow offsets in the chunk from row {self.rows}")
        c.data = <const unsigned char *>a.buffers[2]
        if c.data == NULL and last > 0:  # any offset above 0 points into the data
            raise ValueError("the Arrow array has no data buffer")
        if c.data == NULL: c.data = _ARROW_NO_DATA  # rows all empty at 0 (data_len 0)
        c.data_len = last
        return last - base

    cdef int64_t _fill_views(self, ArrowArray *a, fse_arrow_chunk *c) except -1:
        cdef int64_t n_data = a.n_buffers - 3, k, i, total = 0, n = 0
        cdef int64_t *sizes
        cdef const char *source = <const char *>a.buffers[a.n_buffers - 1]
        cdef const unsigned char *p
        cdef int status = 0
        if n_data > 0:
            if source == NULL:
                raise ValueError("the Arrow view array has no buffer of data sizes")
            if n_data > PY_SSIZE_T_MAX // 8:
                raise ValueError("the Arrow view array has too many data buffers")
            sizes = <int64_t *>self._own(n_data * 8)
            memcpy(sizes, source, n_data * 8)  # memcpy: the sizes need not be aligned
            for k in range(n_data):
                if sizes[k] < 0:
                    raise ValueError(f"data buffer {k} of an Arrow view array has the size {sizes[k]}")
                if sizes[k] > 0 and a.buffers[2 + k] == NULL:
                    raise ValueError(f"data buffer {k} of an Arrow view array is missing")
            c.sizes = sizes
            c.data_ptrs = <const unsigned char * const *>&a.buffers[2]
        c.n_data = n_data
        with nogil:
            for i in range(a.length):
                if fse_arrow_row(c, i, &p, &n) != 0:
                    status = 1
                    break
                if n > INT64_MAX - total:
                    status = 2
                    break
                total += n
        if status == 2:
            raise OverflowError("the Arrow column holds more than 2**63 bytes")
        if status:
            raise ValueError(f"the view of row {self.rows + i} of the Arrow column points outside its data")
        return total

    cdef int _check_range(self, Py_ssize_t first, Py_ssize_t last) except -1:
        if self.released:
            raise ValueError("the Arrow column was released")
        if not 0 <= first <= last <= self.rows:
            raise ValueError(f"range [{first}, {last}) is outside the {self.rows} rows")
        return 0

    @property
    def chunk_starts(self):
        """The first row of every chunk, then the number of rows (int64)."""
        return _int64_copy(self._starts, self._n + 1)

    @property
    def byte_starts(self):
        """The text bytes before every chunk, then all of them (int64)."""
        return _int64_copy(self._byte_starts, self._n + 1)

    def row_bytes(self, Py_ssize_t row):
        """A copy of the bytes of one row."""
        cdef const unsigned char *p
        cdef int64_t n
        self._check_range(row, row + 1)
        _row_at(self._chunks, self._starts, self._n, row, &p, &n)
        return PyBytes_FromStringAndSize(<const char *>p, n)

    def row_address(self, Py_ssize_t row):
        """The address fse reads row from (for the tests: no copy of the text)."""
        cdef const unsigned char *p
        cdef int64_t n
        self._check_range(row, row + 1)
        _row_at(self._chunks, self._starts, self._n, row, &p, &n)
        return <uintptr_t>p

    def span_bytes(self, Py_ssize_t first, Py_ssize_t last):
        """The text bytes of the rows first to last - 1."""
        self._check_range(first, last)
        cdef Py_ssize_t row = first
        cdef _Cursor at
        cdef const unsigned char *p
        cdef int64_t n, total = 0
        cdef int status = 0
        self._readers += 1
        try:
            with nogil:
                _reader_hold()
                if row < last:
                    _seek(&at, self._chunks, self._starts, self._n, row)
                while row < last:
                    if row >= at.end:
                        _seek(&at, self._chunks, self._starts, self._n, row)
                    if fse_arrow_row(at.chunk, row - at.base, &p, &n) != 0:
                        status = 1
                        break
                    total += n
                    row += 1
        finally:
            self._readers -= 1
        if status:
            raise ValueError(f"invalid Arrow offsets or view at row {row}")
        return total

    def scan(self, Py_ssize_t first, Py_ssize_t last, Py_ssize_t chunk_bytes,
             Py_ssize_t chunk_rows=CHUNK_MAX_ROWS):
        """Validate and count the rows first to last - 1 without the GIL, as scan_spans.

        Returns (sentences, tokens, empty, chunks, error_byte, error_sentence): chunks of
        (start, end, first_sentence) in rows of the column and sentences relative to first;
        error_byte is the bad byte within the row, or -1.
        """
        self._check_range(first, last)
        if chunk_bytes < 1:
            raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
        if not 1 <= chunk_rows <= CHUNK_MAX_ROWS:
            raise ValueError(f"chunk_rows must be between 1 and {CHUNK_MAX_ROWS}, got {chunk_rows}")
        cdef Py_ssize_t pos = first, chunk_start = first, chunk_first = 0
        cdef _Cursor at
        cdef Py_ssize_t sentences = 0, tokens = 0, empty = 0, count, error = -1
        cdef int64_t n, in_chunk = 0
        cdef const unsigned char *p
        cdef int status = 0
        cdef _Buffer bounds
        bounds.data = NULL
        bounds.size = 0
        bounds.capacity = 0
        self._readers += 1
        try:
            with nogil:
                _reader_hold()
                if pos < last:
                    _seek(&at, self._chunks, self._starts, self._n, pos)
                while pos < last:
                    if pos >= at.end:
                        _seek(&at, self._chunks, self._starts, self._n, pos)
                    if fse_arrow_row(at.chunk, pos - at.base, &p, &n) != 0:
                        status = _ARROW_INVALID
                        break
                    count = _arrow_tokens(p, n, &error)
                    if error >= 0:
                        break
                    tokens += count
                    if count == 0:
                        empty += 1
                    sentences += 1
                    in_chunk += n
                    pos += 1
                    if (in_chunk >= chunk_bytes or sentences - chunk_first >= chunk_rows) and pos < last:
                        if (
                            _push_i64_nogil(&bounds, chunk_start) != 0
                            or _push_i64_nogil(&bounds, pos) != 0
                            or _push_i64_nogil(&bounds, chunk_first) != 0
                        ):
                            status = ENCODE_NO_MEMORY
                            break
                        chunk_start = pos
                        chunk_first = sentences
                        in_chunk = 0
            if status == ENCODE_NO_MEMORY:
                raise MemoryError("could not grow the list of chunks")
            if status == _ARROW_INVALID:
                raise ValueError(f"invalid Arrow offsets or view at row {pos}")
            if error >= 0:
                return sentences, tokens, empty, [], error, sentences
            if chunk_start < last:
                _push_i64(&bounds, chunk_start)
                _push_i64(&bounds, last)
                _push_i64(&bounds, chunk_first)
            flat = _to_array(&bounds, np.int64).tolist()
        finally:
            free(bounds.data)
            self._readers -= 1
        return sentences, tokens, empty, [tuple(flat[c : c + 3]) for c in range(0, len(flat), 3)], -1, -1

    cdef int _check_encode(self, TextLookup lookup, Py_ssize_t first, Py_ssize_t last,
                           Workspace ws) except -1:
        if lookup._state != TABLE_READY:
            raise RuntimeError("this TextLookup was not built; make a new one")
        self._check_range(first, last)
        if ws.busy:
            raise RuntimeError("a Workspace is used by one thread at a time")
        return 0

    cdef int _encode_rows(self, TextLookup lookup, Py_ssize_t first, Py_ssize_t last,
                          Workspace ws, _ArrowRow *row) noexcept nogil:
        """Stages 1 and 2 of the rows first to last - 1 into the batch buffers of ws; stops
        at the first row that fails, with its status (the caller holds a reader)."""
        cdef Py_ssize_t pos = first, sentence = 0, bad = -1
        cdef _Cursor at
        cdef const unsigned char *p = NULL
        cdef int64_t n = 0
        _reader_hold()
        cdef int status = _start_batch(ws)
        if pos < last:
            _seek(&at, self._chunks, self._starts, self._n, pos)
        while status == ENCODE_OK and pos < last:
            if pos >= at.end:
                _seek(&at, self._chunks, self._starts, self._n, pos)
            if fse_arrow_row(at.chunk, pos - at.base, &p, &n) != 0:
                status = _ARROW_INVALID
                break
            status = lookup._sentence(p, n, ws, &bad)
            if status == ENCODE_OK:
                sentence += 1
                pos += 1
        row.p, row.n, row.bad, row.sentence = p, n, bad, sentence  # once, at the end
        return status

    def encode(self, TextLookup lookup not None, Py_ssize_t first, Py_ssize_t last, first_row,
               Workspace ws not None):
        """The Batch of the rows first to last - 1, which starts at row first_row."""
        from fse._contract import Batch

        global _nogil_sections
        self._check_encode(lookup, first, last, ws)
        cdef _ArrowRow row
        row.p, row.n, row.bad, row.sentence = NULL, 0, -1, 0
        cdef int status
        ws.busy = True
        self._readers += 1
        try:
            _nogil_sections += 1
            with nogil:
                status = self._encode_rows(lookup, first, last, ws, &row)
            _arrow_encode_status(status, row.p, row.n, row.bad, row.sentence, first_row + row.sentence)
            tokens = _to_array(&ws.tokens, np.uint32)
            offsets = _to_array(&ws.offsets, np.int64)
            if not lookup.fasttext:
                return Batch(first_row, offsets, tokens)
            return Batch(
                first_row, offsets, tokens,
                _to_array(&ws.ngram_offsets, np.int64), _to_array(&ws.ngrams, np.uint32),
            )
        finally:
            ws.busy = False
            self._readers -= 1

    def release(self):
        """Release every array now; the column holds no rows afterwards."""
        if self._readers:
            raise RuntimeError("the Arrow column is being read; release it after the run")
        self._free()
