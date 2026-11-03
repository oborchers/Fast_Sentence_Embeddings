# Primitive int64 columns only. No table projection and no producer dependency.

cdef void _id_schema(ArrowSchema *schema) except *:
    if (_c_string(schema.format, "format") != b"l" or schema.n_children != 0
            or schema.dictionary != NULL or _extension_name(schema.metadata) is not None):
        raise TypeError("doc_ids must be a primitive Arrow int64 column without nulls")


cdef class ArrowIds:
    cdef ArrowArray *_arrays
    cdef int64_t *_starts
    cdef Py_ssize_t _n
    cdef readonly int64_t rows
    cdef bint _released
    cdef int _readers

    def __dealloc__(self):
        self._free()

    cdef void _free(self) noexcept:
        cdef ArrowArray *arrays = self._arrays
        cdef Py_ssize_t k, n = self._n
        self._arrays = NULL
        self._n = 0
        self._released = True
        free(self._starts)
        self._starts = NULL
        for k in range(n):
            if arrays[k].release != NULL:
                arrays[k].release(&arrays[k])
        free(arrays)

    def release(self):
        if self._readers:
            raise RuntimeError("the Arrow document IDs are being read")
        self._free()

    cdef void _take(self, ArrowArray *a) except *:
        cdef ArrowArray *arrays
        cdef int64_t *starts
        cdef const uint8_t *bits
        if a.release == NULL:
            raise ValueError("the Arrow ID array was released already")
        if (a.length < 0 or a.offset < 0 or a.offset > INT64_MAX - a.length
                or a.offset + a.length > PY_SSIZE_T_MAX // 8
                or a.length > INT64_MAX - self.rows):
            raise ValueError("invalid Arrow ID length or offset")
        if (a.n_children != 0 or a.dictionary != NULL or a.n_buffers != 2
                or a.buffers == NULL or a.null_count < -1 or a.null_count > a.length):
            raise ValueError("invalid primitive Arrow ID buffers or null count")
        if a.length:
            bits = <const uint8_t *>a.buffers[0]
            if (bits == NULL and a.null_count != 0) or (bits != NULL and
                    fse_arrow_first_null(bits, a.offset, a.length) >= 0):
                raise ValueError("doc_ids Arrow column must have no null values")
            if a.buffers[1] == NULL:
                raise ValueError("the Arrow ID array has no data buffer")
        if self._n >= PY_SSIZE_T_MAX // sizeof(ArrowArray) - 1:
            raise OverflowError("too many Arrow ID chunks")
        arrays = <ArrowArray *>realloc(self._arrays, (self._n + 1) * sizeof(ArrowArray))
        if arrays == NULL:
            raise MemoryError("could not retain Arrow IDs")
        self._arrays = arrays
        starts = <int64_t *>realloc(self._starts, (self._n + 2) * sizeof(int64_t))
        if starts == NULL:
            raise MemoryError("could not retain Arrow ID offsets")
        self._starts = starts
        self._starts[self._n] = self.rows
        self.rows += a.length
        self._starts[self._n + 1] = self.rows
        memcpy(&self._arrays[self._n], a, sizeof(ArrowArray))
        a.release = NULL
        self._n += 1

    @staticmethod
    def from_array(schema_capsule, array_capsule):
        cdef ArrowSchema *schema = <ArrowSchema *>_capsule_pointer(schema_capsule, b"arrow_schema")
        cdef ArrowArray *array = <ArrowArray *>_capsule_pointer(array_capsule, b"arrow_array")
        _id_schema(schema)
        cdef ArrowIds ids = ArrowIds()
        ids._take(array)
        return ids

    @staticmethod
    def from_stream(capsule):
        cdef ArrowArrayStream *source = <ArrowArrayStream *>_capsule_pointer(capsule, b"arrow_array_stream")
        if source.release == NULL:
            raise ValueError("the Arrow ID stream was released already")
        cdef ArrowIds ids = ArrowIds()
        cdef ArrowArrayStream stream
        cdef ArrowSchema schema
        cdef ArrowArray array
        cdef bint have_schema = False
        cdef int code
        memcpy(&stream, source, sizeof(ArrowArrayStream))
        source.release = NULL
        try:
            if stream.get_schema == NULL or stream.get_next == NULL:
                raise ValueError("the Arrow ID stream has no schema or next callback")
            code = stream.get_schema(&stream, &schema)
            if code != 0:
                _stream_error(&stream, code, "get_schema")
            have_schema = True
            _id_schema(&schema)
            while True:
                array.release = NULL
                code = stream.get_next(&stream, &array)
                if code != 0:
                    _stream_error(&stream, code, "get_next")
                if array.release == NULL:
                    break
                try:
                    ids._take(&array)
                finally:
                    if array.release != NULL:
                        array.release(&array)
        finally:
            if have_schema and schema.release != NULL:
                schema.release(&schema)
            if stream.release != NULL:
                stream.release(&stream)
        return ids

    def block(self, int64_t begin, int64_t end):
        if self._released or not 0 <= begin <= end <= self.rows:
            raise ValueError("doc_ids block is outside the open column")
        self._readers += 1
        try:
            return self._block(begin, end)
        finally:
            self._readers -= 1

    cdef object _block(self, int64_t begin, int64_t end):
        cdef np.ndarray out = np.empty(end - begin, dtype="<i8")
        cdef Py_ssize_t k
        cdef int64_t first, last
        if begin < end:
            k = _chunk_of(self._starts, self._n, begin)
            while k < self._n and self._starts[k] < end:
                first, last = max(begin, self._starts[k]), min(end, self._starts[k + 1])
                if first < last:
                    memcpy(<char *>np.PyArray_DATA(out) + 8 * (first - begin),
                           <const char *>self._arrays[k].buffers[1] + 8 *
                           (self._arrays[k].offset + first - self._starts[k]), 8 * (last - first))
                k += 1
        out.setflags(write=False)
        return out
