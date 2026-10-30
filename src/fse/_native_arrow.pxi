# Included by _native.pyx: Arrow string columns (utf8, large_utf8, utf8_view) as text
# sources, read in place without the GIL (ARCHITECTURE.md, "Text sources").
#
# An ArrowColumn takes the arrays of an Arrow C stream or of one C array out of their
# PyCapsules (moved: the capsule's release is set to NULL), checks every field it can
# before it forms a pointer from it, rejects nulls, and releases each array once, with the
# GIL, in release() or when it is freed. Rows are numbered across its chunks. What it
# cannot check is the producer's obligation: buffers at least as large as their layout
# implies, unchanged while fse reads them.

from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid
from libc.stdint cimport INT64_MAX, int32_t, uint8_t, uintptr_t


cdef extern from "_native_arrow.h":
    struct ArrowSchema:
        const char *format
        const char *name
        const char *metadata
        int64_t flags
        int64_t n_children
        ArrowSchema **children
        ArrowSchema *dictionary
        void (*release)(ArrowSchema *)
        void *private_data

    struct ArrowArray:
        int64_t length
        int64_t null_count
        int64_t offset
        int64_t n_buffers
        int64_t n_children
        const void **buffers
        ArrowArray **children
        ArrowArray *dictionary
        void (*release)(ArrowArray *)
        void *private_data

    struct ArrowArrayStream:
        int (*get_schema)(ArrowArrayStream *, ArrowSchema *)
        int (*get_next)(ArrowArrayStream *, ArrowArray *)
        const char *(*get_last_error)(ArrowArrayStream *)
        void (*release)(ArrowArrayStream *)
        void *private_data

    ctypedef struct fse_arrow_chunk:
        int kind
        int64_t length
        int64_t first
        const void *index
        const unsigned char *data
        int64_t data_len
        int64_t n_data
        const unsigned char * const *data_ptrs
        const int64_t *sizes

    int FSE_ARROW_UTF8
    int FSE_ARROW_LARGE_UTF8
    int FSE_ARROW_UTF8_VIEW
    int fse_arrow_row(const fse_arrow_chunk *c, int64_t i, const unsigned char **p,
                      int64_t *n) noexcept nogil
    int64_t fse_arrow_first_null(const uint8_t *bits, int64_t offset, int64_t length) noexcept nogil


_ARROW_FORMATS = {b"u": FSE_ARROW_UTF8, b"U": FSE_ARROW_LARGE_UTF8, b"vu": FSE_ARROW_UTF8_VIEW}
_ARROW_BINARY = (b"z", b"Z", b"vz")
_ARROW_NULL_HINT = (
    "fill or drop the nulls first (pyarrow.compute.fill_null(column, ''), "
    "Series.fillna('') in pandas, Series.fill_null('') in Polars)"
)
_ARROW_CAST_HINT = (
    "cast it to strings first (pyarrow.compute.cast(column, pyarrow.string()), "
    "Series.astype(str) in pandas, Series.cast(polars.String) in Polars)"
)


cdef enum:
    _ARROW_OK = 0
    _ARROW_INVALID = 10  # an offset or a view outside its data, beyond the codes of encode


cdef void *_capsule_pointer(capsule, const char *name) except NULL:
    if not PyCapsule_IsValid(capsule, name):
        raise TypeError(f"expected a PyCapsule named {name.decode()!r}, not {capsule!r}")
    return PyCapsule_GetPointer(capsule, name)


cdef bytes _c_string(const char *s, what):
    if s == NULL:
        raise ValueError(f"the Arrow schema has no {what}")
    return <bytes>s


cdef int32_t _int32_at(const char *p) noexcept:
    cdef int32_t value
    memcpy(&value, p, 4)
    return value


cdef object _extension_name(const char *metadata):
    """The value of ARROW:extension:name in Arrow schema metadata, or None."""
    if metadata == NULL:
        return None
    cdef int32_t n = _int32_at(metadata), length
    cdef const char *p = metadata + 4
    cdef bytes key
    if n < 0:
        raise ValueError("the Arrow schema metadata is invalid")
    for _ in range(n):
        length = _int32_at(p)
        if length < 0:
            raise ValueError("the Arrow schema metadata is invalid")
        key = p[4 : 4 + length]
        p += 4 + length
        length = _int32_at(p)
        if length < 0:
            raise ValueError("the Arrow schema metadata is invalid")
        if key == b"ARROW:extension:name":
            return p[4 : 4 + length].decode("utf-8", "replace")
        p += 4 + length
    return None


cdef int _arrow_kind(ArrowSchema *schema) except -1:
    """The layout of a string column, or a TypeError that says what to pass instead."""
    if schema.release == NULL:
        raise ValueError("the Arrow schema was released already")
    fmt = _c_string(schema.format, "format")
    cdef int64_t i
    cdef ArrowSchema *child
    if fmt == b"+s":
        names = []
        if schema.n_children > 0 and schema.children != NULL:
            for i in range(schema.n_children):
                child = schema.children[i]
                if child != NULL and child.format != NULL and <bytes>child.format in _ARROW_FORMATS:
                    names.append((<bytes>child.name).decode("utf-8", "replace") if child.name != NULL else "")
        found = f"its string columns are {names}" if names else "it has no string column"
        raise TypeError(
            "texts must be one Arrow string column, not a table of columns: select the "
            f"column, such as table['text'] ({found})"
        )
    if schema.dictionary != NULL:
        raise TypeError(f"the Arrow column is dictionary-encoded: {_ARROW_CAST_HINT}")
    if fmt in _ARROW_BINARY:
        raise TypeError(f"the Arrow column holds binary values, not strings: {_ARROW_CAST_HINT}")
    if fmt not in _ARROW_FORMATS:
        raise TypeError(f"the Arrow column has the format {fmt.decode('utf-8', 'replace')!r}, not a "
                        f"string format: {_ARROW_CAST_HINT}")
    extension = _extension_name(schema.metadata)
    if extension is not None:
        raise TypeError(f"the Arrow column has the extension type {extension!r}: pass its "
                        "storage, such as column.storage in pyarrow")
    if schema.n_children != 0:
        raise ValueError("an Arrow string column has no children")
    return _ARROW_FORMATS[fmt]


cdef Py_ssize_t _arrow_tokens(
    const unsigned char *p, Py_ssize_t n, Py_ssize_t *error
) noexcept nogil:
    """The str.split() tokens of p[0:n], as _count_tokens counts them; sets error to the
    first bad byte. A copy of its own: a second caller of _count_tokens changes how the
    compiler builds scan_spans, whose machine code this part leaves as it is."""
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t tokens = 0
    cdef bint inside = False
    cdef int m
    while i < n:
        m = _utf8_len(p, i, n)
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


cdef inline Py_ssize_t _chunk_of(const int64_t *starts, Py_ssize_t n, int64_t row) noexcept nogil:
    """The chunk that holds row: the last k with starts[k] <= row (starts has n + 1)."""
    cdef Py_ssize_t lo = 0, hi = n - 1, mid
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= row:
            lo = mid
        else:
            hi = mid - 1
    return lo


cdef struct _Cursor:
    const fse_arrow_chunk *chunk
    int64_t base  # the first row of the chunk
    int64_t end  # the first row after it


cdef inline void _seek(_Cursor *at, const fse_arrow_chunk *chunks, const int64_t *starts,
                       Py_ssize_t n, int64_t row) noexcept nogil:
    """Point at to the chunk that holds row (the walk changes chunk only at their ends)."""
    cdef Py_ssize_t k = _chunk_of(starts, n, row)
    at.chunk = &chunks[k]
    at.base = starts[k]
    at.end = starts[k + 1]


cdef int _arrow_encode_status(int status, const unsigned char *p, int64_t n, Py_ssize_t bad,
                              Py_ssize_t sentence, Py_ssize_t row) except -1:
    if status == ENCODE_OK:
        return 0
    if status == ENCODE_NO_MEMORY:
        raise MemoryError("could not grow the buffers of the Workspace")
    if status == _ARROW_INVALID:
        raise ValueError(f"invalid Arrow offsets or view at row {row}")
    if status == ENCODE_TOO_LONG:
        raise ValueError(
            f"row {row} has {n} bytes, more than {MAX_SENTENCE_BYTES}; fse reads one sentence "
            "per string"
        )
    if status == ENCODE_BAD_UTF8:
        _raise_decode_error(
            PyBytes_FromStringAndSize(<const char *>p, n), bad,
            f" in sentence {sentence} of the range (byte {bad} of row {row})",
        )
    raise AssertionError(f"unknown status {status}")


cdef int _stream_error(ArrowArrayStream *stream, int code, what) except -1:
    cdef const char *message = stream.get_last_error(stream) if stream.get_last_error != NULL else NULL
    text = (<bytes>message).decode("utf-8", "replace") if message != NULL else f"error code {code}"
    raise OSError(code, f"the Arrow stream failed in {what}: {text}")
