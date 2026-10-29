# Included by _native.pyx: token lists joined into one UTF-8 buffer, for TokenLists.
#
# One pass over exact lists, tuples and str: every token is checked and written at once
# into a C buffer, in parts of the sentences that native threads run in parallel (reading
# each of millions of scattered str objects once is what the time goes to). The calling
# thread holds the GIL all along, also while it waits for the threads, so no Python code
# runs: items are read as borrowed pointers, nothing calls Python, drops a reference or
# allocates a Python object, and the threads only read memory. The bytes object is made
# from the buffers after the pass. (Free-threaded builds, where the GIL does not keep
# other threads out, would need the input locked; fse does not build them.)

from cpython.object cimport PyObject
from libc.stdint cimport uint32_t


cdef extern from *:
    """
    #if PY_VERSION_HEX < 0x030C0000
    #define FSE_UNICODE_IS_READY(o) PyUnicode_IS_READY(o)
    #else
    #define FSE_UNICODE_IS_READY(o) 1
    #endif
    """
    bint FSE_UNICODE_IS_READY(PyObject *o) noexcept nogil  # 3.11: legacy strings are not


cdef extern from "Python.h":
    ctypedef unsigned char Py_UCS1
    ctypedef unsigned short Py_UCS2
    ctypedef uint32_t Py_UCS4_t "Py_UCS4"
    bint PyList_CheckExact(PyObject *o) noexcept nogil
    bint PyTuple_CheckExact(PyObject *o) noexcept nogil
    bint PyUnicode_CheckExact(PyObject *o) noexcept nogil
    Py_ssize_t PyList_GET_SIZE(PyObject *o) noexcept nogil
    Py_ssize_t PyTuple_GET_SIZE(PyObject *o) noexcept nogil
    PyObject *PyList_GET_ITEM(PyObject *o, Py_ssize_t i) noexcept nogil
    PyObject *PyTuple_GET_ITEM(PyObject *o, Py_ssize_t i) noexcept nogil
    bint PyUnicode_IS_ASCII(PyObject *o) noexcept nogil
    int PyUnicode_KIND(PyObject *o) noexcept nogil
    void *PyUnicode_DATA(PyObject *o) noexcept nogil
    Py_ssize_t PyUnicode_GET_LENGTH(PyObject *o) noexcept nogil
    char *PyBytes_AS_STRING(object o)


cdef extern from "_native_threads.h":
    ctypedef void (*fse_part)(void *arg, int part) noexcept nogil
    int fse_parallel(int n, fse_part fn, void *arg) nogil


# What the appends give besides 0; the worst status of any part decides.
cdef enum:
    _INVALID = -1  # an empty token, whitespace or a lone surrogate: the Python path raises
    _UNSUPPORTED = -2  # an object of another type: the Python path decides
    _NO_MEMORY = -3
    _PART_SENTENCES = 4096  # sentences a thread gets at least


cdef struct _Part:
    PyObject *outer
    Py_ssize_t first, last  # its sentences
    _Buffer text
    _Buffer ends  # int64 ends of its sentences in its text
    int status


cdef inline bint _is_space_code(uint32_t c) noexcept nogil:
    """Whether c is in W (str.isspace), the code points of _is_space."""
    if c < 0x80:
        return (9 <= c <= 13) or (28 <= c <= 32)
    if c < 0x1680:
        return c == 0x85 or c == 0xA0
    return (
        c == 0x1680 or 0x2000 <= c <= 0x200A or c == 0x2028 or c == 0x2029
        or c == 0x202F or c == 0x205F or c == 0x3000
    )


cdef inline uint32_t _code_at(int kind, const void *data, Py_ssize_t i) noexcept nogil:
    if kind == 1:
        return (<const Py_UCS1 *>data)[i]
    if kind == 2:
        return (<const Py_UCS2 *>data)[i]
    return (<const Py_UCS4_t *>data)[i]


cdef inline Py_ssize_t _items(PyObject *o) noexcept nogil:
    return PyList_GET_SIZE(o) if PyList_CheckExact(o) else PyTuple_GET_SIZE(o)


cdef inline PyObject *_item(PyObject *o, Py_ssize_t i) noexcept nogil:
    return PyList_GET_ITEM(o, i) if PyList_CheckExact(o) else PyTuple_GET_ITEM(o, i)


cdef int _grow(_Buffer *b, Py_ssize_t extra) noexcept nogil:
    """_reserve without the GIL: 0, or _NO_MEMORY."""
    cdef Py_ssize_t capacity
    cdef char *data
    if extra > PY_SSIZE_T_MAX - b.size:
        return _NO_MEMORY
    if b.size + extra <= b.capacity:
        return 0
    capacity = b.capacity * 2 if b.capacity <= PY_SSIZE_T_MAX // 2 else PY_SSIZE_T_MAX
    capacity = max(b.size + extra, capacity, 4096)
    data = <char *>realloc(b.data, capacity)
    if data == NULL:
        return _NO_MEMORY
    b.data = data
    b.capacity = capacity
    return 0


cdef inline char *_put(char *out, uint32_t c) noexcept nogil:
    """Write the UTF-8 of the code point c (not a surrogate); the end of what it wrote."""
    if c < 0x80:
        out[0] = <char>c
        return out + 1
    if c < 0x800:
        out[0] = <char>(0xC0 | (c >> 6))
        out[1] = <char>(0x80 | (c & 0x3F))
        return out + 2
    if c < 0x10000:
        out[0] = <char>(0xE0 | (c >> 12))
        out[1] = <char>(0x80 | ((c >> 6) & 0x3F))
        out[2] = <char>(0x80 | (c & 0x3F))
        return out + 3
    out[0] = <char>(0xF0 | (c >> 18))
    out[1] = <char>(0x80 | ((c >> 12) & 0x3F))
    out[2] = <char>(0x80 | ((c >> 6) & 0x3F))
    out[3] = <char>(0x80 | (c & 0x3F))
    return out + 4


cdef int _append_token(_Buffer *b, PyObject *token) noexcept nogil:
    """Append the UTF-8 of a token to b: 0 or a status."""
    cdef Py_ssize_t i, n
    cdef int kind
    cdef const void *data
    cdef char *out
    cdef uint32_t c
    if not PyUnicode_CheckExact(token) or not FSE_UNICODE_IS_READY(token):
        return _UNSUPPORTED
    n = PyUnicode_GET_LENGTH(token)
    if n == 0:
        return _INVALID
    data = PyUnicode_DATA(token)
    if PyUnicode_IS_ASCII(token):
        for i in range(n):
            if _is_space_code((<const Py_UCS1 *>data)[i]):
                return _INVALID
        if _grow(b, n) < 0:
            return _NO_MEMORY
        memcpy(b.data + b.size, data, n)
        b.size += n
        return 0
    if n > PY_SSIZE_T_MAX // 4 or _grow(b, 4 * n) < 0:  # at most 4 bytes per code point
        return _NO_MEMORY
    kind = PyUnicode_KIND(token)
    out = b.data + b.size
    for i in range(n):
        c = _code_at(kind, data, i)
        if _is_space_code(c) or 0xD800 <= c <= 0xDFFF:
            return _INVALID
        out = _put(out, c)
    b.size = out - b.data
    return 0


cdef int _append_sentence(_Buffer *b, PyObject *sentence) noexcept nogil:
    """Append the tokens of a sentence to b, one space between them: 0 or a status."""
    cdef Py_ssize_t t
    cdef int status
    if not (PyList_CheckExact(sentence) or PyTuple_CheckExact(sentence)):
        return _UNSUPPORTED
    for t in range(_items(sentence)):
        if t > 0:
            if _grow(b, 1) < 0:
                return _NO_MEMORY
            b.data[b.size] = 32
            b.size += 1
        status = _append_token(b, _item(sentence, t))
        if status < 0:
            return status
    return 0


cdef void _run_part(void *arg, int index) noexcept nogil:
    cdef _Part *part = (<_Part *>arg) + index
    cdef Py_ssize_t i
    for i in range(part.first, part.last):
        part.status = _append_sentence(&part.text, _item(part.outer, i))
        if part.status == 0 and _grow(&part.ends, sizeof(np.int64_t)) < 0:
            part.status = _NO_MEMORY
        if part.status < 0:
            return
        (<np.int64_t *>(part.ends.data + part.ends.size))[0] = part.text.size
        part.ends.size += sizeof(np.int64_t)


cdef object _joined(_Part *parts, int n_parts, Py_ssize_t n):
    """The bytes and the offsets of the finished parts."""
    cdef Py_ssize_t total = 0, at = 0, k = 1, j
    cdef int p
    cdef np.int64_t[::1] offsets
    cdef char *out
    for p in range(n_parts):
        if parts[p].text.size > PY_SSIZE_T_MAX - total:
            raise MemoryError("the joined token lists would exceed the address space")
        total += parts[p].text.size
    data = PyBytes_FromStringAndSize(NULL, total)
    starts = np.empty(n + 1, dtype=np.int64)
    offsets = starts
    out = PyBytes_AS_STRING(data)
    offsets[0] = 0
    for p in range(n_parts):
        if parts[p].text.size:
            memcpy(out + at, parts[p].text.data, parts[p].text.size)
        for j in range(parts[p].ends.size // <Py_ssize_t>sizeof(np.int64_t)):
            offsets[k] = at + (<np.int64_t *>parts[p].ends.data)[j]
            k += 1
        at += parts[p].text.size
    return data, starts


def join_token_lists(sentences, int threads=1):
    """The sentences, each a list or tuple of str tokens, as the UTF-8 bytes of the tokens
    joined by one space per sentence, and the int64 offsets of the sentences in them,
    read by up to ``threads`` native threads.

    Returns None if an object is not of these exact types (subclasses, a numpy array of
    sentences, a legacy string of 3.11), and False if a token is empty, holds whitespace
    (str.isspace) or a lone surrogate; the caller then takes the Python path, which decides
    and raises its errors. Each part stops at its first problem, so with both kinds in the
    input the verdict depends on how it was split; False still always means an invalid
    token of an exact str, on which the Python path raises.
    """
    cdef PyObject *outer = <PyObject *>sentences
    cdef Py_ssize_t n, size
    cdef int p, n_parts, status = 0
    cdef _Part *parts
    if threads < 1:
        raise ValueError(f"threads must be at least 1, not {threads}")
    if not (PyList_CheckExact(outer) or PyTuple_CheckExact(outer)):
        return None
    n = _items(outer)
    n_parts = <int>max(1, min(threads, n // _PART_SENTENCES))
    parts = <_Part *>calloc(n_parts, sizeof(_Part))
    if parts == NULL:
        raise MemoryError("could not allocate the parts of the token lists")
    try:
        size = n // n_parts
        for p in range(n_parts):
            parts[p].outer = outer
            parts[p].first = p * size
            parts[p].last = n if p == n_parts - 1 else (p + 1) * size
        if fse_parallel(n_parts, _run_part, parts) < 0:
            raise MemoryError("could not start the threads for the token lists")
        for p in range(n_parts):
            status = min(status, parts[p].status)
        if status == _NO_MEMORY:
            raise MemoryError("could not grow the buffer of the token lists")
        if status < 0:
            return False if status == _INVALID else None
        # The items are not read any more, so Python code may run from here.
        return _joined(parts, n_parts, n)
    finally:
        for p in range(n_parts):
            free(parts[p].text.data)
            free(parts[p].ends.data)
        free(parts)
