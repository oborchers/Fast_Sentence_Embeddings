# Included by _native.pyx: token lists joined into one UTF-8 buffer, for TokenListSource.
#
# One pass over exact lists, tuples and str with the GIL held: every token is checked and
# written at once into a C buffer (reading each of millions of scattered str objects once
# is what the time goes to). Items are read as borrowed pointers, and no Python code can
# run while they are held: no calls, no reference counts dropped, and no Python object is
# allocated, so the garbage collector cannot run a finalizer. The bytes object is made
# from the buffer after the pass.

from cpython.object cimport PyObject
from libc.stdint cimport uint32_t


cdef extern from *:
    """
    #if PY_VERSION_HEX < 0x030C0000
    #define FSE_UNICODE_READY(o) PyUnicode_READY(o)
    #else
    #define FSE_UNICODE_READY(o) 0
    #endif
    """
    int FSE_UNICODE_READY(PyObject *o)  # 3.11: legacy strings get their canonical form


cdef extern from "Python.h":
    ctypedef unsigned char Py_UCS1
    ctypedef unsigned short Py_UCS2
    ctypedef uint32_t Py_UCS4_t "Py_UCS4"
    bint PyList_CheckExact(PyObject *o)
    bint PyTuple_CheckExact(PyObject *o)
    bint PyUnicode_CheckExact(PyObject *o)
    Py_ssize_t PyList_GET_SIZE(PyObject *o)
    Py_ssize_t PyTuple_GET_SIZE(PyObject *o)
    PyObject *PyList_GET_ITEM(PyObject *o, Py_ssize_t i)
    PyObject *PyTuple_GET_ITEM(PyObject *o, Py_ssize_t i)
    bint PyUnicode_IS_ASCII(PyObject *o)
    int PyUnicode_KIND(PyObject *o)
    void *PyUnicode_DATA(PyObject *o)
    Py_ssize_t PyUnicode_GET_LENGTH(PyObject *o)


# What _append_token and _append_sentence give besides 0.
cdef enum:
    _INVALID = -1  # an empty token, whitespace or a lone surrogate: the Python path raises
    _UNSUPPORTED = -2  # an object of another type: the Python path decides
    _FAILED = -3  # an exception is set


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


cdef inline Py_ssize_t _items(PyObject *o) noexcept:
    return PyList_GET_SIZE(o) if PyList_CheckExact(o) else PyTuple_GET_SIZE(o)


cdef inline PyObject *_item(PyObject *o, Py_ssize_t i) noexcept:
    return PyList_GET_ITEM(o, i) if PyList_CheckExact(o) else PyTuple_GET_ITEM(o, i)


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


cdef int _append_token(_Buffer *b, PyObject *token) except? _FAILED:
    """Append the UTF-8 of a token to b: 0, _INVALID or _UNSUPPORTED."""
    cdef Py_ssize_t i, n
    cdef int kind
    cdef const void *data
    cdef char *out
    cdef uint32_t c
    if not PyUnicode_CheckExact(token):
        return _UNSUPPORTED
    if FSE_UNICODE_READY(token) < 0:
        return _FAILED
    n = PyUnicode_GET_LENGTH(token)
    if n == 0:
        return _INVALID
    data = PyUnicode_DATA(token)
    if PyUnicode_IS_ASCII(token):
        for i in range(n):
            if _is_space_code((<const Py_UCS1 *>data)[i]):
                return _INVALID
        _reserve(b, n)
        memcpy(b.data + b.size, data, n)
        b.size += n
        return 0
    if n > PY_SSIZE_T_MAX // 4:
        raise MemoryError("the joined token lists would exceed the address space")
    _reserve(b, 4 * n)  # at most 4 bytes per code point
    kind = PyUnicode_KIND(token)
    out = b.data + b.size
    for i in range(n):
        c = _code_at(kind, data, i)
        if _is_space_code(c) or 0xD800 <= c <= 0xDFFF:
            return _INVALID
        out = _put(out, c)
    b.size = out - b.data
    return 0


cdef int _append_sentence(_Buffer *b, PyObject *sentence) except? _FAILED:
    """Append the tokens of a sentence to b, one space between them: 0 or a status."""
    cdef Py_ssize_t t
    cdef int status
    if not (PyList_CheckExact(sentence) or PyTuple_CheckExact(sentence)):
        return _UNSUPPORTED
    for t in range(_items(sentence)):
        if t > 0:
            _reserve(b, 1)
            b.data[b.size] = 32
            b.size += 1
        status = _append_token(b, _item(sentence, t))
        if status < 0:
            return status
    return 0


def join_token_lists(sentences):
    """The sentences, each a list or tuple of str tokens, as the UTF-8 bytes of the tokens
    joined by one space per sentence, and the int64 offsets of the sentences in them.

    Returns None if an object is not of these exact types (subclasses, a numpy array of
    sentences), and False if a token is empty, holds whitespace (str.isspace) or a lone
    surrogate; the caller then takes the Python path, which decides and raises its errors.
    """
    cdef PyObject *outer = <PyObject *>sentences
    cdef Py_ssize_t i, n
    cdef int status = 0
    cdef _Buffer text
    cdef _Buffer offsets
    if not (PyList_CheckExact(outer) or PyTuple_CheckExact(outer)):
        return None
    n = _items(outer)
    text.data = offsets.data = NULL
    text.size = text.capacity = offsets.size = offsets.capacity = 0
    try:
        _reserve(&offsets, (n + 1) * sizeof(np.int64_t))  # n items fit in memory already
        _push_i64(&offsets, 0)
        for i in range(n):
            status = _append_sentence(&text, _item(outer, i))
            if status < 0:
                break
            _push_i64(&offsets, text.size)
        if status < 0:
            return None if status == _UNSUPPORTED else False
        # The items are not read any more, so Python code may run from here.
        data = PyBytes_FromStringAndSize(text.data, text.size)
        starts = np.empty(n + 1, dtype=np.int64)
        memcpy(np.PyArray_DATA(starts), offsets.data, offsets.size)
    finally:
        free(text.data)
        free(offsets.data)
    return data, starts
