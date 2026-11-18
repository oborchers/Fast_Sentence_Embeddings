# Included after _native_strings.pxi. All Unicode inspection and ownership changes
# happen with the GIL. Tasks without it read only immutable character bytes.

from libc.stdint cimport int64_t

cdef extern from "Python.h":
    object PyTuple_New(Py_ssize_t n)
    void PyTuple_SET_ITEM(PyObject *o, Py_ssize_t i, PyObject *item)
    void Py_IncRef(PyObject *o)

cdef int _copy_fail_stage = 0
cdef object _copy_between = None
cdef object _copy_check = None


cdef struct _CopyRow:
    const char *data
    Py_ssize_t length


cdef struct _CopyRange:
    _CopyRow *rows
    const int64_t *offsets
    char *out
    Py_ssize_t first, last
    bint *copied
    Py_ssize_t *calls


cdef class StringCopy:
    """Opaque owner of a snapshot, descriptors and bytes written only by tasks."""
    cdef object __weakref__
    cdef object snapshot, data, offsets
    cdef _CopyRow *rows
    cdef Py_ssize_t finished, copy_calls

    def __dealloc__(self):
        fse_join_free(self.rows)

    @property
    def positions(self):
        return self.offsets


cdef object _copy_snapshot(texts, Py_ssize_t n):
    cdef PyObject *item
    cdef Py_ssize_t i
    if _copy_fail_stage == 1:
        raise MemoryError("could not allocate the snapshot of the strings")
    if PyTuple_CheckExact(<PyObject *>texts):
        return texts
    snapshot = PyTuple_New(n)
    if _copy_between is not None:
        _copy_between()
    if PyList_GET_SIZE(<PyObject *>texts) != n:
        return None
    # No allocation or Python execution between the size recheck and the last INCREF.
    for i in range(n):
        item = PyList_GET_ITEM(<PyObject *>texts, i)
        Py_IncRef(item)
        PyTuple_SET_ITEM(<PyObject *>snapshot, i, item)
    return snapshot


def copy_strings(texts, Py_ssize_t max_rows):
    """Prepare exact ready ASCII rows for task-side copying, else return None."""
    cdef Py_ssize_t n, i, length, total = 0
    cdef PyObject *text
    cdef int64_t[::1] offsets
    cdef StringCopy owner
    if not (PyList_CheckExact(<PyObject *>texts) or PyTuple_CheckExact(<PyObject *>texts)):
        return None
    n = _items(<PyObject *>texts)
    if not 1 <= n < max_rows:
        return None
    if n >= PY_SSIZE_T_MAX // <Py_ssize_t>sizeof(int64_t):
        raise MemoryError("the offsets of the strings would exceed the address space")
    owner = StringCopy()
    owner.snapshot = _copy_snapshot(texts, n)
    if owner.snapshot is None:
        return None
    if n > PY_SSIZE_T_MAX // <Py_ssize_t>sizeof(_CopyRow):
        raise MemoryError("the descriptors of the strings would exceed the address space")
    if _copy_fail_stage != 2:
        owner.rows = <_CopyRow *>fse_join_calloc(n, sizeof(_CopyRow))
    if owner.rows == NULL:
        raise MemoryError("could not allocate the descriptors of the strings")
    if _copy_fail_stage == 3:
        raise MemoryError("could not allocate the offsets of the strings")
    owner.offsets = np.empty(n + 1, dtype=np.int64)
    offsets = owner.offsets
    offsets[0] = 0
    for i in range(n):
        text = PyTuple_GET_ITEM(<PyObject *>owner.snapshot, i)
        if not PyUnicode_CheckExact(text) or not FSE_UNICODE_IS_READY(text):
            return None
        if not PyUnicode_IS_ASCII(text):
            return None
        length = PyUnicode_GET_LENGTH(text)
        if length > PY_SSIZE_T_MAX - total:
            raise MemoryError("the joined strings would exceed the address space")
        owner.rows[i].data = <const char *>PyUnicode_DATA(text)
        owner.rows[i].length = length
        total += length
        offsets[i + 1] = total
        assert offsets[i + 1] - offsets[i] == owner.rows[i].length
    if _copy_fail_stage == 4:
        raise MemoryError("could not allocate the bytes of the strings")
    owner.data = PyBytes_FromStringAndSize(NULL, total)
    owner.offsets.flags.writeable = False
    return owner


cdef void _copy_range(_CopyRange *span) noexcept nogil:
    cdef Py_ssize_t i
    if span == NULL or span.copied[0]:
        return
    for i in range(span.first, span.last):
        if span.rows[i].length:
            memcpy(span.out + span.offsets[i], span.rows[i].data, span.rows[i].length)
    span.copied[0] = True
    span.calls[0] += 1


def _string_copy_between(callback):
    """Test only: run after allocating a list snapshot, before rechecking its size."""
    global _copy_between
    previous, _copy_between = _copy_between, callback
    return previous


def _string_copy_fail_stage(int stage):
    """Test only: fail snapshot, descriptors, offsets or bytes (1 to 4; 0 disables)."""
    global _copy_fail_stage
    previous, _copy_fail_stage = _copy_fail_stage, stage
    return previous


def _test_string_copy_result(StringCopy owner not None):
    """Test only: inspect bytes only after all rows have been copied by their tasks."""
    if owner.finished != len(owner.snapshot):
        raise AssertionError("the string tasks have not copied every row")
    return owner.data, owner.offsets


def _string_copy_check(callback):
    """Test only: run immediately before a copied attempt checks its encode range."""
    global _copy_check
    previous, _copy_check = _copy_check, callback
    return previous


cdef void _check_string_copy(StringCopy owner) except *:
    # A default-constructed extension object has NULL descriptors and None fields.
    # Reject it before any PyArray_DATA/PyBytes_AS_STRING macro sees those fields.
    if (owner.rows == NULL or not isinstance(owner.snapshot, tuple)
            or type(owner.data) is not bytes or not isinstance(owner.offsets, np.ndarray)):
        raise ValueError("StringCopy is not initialized")
    if (owner.offsets.dtype != np.int64 or owner.offsets.ndim != 1
            or len(owner.offsets) != len(owner.snapshot) + 1):
        raise ValueError("StringCopy is not initialized")


def _test_string_copy_calls(StringCopy owner not None):
    """Test only: actual native copies completed by this owner's task attempts."""
    return owner.copy_calls
