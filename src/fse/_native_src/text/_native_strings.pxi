# Included after _native_lists.pxi. Two passes over exact strings, one final buffer.
# Allocate offsets and acquire their view before fixing the input bounds: allocation
# may run Python finalizers. After rechecking the length, no Python can run until
# every borrowed input pointer has been read. PyBytes allocation does not run GC.
# Never inspect the UTF-8 cache, which other native code may populate.

cdef unsigned _string_second_starts = 0
cdef int _string_fail_stage = 0
cdef int _string_mismatch = 0
cdef int _string_last_parts = 0  # Test-only observation, never a selection input.
cdef bint _string_last_estimated = False

cdef enum:
    _MISMATCH = -4
    _STRING_PART_BYTES = 512 * 1024
    _STRING_SIZE_ROWS = 65536


cdef int _string_parts(PyObject *outer, Py_ssize_t n, int threads) noexcept nogil:
    """Bounded UTF-8 upper estimate; actual sizing still validates every row."""
    global _string_last_estimated
    cdef Py_ssize_t i, length, estimate = 0, cap = PY_SSIZE_T_MAX
    cdef int weight, kind
    cdef PyObject *text
    cdef int by_rows = <int>max(1, min(threads, n // _PART_SENTENCES))
    if threads == 1 or not n:
        return 1
    if n > _STRING_SIZE_ROWS or by_rows == threads:
        return by_rows
    if threads <= PY_SSIZE_T_MAX // _STRING_PART_BYTES:
        cap = <Py_ssize_t>threads * _STRING_PART_BYTES
    _string_last_estimated = True
    for i in range(n):
        text = _item(outer, i)
        if not PyUnicode_CheckExact(text) or not FSE_UNICODE_IS_READY(text):
            return by_rows
        length = PyUnicode_GET_LENGTH(text)
        kind = PyUnicode_KIND(text)
        weight = 1 if PyUnicode_IS_ASCII(text) else (2 if kind == 1 else (3 if kind == 2 else 4))
        # Divide before multiplying: even the upper estimate must never overflow.
        if length > (cap - estimate) // weight:
            estimate = cap
            break
        estimate += length * weight
        if estimate == cap:
            break
    return <int>max(1, min(threads, n, max(by_rows, estimate // _STRING_PART_BYTES)))


cdef struct _StringPart:
    PyObject *outer
    Py_ssize_t first, last, total, base
    np.int64_t *offsets
    char *out
    int status


cdef inline Py_ssize_t _utf8_width(uint32_t c) noexcept nogil:
    return 1 if c < 0x80 else (2 if c < 0x800 else (3 if c < 0x10000 else 4))


cdef Py_ssize_t _string_size(PyObject *text) noexcept nogil:
    cdef Py_ssize_t i, n, total = 0
    cdef int kind
    cdef const void *data
    cdef uint32_t c
    if not PyUnicode_CheckExact(text) or not FSE_UNICODE_IS_READY(text):
        return _UNSUPPORTED
    n = PyUnicode_GET_LENGTH(text)
    if PyUnicode_IS_ASCII(text):
        return n
    if n > PY_SSIZE_T_MAX // 4:
        return _NO_MEMORY
    kind = PyUnicode_KIND(text)
    data = PyUnicode_DATA(text)
    for i in range(n):
        c = _code_at(kind, data, i)
        if 0xD800 <= c <= 0xDFFF:
            return _INVALID
        total += _utf8_width(c)
    return total


cdef void _size_strings(void *arg, int index) noexcept nogil:
    cdef _StringPart *part = (<_StringPart *>arg) + index
    cdef Py_ssize_t i, size
    for i in range(part.first, part.last):
        size = _string_size(_item(part.outer, i))
        if size < 0:
            part.status = <int>size
            return
        if size > PY_SSIZE_T_MAX - part.total:
            part.status = _NO_MEMORY
            return
        part.total += size
        part.offsets[i + 1] = part.total


cdef int _write_string(PyObject *text, char *out, Py_ssize_t span) noexcept nogil:
    """Check each write against its reserved span before touching the target."""
    cdef Py_ssize_t i, n = PyUnicode_GET_LENGTH(text), width, at = 0
    cdef int kind
    cdef const void *data = PyUnicode_DATA(text)
    cdef uint32_t c
    if PyUnicode_IS_ASCII(text):
        if n != span:
            return _MISMATCH
        if n:
            memcpy(out, data, n)
        return 0
    kind = PyUnicode_KIND(text)
    for i in range(n):
        c = _code_at(kind, data, i)
        width = _utf8_width(c)
        if width > span - at:
            return _MISMATCH
        _put(out + at, c)
        at += width
    return 0 if at == span else _MISMATCH


cdef void _write_strings(void *arg, int index) noexcept nogil:
    cdef _StringPart *part = (<_StringPart *>arg) + index
    cdef Py_ssize_t i, end, at = 0
    for i in range(part.first, part.last):
        end = part.offsets[i + 1]
        if end < at or end > part.total:
            part.status = _MISMATCH
            return
        part.status = _write_string(_item(part.outer, i), part.out + part.base + at, end - at)
        if part.status:
            return
        part.offsets[i + 1] = part.base + end
        at = end
    if at != part.total:
        part.status = _MISMATCH


cdef object _join_strings_direct(texts, int threads):
    global fse_test_failed_starts
    global fse_test_fail_bookkeeping
    global _string_last_parts, _string_last_estimated
    cdef PyObject *outer = <PyObject *>texts
    cdef Py_ssize_t n, size, total = 0
    cdef int p, n_parts, status = 0
    cdef unsigned first_starts = fse_test_failed_starts
    cdef _StringPart *parts = NULL
    cdef np.int64_t[::1] offsets
    cdef char *out
    starts = data = None
    _string_last_parts = 0
    _string_last_estimated = False
    if threads < 1:
        raise ValueError(f"threads must be at least 1, not {threads}")
    if not (PyList_CheckExact(outer) or PyTuple_CheckExact(outer)):
        return None
    n = _items(outer)
    # Check both the row count and the offsets' byte size before n + 1 is formed.
    if n >= PY_SSIZE_T_MAX // <Py_ssize_t>sizeof(np.int64_t):
        raise MemoryError("the offsets of the strings would exceed the address space")
    try:
        if _string_fail_stage == 1:
            raise MemoryError("could not allocate the offsets of the strings")
        starts = np.empty(n + 1, dtype=np.int64)
        offsets = starts
        # Both operations above may run Python. Fix bounds only after both complete.
        if _items(outer) != n:
            return None
        n_parts = _string_parts(outer, n, threads)
        _string_last_parts = n_parts
        if n_parts > PY_SSIZE_T_MAX // <Py_ssize_t>sizeof(_StringPart):
            raise MemoryError("the parts of the strings would exceed the address space")
        if _string_fail_stage != 2:
            parts = <_StringPart *>fse_join_calloc(n_parts, sizeof(_StringPart))
        if parts == NULL:
            raise MemoryError("could not allocate the parts of the strings")
        size = n // n_parts
        for p in range(n_parts):
            parts[p].outer = outer
            parts[p].first = p * size
            parts[p].last = n if p == n_parts - 1 else (p + 1) * size
            parts[p].offsets = &offsets[0]
        fse_test_fail_bookkeeping = _string_fail_stage == 3
        if fse_parallel(n_parts, _size_strings, parts) < 0:
            raise MemoryError("could not start the sizing threads for the strings")
        for p in range(n_parts):
            status = min(status, parts[p].status)
        if status == _NO_MEMORY:
            raise MemoryError("the joined strings would exceed the address space")
        if status < 0:
            return None
        for p in range(n_parts):
            parts[p].base = total
            if parts[p].total > PY_SSIZE_T_MAX - total:
                raise MemoryError("the joined strings would exceed the address space")
            total += parts[p].total
        if _string_fail_stage == 4:
            raise MemoryError("could not allocate the bytes of the strings")
        data = PyBytes_FromStringAndSize(NULL, total)
        out = PyBytes_AS_STRING(data)
        for p in range(n_parts):
            parts[p].out = out
        # Test only: corrupt a reserved row span in either direction, without Python.
        if n and _string_mismatch:
            if _string_mismatch > 0:
                if offsets[1] < PY_SSIZE_T_MAX:
                    offsets[1] += 1
            else:
                offsets[1] -= 1
        fse_test_failed_starts = _string_second_starts
        fse_test_fail_bookkeeping = _string_fail_stage == 5
        if fse_parallel(n_parts, _write_strings, parts) < 0:
            raise MemoryError("could not start the writing threads for the strings")
        for p in range(n_parts):
            if parts[p].status == _MISMATCH:
                raise AssertionError("the string size changed between join passes")
        offsets[0] = 0
        return data, starts
    finally:
        # On success the tuple owns the Python objects. On abort these local references
        # are the only owners. Release the view before its array, and always free parts.
        offsets = None
        starts = data = None
        fse_join_free(parts)
        fse_test_failed_starts = first_starts
        fse_test_fail_bookkeeping = 0


def _test_join_resources():
    """Actual live native part/bookkeeping allocations. Python objects use weakref/tracemalloc tests."""
    return fse_test_join_live


def _test_join_string_parts():
    """Parts actually selected by the last string join, zero before selection."""
    return _string_last_parts


def _test_join_string_estimated():
    """Whether the last string join entered the size pre-pass."""
    return bool(_string_last_estimated)
