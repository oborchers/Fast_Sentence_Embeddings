# Included by _native.pyx: the int8 prefilter, as fse._reference_prefilter.

# ---------------------------------------------------------------------------------------
# The slack of a query and one pass over a block of stored int8 rows
# (docs/architecture/numerical-contract.md, "int8 prefilter (no new arithmetic)").
# ---------------------------------------------------------------------------------------

from libc.stdint cimport int8_t, int32_t

cdef extern from "_native_kernel.h":
    int fse_i8_compiled(int variant) nogil
    int fse_i8_runnable(int variant) nogil
    void fse_int8_bounds(int64_t q, int dim, const float *units, const int8_t *qcodes,
                         const float *qscales, double *slack, int32_t *ok) nogil
    int fse_int8_pass(int variant, const int8_t *codes, const float *inverse, int64_t n, int dim,
                      const int8_t *qcodes, const float *qscales, int q, int k,
                      const double *slack, const double *floor_in, double *heap, int32_t *heap_n,
                      double *thr, int64_t *rows, double *upper, int64_t cap, int64_t *count,
                      int64_t *unsafe_row) nogil

_PREFILTER_VARIANTS = {"portable": 0, "avx2": 1, "neon-sdot": 2}
_PREFILTER_STATUS = ("ok", "overflow", "unsafe")
# The bound is proven for at most 1024 dimensions (the integer sums stay below 2^24).
_PREFILTER_MAX_DIM = 1024
_PREFILTER_INT_MAX = 2**31 - 1
_PREFILTER_MAX_QUERIES = 2**20  # the kernel indexes queries with int, two at a time


def prefilter_variants():
    """The variants of the int8 prefilter built into this binary, the fastest last."""
    return tuple(name for name, code in _PREFILTER_VARIANTS.items() if fse_i8_compiled(code))


def prefilter_can_run(variant):
    """Whether ``variant`` is built into this binary and this CPU can run it."""
    code = _PREFILTER_VARIANTS.get(variant)
    return code is not None and bool(fse_i8_runnable(code))


def _prefilter_code(variant):
    """The code of ``variant``, or of the fastest variant this CPU runs if ``None``."""
    if variant is None:
        return max(code for name, code in _PREFILTER_VARIANTS.items() if prefilter_can_run(name))
    if variant not in _PREFILTER_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(_PREFILTER_VARIANTS)}, not {variant!r}")
    code = _PREFILTER_VARIANTS[variant]
    if not fse_i8_compiled(code):
        raise ValueError(f"this build has no {variant} prefilter; built: {prefilter_variants()}")
    if not fse_i8_runnable(code):
        raise ValueError(f"this CPU cannot run the {variant} prefilter")
    return code


def _prefilter_array(value, dtype, shape, name):
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != dtype
        or value.shape != shape
        or not value.flags.c_contiguous
    ):
        raise ValueError(f"{name} must be a C-contiguous {np.dtype(dtype)} array of shape {shape}")


def _prefilter_queries(qcodes, qscales):
    """The shape of the int8 queries, checked: (queries, dimension)."""
    if (
        not isinstance(qcodes, np.ndarray)
        or qcodes.dtype != np.int8
        or qcodes.ndim != 2
        or not qcodes.size
        or not qcodes.flags.c_contiguous
    ):
        raise ValueError("qcodes must be a nonempty C-contiguous int8 matrix")
    if qcodes.shape[0] > _PREFILTER_MAX_QUERIES:
        raise ValueError(f"the prefilter takes at most {_PREFILTER_MAX_QUERIES} queries")
    if qcodes.shape[1] > _PREFILTER_MAX_DIM:
        raise ValueError(f"the bound is proven for at most {_PREFILTER_MAX_DIM} dimensions")
    if qcodes.min() < -127:
        raise ValueError("the codes of a query lie in [-127, 127]")
    _prefilter_array(qscales, np.float32, (qcodes.shape[0],), "qscales")
    if not (np.isfinite(qscales).all() and (qscales >= 0).all()):
        raise ValueError("the scales of the queries must be finite and not negative")
    return qcodes.shape


def int8_bounds(units, qcodes, qscales):
    """``fse._reference_prefilter.bounds`` in the kernel: the slack ``E`` (float64) and
    whether the query is in the domain (bool), per query."""
    with _KernelEnv():
        shape = _prefilter_queries(qcodes, qscales)
        _prefilter_array(units, np.float32, shape, "units")
        slack = np.empty(shape[0], dtype=np.float64)
        ok = np.empty(shape[0], dtype=np.int32)
        _int8_bounds_into(units, qcodes, qscales, slack, ok)
    return slack, ok.astype(np.bool_)


cdef void _int8_bounds_into(units, qcodes, qscales, slack, ok):
    cdef int64_t q = qcodes.shape[0]
    cdef int dim = qcodes.shape[1]
    cdef const float *u = <const float *>np.PyArray_DATA(units)
    cdef const int8_t *c = <const int8_t *>np.PyArray_DATA(qcodes)
    cdef const float *s = <const float *>np.PyArray_DATA(qscales)
    cdef double[::1] e = slack
    cdef int32_t[::1] o = ok
    with nogil:
        fse_int8_bounds(q, dim, u, c, s, &e[0], &o[0])


def int8_pass(stored, inverse, qcodes, qscales, slack, threshold, k, capacity=None, *, variant=None):
    """``fse._reference_prefilter.int8_pass`` in the kernel: ``(status, row, heap, rows,
    upper)``, status "ok", "overflow" or "unsafe". ``capacity`` bounds the kept rows per
    query (default ``max(4 * k, 4096)``); ``variant`` picks the integer kernel (tests)."""
    with _KernelEnv():
        return _int8_pass(stored, inverse, qcodes, qscales, slack, threshold, k, capacity, variant)


def _int8_pass(stored, inverse, qcodes, qscales, slack, threshold, k, capacity, variant):
    cdef int code = _prefilter_code(variant)
    shape = _prefilter_queries(qcodes, qscales)
    width, dim = int(shape[0]), int(shape[1])
    if (
        not isinstance(stored, np.ndarray)
        or stored.dtype != np.int8
        or stored.ndim != 2
        or stored.shape[1] != dim
        or not stored.flags.c_contiguous
    ):
        raise ValueError("stored must be a C-contiguous int8 matrix with the dimension of the queries")
    n = int(stored.shape[0])
    _prefilter_array(inverse, np.float32, (n,), "inverse")
    _prefilter_array(slack, np.float64, (width,), "slack")
    _prefilter_array(threshold, np.float64, (width,), "threshold")
    if not (np.isfinite(slack).all() and (slack >= 0).all()):
        raise ValueError("the slack of every query must be finite and not negative")
    if np.isnan(threshold).any():
        raise ValueError("a threshold must not be NaN")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= _PREFILTER_INT_MAX:
        raise ValueError(f"k must be an integer from 1 to {_PREFILTER_INT_MAX}")
    k = int(k)
    if capacity is None:
        capacity = max(4 * k, 4096)
    if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or not 1 <= capacity <= _PREFILTER_INT_MAX:
        raise ValueError(f"capacity must be an integer from 1 to {_PREFILTER_INT_MAX}")
    capacity = int(capacity)
    heap = np.empty((width, k), dtype=np.float64)
    rows = np.empty((width, capacity), dtype=np.int64)
    upper = np.empty((width, capacity), dtype=np.float64)
    count = np.empty(width, dtype=np.int64)
    cdef int64_t bad = -1
    cdef int status = _int8_pass_into(code, stored, inverse, qcodes, qscales, slack, threshold, k,
                                      capacity, heap, rows, upper, count, &bad)
    if status == 2:
        return "unsafe", int(bad), None, None, None
    if status == 1:
        return "overflow", None, None, None, None
    kept = [int(count[j]) for j in range(width)]
    return (
        "ok",
        None,
        heap,
        tuple(rows[j, : kept[j]].copy() for j in range(width)),
        tuple(upper[j, : kept[j]].copy() for j in range(width)),
    )


cdef int _int8_pass_into(int code, stored, inverse, qcodes, qscales, slack, threshold, int k,
                         int64_t capacity, heap, rows, upper, count, int64_t *bad):
    cdef int64_t n = stored.shape[0]
    cdef int dim = stored.shape[1]
    cdef int q = qcodes.shape[0]
    cdef int32_t[::1] heap_n = np.empty(q, dtype=np.int32)
    cdef double[::1] thr = np.empty(q, dtype=np.float64)
    cdef const int8_t *codes = <const int8_t *>np.PyArray_DATA(stored)
    cdef const float *inv = <const float *>np.PyArray_DATA(inverse)
    cdef const int8_t *qc = <const int8_t *>np.PyArray_DATA(qcodes)
    cdef const float *qs = <const float *>np.PyArray_DATA(qscales)
    cdef const double *sl = <const double *>np.PyArray_DATA(slack)
    cdef const double *th = <const double *>np.PyArray_DATA(threshold)
    cdef double[:, ::1] h = heap
    cdef int64_t[:, ::1] r = rows
    cdef double[:, ::1] u = upper
    cdef int64_t[::1] c = count
    cdef int status
    with nogil:
        status = fse_int8_pass(code, codes, inv, n, dim, qc, qs, q, k, sl, th, &h[0, 0],
                               &heap_n[0], &thr[0], &r[0, 0], &u[0, 0], capacity, &c[0], bad)
    return status
