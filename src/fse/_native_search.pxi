# Included by _native.pyx: inverse norms, cosine and score of many vectors.

# ---------------------------------------------------------------------------------------
# The search, as fse._reference.inverse_norms, cosines and scores.
# ---------------------------------------------------------------------------------------

cdef extern from "_native_kernel.h":
    int64_t fse_inverse_norms(int dtype, const void *stored, int64_t n, int dim, float *out,
                              float *scratch) nogil
    int64_t fse_cosines(int isa, int dtype, const void *stored, const float *inverse, int64_t n,
                        int dim, const float *units, int q, float *out, float *scratch) nogil
    int64_t fse_scores(int isa, int dtype, const void *stored, const float *scales, int64_t n,
                       int dim, const float *weights, const float *biases, int m, float *out,
                       float *scratch) nogil


def inverse_norms(stored):
    """``fse._reference.inverse_norms`` in the kernel: one float32 per stored row."""
    with _KernelEnv():
        stored = _rows(stored, "stored")
        n, dim = stored.shape[0], _check_dim(stored.shape[1])
        out = np.empty(n, dtype=np.float32)
        if n == 0:
            return out
        bad = _inverse_norms_into(stored, n, dim, out)
    if bad >= 0:
        raise ValueError(f"the inverse norm of row {bad} is beyond float32")
    return out


cdef int64_t _inverse_norms_into(stored, int64_t n, int dim, out):
    cdef float[::1] o = out
    cdef float[::1] scratch = np.empty(max(dim, 1), dtype=np.float32)
    cdef const void *data = np.PyArray_DATA(stored)
    cdef int dtype = _DTYPES[stored.dtype]
    cdef int64_t bad
    with nogil:
        bad = fse_inverse_norms(dtype, data, n, dim, &o[0], &scratch[0])
    return bad


def _matrix_of(value, name, dim):
    with np.errstate(over="ignore", invalid="ignore"):  # as the reference
        value = np.ascontiguousarray(value, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix with one vector per row, not shape {value.shape}")
    if value.shape[1] != dim:
        raise ValueError(f"{name} must hold {dim} values per vector, not {value.shape[1]}")
    return value


def cosines(stored, inverse, queries, *, isa=None):
    """``fse._reference.cosines`` in the kernel: a (rows, queries) float32 array."""
    with _KernelEnv():
        return _cosines(stored, inverse, queries, isa)


def _cosines(stored, inverse, queries, isa):
    from fse._contract import Normalize

    cdef int code = _check_isa(isa)
    stored = _rows(stored, "stored")
    cdef int64_t n = stored.shape[0]
    cdef int dim = _check_dim(stored.shape[1])
    inverse = _float32_vector(inverse, "inverse", n)
    queries = _matrix_of(queries, "queries", dim)
    for j in range(queries.shape[0]):
        if not (queries[j] != 0).any():
            raise ValueError(f"query {j} is the zero vector and has no direction")
    units = np.empty_like(queries)
    for j in range(queries.shape[0]):
        try:
            units[j] = _postprocess(queries[j : j + 1], [Normalize()], 0, isa)[0]
        except ValueError as error:
            raise ValueError(f"query {j}: {error}") from None
    out = np.empty((n, queries.shape[0]), dtype=np.float32)
    if n == 0 or queries.shape[0] == 0:
        return out
    cdef int64_t bad = _cosines_into(code, stored, inverse, units, out)
    if bad >= 0:
        raise ValueError(f"query {bad // n}: the cosine of row {bad % n} is not finite")
    return out


cdef int64_t _cosines_into(int code, stored, inverse, units, out):
    cdef int64_t n = stored.shape[0]
    cdef int dim = stored.shape[1]
    cdef int q = units.shape[0]
    cdef const float[::1] inv = inverse
    cdef const float[:, ::1] u = units
    cdef float[:, ::1] o = out
    cdef float[::1] scratch = np.empty(max(dim, 1), dtype=np.float32)
    cdef const void *data = np.PyArray_DATA(stored)
    cdef int dtype = _DTYPES[stored.dtype]
    cdef int64_t bad
    with nogil:
        bad = fse_cosines(code, dtype, data, &inv[0], n, dim, &u[0, 0] if dim else NULL, q,
                          &o[0, 0], &scratch[0])
    return bad


def scores(stored, scales, weights, bias=0.0, *, isa=None):
    """``fse._reference.scores`` in the kernel: a (rows, vectors) float32 array; ``bias`` is
    one number or one per weight vector."""
    with _KernelEnv():
        return _scores(stored, scales, weights, bias, isa)


def _scores(stored, scales, weights, bias, isa):
    cdef int code = _check_isa(isa)
    stored = _rows(stored, "stored")
    cdef int64_t n = stored.shape[0]
    cdef int dim = _check_dim(stored.shape[1])
    weights = _matrix_of(weights, "weights", dim)
    cdef bint int8 = stored.dtype == np.int8
    if int8 != (scales is not None):
        raise ValueError("scales are given exactly for int8 rows")
    if int8:
        scales = _float32_vector(scales, "scales", n)
    with np.errstate(over="ignore", invalid="ignore"):  # as the reference
        biases = np.ascontiguousarray(np.broadcast_to(np.asarray(bias, dtype=np.float32), (weights.shape[0],)))
    out = np.empty((n, weights.shape[0]), dtype=np.float32)
    if n == 0 or weights.shape[0] == 0:
        return out
    cdef int64_t bad = _scores_into(code, stored, scales, weights, biases, out)
    if bad >= 0:
        raise ValueError(f"weights {bad // n}: the score of row {bad % n} overflows float32")
    return out


cdef int64_t _scores_into(int code, stored, scales, weights, biases, out):
    cdef int64_t n = stored.shape[0]
    cdef int dim = stored.shape[1]
    cdef int m = weights.shape[0]
    cdef const float[:, ::1] w = weights
    cdef const float[::1] b = biases
    cdef float[:, ::1] o = out
    cdef float[::1] scratch = np.empty(max(dim, 1), dtype=np.float32)
    cdef const void *data = np.PyArray_DATA(stored)
    cdef const float *sc = <const float *>np.PyArray_DATA(scales) if scales is not None else NULL
    cdef int dtype = _DTYPES[stored.dtype]
    cdef int64_t bad
    with nogil:
        bad = fse_scores(code, dtype, data, sc, n, dim, &w[0, 0] if dim else NULL, &b[0], m,
                         &o[0, 0], &scratch[0])
    return bad
