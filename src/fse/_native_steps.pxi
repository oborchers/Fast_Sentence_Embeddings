# Included by _native.pyx: dot16, the postprocessing steps and score.

# ---------------------------------------------------------------------------------------
# dot16, the postprocessing steps and score, as in fse._reference.
# ---------------------------------------------------------------------------------------


def _float32_rows(rows, name="rows"):
    if not isinstance(rows, np.ndarray) or rows.dtype != np.float32 or rows.ndim != 2:
        raise ValueError(f"{name} must be a float32 matrix")
    return np.ascontiguousarray(rows)


def dot16(rows, other, *, isa=None):
    """dot16 of every row with ``other`` (one vector, or one row per row)."""
    with _KernelEnv():
        return _dot16(rows, other, isa)


def _dot16(rows, other, isa):
    cdef int code = _check_isa(isa)
    rows = _float32_rows(rows)
    other = np.ascontiguousarray(np.broadcast_to(np.asarray(other, dtype=np.float32), rows.shape))
    cdef const float[:, ::1] a = rows
    cdef const float[:, ::1] b = other
    out = np.empty(rows.shape[0], dtype=np.float32)
    cdef float[::1] o = out
    cdef int n = _check_dim(rows.shape[1])
    if a.shape[0] == 0 or n == 0:
        out[:] = 0
        return out
    with nogil:
        fse_dot16_rows(code, &a[0, 0], &b[0, 0], a.shape[0], n, &o[0])
    return out


def postprocess(rows, steps, first_row=0, *, isa=None):
    """``fse._reference.postprocess`` in the kernel: the same rows, bits and errors."""
    with _KernelEnv():
        return _postprocess(rows, steps, first_row, isa)


def postprocess_pooled(rows, steps, first_row=0, *, isa=None):
    """``postprocess`` of rows the kernel has just pooled, in place: ``rows`` (writable
    C-contiguous float32, finite as pooling leaves them) are neither copied nor checked
    again. The result is ``rows`` itself, or a new array after a ``project``."""
    if not (
        isinstance(rows, np.ndarray)
        and rows.dtype == np.float32
        and rows.ndim == 2
        and rows.flags.c_contiguous
        and rows.flags.writeable
    ):
        raise ValueError("pooled rows must be a writable C-contiguous float32 matrix")
    steps = tuple(steps)  # iterated twice: the checks and the steps
    for step in steps:  # written in place, the rows must not share memory with a step
        for name in ("components", "weights", "basis"):
            array = getattr(step, name, None)
            if array is not None and np.may_share_memory(rows, array):
                raise ValueError(f"the pooled rows overlap the {name} of a step")
    with _KernelEnv():
        _check_dim(rows.shape[1])
        return _apply_steps(rows, steps, first_row, _check_isa(isa))


def _postprocess(rows, steps, first_row, isa):
    cdef int code = _check_isa(isa)
    out = np.array(rows, dtype=np.float32, copy=True, order="C")
    if out.ndim != 2:
        raise ValueError("rows must be a matrix")
    _check_dim(out.shape[1])
    _check_finite_rows(out, first_row)
    return _apply_steps(out, steps, first_row, code)


def _apply_steps(out, steps, first_row, int code):
    """The steps on ``out`` in place (a project gives a new array): the rows of each."""
    from fse._contract import Normalize, Project, RemoveComponents, check_steps

    cdef float[:, ::1] view
    cdef const float[:, ::1] params
    cdef const float[::1] weights
    cdef float[::1] scratch
    cdef float[:, ::1] projected
    cdef int64_t bad, unnormalizable
    cdef int64_t n
    cdef int dim, k
    for step in check_steps(steps, out.shape[1]):
        n, dim = out.shape[0], out.shape[1]
        if n == 0:
            if isinstance(step, Project):
                out = np.empty((0, step.output_dim), dtype=np.float32)
            continue
        view = out
        if isinstance(step, RemoveComponents):
            params = step.components
            k = params.shape[0]
            # The kernel reads through raw pointers: check the shapes again, as numpy lets
            # a caller reshape the arrays of a checked step.
            if not 1 <= k <= dim or params.shape[1] != dim:
                raise ValueError(f"components must have shape (k, {dim}) with 1 <= k <= {dim}")
            if step.weights is not None and step.weights.shape != (k,):
                raise ValueError(f"weights must have shape ({k},)")
            scratch = np.empty(dim, dtype=np.float32)
            if step.weights is not None:
                weights = step.weights
                with nogil:
                    bad = fse_remove_components(code, &view[0, 0], n, dim, &params[0, 0], k, &weights[0], &scratch[0])
            else:
                with nogil:
                    bad = fse_remove_components(code, &view[0, 0], n, dim, &params[0, 0], k, NULL, &scratch[0])
        elif isinstance(step, Project):
            params = step.basis
            k = params.shape[0]
            if not 1 <= k <= dim or params.shape[1] != dim:
                raise ValueError(f"basis must have shape (k, {dim}) with 1 <= k <= {dim}")
            result = np.empty((n, k), dtype=np.float32)
            projected = result
            with nogil:
                bad = fse_project(code, &view[0, 0], n, dim, &params[0, 0], k, &projected[0, 0])
            out = result
        else:
            with nogil:
                bad = fse_normalize(code, &view[0, 0], n, dim, &unnormalizable)
            if unnormalizable >= 0:
                raise ValueError(
                    f"row {first_row + unnormalizable} cannot be normalized: the sum of its "
                    "squares underflows or overflows float32"
                )
        if bad >= 0:
            raise ValueError(f"row {first_row + bad} contains NaN or infinite values")
    return out


def _check_finite_rows(rows, first_row):
    bad = np.flatnonzero(~np.isfinite(rows).all(axis=1))
    if len(bad):
        raise ValueError(f"row {first_row + int(bad[0])} contains NaN or infinite values")


def score(stored, scales, weights, bias=0.0, *, isa=None):
    """``fse._reference.score`` in the kernel: dot16(stored, w) * scale + bias per row."""
    with _KernelEnv():
        return _score(stored, scales, weights, bias, isa)


def _score(stored, scales, weights, bias, isa):
    cdef int code = _check_isa(isa)
    stored = _rows(stored, "stored")
    cdef Py_ssize_t n = stored.shape[0]
    cdef int dim = _check_dim(stored.shape[1])
    with np.errstate(over="ignore", invalid="ignore"):  # as the reference
        weights = np.ascontiguousarray(weights, dtype=np.float32)
    if weights.shape != (dim,):
        raise ValueError(f"weights must hold {dim} values")
    cdef bint int8 = stored.dtype == np.int8
    if int8 != (scales is not None):
        raise ValueError("scales are given exactly for int8 rows")
    if int8:
        scales = _float32_vector(scales, "scales", n)
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    cdef float[::1] o = out
    cdef const float[::1] w = weights
    cdef float[::1] scratch = np.empty(max(dim, 1), dtype=np.float32)
    cdef const void *data = np.PyArray_DATA(stored)
    cdef const float *sc = <const float *>np.PyArray_DATA(scales) if int8 else NULL
    cdef int dtype = _DTYPES[stored.dtype]
    cdef float b
    with np.errstate(over="ignore", invalid="ignore"):  # as the reference
        b = np.float32(bias)
    cdef int64_t bad
    with nogil:
        bad = fse_score(code, dtype, data, sc, n, dim, &w[0] if dim else NULL, b, &o[0], &scratch[0])
    if bad >= 0:
        raise ValueError(f"the score of row {bad} overflows float32")
    return out
