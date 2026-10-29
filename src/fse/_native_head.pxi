# Included by _native.pyx: linear heads, as fse._reference_head.

# ---------------------------------------------------------------------------------------
# The logits of a head, exp@1, the links and the labels (linear@1).
# ---------------------------------------------------------------------------------------

cdef extern from "_native_kernel.h":
    int64_t fse_head_bias(int isa, const float *rows, int64_t n, int k, const float *bias,
                          float *out) nogil
    int64_t fse_head_proba(int isa, int link, const float *logits, int64_t n, int k, float *out) nogil
    int64_t fse_head_labels(const float *logits, int64_t n, int k, int64_t *out) nogil
    void fse_exp1_rows(int isa, const float *x, int64_t n, float *out) nogil


def exp1(x, *, isa=None):
    """``fse._reference_head.exp`` in the kernel: exp@1 of every value of a float32 array."""
    cdef int code = _check_isa(isa)
    if not isinstance(x, np.ndarray) or x.dtype != np.float32:
        raise ValueError("x must be a float32 array")
    flat = np.ascontiguousarray(x).reshape(-1)
    out = np.empty_like(flat)
    cdef const float[::1] a = flat
    cdef float[::1] o = out
    cdef int64_t n = flat.shape[0]
    if n:
        with nogil:
            fse_exp1_rows(code, &a[0], n, &o[0])
    return out.reshape(x.shape)


def _matrix32(value, name, rows, cols, writable=False):
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.shape != (rows, cols)
        or not value.flags.c_contiguous
        or (writable and not value.flags.writeable)
    ):
        kind = "a writable " if writable else "a "
        raise ValueError(f"{name} must be {kind}C-contiguous float32 array of shape {(rows, cols)}")
    return value


def head_logits(rows, weights, bias, out, first_row=0, *, isa=None):
    """``fse._reference_head.logits`` in the kernel, into ``out`` (n, k): with ``weights``
    (k, d) the scores of the float32 ``rows``, else ``rows`` (n, k) plus ``bias`` (``out``
    may be ``rows``). A row whose logits are not finite raises, named from ``first_row``."""
    with _KernelEnv():
        _head_logits(rows, weights, bias, out, first_row, isa)


def _head_logits(rows, weights, bias, out, first_row, isa):
    cdef int code = _check_isa(isa)
    if not isinstance(rows, np.ndarray) or rows.ndim != 2:
        raise ValueError("rows must be a float32 matrix")
    cdef int64_t n = rows.shape[0]
    cdef int k = bias.shape[0] if isinstance(bias, np.ndarray) and bias.ndim == 1 else -1
    if k < 1 or bias.dtype != np.float32 or not bias.flags.c_contiguous:
        raise ValueError("bias must be a C-contiguous float32 vector of at least one value")
    _matrix32(out, "out", n, k, writable=True)
    if weights is None:
        _matrix32(rows, "rows", n, k)
    else:
        _matrix32(rows, "rows", n, _check_dim(rows.shape[1]))
        _matrix32(weights, "weights", k, rows.shape[1])
    if n == 0:
        return
    cdef int64_t bad
    cdef const float[:, ::1] r = rows
    cdef const float[::1] b = bias
    cdef float[:, ::1] o = out
    if weights is None:
        with nogil:
            bad = fse_head_bias(code, &r[0, 0], n, k, &b[0], &o[0, 0])
    else:
        bad = _scores_into(code, rows, None, weights, bias, out)
        if bad >= 0:
            bad = int(np.flatnonzero(~np.isfinite(out).all(axis=1))[0])
    if bad >= 0:
        raise ValueError(f"the logits of row {first_row + bad} are not finite")


def head_outputs(logits, kind, link, out, first_row=0, *, isa=None):
    """``fse._reference_head.proba`` (``kind`` "proba", with a link) or ``labels`` ("labels")
    of finished ``logits`` (n, k) into ``out``: float32 (n, 2) for sigmoid, (n, k) for the
    other links, int64 (n,) for labels. Logits that are not finite raise, naming the row."""
    with _KernelEnv():
        _head_outputs(logits, kind, link, out, first_row, _check_isa(isa))


def _head_outputs(logits, kind, link, out, first_row, int isa):
    from fse._contract import Link

    if not isinstance(logits, np.ndarray) or logits.ndim != 2 or logits.shape[1] < 1:
        raise ValueError("logits must be a float32 matrix")
    cdef int64_t n = logits.shape[0]
    cdef int k = logits.shape[1]
    _matrix32(logits, "logits", n, k)
    cdef const float[:, ::1] z = logits
    cdef float[:, ::1] o
    cdef int64_t[::1] labels
    cdef int code
    cdef int64_t bad = -1
    if kind == "labels":
        if (
            not isinstance(out, np.ndarray)
            or out.dtype != np.int64
            or out.shape != (n,)
            or not out.flags.c_contiguous
            or not out.flags.writeable
        ):
            raise ValueError(f"out must be a writable int64 vector of {n} values")
        if n:
            labels = out
            with nogil:
                bad = fse_head_labels(&z[0, 0], n, k, &labels[0])
    elif kind == "proba":
        codes = {Link.SIGMOID: 0, Link.SOFTMAX: 1, Link.OVR: 2}
        if link not in codes:
            raise ValueError(f"link must be one of {[str(member) for member in Link]}")
        if (link is Link.SIGMOID) != (k == 1):
            want = "one logit" if link is Link.SIGMOID else "at least two logits"
            raise ValueError(f"{link} takes {want} per row")
        code = codes[link]
        _matrix32(out, "out", n, 2 if k == 1 else k, writable=True)
        if n:
            o = out
            with nogil:
                bad = fse_head_proba(isa, code, &z[0, 0], n, k, &o[0, 0])
    else:
        raise ValueError(f"kind must be 'proba' or 'labels', not {kind!r}")
    if bad >= 0:
        raise ValueError(f"the logits of row {first_row + bad} are not finite")
