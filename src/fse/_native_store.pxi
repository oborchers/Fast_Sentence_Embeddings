# Included by _native.pyx: stage 5, the storage of finished rows.

# ---------------------------------------------------------------------------------------
# quantize_rows@1 in one pass per row, as fse._reference.quantize_rows.
# ---------------------------------------------------------------------------------------

from libc.stdint cimport int8_t, uint16_t

cdef extern from "_native_kernel.h":
    void fse_store_f16(int isa, const float *rows, int64_t n, int dim, uint16_t *codes, int64_t *bad,
                       int64_t *beyond) nogil
    void fse_store_i8(int isa, const float *rows, int64_t n, int dim, int8_t *codes, float *scales,
                      int64_t *bad) nogil


def _store_target(array, dtype, shape, name):
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != dtype
        or array.shape != shape
        or not array.flags.c_contiguous
        or not array.flags.writeable
    ):
        raise ValueError(f"{name} must be a writable C-contiguous {np.dtype(dtype)} array of shape {shape}")


def store_rows(rows, codes, scales=None, first_row=0, *, isa=None):
    """float32 ``rows`` into float16 or int8 ``codes`` (and one scale per row in ``scales``
    for int8) by quantize_rows@1, as ``fse._reference.quantize_rows``. The first row that is
    not finite raises, else for float16 the first row beyond +-65504, named from
    ``first_row``."""
    cdef int code = _check_isa(isa)
    if not isinstance(rows, np.ndarray) or rows.dtype != np.float32 or rows.ndim != 2:
        raise ValueError("rows must be a float32 matrix")
    if not rows.flags.c_contiguous:
        raise ValueError("rows must be C-contiguous")
    int8 = isinstance(codes, np.ndarray) and codes.dtype == np.int8
    _store_target(codes, np.int8 if int8 else np.float16, rows.shape, "codes")
    if int8:
        _store_target(scales, np.float32, (rows.shape[0],), "scales")
    elif scales is not None:
        raise ValueError("float16 codes have no scales")
    if any(np.may_share_memory(a, b) for a, b in ((rows, codes), (rows, scales), (codes, scales)) if b is not None):
        raise ValueError("rows, codes and scales must not share memory")
    cdef int64_t n = rows.shape[0]
    cdef int dim = _check_dim(rows.shape[1])
    cdef int64_t bad = -1, beyond = -1
    cdef const float[:, ::1] r
    cdef int8_t[:, ::1] c8
    cdef uint16_t[:, ::1] c16
    cdef float[::1] s
    global _nogil_sections
    if n and dim:
        r = rows
        if int8:
            c8 = codes
            s = scales
            _nogil_sections += 1
            with nogil:
                fse_store_i8(code, &r[0, 0], n, dim, &c8[0, 0], &s[0], &bad)
        else:
            c16 = codes.view(np.uint16)
            _nogil_sections += 1
            with nogil:
                fse_store_f16(code, &r[0, 0], n, dim, &c16[0, 0], &bad, &beyond)
    elif int8:
        scales[:] = 0  # rows without values: absmax 0
    if bad >= 0:
        raise ValueError(f"row {first_row + bad} contains NaN or infinite values")
    if beyond >= 0:
        raise ValueError(f"row {first_row + beyond} has values beyond the float16 range of +-65504")
