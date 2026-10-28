# Included by _native.pyx: The weighted mean kernel, its ISA dispatch and FP environment.

# ---------------------------------------------------------------------------------------
# The weighted mean kernel (weighted_mean@1): stages 2 and 3 of a batch, without the GIL.
# ---------------------------------------------------------------------------------------

from libc.stdint cimport int64_t, uint32_t, uint64_t

cdef extern from "_native_kernel.h":
    ctypedef struct fse_params:
        int dtype
        int dim
        size_t row_bytes
        const void *rows
        const float *scales
        const float *weights
        const void *ngram_rows
        const float *ngram_scales
        float oov_weight
    ctypedef struct fse_batch:
        int64_t n_sentences
        const int64_t *offsets
        const uint32_t *tokens
        const int64_t *ngram_offsets
        const uint32_t *ngrams
    float fse_half_table[65536]
    int fse_has_isa(int isa) nogil
    int64_t fse_run(int isa, const fse_params *p, const fse_batch *b, float *out,
                    int64_t *counts, int finish) nogil
    void fse_dot16_rows(int isa, const float *a, const float *b, int64_t n, int dim, float *out) nogil
    uint64_t fse_test_fpenv_set(int flush, int round_up)
    uint64_t fse_fpenv_push()
    void fse_fpenv_pop(uint64_t saved)
    void fse_test_fpenv_restore(uint64_t saved)
    uint64_t fse_test_fpenv_get()
    int64_t fse_remove_components(int isa, float *rows, int64_t n, int dim,
                                  const float *components, int k, const float *weights,
                                  float *scratch) nogil
    int64_t fse_project(int isa, const float *rows, int64_t n, int dim, const float *basis,
                        int k, float *out) nogil
    int64_t fse_normalize(int isa, float *rows, int64_t n, int dim, int64_t *unnormalizable) nogil
    int64_t fse_score(int isa, int dtype, const void *stored, const float *scales, int64_t n,
                      int dim, const float *w, float bias, float *out, float *scratch) nogil

_ISAS = {"portable": 0, "avx2": 1, "neon": 2}
_DTYPES = {np.dtype(np.float32): 0, np.dtype(np.float16): 1, np.dtype(np.int8): 2}

with np.errstate(invalid="ignore"):
    # The bit patterns include NaN payloads, whose conversion numpy reports.
    _half_values = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
for _i in range(65536):
    fse_half_table[_i] = _half_values[_i]
del _half_values, _i


def available_isas():
    """The instruction sets of the kernel this CPU runs, the fastest last."""
    return tuple(name for name, isa in _ISAS.items() if fse_has_isa(isa))


def default_isa():
    """The instruction set the kernel uses unless told otherwise."""
    isas = available_isas()
    return isas[len(isas) - 1]


# The kernel indexes a row with int, and its loops step up to 32 values past an index.
MAX_DIM = 1 << 30


def _check_dim(dim):
    if dim > MAX_DIM:
        raise ValueError(f"the kernel supports at most {MAX_DIM} dimensions, not {dim}")
    return dim


cdef class _KernelEnv:
    """The floating-point environment of the kernel for a whole call: the arithmetic and the
    numpy conversions of the wrapper (bias, oov_weight, input arrays) run in it."""

    cdef uint64_t saved

    def __enter__(self):
        self.saved = fse_fpenv_push()

    def __exit__(self, *exc):
        fse_fpenv_pop(self.saved)
        return False


def kernel_env():
    """A context manager that runs its block in the floating-point environment of the
    kernel (round to nearest, no flush-to-zero) and restores the thread's afterwards, also
    after an error: for numpy arithmetic of the contract, such as the storage of rows."""
    return _KernelEnv()


def _test_fpenv_set(flush=False, round_up=False):
    """For the tests: set flush-to-zero and rounding up; returns the previous environment."""
    return fse_test_fpenv_set(1 if flush else 0, 1 if round_up else 0)


def _test_fpenv_restore(saved):
    """For the tests: restore an environment returned by _test_fpenv_set."""
    fse_test_fpenv_restore(saved)


def _test_fpenv_get():
    """For the tests: the floating-point control register (MXCSR or FPCR)."""
    return fse_test_fpenv_get()


def _check_isa(isa):
    if isa is None:
        return _ISAS[default_isa()]
    if isa not in _ISAS:
        raise ValueError(f"isa must be one of {sorted(_ISAS)}, not {isa!r}")
    if not fse_has_isa(_ISAS[isa]):
        raise ValueError(f"this CPU cannot run the {isa} kernel; available: {available_isas()}")
    return _ISAS[isa]


def _float32_vector(value, name, length):
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != (length,):
        raise ValueError(f"{name} must be a float32 vector of {length} values")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _rows(value, name):
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.dtype not in _DTYPES:
        raise ValueError(f"{name} must be a matrix of float32, float16 or int8 rows")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


cdef class WeightedMean:
    """The parameters of weighted_mean@1 on one set of word rows, checked once.

    ``vectors`` (and ``ngram_vectors``) are the stored rows as float32, float16 or int8
    codes, with ``scales`` (``ngram_scales``) for int8; ``weights`` one float32 weight per
    word; ``oov_weight`` the weight of unknown FastText words. ``pool`` and ``accumulate``
    give the bits of ``fse._reference``; several threads may call them at once.
    """

    cdef fse_params params
    cdef int isa
    cdef readonly int dim
    cdef readonly object isa_name
    cdef Py_ssize_t n_words
    cdef Py_ssize_t n_buckets
    cdef bint fasttext
    cdef object _keep

    def __init__(self, vectors, weights, scales=None, ngram_vectors=None, ngram_scales=None,
                 oov_weight=None, *, isa=None):
        self.isa = _check_isa(isa)
        self.isa_name = next(k for k, v in _ISAS.items() if v == self.isa)
        vectors = _rows(vectors, "vectors")
        self.n_words = vectors.shape[0]
        self.dim = _check_dim(vectors.shape[1])
        if self.dim < 1:
            raise ValueError("vectors need at least one dimension")
        int8 = vectors.dtype == np.int8
        weights = _float32_vector(weights, "weights", self.n_words)
        if int8 != (scales is not None):
            raise ValueError("scales are given exactly for int8 vectors")
        if int8:
            scales = _float32_vector(scales, "scales", self.n_words)
        self.fasttext = ngram_vectors is not None
        if self.fasttext:
            ngram_vectors = _rows(ngram_vectors, "ngram_vectors")
            if ngram_vectors.dtype != vectors.dtype or ngram_vectors.shape[1] != self.dim:
                raise ValueError("ngram_vectors must have the dtype and dimension of vectors")
            self.n_buckets = ngram_vectors.shape[0]
            if int8 != (ngram_scales is not None):
                raise ValueError("ngram_scales are given exactly for int8 ngram vectors")
            if int8:
                ngram_scales = _float32_vector(ngram_scales, "ngram_scales", self.n_buckets)
        elif ngram_scales is not None:
            raise ValueError("ngram_scales without ngram_vectors")
        self.params.dtype = _DTYPES[vectors.dtype]
        self.params.dim = self.dim
        self.params.row_bytes = <size_t>(self.dim * vectors.itemsize)
        self.params.rows = np.PyArray_DATA(vectors)
        self.params.scales = <const float *>np.PyArray_DATA(scales) if int8 else NULL
        self.params.weights = <const float *>np.PyArray_DATA(weights)
        self.params.ngram_rows = np.PyArray_DATA(ngram_vectors) if self.fasttext else NULL
        self.params.ngram_scales = (
            <const float *>np.PyArray_DATA(ngram_scales) if self.fasttext and int8 else NULL
        )
        # As the reference: float32(oov_weight); None only without unknown words.
        with _KernelEnv(), np.errstate(over="ignore", invalid="ignore"):  # as the reference
            self.params.oov_weight = np.float32(oov_weight) if oov_weight is not None else 0
        self._keep = (vectors, weights, scales, ngram_vectors, ngram_scales, oov_weight)

    def _check_batch(self, batch):
        from fse._contract import OOV_TOKEN, Batch, _check_offsets

        if not isinstance(batch, Batch):
            raise TypeError(f"batch must be a Batch, not {type(batch).__name__}")
        tokens = batch.tokens
        # Batch checked its arrays when it was made; the kernel reads through raw pointers,
        # so the offsets are checked again here.
        _check_offsets(batch.offsets, len(tokens), "offsets")
        unknown = tokens == OOV_TOKEN
        if unknown.any():
            if batch.ngrams is None or batch.ngram_offsets is None:
                raise ValueError("an unknown FastText word needs ngram buckets")
            n_unknown = int(np.count_nonzero(unknown))
            if len(batch.ngram_offsets) != n_unknown + 1:
                raise ValueError(f"ngram_offsets must have {n_unknown + 1} entries for {n_unknown} unknown words")
            _check_offsets(batch.ngram_offsets, len(batch.ngrams), "ngram_offsets")
            if not self.fasttext:
                raise ValueError("an unknown FastText word needs ngram vectors")
            if self._keep[5] is None:
                raise ValueError("an unknown FastText word needs oov_weight")
            if len(batch.ngrams) and int(batch.ngrams.max()) >= self.n_buckets:
                raise ValueError(f"ngram buckets must be below {self.n_buckets}")
        known = tokens[~unknown]
        if len(known) and int(known.max()) >= self.n_words:
            raise ValueError(f"token ids must be below {self.n_words}, the number of words")

    cdef int64_t _run(self, batch, float[:, ::1] out, int64_t *counts, int finish) except? -2:
        cdef fse_batch b
        cdef const np.int64_t[::1] offsets = batch.offsets
        cdef const np.uint32_t[::1] tokens = batch.tokens
        cdef const np.int64_t[::1] ngram_offsets
        cdef const np.uint32_t[::1] ngrams
        b.n_sentences = offsets.shape[0] - 1
        b.offsets = <const int64_t *>&offsets[0]
        b.tokens = <const uint32_t *>&tokens[0] if tokens.shape[0] else NULL
        b.ngram_offsets = NULL
        b.ngrams = NULL
        if batch.ngram_offsets is not None:
            ngram_offsets = batch.ngram_offsets
            ngrams = batch.ngrams
            b.ngram_offsets = <const int64_t *>&ngram_offsets[0]
            b.ngrams = <const uint32_t *>&ngrams[0] if ngrams.shape[0] else NULL
        cdef float *target = &out[0, 0] if out.shape[0] else NULL
        cdef int64_t bad
        if b.n_sentences == 0:
            return -1
        with nogil:
            bad = fse_run(self.isa, &self.params, &b, target, counts, finish)
        return bad

    def _out(self, batch, out):
        n = batch.n_sentences
        if out is None:
            return np.empty((n, self.dim), dtype=np.float32)
        if (
            not isinstance(out, np.ndarray)
            or out.dtype != np.float32
            or out.shape != (n, self.dim)
            or not out.flags.c_contiguous
            or not out.flags.writeable
        ):
            raise ValueError(f"out must be a writable C-contiguous float32 array of shape {(n, self.dim)}")
        for kept in self._keep[:5]:
            if kept is not None and np.may_share_memory(out, kept):
                raise ValueError("out must not share memory with the vectors or weights")
        for kept in (batch.offsets, batch.tokens, batch.ngram_offsets, batch.ngrams):
            if kept is not None and np.may_share_memory(out, kept):
                raise ValueError("out must not share memory with the batch")
        return out

    def pool(self, batch, out=None):
        """weighted_mean@1 of every sentence of ``batch``, into ``out`` or a new array."""
        self._check_batch(batch)
        out = self._out(batch, out)
        bad = self._run(batch, out, NULL, 1)
        if bad >= 0:
            raise ValueError(f"row {batch.first_row + bad} contains NaN or infinite values")
        return out

    def accumulate(self, batch):
        """The accumulators and the token counts of every sentence of ``batch``."""
        self._check_batch(batch)
        accs = np.empty((batch.n_sentences, self.dim), dtype=np.float32)
        counts = np.zeros(batch.n_sentences, dtype=np.int64)
        cdef np.int64_t[::1] view = counts
        bad = self._run(batch, accs, <int64_t *>&view[0] if counts.shape[0] else NULL, 0)
        if bad >= 0:
            raise ValueError(f"row {batch.first_row + bad} contains NaN or infinite values")
        return accs, counts


def pool_weighted_mean(batch, vectors, weights, scales=None, ngram_vectors=None,
                       ngram_scales=None, oov_weight=None, *, isa=None):
    """``weighted_mean@1`` with the signature of ``fse._reference.pool_weighted_mean``."""
    return WeightedMean(vectors, weights, scales, ngram_vectors, ngram_scales, oov_weight, isa=isa).pool(batch)


def accumulate(batch, vectors, weights, scales=None, ngram_vectors=None, ngram_scales=None,
               oov_weight=None, *, isa=None):
    """``fse._reference.accumulate`` in the kernel."""
    return WeightedMean(vectors, weights, scales, ngram_vectors, ngram_scales, oov_weight, isa=isa).accumulate(batch)
