# Included by _native.pyx: unigram@1, its state, the normalizer steps, the lattice and one
# sentence.

# ---------------------------------------------------------------------------------------
# unigram@1 (ARCHITECTURE.md, "unigram@1"), as in fse._reference_unigram. Correct first,
# not fast: every step is its own pass over the piece, and the candidates of a start are
# found by one lookup of the pieces and one of their proper prefixes per code point.
# ---------------------------------------------------------------------------------------

from libc.stdint cimport uint8_t

cdef extern from "_native_unigram.h":
    int fse_utf8_decode(const unsigned char *p, int64_t i, int64_t end, uint32_t *cp) nogil
    ctypedef struct fse_uni_cand:
        int64_t end
        double score
        uint32_t id
    uint8_t fse_gc_props(uint32_t cp) nogil
    int64_t fse_gc_next(const uint32_t *cps, int64_t i, int64_t n) nogil
    int fse_cm_find(const uint32_t *units, int64_t n_units, int64_t root,
                    const unsigned char *key, int64_t length, uint32_t *value) nogil
    double fse_uni_unk_score(const double *scores, int64_t n) nogil
    int64_t fse_uni_viterbi(int64_t size, const int64_t *pos, const int64_t *first,
                            int64_t count, const fse_uni_cand *cands, double *best,
                            int64_t *best_start, uint32_t *best_id) nogil
    unsigned FSE_GC_WHITE_SPACE
    int FSE_UNI_GRAPHEME_BYTES

# The normalizer steps of the state, in the order they apply; a row of _UNI_ROW int64:
# kind, the offset and length of the first bytes (pattern) and of the second (content)
# in the pool, and a number (min of runs; 1 left + 2 right for strip).
cdef enum:
    UNI_PRECOMPILED = 0
    UNI_REPLACE = 1
    UNI_RUNS_SPACE = 2
    UNI_RUNS_WHITE_SPACE = 3
    UNI_STRIP = 4
    _UNI_ROW = 6


def _grapheme_props(code_points):
    """The byte of the native grapheme table of every uint32 code point; for tests."""
    cdef const uint32_t[::1] cps = np.ascontiguousarray(code_points, dtype=np.uint32)
    out = np.empty(cps.shape[0], dtype=np.uint8)
    cdef uint8_t[::1] o = out
    cdef Py_ssize_t i
    with nogil:
        for i in range(cps.shape[0]):
            o[i] = fse_gc_props(cps[i])
    return out


def _grapheme_boundaries(code_points):
    """The indices with a grapheme boundary before them (0 < i < n), as
    fse._reference_unigram.boundaries; for tests."""
    cps = np.ascontiguousarray(code_points, dtype=np.uint32)
    cdef const uint32_t[::1] c = cps
    cdef int64_t n = c.shape[0], i = 0
    out = []
    while i < n:
        i = fse_gc_next(&c[0], i, n)
        if i < n:
            out.append(i)
    return out


def _added_parts(data, offsets, added_ids):
    """The added tokens by length, longest first: their bytes (with a NUL after), offsets
    and ids, the bytes they start with, and that byte if all start with one, else -1."""
    if not isinstance(added_ids, tuple):
        raise TypeError(f"added_ids must be a tuple of ints, not {type(added_ids).__name__}")
    n = len(offsets) - 1
    checked = [_wp_int(i, "an added id", 0) for i in added_ids]
    if len(set(checked)) != len(checked) or any(i >= n for i in checked):
        raise ValueError(f"the added ids must be unique ids below {n}, not {list(added_ids)}")
    tokens = [data[offsets[i] : offsets[i + 1]].tobytes() for i in checked]
    for i, t in zip(checked, tokens):
        if not t:
            raise ValueError("an added token must not be empty")
        try:  # the scan matches bytes, so a token that is no UTF-8 would hide bad input
            t.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"the added token {i} is not valid UTF-8: {t!r}") from None
    order = sorted(range(len(tokens)), key=lambda j: -len(tokens[j]))
    joined = np.frombuffer(b"".join(tokens[j] for j in order) + b"\0", np.uint8)
    lengths = np.zeros(len(tokens) + 1, np.int64)
    np.cumsum([len(tokens[j]) for j in order], out=lengths[1:])
    ids = np.array([checked[j] for j in order], dtype=np.uint32)
    starts = {t[0] for t in tokens}
    return joined, lengths, ids, starts, (next(iter(starts)) if len(starts) == 1 else -1)


def _proper_prefixes(data, offsets):
    """The prefixes of the pieces that end on a code point boundary, are no piece and are
    shorter than their piece: with the pieces, the stop signal of the candidates."""
    raw = data.tobytes()
    pieces = {raw[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)}
    out = set()
    for piece in pieces:
        for k in range(1, len(piece)):
            if piece[k] & 0xC0 != 0x80 and piece[:k] not in pieces:
                out.add(piece[:k])
    ordered = sorted(out)
    lengths = np.zeros(len(ordered) + 1, np.int64)
    np.cumsum([len(p) for p in ordered], out=lengths[1:])
    return np.frombuffer(b"".join(ordered), np.uint8).copy(), lengths


def _steps_parts(steps):
    """The rows of the normalizer steps and the pool of their bytes."""
    if not isinstance(steps, tuple):
        raise TypeError(f"steps must be a tuple, not {type(steps).__name__}")
    pool = bytearray()
    rows = []

    def put(value, name):
        if not isinstance(value, bytes):
            raise TypeError(f"{name} must be bytes, not {type(value).__name__}")
        value.decode("utf-8")  # raises for bytes that are no UTF-8
        start = len(pool)
        pool.extend(value)
        return start, len(value)

    for step in steps:
        kind = step[0] if isinstance(step, tuple) and step else None
        if kind == "precompiled" and len(step) == 1:
            rows.append((UNI_PRECOMPILED, 0, 0, 0, 0, 0))
        elif kind == "replace" and len(step) == 3:
            if not step[1]:
                raise ValueError("the pattern of replace must not be empty")
            rows.append((UNI_REPLACE, *put(step[1], "a pattern"), *put(step[2], "a content"), 0))
        elif kind == "runs" and len(step) == 4 and step[1] in ("space", "white_space"):
            code = UNI_RUNS_SPACE if step[1] == "space" else UNI_RUNS_WHITE_SPACE
            rows.append((code, 0, 0, *put(step[3], "a content"), _wp_int(step[2], "min", 1)))
        elif kind == "strip" and len(step) == 3 and all(isinstance(v, bool) for v in step[1:]):
            rows.append((UNI_STRIP, 0, 0, 0, 0, int(step[1]) + 2 * int(step[2])))
        else:
            raise ValueError(f"no normalizer step {step!r}")
    return np.array(rows, dtype=np.int64).reshape(-1, _UNI_ROW), np.frombuffer(bytes(pool) + b"\0", np.uint8)


cdef class UnigramState:
    """The state of unigram@1 for one vocabulary, built once under the GIL and shared by
    all workers: the VocabTable of the pieces (whose index is the id), a VocabTable of
    their proper prefixes, the float64 scores and unk_score, the charsmap, the normalizer
    steps, the replacement of Metaspace and the added tokens, longest first.

    ``vocab_bytes`` and ``vocab_offsets`` are the pieces of ``table`` in id order (checked:
    every piece is found at its id); ``scores`` one finite float64 per piece; ``steps`` a
    tuple of ("precompiled",), ("replace", pattern, content), ("runs", "space" or
    "white_space", min, content) and ("strip", left, right), bytes as UTF-8; ``charsmap``
    None or (units, values, root) of a blob that fse._charsmap checked (the search is
    bounds checked anyway). ``max_tokens`` and ``max_chars`` of 0 cut nothing.
    """

    cdef readonly VocabTable table
    cdef VocabTable prefixes
    cdef object _scores, _units, _values, _ops, _pool, _replacement
    cdef object _added_bytes, _added_offsets, _added_ids
    cdef const double *scores
    cdef const uint32_t *units
    cdef int64_t n_units, root
    cdef const unsigned char *values
    cdef int64_t n_values
    cdef bint has_charsmap
    cdef const int64_t *ops
    cdef int64_t n_ops
    cdef const unsigned char *pool
    cdef const unsigned char *replacement
    cdef int64_t n_replacement, max_piece
    cdef const unsigned char *added_bytes
    cdef const int64_t *added_offsets
    cdef const uint32_t *added_ids
    cdef int64_t n_added
    cdef unsigned char first[256]
    cdef int single_first
    cdef readonly uint32_t unk_id
    cdef readonly double unk_score
    cdef readonly int64_t max_tokens, max_chars
    cdef readonly bint drop_unknown
    cdef int _state

    def __init__(self, VocabTable table not None, vocab_bytes, vocab_offsets, scores, unk_id,
                 added_ids, steps, charsmap, replacement, max_tokens=0, max_chars=0,
                 drop_unknown=True):
        if self._state != TABLE_NEW:
            raise RuntimeError("a UnigramState is built once; make a new one")
        self._state = TABLE_BUILDING
        try:
            self._build(table, vocab_bytes, vocab_offsets, scores, unk_id, added_ids, steps,
                        charsmap, replacement, max_tokens, max_chars, drop_unknown)
        except BaseException:
            self._state = TABLE_FAILED
            raise
        self._state = TABLE_READY

    cdef _build(self, VocabTable table, vocab_bytes, vocab_offsets, scores, unk_id, added_ids,
                steps, charsmap, replacement, max_tokens, max_chars, drop_unknown):
        table._check_ready()
        data, offsets = _vocab_arrays(vocab_bytes, vocab_offsets)
        cdef Py_ssize_t n = len(offsets) - 1
        if n != len(table):
            raise ValueError(f"the table holds {len(table)} pieces, the vocabulary {n}")
        _check_ids(table, data, offsets)
        if _wp_int(unk_id, "unk_id", 0) >= n:
            raise ValueError(f"unk_id is {unk_id}, but the vocabulary holds {n} pieces")
        self.unk_id = <uint32_t>unk_id
        self.max_tokens = _wp_int(max_tokens, "max_tokens", 0)
        self.max_chars = _wp_int(max_chars, "max_chars", 0)
        if not isinstance(drop_unknown, bool):
            raise TypeError(f"drop_unknown must be a bool, not {type(drop_unknown).__name__}")
        self.drop_unknown = drop_unknown
        self._build_scores(scores, n)
        self._build_steps(steps, charsmap, replacement)
        self._build_added(data, offsets, added_ids)
        prefix_bytes, prefix_offsets = _proper_prefixes(data, offsets)
        self.prefixes = VocabTable(prefix_bytes, prefix_offsets)
        self.max_piece = int(np.diff(offsets).max())
        self.table = table

    cdef _build_scores(self, scores, Py_ssize_t n):
        if (not isinstance(scores, np.ndarray) or scores.dtype != np.dtype("<f8")
                or scores.shape != (n,)):
            raise ValueError(f"scores must be a float64 vector of the {n} pieces")
        self._scores = np.ascontiguousarray(scores).copy()
        if not np.isfinite(self._scores).all():
            raise ValueError("every score must be finite")
        self.scores = <const double *>np.PyArray_DATA(self._scores)
        with nogil:  # in the environment of the kernels, as every sum of the Viterbi
            self.unk_score = fse_uni_unk_score(self.scores, n)

    cdef _build_steps(self, steps, charsmap, replacement):
        ops, pool = _steps_parts(steps)
        kinds = ops[:, 0].tolist()
        if (charsmap is None) != (UNI_PRECOMPILED not in kinds):
            raise ValueError("a charsmap is given exactly with a precompiled step")
        if kinds.count(UNI_PRECOMPILED) > 1:
            raise ValueError("unigram@1 takes at most one precompiled step")
        self._ops, self._pool = ops, pool
        self.ops = <const int64_t *>np.PyArray_DATA(ops)
        self.n_ops = len(ops)
        self.pool = <const unsigned char *>np.PyArray_DATA(pool)
        if charsmap is not None:
            units, values, root = charsmap
            if (not isinstance(units, np.ndarray) or units.dtype != np.uint32 or units.ndim != 1
                    or not len(units) or not isinstance(values, bytes)):
                raise ValueError("a charsmap is (units, values, root): a uint32 vector and bytes")
            self._units = np.ascontiguousarray(units).copy()
            self._values = values + b"\0"
            self.units = <const uint32_t *>np.PyArray_DATA(self._units)
            self.n_units = len(self._units)
            self.root = _wp_int(root, "root", 0)
            self.values = self._values
            self.n_values = len(values)
            self.has_charsmap = True
        if not isinstance(replacement, bytes) or not replacement:
            raise ValueError("the replacement of Metaspace must be non-empty bytes")
        replacement.decode("utf-8")
        self._replacement = replacement
        self.replacement = self._replacement
        self.n_replacement = len(replacement)

    cdef _build_added(self, data, offsets, added_ids):
        joined, lengths, ids, starts, single = _added_parts(data, offsets, added_ids)
        self._added_bytes, self._added_offsets, self._added_ids = joined, lengths, ids
        memset(self.first, 0, 256)
        for b in starts:
            self.first[b] = 1
        self.single_first = single
        self.n_added = len(ids)
        self.added_bytes = <const unsigned char *>np.PyArray_DATA(joined)
        self.added_offsets = <const int64_t *>np.PyArray_DATA(lengths)
        self.added_ids = <const uint32_t *>np.PyArray_DATA(ids)

    cdef inline int _check_ready(self) except -1:
        if self._state != TABLE_READY:
            raise RuntimeError("this UnigramState was not built; make a new one")
        return 0

    def __reduce__(self):
        raise TypeError("a UnigramState cannot be pickled; build it from the encoder")

