# Included by _native.pyx: wordpiece@1, its state, the greedy match and one sentence.

# ---------------------------------------------------------------------------------------
# wordpiece@1 (ARCHITECTURE.md, "wordpiece@1"), as in fse._reference_wordpiece. Correct
# first, not fast: the added-token scan, the normalization and the greedy match are plain.
# ---------------------------------------------------------------------------------------

cdef extern from "_native_wordpiece.h":
    int64_t fse_wp_added_at(const unsigned char *p, int64_t i, int64_t n,
                            const unsigned char *first, const unsigned char *bytes,
                            const int64_t *offsets, int64_t count) nogil
    int64_t fse_wp_next_candidate(const unsigned char *p, int64_t i, int64_t n,
                                  const unsigned char *first, int single) nogil
    int64_t fse_wp_normalize(const unsigned char *p, int64_t n, unsigned char *out,
                             uint32_t *run) nogil
    int64_t fse_wp_normalize_ascii(const unsigned char *p, int64_t n, unsigned char *out,
                                   uint32_t *run) nogil
    int fse_wp_ascii_init() nogil
    int fse_wp_next_word(const unsigned char *q, int64_t m, int64_t *pos, int64_t *start,
                         int64_t *end) nogil
    int64_t fse_wp_validate(const unsigned char *p, int64_t i, int64_t n) nogil
    int64_t fse_wp_char_cut(const unsigned char *p, int64_t n, int64_t max_chars) nogil

# Parameters stay below this, as they are int64 in the state.
cdef int64_t _WP_LIMIT = 1 << 62
# The ASCII tables of the fast paths, derived from the Unicode table once (0, or 1 + the
# first byte they cannot hold, which WordPieceState refuses).
cdef int _WP_ASCII = fse_wp_ascii_init()


cdef int64_t _wp_int(value, name, int64_t low) except -1:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, not {type(value).__name__}")
    if not low <= value < _WP_LIMIT:
        raise ValueError(f"{name} must be between {low} and 2**62 - 1, not {value}")
    return value


def _vocab_arrays(vocab_bytes, vocab_offsets):
    """Own copies of a vocabulary, checked as VocabTable checks them."""
    if (not isinstance(vocab_bytes, np.ndarray) or vocab_bytes.dtype != np.uint8
            or vocab_bytes.ndim != 1):
        raise ValueError("vocab_bytes must be a uint8 vector")
    if (not isinstance(vocab_offsets, np.ndarray) or vocab_offsets.dtype != np.int64
            or vocab_offsets.ndim != 1 or len(vocab_offsets) == 0):
        raise ValueError("vocab_offsets must be a non-empty int64 vector")
    data = np.ascontiguousarray(vocab_bytes).copy()
    offsets = np.ascontiguousarray(vocab_offsets).copy()
    if offsets[0] != 0 or offsets[len(offsets) - 1] != len(data) or (offsets[1:] < offsets[:-1]).any():
        raise ValueError(f"vocab_offsets must run from 0 to {len(data)} without decreasing")
    return data, offsets


cdef class WordPieceState:
    """The state of wordpiece@1 for one vocabulary, built once under the GIL and shared
    by all workers: the VocabTable of all tokens (whose index is the token id), a second
    table of the continuation tokens without their prefix with the id of each, and the
    added tokens, longest first.

    ``vocab_bytes`` and ``vocab_offsets`` are the tokens of ``table`` in id order (checked:
    every token is found at its id); ``added_ids`` the ids of the added tokens, whose
    contents are their tokens. ``max_tokens`` and ``max_chars`` of 0 cut nothing.

    ``_generic`` (private, for the tests, never persisted) takes the generic path of every
    step, without the fast paths, which give the same ids.
    """

    cdef readonly VocabTable table
    cdef VocabTable continuation
    cdef object _continuation_ids, _added_bytes, _added_offsets, _added_ids
    cdef const uint32_t *cont_ids
    cdef const unsigned char *added_bytes
    cdef const int64_t *added_offsets
    cdef const uint32_t *added_ids
    cdef int64_t n_added
    cdef unsigned char first[256]
    cdef int single_first
    cdef bint generic
    cdef readonly uint32_t unk_id
    cdef readonly int64_t max_input_chars, max_tokens, max_chars
    cdef readonly bint drop_unknown
    cdef int _state

    def __init__(self, VocabTable table not None, vocab_bytes, vocab_offsets, unk_id, prefix,
                 max_input_chars_per_word, added_ids, max_tokens=0, max_chars=0,
                 drop_unknown=True, *, _generic=False):
        if self._state != TABLE_NEW:
            raise RuntimeError("a WordPieceState is built once; make a new one")
        self._state = TABLE_BUILDING
        try:
            if not isinstance(_generic, bool):
                raise TypeError(f"_generic must be a bool, not {type(_generic).__name__}")
            self.generic = _generic
            self._build(table, vocab_bytes, vocab_offsets, unk_id, prefix,
                        max_input_chars_per_word, added_ids, max_tokens, max_chars, drop_unknown)
        except BaseException:
            self._state = TABLE_FAILED
            raise
        self._state = TABLE_READY

    cdef _build(self, VocabTable table, vocab_bytes, vocab_offsets, unk_id, prefix,
                max_input_chars_per_word, added_ids, max_tokens, max_chars, drop_unknown):
        table._check_ready()
        if _WP_ASCII:
            raise RuntimeError(f"the Unicode table of wordpiece@1 has an ASCII byte {_WP_ASCII - 1} "
                               "of a shape the fast paths cannot take")
        data, offsets = _vocab_arrays(vocab_bytes, vocab_offsets)
        cdef Py_ssize_t n = len(offsets) - 1
        if n != len(table):
            raise ValueError(f"the table holds {len(table)} tokens, the vocabulary {n}")
        _check_ids(table, data, offsets)
        # Checked as the full int, then stored: a cast first would wrap 2**32 + 1 to 1.
        if _wp_int(unk_id, "unk_id", 0) >= n:
            raise ValueError(f"unk_id is {unk_id}, but the vocabulary holds {n} tokens")
        self.unk_id = <uint32_t>unk_id
        if not isinstance(prefix, str) or not prefix:
            raise ValueError(f"the prefix must be a non-empty str, not {prefix!r}")
        self.max_input_chars = _wp_int(max_input_chars_per_word, "max_input_chars_per_word", 1)
        self.max_tokens = _wp_int(max_tokens, "max_tokens", 0)
        self.max_chars = _wp_int(max_chars, "max_chars", 0)
        if not isinstance(drop_unknown, bool):
            raise TypeError(f"drop_unknown must be a bool, not {type(drop_unknown).__name__}")
        self.drop_unknown = drop_unknown
        self._build_continuation(data, offsets, prefix.encode("utf-8"))
        self._build_added(data, offsets, added_ids)
        self.table = table

    cdef _build_continuation(self, data, offsets, bytes prefix):
        """The continuation tokens without their prefix, and their ids."""
        cdef Py_ssize_t k = len(prefix)
        raw = data.tobytes()
        starts = offsets[:-1]
        ids = [i for i in range(len(starts)) if raw.startswith(prefix, starts[i], offsets[i + 1])]
        pieces = [raw[starts[i] + k : offsets[i + 1]] for i in ids]
        lengths = np.zeros(len(pieces) + 1, np.int64)
        np.cumsum([len(p) for p in pieces], out=lengths[1:])
        self.continuation = VocabTable(np.frombuffer(b"".join(pieces), np.uint8).copy(), lengths)
        self._continuation_ids = np.array(ids, dtype=np.uint32)
        self.cont_ids = <const uint32_t *>np.PyArray_DATA(self._continuation_ids)

    cdef _build_added(self, data, offsets, added_ids):
        """The added tokens by length, longest first, and the bytes they start with."""
        if not isinstance(added_ids, tuple):
            raise TypeError(f"added_ids must be a tuple of ints, not {type(added_ids).__name__}")
        cdef Py_ssize_t n = len(offsets) - 1
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
        self._added_bytes = np.frombuffer(b"".join(tokens[j] for j in order) + b"\0", np.uint8)
        self._added_offsets = np.zeros(len(tokens) + 1, np.int64)
        np.cumsum([len(tokens[j]) for j in order], out=self._added_offsets[1:])
        self._added_ids = np.array([checked[j] for j in order], dtype=np.uint32)
        memset(self.first, 0, 256)
        for t in tokens:
            self.first[t[0]] = 1
        starts = {t[0] for t in tokens}
        self.single_first = starts.pop() if len(starts) == 1 else -1
        self.n_added = len(tokens)
        self.added_bytes = <const unsigned char *>np.PyArray_DATA(self._added_bytes)
        self.added_offsets = <const int64_t *>np.PyArray_DATA(self._added_offsets)
        self.added_ids = <const uint32_t *>np.PyArray_DATA(self._added_ids)

    cdef inline int _check_ready(self) except -1:
        if self._state != TABLE_READY:
            raise RuntimeError("this WordPieceState was not built; make a new one")
        return 0

    def __reduce__(self):
        raise TypeError("a WordPieceState cannot be pickled; build it from the encoder")


cdef _check_ids(VocabTable table, data, offsets):
    """Every token of the vocabulary is found in the table at its id."""
    cdef const unsigned char[::1] d = data if len(data) else np.zeros(1, np.uint8)
    cdef const np.int64_t[::1] o = offsets
    cdef Py_ssize_t i, bad = -1
    with nogil:
        for i in range(o.shape[0] - 1):
            if _vocab_lookup(&table.view, &d[0] + o[i], o[i + 1] - o[i]) != <uINT_t>i:
                bad = i
                break
    if bad >= 0:
        raise ValueError(f"token {bad} of the vocabulary is not token {bad} of the table")


cdef inline Py_ssize_t _wp_word(WordPieceState st, const unsigned char *w, Py_ssize_t n,
                                uINT_t *ids) noexcept nogil:
    """Step 5 for the word w[0:n] of the normalized buffer (16 bytes readable after any
    piece): its ids, greedy from the left, or one unk_id; returns how many."""
    cdef Py_ssize_t chars = 0, k, start = 0, end, count = 0
    cdef uINT_t index
    for k in range(n):
        chars += (w[k] & 0xC0) != 0x80
    if chars > st.max_input_chars:
        ids[0] = st.unk_id
        return 1
    while start < n:
        end = n
        while True:
            if start == 0:
                index = _vocab_lookup(&st.table.view, w, end, True)
            else:
                index = _vocab_lookup(&st.continuation.view, w + start, end - start, True)
                if index != NOT_FOUND:
                    index = st.cont_ids[index]
            if index != NOT_FOUND:
                break
            end -= 1
            while end > start and (w[end] & 0xC0) == 0x80:
                end -= 1
            if end == start:
                ids[0] = st.unk_id
                return 1
        ids[count] = index
        count += 1
        start = end
    return count


cdef enum:
    WP_GO_ON = 0
    WP_FULL = 1
    WP_BAD = 2


cdef inline int _wp_piece(WordPieceState st, const unsigned char *p, Py_ssize_t n, Workspace ws,
                          uINT_t *ids, Py_ssize_t *count, Py_ssize_t *bad) noexcept nogil:
    """Steps 3 to 5 for the piece p[0:n] between added tokens; WP_FULL once the token cut
    is reached after a whole word."""
    cdef unsigned char *q = <unsigned char *>ws.norm.data
    cdef int64_t m = (fse_wp_normalize(p, n, q, <uint32_t *>ws.run.data) if st.generic
                      else fse_wp_normalize_ascii(p, n, q, <uint32_t *>ws.run.data))
    cdef int64_t pos = 0, start, end
    if m < 0:
        bad[0] = -1 - m
        return WP_BAD
    while fse_wp_next_word(q, m, &pos, &start, &end):
        count[0] += _wp_word(st, q + start, end - start, ids + count[0])
        if st.max_tokens and count[0] >= st.max_tokens:
            return WP_FULL
    return WP_GO_ON


cdef int _wordpiece_sentence(WordPieceState st, const unsigned char *p, Py_ssize_t n,
                             Workspace ws, Py_ssize_t *bad) noexcept nogil:
    """Append the ids of the sentence p[0:n] (steps 1 to 6) to the batch buffers of ws."""
    cdef Py_ssize_t limit = n, pos = 0, piece = 0, done = 0, count = 0, kept = 0, k, i
    cdef int64_t found, cut
    cdef int status = WP_GO_ON
    cdef uINT_t *ids
    if st.max_chars:  # the whole sentence is validated here
        cut = fse_wp_char_cut(p, n, st.max_chars)
        if cut < 0:
            bad[0] = -1 - cut
            return ENCODE_BAD_UTF8
        limit = cut
    ws.norm.size = 0
    ws.run.size = 0
    # At most 3 normalized bytes per byte in (and 32 to read words 16 bytes wide), so as
    # many items of a run and ids, and one id per added token.
    if (_ensure(&ws.norm, 3 * limit + 32) or _ensure(&ws.run, 12 * limit + 4)
            or _ensure(&ws.tokens, (4 * limit + 1) * <Py_ssize_t>sizeof(uINT_t))):
        return ENCODE_NO_MEMORY
    ids = <uINT_t *>(ws.tokens.data + ws.tokens.size)
    while pos < limit and status == WP_GO_ON:
        if not st.generic:  # straight to the next byte a token may start at
            pos = fse_wp_next_candidate(p, pos, limit, st.first, st.single_first) if st.n_added else limit
            if pos == limit:
                break
        found = fse_wp_added_at(p, pos, limit, st.first, st.added_bytes, st.added_offsets,
                                st.n_added) if st.n_added else -1
        if found < 0:
            pos += 1
            continue
        status = _wp_piece(st, p + piece, pos - piece, ws, ids, &count, bad)
        done = pos
        if status == WP_GO_ON:
            ids[count] = st.added_ids[found]
            count += 1
            pos += st.added_offsets[found + 1] - st.added_offsets[found]
            piece = done = pos
            if st.max_tokens and count >= st.max_tokens:
                status = WP_FULL
    if status == WP_GO_ON:
        status = _wp_piece(st, p + piece, limit - piece, ws, ids, &count, bad)
        done = limit
    if status == WP_BAD:
        bad[0] += piece
        return ENCODE_BAD_UTF8
    if not st.max_chars and done < n:  # the rest of a sentence the token cut ended early
        found = fse_wp_validate(p, done, n)
        if found >= 0:
            bad[0] = found
            return ENCODE_BAD_UTF8
    if st.max_tokens and count > st.max_tokens:
        count = st.max_tokens
    ws.tokens_seen += count
    if st.drop_unknown:
        for i in range(count):
            if ids[i] != st.unk_id:
                ids[kept] = ids[i]
                kept += 1
        count = kept
    ws.tokens.size += count * sizeof(uINT_t)
    if _push_i64_nogil(&ws.offsets, ws.tokens.size // sizeof(uINT_t)):
        return ENCODE_NO_MEMORY
    return ENCODE_OK
