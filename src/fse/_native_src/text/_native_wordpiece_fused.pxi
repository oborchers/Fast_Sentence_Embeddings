# Included by _native.pyx: shared ASCII piece driver for plain and watched WordPiece.

cdef extern from "_native_wordpiece_fused.h":
    ctypedef struct fse_wp_ascii_scan:
        int64_t pos, available, start, end
        uint64_t low, other, key[2]
    int fse_wp_ascii_next(fse_wp_ascii_scan *s, const unsigned char *p,
                          int64_t n, unsigned char *q) nogil


cdef inline uint64_t _wp_short_hash(fse_wp_ascii_scan *s) noexcept nogil:
    cdef uint64_t h = _mix(_mix(0x9E3779B97F4A7C15ULL ^ <uint64_t>(s.end - s.start),
                                s.key[0]), s.key[1])
    h *= 0x94D049BB133111EBULL
    return h ^ (h >> 32)


cdef inline Py_ssize_t _wp_ascii_word(WordPieceState st, const unsigned char *q,
                                      fse_wp_ascii_scan *s, uINT_t *ids) noexcept nogil:
    """Whole-word probe from register keys, or the unchanged deferred long-word hash."""
    cdef Py_ssize_t n = s.end - s.start
    cdef uint64_t h
    cdef uint32_t index
    if n > st.max_input_chars:  # every printable ASCII byte is one code point
        ids[0] = st.unk_id
        return 1
    if n <= INLINE_KEY_BYTES:
        h = _wp_short_hash(s)
    else:
        # The final length seeds the first nonlinear mix. Re-read only long words.
        h = _hash_word(q + s.start, n, <np.uint64_t *>s.key, True)
    index = _vocab_lookup_hashed(&st.table.view, q + s.start, n, h, <np.uint64_t *>s.key)
    if index != NOT_FOUND:
        ids[0] = index
        return 1
    return _wp_trie_tail(st, q + s.start, n, ids)


cdef inline int _wp_piece_fused(WordPieceState st, const unsigned char *p, Py_ssize_t n,
                                Workspace ws, uINT_t *ids, Py_ssize_t *count,
                                Py_ssize_t *bad, Py_ssize_t stop_at) noexcept nogil:
    """One shared stop rule: unlimited, the token limit, or the watched flag limit.

    On replay, all provisional ids are overwritten from the span's original count.
    q is wholly rewritten by normalization, with an empty mark run. No workspace
    counters are committed here. Even a token stop validates the entire raw span.
    """
    cdef unsigned char *q = <unsigned char *>ws.norm.data
    cdef fse_wp_ascii_scan scan
    memset(&scan, 0, sizeof(fse_wp_ascii_scan))
    cdef Py_ssize_t initial = count[0]
    cdef int status
    cdef int64_t m, pos = 0, start, end, invalid
    while True:
        status = fse_wp_ascii_next(&scan, p, n, q)
        if status < 0:
            break
        if status == 0:
            return WP_GO_ON
        count[0] += _wp_ascii_word(st, q, &scan, ids + count[0])
        if stop_at and count[0] >= stop_at:
            invalid = fse_wp_validate(p, scan.pos, n)
            if invalid >= 0:
                bad[0] = invalid
                count[0] = initial
                return WP_BAD
            return WP_FULL
    count[0] = initial
    m = fse_wp_normalize_ascii(p, n, q, <uint32_t *>ws.run.data)
    if m < 0:
        bad[0] = -1 - m
        return WP_BAD
    while fse_wp_next_word_ascii(q, m, &pos, &start, &end):
        count[0] += _wp_word_trie(st, q + start, end - start, ids + count[0])
        if stop_at and count[0] >= stop_at:
            return WP_FULL
    return WP_GO_ON


def _wordpiece_ascii_probe(VocabTable table not None, bytes text):
    """Private differential probe of register keys and table hits, without a trie."""
    table._check_ready()
    cdef const unsigned char *p = text
    cdef Py_ssize_t n = len(text), length
    normalized = np.empty(n + 32, np.uint8)
    cdef unsigned char *q = <unsigned char *>np.PyArray_DATA(normalized)
    cdef fse_wp_ascii_scan scan
    cdef np.uint64_t key[2]
    cdef uint64_t h, expected
    cdef int status
    cdef uINT_t index
    memset(&scan, 0, sizeof(fse_wp_ascii_scan))
    out = []
    while True:
        status = fse_wp_ascii_next(&scan, p, n, q)
        if status <= 0:
            if status < 0:
                raise ValueError("probe requires printable ASCII without '['")
            return out
        length = scan.end - scan.start
        expected = _hash_word(q + scan.start, length, key, True)
        h = _wp_short_hash(&scan) if length <= INLINE_KEY_BYTES else expected
        index = _vocab_lookup_hashed(&table.view, q + scan.start, length, h,
                                     <np.uint64_t *>scan.key)
        out.append((scan.start, scan.end, None if index == NOT_FOUND else index,
                    h == expected, length > INLINE_KEY_BYTES or
                    (scan.key[0] == key[0] and scan.key[1] == key[1])))
