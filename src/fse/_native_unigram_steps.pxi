# Included by _native.pyx: the passes of unigram@1 over one sentence: the normalizer
# steps, Metaspace, the candidates and the lattice, and the sentence (steps 1 to 6).

cdef inline int _uni_put(_Buffer *b, const unsigned char *src, Py_ssize_t k) noexcept nogil:
    """Append k bytes to b; -1 if memory runs out."""
    if k <= 0:
        return 0
    if _ensure(b, k):
        return -1
    memcpy(b.data + b.size, src, k)
    b.size += k
    return 0


cdef inline int _uni_cp(const unsigned char *s, Py_ssize_t i, Py_ssize_t n, uint32_t *cp) noexcept nogil:
    """The length of the code point at s[i] of valid UTF-8 (the sentence was validated)."""
    return fse_utf8_decode(s, i, n, cp)


cdef inline bint _uni_space(uint32_t cp) noexcept nogil:
    return (fse_gc_props(cp) & FSE_GC_WHITE_SPACE) != 0


cdef int _uni_value(UnigramState st, uint32_t start, _Buffer *dst) noexcept nogil:
    """The value of a charsmap match: from start to the next NUL or the end of the values."""
    cdef const unsigned char *end
    if start >= st.n_values:
        return 0
    end = <const unsigned char *>memchr(st.values + start, 0, st.n_values - start)
    return _uni_put(dst, st.values + start, (end - st.values if end != NULL else st.n_values) - start)


cdef int _uni_precompiled(UnigramState st, const unsigned char *s, Py_ssize_t n, _Buffer *dst,
                          Workspace ws) noexcept nogil:
    """The step precompiled: a grapheme of fewer than 6 bytes becomes the value of its
    shortest prefix key, otherwise each code point becomes its key's value, or stays."""
    cdef Py_ssize_t i = 0, count = 0, g = 0, j, c
    cdef uint32_t cp, value
    cdef uint32_t *cps
    cdef int64_t *at
    ws.run.size = 0
    ws.word.size = 0
    if _ensure(&ws.run, (n + 1) * <Py_ssize_t>sizeof(uint32_t)) or _ensure(&ws.word, (n + 1) * <Py_ssize_t>sizeof(int64_t)):
        return -1
    cps = <uint32_t *>ws.run.data
    at = <int64_t *>ws.word.data
    while i < n:
        at[count] = i
        i += _uni_cp(s, i, n, &cps[count])
        count += 1
    at[count] = n
    while g < count:
        j = fse_gc_next(cps, g, count)
        if (at[j] - at[g] < FSE_UNI_GRAPHEME_BYTES
                and fse_cm_find(st.units, st.n_units, st.root, s + at[g], at[j] - at[g], &value)):
            if _uni_value(st, value, dst):
                return -1
        else:
            for c in range(g, j):
                if fse_cm_find(st.units, st.n_units, st.root, s + at[c], at[c + 1] - at[c], &value):
                    if _uni_value(st, value, dst):
                        return -1
                elif _uni_put(dst, s + at[c], at[c + 1] - at[c]):
                    return -1
        g = j
    return 0


cdef int _uni_replace(const unsigned char *s, Py_ssize_t n, const unsigned char *pattern,
                      Py_ssize_t k, const unsigned char *content, Py_ssize_t m,
                      _Buffer *dst) noexcept nogil:
    """The step replace: every occurrence of the pattern, left to right without overlap."""
    cdef Py_ssize_t i = 0, last = 0
    while i + k <= n:
        if s[i] == pattern[0] and memcmp(s + i, pattern, k) == 0:
            if _uni_put(dst, s + last, i - last) or _uni_put(dst, content, m):
                return -1
            i += k
            last = i
        else:
            i += 1
    return _uni_put(dst, s + last, n - last)


cdef int _uni_runs(const unsigned char *s, Py_ssize_t n, bint white, int64_t minimum,
                   const unsigned char *content, Py_ssize_t m, _Buffer *dst) noexcept nogil:
    """The step runs: every maximal run of at least minimum code points of the class (a
    space, or White_Space) becomes the content."""
    cdef Py_ssize_t i = 0, j, length, runs
    cdef uint32_t cp
    while i < n:
        j = i
        runs = 0
        while j < n:
            length = _uni_cp(s, j, n, &cp)
            if not (cp == 32 if not white else _uni_space(cp)):
                break
            j += length
            runs += 1
        if runs >= minimum:
            if _uni_put(dst, content, m):
                return -1
        elif _uni_put(dst, s + i, j - i):
            return -1
        if j == i:
            j += _uni_cp(s, i, n, &cp)
            if _uni_put(dst, s + i, j - i):
                return -1
        i = j
    return 0


cdef int _uni_strip(const unsigned char *s, Py_ssize_t n, int64_t sides, _Buffer *dst) noexcept nogil:
    """The step strip: leading (sides & 1) and trailing (sides & 2) White_Space."""
    cdef Py_ssize_t start = 0, end = n, k, length
    cdef uint32_t cp
    while sides & 1 and start < end:
        length = _uni_cp(s, start, n, &cp)
        if not _uni_space(cp):
            break
        start += length
    while sides & 2 and end > start:
        k = end - 1
        while k > start and (s[k] & 0xC0) == 0x80:
            k -= 1
        _uni_cp(s, k, n, &cp)
        if not _uni_space(cp):
            break
        end = k
    return _uni_put(dst, s + start, end - start)


cdef int _uni_metaspace(UnigramState st, const unsigned char *s, Py_ssize_t n, _Buffer *dst) noexcept nogil:
    """Metaspace: every U+0020 becomes the replacement, which goes in front unless the
    piece then starts with it; an empty piece stays empty."""
    cdef Py_ssize_t i, last = 0
    cdef Py_ssize_t r = st.n_replacement
    if n == 0:
        return 0
    if not (s[0] == 32 or (n >= r and memcmp(s, st.replacement, r) == 0)):
        if _uni_put(dst, st.replacement, r):
            return -1
    for i in range(n):
        if s[i] == 32:
            if _uni_put(dst, s + last, i - last) or _uni_put(dst, st.replacement, r):
                return -1
            last = i + 1
    return _uni_put(dst, s + last, n - last)


cdef int _uni_normalize(UnigramState st, const unsigned char *p, Py_ssize_t n, Workspace ws,
                        _Buffer **out) noexcept nogil:
    """Steps 3 and 4 of the piece p[0:n]: each step from one buffer into the other; *out
    is the buffer with the pre-token."""
    cdef _Buffer *a = &ws.norm
    cdef _Buffer *b = &ws.cand
    cdef _Buffer *t
    cdef Py_ssize_t k
    cdef const int64_t *row
    cdef int err = 0
    a.size = 0
    if _uni_put(a, p, n):
        return -1
    for k in range(st.n_ops):
        row = st.ops + k * _UNI_ROW
        b.size = 0
        if row[0] == UNI_PRECOMPILED:
            err = _uni_precompiled(st, <const unsigned char *>a.data, a.size, b, ws)
        elif row[0] == UNI_REPLACE:
            err = _uni_replace(<const unsigned char *>a.data, a.size, st.pool + row[1], row[2],
                               st.pool + row[3], row[4], b)
        elif row[0] == UNI_STRIP:
            err = _uni_strip(<const unsigned char *>a.data, a.size, row[5], b)
        else:
            err = _uni_runs(<const unsigned char *>a.data, a.size, row[0] == UNI_RUNS_WHITE_SPACE,
                            row[5], st.pool + row[3], row[4], b)
        if err:
            return -1
        t = a
        a = b
        b = t
    b.size = 0
    if _uni_metaspace(st, <const unsigned char *>a.data, a.size, b):
        return -1
    # 32 bytes of slack, so a candidate can be read 16 bytes wide.
    if _ensure(b, 32):
        return -1
    out[0] = b
    return 0


cdef enum:
    UNI_GO_ON = 0
    UNI_NO_MEMORY = 1
    UNI_BROKEN = 2


cdef inline uINT_t _uni_lookup(UnigramState st, const unsigned char *q, Py_ssize_t k) noexcept nogil:
    return _vocab_lookup(&st.table.view, q, k, True)


cdef int _uni_candidates(UnigramState st, const unsigned char *q, Py_ssize_t m, Workspace ws,
                         Py_ssize_t *count) noexcept nogil:
    """The candidates of every code point boundary of q[0:m] (step 5): every piece that is
    a prefix of the rest, shortest first, then the unknown node if no piece is the code
    point at the start. pos and first of the starts go to ws.starts and ws.ends."""
    cdef Py_ssize_t s = 0, e, size, k = 0, total = 0
    cdef bint single
    cdef uINT_t index, prefix
    cdef uint32_t cp
    cdef fse_uni_cand *cands
    ws.starts.size = 0
    ws.ends.size = 0
    ws.cands.size = 0
    if (_ensure(&ws.starts, (m + 1) * <Py_ssize_t>sizeof(int64_t))
            or _ensure(&ws.ends, (m + 2) * <Py_ssize_t>sizeof(int64_t))):
        return UNI_NO_MEMORY
    while s < m:
        (<int64_t *>ws.starts.data)[k] = s
        (<int64_t *>ws.ends.data)[k] = total
        size = _uni_cp(q, s, m, &cp)
        single = False
        e = s
        while e < m and e - s < st.max_piece:
            e += _uni_cp(q, e, m, &cp)
            index = _uni_lookup(st, q + s, e - s)
            if index == NOT_FOUND:
                prefix = _vocab_lookup(&st.prefixes.view, q + s, e - s, True)
                if prefix == NOT_FOUND:
                    break
                continue
            if _ensure(&ws.cands, (total + 1) * <Py_ssize_t>sizeof(fse_uni_cand)):
                return UNI_NO_MEMORY
            cands = <fse_uni_cand *>ws.cands.data
            cands[total].end = e
            cands[total].id = index
            cands[total].score = st.scores[index]
            total += 1
            single = single or e - s == size
        if not single:
            if _ensure(&ws.cands, (total + 1) * <Py_ssize_t>sizeof(fse_uni_cand)):
                return UNI_NO_MEMORY
            cands = <fse_uni_cand *>ws.cands.data
            cands[total].end = s + size
            cands[total].id = st.unk_id
            cands[total].score = st.unk_score
            total += 1
        s += size
        k += 1
    (<int64_t *>ws.ends.data)[k] = total
    count[0] = k
    return UNI_GO_ON


cdef int _uni_piece(UnigramState st, const unsigned char *p, Py_ssize_t n, Workspace ws,
                    Py_ssize_t *count) noexcept nogil:
    """Steps 3 to 5 of the piece p[0:n] between added tokens: its ids are appended after
    the count ids of the sentence so far in ws.tokens."""
    cdef _Buffer *pre
    cdef const unsigned char *q
    cdef Py_ssize_t m, starts, end, start, run_end, run_start, k, found = 0
    cdef uINT_t *ids
    cdef double *best
    cdef int64_t *best_start
    cdef uint32_t *best_id
    cdef uint32_t *reversed_ids
    cdef uINT_t joined
    if _uni_normalize(st, p, n, ws, &pre):
        return UNI_NO_MEMORY
    q = <const unsigned char *>pre.data
    m = pre.size
    if m == 0:
        return UNI_GO_ON
    if m > (PY_SSIZE_T_MAX // 32) - 2:
        return UNI_NO_MEMORY
    if _uni_candidates(st, q, m, ws, &starts):
        return UNI_NO_MEMORY
    # The lattice: m + 1 doubles, int64 and uint32, each part aligned to 8 bytes.
    ws.lattice.size = 0
    if _ensure(&ws.lattice, 24 * (m + 1)) or _ensure(&ws.run, (m + 1) * <Py_ssize_t>sizeof(uint32_t)):
        return UNI_NO_MEMORY
    best = <double *>ws.lattice.data
    best_start = <int64_t *>(ws.lattice.data + 8 * (m + 1))
    best_id = <uint32_t *>(ws.lattice.data + 16 * (m + 1))
    if fse_uni_viterbi(m, <const int64_t *>ws.starts.data, <const int64_t *>ws.ends.data, starts,
                       <const fse_uni_cand *>ws.cands.data, best, best_start, best_id) >= 0:
        return UNI_BROKEN
    # Back from the end: nodes of unk_id join, and the joined bytes are looked up again.
    reversed_ids = <uint32_t *>ws.run.data
    end = m
    run_end = -1
    run_start = 0
    while end > 0:
        start = best_start[end]
        if not 0 <= start < end:
            return UNI_BROKEN
        if best_id[end] == st.unk_id:
            if run_end < 0:
                run_end = end
            run_start = start
        else:
            if run_end >= 0:
                joined = _uni_lookup(st, q + run_start, run_end - run_start)
                reversed_ids[found] = st.unk_id if joined == NOT_FOUND else joined
                found += 1
                run_end = -1
            reversed_ids[found] = best_id[end]
            found += 1
        end = start
    if run_end >= 0:
        joined = _uni_lookup(st, q + run_start, run_end - run_start)
        reversed_ids[found] = st.unk_id if joined == NOT_FOUND else joined
        found += 1
    if _ensure(&ws.tokens, (count[0] + found) * <Py_ssize_t>sizeof(uINT_t)):
        return UNI_NO_MEMORY
    ids = <uINT_t *>(ws.tokens.data + ws.tokens.size)
    for k in range(found):
        ids[count[0] + k] = reversed_ids[found - 1 - k]
    count[0] += found
    return UNI_GO_ON


cdef int _unigram_sentence(UnigramState st, const unsigned char *p, Py_ssize_t n,
                           Workspace ws, Py_ssize_t *bad) noexcept nogil:
    """Append the ids of the sentence p[0:n] (steps 1 to 6) to the batch buffers of ws."""
    cdef Py_ssize_t limit = n, pos = 0, piece = 0, count = 0, kept = 0, i
    cdef int64_t found, cut
    cdef int status = UNI_GO_ON
    cdef uINT_t *ids
    if _ensure(&ws.tokens, sizeof(uINT_t)):
        return ENCODE_NO_MEMORY
    if st.max_chars:  # the whole sentence is validated here
        cut = fse_wp_char_cut(p, n, st.max_chars)
        if cut < 0:
            bad[0] = -1 - cut
            return ENCODE_BAD_UTF8
        limit = cut
    else:
        found = fse_wp_validate(p, 0, n)
        if found >= 0:
            bad[0] = found
            return ENCODE_BAD_UTF8
    while pos < limit and status == UNI_GO_ON:
        pos = fse_wp_next_candidate(p, pos, limit, st.first, st.single_first) if st.n_added else limit
        if pos == limit:
            break
        found = fse_wp_added_at(p, pos, limit, st.first, st.added_bytes, st.added_offsets, st.n_added)
        if found < 0:
            pos += 1
            continue
        status = _uni_piece(st, p + piece, pos - piece, ws, &count)
        if status == UNI_GO_ON:
            if _ensure(&ws.tokens, (count + 1) * <Py_ssize_t>sizeof(uINT_t)):
                return ENCODE_NO_MEMORY
            (<uINT_t *>(ws.tokens.data + ws.tokens.size))[count] = st.added_ids[found]
            count += 1
            pos += st.added_offsets[found + 1] - st.added_offsets[found]
            piece = pos
    if status == UNI_GO_ON:
        status = _uni_piece(st, p + piece, limit - piece, ws, &count)
    if status == UNI_NO_MEMORY:
        return ENCODE_NO_MEMORY
    if status == UNI_BROKEN:
        return ENCODE_BROKEN
    if st.max_tokens and count > st.max_tokens:
        count = st.max_tokens
    ws.tokens_seen += count
    ids = <uINT_t *>(ws.tokens.data + ws.tokens.size)
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
