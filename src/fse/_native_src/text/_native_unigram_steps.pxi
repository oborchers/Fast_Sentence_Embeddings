# Included by _native.pyx: candidates, lattice and sentence of unigram@1.

cdef enum:
    UNI_GO_ON = 0
    UNI_NO_MEMORY = 1
    UNI_BROKEN = 2


cdef inline uINT_t _uni_lookup(UnigramState st, const unsigned char *q, Py_ssize_t k) noexcept nogil:
    return _vocab_lookup(&st.table.view, q, k, True)


cdef int _uni_candidates_generic(UnigramState st, const unsigned char *q, Py_ssize_t m, Workspace ws,
                         Py_ssize_t *count) noexcept nogil:
    """The candidates of every code point boundary of q[0:m] (step 5): every piece that is
    a prefix of the rest, shortest first, then the unknown node if no piece is the code
    point at the start. pos and first of the starts go to ws.starts and ws.ends."""
    cdef Py_ssize_t s = 0, e, size, length, k = 0, total = 0
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
        if size == 0:
            return UNI_BROKEN
        single = False
        e = s
        while e < m and e - s < st.max_piece:
            length = _uni_cp(q, e, m, &cp)
            if length == 0:
                return UNI_BROKEN
            e += length
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


cdef int _uni_candidates(UnigramState st, const unsigned char *q, Py_ssize_t m, Workspace ws,
                         Py_ssize_t *count) noexcept nogil:
    """The same ordered offers as the generic path, with one byte-trie walk per start."""
    cdef Py_ssize_t s = 0, size, k = 0, total = 0, bound
    cdef uint32_t cp
    cdef int single
    cdef fse_uni_cand *cands
    if m > (PY_SSIZE_T_MAX // 32) - 2:
        return UNI_NO_MEMORY
    if st.generic:
        return _uni_candidates_generic(st, q, m, ws, count)
    ws.starts.size = ws.ends.size = ws.cands.size = 0
    if (_ensure(&ws.starts, (m + 1) * <Py_ssize_t>sizeof(int64_t))
            or _ensure(&ws.ends, (m + 2) * <Py_ssize_t>sizeof(int64_t))):
        return UNI_NO_MEMORY
    while s < m:
        size = _uni_cp(q, s, m, &cp)
        if size == 0:
            return UNI_BROKEN
        (<int64_t *>ws.starts.data)[k] = s
        (<int64_t *>ws.ends.data)[k] = total
        bound = min(m - s, st.max_piece) + 1
        # cands.size stays zero. Reserve for the previous starts too, checking the
        # addition and byte multiplication before evaluating either one.
        if total > PY_SSIZE_T_MAX // <Py_ssize_t>sizeof(fse_uni_cand) - bound:
            return UNI_NO_MEMORY
        if _ensure(&ws.cands, (total + bound) * <Py_ssize_t>sizeof(fse_uni_cand)):
            return UNI_NO_MEMORY
        cands = <fse_uni_cand *>ws.cands.data  # growth may have moved the buffer
        total += fse_uni_candidates(&st.trie, q, m, s, size, st.scores, cands + total, &single)
        if not single:
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
    cdef int err = _uni_normalize(st, p, n, ws, &pre)
    if err:
        return UNI_BROKEN if err == _UNI_BAD_TEXT else UNI_NO_MEMORY
    q = <const unsigned char *>pre.data
    m = pre.size
    if m == 0:
        return UNI_GO_ON
    if m > (PY_SSIZE_T_MAX // 32) - 2:
        return UNI_NO_MEMORY
    err = _uni_candidates(st, q, m, ws, &starts)
    if err:
        return err
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
