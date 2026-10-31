# Included by _native.pyx: the passes of unigram@1 over one sentence: the normalizer
# steps and Metaspace (steps 3 and 4).

def _uni_fuse_replaces(ops, pool, enabled):
    """Fuse a maximal one-byte run only when later patterns cannot hit earlier output.
    Rows have already passed _steps_parts' UTF-8 and shape validation.
    """
    rows, maps = [], []
    i = 0
    while i < len(ops):
        end = i
        while end < len(ops) and ops[end, 0] == UNI_REPLACE and ops[end, 2] == 1:
            end += 1
        if enabled and end - i > 1 and _uni_independent_replaces(ops[i:end], pool):
            table = np.full((256, 2), -1, dtype=np.int64)
            for row in ops[i:end]:
                table[pool[row[1]]] = row[3:5]
            rows.append((UNI_REPLACE_BYTES, len(maps) * 512, 0, 0, 0, 0))
            maps.append(table)
        else:
            end = max(i + 1, end)
            rows.extend(ops[i:end])
        i = end
    return (np.array(rows, dtype=np.int64).reshape(-1, _UNI_ROW),
            np.array(maps, dtype=np.int64).reshape(-1))


def _uni_independent_replaces(rows, pool):
    patterns = [int(pool[row[1]]) for row in rows]
    if len(set(patterns)) != len(patterns):
        return False
    for i, row in enumerate(rows):
        if any(p in pool[row[3]:row[3] + row[4]] for p in patterns[i + 1:]):
            return False
    return True


cdef int _uni_replace_bytes(const unsigned char *s, Py_ssize_t n,
                            const int64_t *table, const unsigned char *pool,
                            _Buffer *dst) noexcept nogil:
    """Map original bytes independently and copy unchanged spans together."""
    cdef Py_ssize_t i, last = 0
    cdef const int64_t *entry
    for i in range(n):
        entry = table + 2 * s[i]
        if entry[0] >= 0:
            if _uni_put(dst, s + last, i - last) or _uni_put(dst, pool + entry[0], entry[1]):
                return -1
            last = i + 1
    return _uni_put(dst, s + last, n - last)


cdef inline int _uni_put(_Buffer *b, const unsigned char *src, Py_ssize_t k) noexcept nogil:
    """Append k bytes to b; -1 if memory runs out."""
    if k <= 0:
        return 0
    if _ensure(b, k):
        return -1
    memcpy(b.data + b.size, src, k)
    b.size += k
    return 0


# What a pass returns besides 0: memory ran out, or the text it was given is no UTF-8 (only
# a value of an unchecked charsmap could make it so; it is an error, never a loop that
# stands still).
cdef enum:
    _UNI_NO_MEMORY = -1
    _UNI_BAD_TEXT = -2


cdef inline int _uni_cp(const unsigned char *s, Py_ssize_t i, Py_ssize_t n, uint32_t *cp) noexcept nogil:
    """The length of the code point at s[i], or 0 if it is no UTF-8 (the sentence was
    validated, and the values of the charsmap when the state was built)."""
    return fse_utf8_decode(s, i, n, cp)


cdef inline bint _uni_space(uint32_t cp) noexcept nogil:
    return (fse_gc_props(cp) & FSE_GC_WHITE_SPACE) != 0


cdef int _uni_value(UnigramState st, uint32_t start, _Buffer *dst) noexcept nogil:
    """The value of a charsmap match: from start to the next NUL or the end of the values."""
    cdef const unsigned char *end
    if start > st.n_values:  # past the values: the blob is broken
        return _UNI_BAD_TEXT
    if start == st.n_values:  # the end of the values is the empty string
        return 0
    if (st.values[start] & 0xC0) == 0x80:  # inside a code point: the blob is broken
        return _UNI_BAD_TEXT
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
    cdef int err
    ws.run.size = 0
    ws.word.size = 0
    if _ensure(&ws.run, (n + 1) * <Py_ssize_t>sizeof(uint32_t)) or _ensure(&ws.word, (n + 1) * <Py_ssize_t>sizeof(int64_t)):
        return -1
    cps = <uint32_t *>ws.run.data
    at = <int64_t *>ws.word.data
    while i < n:
        at[count] = i
        c = _uni_cp(s, i, n, &cps[count])
        if c == 0:
            return _UNI_BAD_TEXT
        i += c
        count += 1
    at[count] = n
    while g < count:
        j = fse_gc_next(cps, g, count)
        if (at[j] - at[g] < FSE_UNI_GRAPHEME_BYTES
                and fse_cm_find(st.units, st.n_units, st.root, s + at[g], at[j] - at[g], &value)):
            c = _uni_value(st, value, dst)
            if c:
                return c
        else:
            for c in range(g, j):
                if fse_cm_find(st.units, st.n_units, st.root, s + at[c], at[c + 1] - at[c], &value):
                    err = _uni_value(st, value, dst)
                    if err:
                        return err
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
            if length == 0:
                return _UNI_BAD_TEXT
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
            length = _uni_cp(s, i, n, &cp)
            if length == 0:
                return _UNI_BAD_TEXT
            j += length
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
        if length == 0:
            return _UNI_BAD_TEXT
        if not _uni_space(cp):
            break
        start += length
    while sides & 2 and end > start:
        k = end - 1
        while k > start and (s[k] & 0xC0) == 0x80:
            k -= 1
        if _uni_cp(s, k, n, &cp) == 0:
            return _UNI_BAD_TEXT
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
        elif row[0] == UNI_REPLACE_BYTES:
            err = _uni_replace_bytes(<const unsigned char *>a.data, a.size,
                                     st.byte_maps + row[1], st.pool, b)
        elif row[0] == UNI_STRIP:
            err = _uni_strip(<const unsigned char *>a.data, a.size, row[5], b)
        else:
            err = _uni_runs(<const unsigned char *>a.data, a.size, row[0] == UNI_RUNS_WHITE_SPACE,
                            row[5], st.pool + row[3], row[4], b)
        if err:
            return err
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


