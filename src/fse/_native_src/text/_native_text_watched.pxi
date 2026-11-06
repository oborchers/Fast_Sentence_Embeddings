# Included by _native.pyx: The sentence functions that record what the length rule cut.
#
# They are copies of the default ones (TextLookup._sentence, _wordpiece_sentence,
# _unigram_sentence, _wp_piece), which stay as they were before the length report and know
# nothing of it, so the encoding without a report runs the same machine code. The loop of a
# batch picks one or the other once (``ws.watch``). Recording changes no id, offset or other
# state: a sentence at the token cut is tokenized a second time (_wp_over) in the scratch
# of the ids.

cdef inline int _wp_piece_until(WordPieceState st, const unsigned char *p, Py_ssize_t n, Workspace ws,
                          uINT_t *ids, Py_ssize_t *count, Py_ssize_t *bad,
                          Py_ssize_t stop_at) noexcept nogil:
    """Steps 3 to 5 for the piece p[0:n] between added tokens; WP_FULL once ``stop_at`` ids
    (0: no limit) are reached after a whole word."""
    cdef unsigned char *q = <unsigned char *>ws.norm.data
    cdef int64_t m = (fse_wp_normalize(p, n, q, <uint32_t *>ws.run.data) if st.generic
                      else fse_wp_normalize_ascii(p, n, q, <uint32_t *>ws.run.data))
    cdef int64_t pos = 0, start, end
    if m < 0:
        bad[0] = -1 - m
        return WP_BAD
    if st.generic:
        while fse_wp_next_word(q, m, &pos, &start, &end):
            count[0] += _wp_word(st, q + start, end - start, ids + count[0])
            if stop_at and count[0] >= stop_at:
                return WP_FULL
        return WP_GO_ON
    while fse_wp_next_word_ascii(q, m, &pos, &start, &end):
        count[0] += _wp_word_trie(st, q + start, end - start, ids + count[0])
        if stop_at and count[0] >= stop_at:
            return WP_FULL
    return WP_GO_ON


cdef int _wp_run(WordPieceState st, const unsigned char *p, Py_ssize_t limit, Workspace ws,
                 uINT_t *ids, Py_ssize_t stop_at, Py_ssize_t *count, Py_ssize_t *done,
                 Py_ssize_t *bad) noexcept nogil:
    """The loop of _wordpiece_sentence for the second run of a sentence that stopped at the
    token cut (_wp_over): ``count`` ids into ``ids``, stopping at ``stop_at`` (0: never);
    ``done`` is where it read to. It writes only ids, so it repeats the first run."""
    cdef Py_ssize_t pos = 0, piece = 0
    cdef int64_t found
    cdef int status = WP_GO_ON
    count[0] = 0
    done[0] = 0
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
        status = _wp_piece_until(st, p + piece, pos - piece, ws, ids, count, bad, stop_at)
        done[0] = pos
        if status == WP_GO_ON:
            ids[count[0]] = st.added_ids[found]
            count[0] += 1
            pos += st.added_offsets[found + 1] - st.added_offsets[found]
            piece = done[0] = pos
            if stop_at and count[0] >= stop_at:
                status = WP_FULL
    if status == WP_GO_ON:
        status = _wp_piece_until(st, p + piece, limit - piece, ws, ids, count, bad, stop_at)
        done[0] = limit
    if status == WP_BAD:
        bad[0] += piece
    return status


cdef Py_ssize_t _wp_over(WordPieceState st, const unsigned char *p, Py_ssize_t limit,
                         Workspace ws, uINT_t *ids, Py_ssize_t count, Py_ssize_t *redone,
                         Py_ssize_t *bad) noexcept nogil:
    """The ids beyond ``st.max_tokens`` once the first run of ``count`` ids stopped at the cut:
    for the flags up to one more (0, or 1 and more), when counting all; -1 for invalid UTF-8.
    The first ``count`` ids stay as they are. ``redone`` is where the second run read to
    (not set if none ran: more than the limit already is the flag)."""
    cdef Py_ssize_t total = 0, over = count - st.max_tokens
    if ws.watch == 1 and over > 0:
        return over
    if _wp_run(st, p, limit, ws, ids, 0 if ws.watch == 2 else st.max_tokens + 1,
               &total, redone, bad) == WP_BAD:
        return -1
    return total - st.max_tokens


cdef int _wordpiece_sentence_watched(WordPieceState st, const unsigned char *p, Py_ssize_t n,
                                     Workspace ws, Py_ssize_t *bad) noexcept nogil:
    """_wordpiece_sentence with the record of the length rule."""
    cdef Py_ssize_t limit = n, done = 0, count = 0, kept = 0, i, over = 0, redone = 0
    cdef int64_t found, cut
    cdef int status
    cdef int kind = 0
    cdef uINT_t *ids
    if st.max_chars:  # the whole sentence is validated here
        cut = (fse_wp_char_cut(p, n, st.max_chars) if st.generic
               else fse_wp_char_cut_ascii(p, n, st.max_chars))
        if cut < 0:
            bad[0] = -1 - cut
            return ENCODE_BAD_UTF8
        limit = cut
    ws.norm.size = 0
    ws.run.size = 0
    if (_ensure(&ws.norm, 3 * limit + 32) or _ensure(&ws.run, 12 * limit + 4)
            or _ensure(&ws.tokens, (4 * limit + 1) * <Py_ssize_t>sizeof(uINT_t))):
        return ENCODE_NO_MEMORY
    ids = <uINT_t *>(ws.tokens.data + ws.tokens.size)
    status = _wp_run(st, p, limit, ws, ids, st.max_tokens, &count, &done, bad)
    if status == WP_BAD:
        return ENCODE_BAD_UTF8
    if status == WP_FULL:  # the cut was reached: is there more behind it?
        over = _wp_over(st, p, limit, ws, ids, count, &redone, bad)
        if over < 0:
            return ENCODE_BAD_UTF8
        if redone:
            done = redone
    if not st.max_chars and done < n:  # the rest of a sentence the token cut ended early
        found = fse_wp_validate(p, done, n)
        if found >= 0:
            bad[0] = found
            return ENCODE_BAD_UTF8
    if st.max_tokens and count > st.max_tokens:
        count = st.max_tokens
    ws.tokens_seen += count
    kept = count
    if st.drop_unknown:
        kept = 0
        for i in range(count):
            if ids[i] != st.unk_id:
                ids[kept] = ids[i]
                kept += 1
    if limit < n:
        kind = CUT_CHARS
    if over > 0:
        kind |= CUT_TOKENS
    status = _note_row(ws, kind, count - kept, over,
                       _code_points(p + limit, n - limit) if ws.watch == 2 else 0,
                       st.max_chars, st.max_tokens)
    if status != ENCODE_OK:
        return status
    ws.tokens.size += kept * sizeof(uINT_t)
    if _push_i64_nogil(&ws.offsets, ws.tokens.size // sizeof(uINT_t)):
        return ENCODE_NO_MEMORY
    return ENCODE_OK


cdef int _unigram_sentence_watched(UnigramState st, const unsigned char *p, Py_ssize_t n,
                           Workspace ws, Py_ssize_t *bad) noexcept nogil:
    """Append the ids of the sentence p[0:n] (steps 1 to 6) to the batch buffers of ws."""
    cdef Py_ssize_t limit = n, pos = 0, piece = 0, count = 0, kept = 0, i, over = 0
    cdef int64_t found, cut
    cdef int status = UNI_GO_ON
    cdef int kind = 0
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
        over = count - st.max_tokens
        count = st.max_tokens
    ws.tokens_seen += count
    ids = <uINT_t *>(ws.tokens.data + ws.tokens.size)
    kept = count
    if st.drop_unknown:
        kept = 0
        for i in range(count):
            if ids[i] != st.unk_id:
                ids[kept] = ids[i]
                kept += 1
    if ws.watch:
        if limit < n:
            kind = CUT_CHARS
        if over > 0:
            kind |= CUT_TOKENS
        status = _note_row(ws, kind, count - kept, over,
                           _code_points(p + limit, n - limit) if ws.watch == 2 else 0,
                           st.max_chars, st.max_tokens)
        if status != ENCODE_OK:
            return status
    ws.tokens.size += kept * sizeof(uINT_t)
    if _push_i64_nogil(&ws.offsets, ws.tokens.size // sizeof(uINT_t)):
        return ENCODE_NO_MEMORY
    return ENCODE_OK


cdef int _words_sentence_watched(TextLookup self, const unsigned char *p, Py_ssize_t n,
                                 Workspace ws, Py_ssize_t *bad) noexcept nogil:
    """_sentence for fse_words@1 and external: no length rule, the unknown words are all that
    is lost."""
    cdef fse_tokens out
    cdef Py_ssize_t k, start, length, cap, kept
    cdef bint padded
    cdef uINT_t *ids
    cdef int64_t err
    cdef uINT_t index
    cdef size_t written
    cdef int status
    ws.cand.size = 0
    ws.starts.size = 0
    ws.ends.size = 0
    # 32 bytes of slack, so every token of the candidates can be read 16 bytes wide.
    if (_ensure(&ws.cand, 2 * n + 32) or _ensure(&ws.starts, (n + 1) * <Py_ssize_t>sizeof(int64_t))
            or _ensure(&ws.ends, (n + 1) * <Py_ssize_t>sizeof(int64_t))):
        return ENCODE_NO_MEMORY
    out.starts = <int64_t *>ws.starts.data
    out.ends = <int64_t *>ws.ends.data
    err = fse_tokenize(self.tokenizer, p, n, <unsigned char *>ws.cand.data, &out)
    # fse_words@1 tokens lie in the candidate buffer, external ones in the input.
    padded = self.tokenizer == FSE_TOKENIZER_FSE_WORDS
    if err >= 0:
        bad[0] = err
        return ENCODE_BAD_UTF8
    ws.tokens_seen += out.n_tokens
    if _ensure(&ws.tokens, out.n_tokens * <Py_ssize_t>sizeof(uINT_t)):
        return ENCODE_NO_MEMORY
    ids = <uINT_t *>(ws.tokens.data + ws.tokens.size)
    kept = 0
    for k in range(out.n_tokens):
        start = out.starts[k]
        length = out.ends[k] - start
        index = _vocab_lookup(&self.table.view, out.bytes + start, length, padded)
        if index != NOT_FOUND:
            ids[kept] = index
            kept += 1
        elif self.fasttext:
            ids[kept] = OOV_TOKEN
            kept += 1
            # The ngrams of "<token>": fewer than (length + 2) * max_n of them.
            ws.word.size = 0
            if _ensure(&ws.word, length + 2):
                return ENCODE_NO_MEMORY
            ws.word.data[0] = b"<"
            memcpy(ws.word.data + 1, out.bytes + start, length)
            ws.word.data[length + 1] = b">"
            cap = min(<Py_ssize_t>self.max_ngrams, (length + 2) * <Py_ssize_t>self.max_n)
            if _ensure(&ws.ngrams, cap * <Py_ssize_t>sizeof(uINT_t)):
                return ENCODE_NO_MEMORY
            written = ft_ngram_hashes_into(
                <const unsigned char *>ws.word.data, length + 2, self.min_n, self.max_n,
                self.bucket, <uINT_t *>(ws.ngrams.data + ws.ngrams.size), cap,
            )
            ws.ngrams.size += written * sizeof(uINT_t)
            if _push_i64_nogil(&ws.ngram_offsets, ws.ngrams.size // sizeof(uINT_t)):
                return ENCODE_NO_MEMORY
    if ws.watch:  # no length rule: the unknown words are all that is lost
        status = _note_row(ws, 0, 0 if self.fasttext else out.n_tokens - kept, 0, 0, 0, 0)
        if status != ENCODE_OK:
            return status
    ws.tokens.size += kept * sizeof(uINT_t)
    if _push_i64_nogil(&ws.offsets, ws.tokens.size // sizeof(uINT_t)):
        return ENCODE_NO_MEMORY
    return ENCODE_OK


cdef int _sentence_watched(TextLookup self, const unsigned char *p, Py_ssize_t n,
                           Workspace ws, Py_ssize_t *bad) noexcept nogil:
    """TextLookup._sentence with the record of the length rule."""
    if n > _MAX_SENTENCE_BYTES:
        return ENCODE_TOO_LONG
    if self.tokenizer == FSE_TOKENIZER_WORDPIECE:
        return _wordpiece_sentence_watched(self.wordpiece, p, n, ws, bad)
    if self.tokenizer == FSE_TOKENIZER_UNIGRAM:
        return _unigram_sentence_watched(self.unigram, p, n, ws, bad)
    return _words_sentence_watched(self, p, n, ws, bad)
