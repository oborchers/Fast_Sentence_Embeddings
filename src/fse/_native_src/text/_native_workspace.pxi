# Included by _native.pyx: The workspace of a worker and what it records of the length rule.

# The cuts of a sentence, as the bits of a row of Workspace.cuts.
cdef enum:
    CUT_CHARS = 1
    CUT_TOKENS = 2


cdef class Workspace:
    """The buffers one worker reuses for every batch it encodes from text.

    A Workspace is used by one thread at a time; give every worker its own.

    ``watch`` 0 records nothing (the default, bit for bit the encoding without a report),
    1 records which sentences the length rule cut and 2 also how much it cut.
    ``raise_cut`` stops at the first sentence the rule would cut (``TruncationError``);
    it needs ``watch`` of 1 or more. ``store_rows`` keeps the record of every sentence
    (``take_lengths``), otherwise only its sums.
    """

    cdef _Buffer cand, starts, ends, word, tokens, offsets, ngram_offsets, ngrams, norm, run
    cdef _Buffer lattice, cands  # the lattice and the candidates of unigram@1
    cdef bint busy
    # The tokens the tokenizer gave in the last encode, unknown words included
    # (benchmarks/bench_vocab.py measures per token with it).
    cdef readonly int64_t tokens_seen
    # What the length rule did (below the fields above, so those keep their places).
    cdef _Buffer cuts, dropped, cut_tokens, cut_chars  # one entry per sentence, if store_rows
    cdef readonly int watch
    cdef readonly bint raise_cut, store_rows
    # The cut that stopped the encode (a CUT_ kind) and its limit, for the error.
    cdef int cut_kind
    cdef int64_t cut_limit
    # The sums of the batch: sentences cut by characters, by tokens, by either, the unknown
    # tokens dropped and the tokens and characters cut.
    cdef int64_t n_char, n_token, n_either, dropped_sum, cut_token_sum, cut_char_sum

    def __init__(self, int watch=0, bint raise_cut=False, bint store_rows=False):
        if not 0 <= watch <= 2:
            raise ValueError(f"watch must be 0, 1 or 2, not {watch}")
        if raise_cut and watch == 0:
            raise ValueError("raise_cut needs watch of 1 or 2")
        if store_rows and watch == 0:
            raise ValueError("store_rows needs watch of 1 or 2")
        self.watch = watch
        self.raise_cut = raise_cut
        self.store_rows = store_rows

    def __dealloc__(self):
        free(self.cand.data)
        free(self.starts.data)
        free(self.ends.data)
        free(self.word.data)
        free(self.tokens.data)
        free(self.offsets.data)
        free(self.ngram_offsets.data)
        free(self.ngrams.data)
        free(self.norm.data)
        free(self.run.data)
        free(self.lattice.data)
        free(self.cands.data)
        free(self.cuts.data)
        free(self.dropped.data)
        free(self.cut_tokens.data)
        free(self.cut_chars.data)

    def __reduce__(self):
        raise TypeError("a Workspace belongs to one worker and cannot be pickled")

    def take_lengths(self):
        """What the length rule did to the sentences of the last encode: ``(sums, rows)``.
        ``sums`` is (sentences cut by characters, by tokens, by either, empty sentences,
        tokens, unknown tokens dropped, tokens cut, characters cut); ``rows`` is None
        without ``store_rows``, else (cuts, dropped, tokens, tokens cut, characters cut)
        as arrays of one entry per sentence (cuts holds the bits 1 for characters and 2 for
        tokens and is uint8; the last two are None at ``watch`` 1)."""
        cdef Py_ssize_t n = self.offsets.size // <Py_ssize_t>sizeof(int64_t) - 1, i
        cdef const int64_t *o = <const int64_t *>self.offsets.data
        cdef int64_t empty = 0, last = 0, width
        cdef int64_t[::1] view
        tokens = np.empty(max(n, 0), dtype=np.int64) if self.store_rows else None
        if self.store_rows and n > 0:  # a typed view of an empty array has no address
            view = tokens
        for i in range(max(n, 0)):  # no numpy call: a worker holds the GIL here
            width = o[i + 1] - o[i]
            empty += width == 0
            if self.store_rows:
                view[i] = width
        if n > 0:
            last = o[n]
        sums = (
            self.n_char, self.n_token, self.n_either, empty, last, self.dropped_sum,
            self.cut_token_sum, self.cut_char_sum,
        )
        if not self.store_rows:
            return sums, None
        counted = self.watch == 2
        return sums, (
            _to_array(&self.cuts, np.uint8), _to_array(&self.dropped, np.int64), tokens,
            _to_array(&self.cut_tokens, np.int64) if counted else None,
            _to_array(&self.cut_chars, np.int64) if counted else None,
        )


cdef inline int _start_lengths(Workspace ws) noexcept nogil:
    """Empty the record of the length rule at the start of a batch."""
    ws.cut_kind = 0
    ws.cut_limit = 0
    ws.n_char = ws.n_token = ws.n_either = 0
    ws.dropped_sum = ws.cut_token_sum = ws.cut_char_sum = 0
    ws.cuts.size = ws.dropped.size = ws.cut_tokens.size = ws.cut_chars.size = 0
    return ENCODE_OK


cdef inline int _note_row(Workspace ws, int kind, int64_t dropped, int64_t cut_tokens,
                          int64_t cut_chars, int64_t max_chars, int64_t max_tokens) noexcept nogil:
    """Record the sentence just encoded (call it before its offset is pushed): ``kind`` has the
    CUT_ bits of the rules that cut it, ``dropped`` the unknown tokens it lost, and
    ``cut_tokens`` and ``cut_chars`` how much the rules cut (watch 2). With ``raise_cut``
    a cut sentence stops the encode: ENCODE_TRUNCATED, the characters first."""
    if kind and ws.raise_cut:
        if kind & CUT_CHARS:
            ws.cut_kind, ws.cut_limit = CUT_CHARS, max_chars
        else:
            ws.cut_kind, ws.cut_limit = CUT_TOKENS, max_tokens
        return ENCODE_TRUNCATED
    ws.n_char += (kind & CUT_CHARS) != 0
    ws.n_token += (kind & CUT_TOKENS) != 0
    ws.n_either += kind != 0
    ws.dropped_sum += dropped
    ws.cut_token_sum += cut_tokens
    ws.cut_char_sum += cut_chars
    if not ws.store_rows:
        return ENCODE_OK
    if _ensure(&ws.cuts, 1):
        return ENCODE_NO_MEMORY
    ws.cuts.data[ws.cuts.size] = kind
    ws.cuts.size += 1
    if _push_i64_nogil(&ws.dropped, dropped):
        return ENCODE_NO_MEMORY
    if ws.watch == 2:
        if _push_i64_nogil(&ws.cut_tokens, cut_tokens) or _push_i64_nogil(&ws.cut_chars, cut_chars):
            return ENCODE_NO_MEMORY
    return ENCODE_OK


cdef inline int64_t _code_points(const unsigned char *p, int64_t n) noexcept nogil:
    """The code points of the valid UTF-8 text p[0:n]."""
    cdef int64_t i, chars = 0
    for i in range(n):
        chars += (p[i] & 0xC0) != 0x80
    return chars


cdef int _truncation_error(Workspace ws, row) except -1:
    """Raise the TruncationError of the cut that stopped an encode, at ``row`` of the run."""
    from fse._contract import TruncationError

    cut = "characters" if ws.cut_kind == CUT_CHARS else "tokens"
    raise TruncationError(
        f"sentence {row} has more than the {ws.cut_limit} {cut} of the encoder's length rule "
        f"and would be cut; split it, or pass on_truncation=\"keep\" to cut it as the model does",
        row=row, cut=cut, limit=ws.cut_limit,
    )
