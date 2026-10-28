# Included by _native.pyx: The tokenizers and the text lookup.

# ---------------------------------------------------------------------------------------
# Tokenizers (ARCHITECTURE.md, "Tokenizers"), as in fse._reference_text.
# ---------------------------------------------------------------------------------------

cdef extern from "_native_text.h":
    ctypedef struct fse_tokens:
        const unsigned char *bytes
        int64_t *starts
        int64_t *ends
        int64_t n_tokens
    void fse_text_init()
    uint32_t fse_simple_lower(uint32_t cp) nogil
    int64_t fse_tokenize(int tokenizer, const unsigned char *p, int64_t n, unsigned char *cand,
                         fse_tokens *out) nogil
    int FSE_TOKENIZER_EXTERNAL
    int FSE_TOKENIZER_FSE_WORDS

fse_text_init()

TOKENIZER_EXTERNAL = FSE_TOKENIZER_EXTERNAL
TOKENIZER_FSE_WORDS = FSE_TOKENIZER_FSE_WORDS
_TOKENIZERS = {"external": FSE_TOKENIZER_EXTERNAL, "fse_words@1": FSE_TOKENIZER_FSE_WORDS}


def _tokenizer_code(tokenizer):
    from fse._contract import check_tokenizer

    return _TOKENIZERS[check_tokenizer(tokenizer)]


def _simple_lower(code_points):
    """The simple lowercase of every uint32 code point, by the native table; for tests."""
    cdef const uint32_t[::1] cps = np.ascontiguousarray(code_points, dtype=np.uint32)
    out = np.empty(cps.shape[0], dtype=np.uint32)
    cdef uint32_t[::1] o = out
    cdef Py_ssize_t i
    with nogil:
        for i in range(cps.shape[0]):
            o[i] = fse_simple_lower(cps[i])
    return out


def _raise_decode_error(data, position, where):
    """Raise CPython's error for invalid UTF-8 data, whose first bad byte the native
    decoder found at position; a disagreement between the two is a bug and says so."""
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as err:
        if err.start != position:
            raise AssertionError(
                f"the native decoder stopped at byte {position}, CPython at {err.start}"
            ) from err
        raise UnicodeDecodeError(err.encoding, err.object, err.start, err.end, err.reason + where) from None
    raise AssertionError(f"the native decoder rejected valid UTF-8 at byte {position}")


def tokenize(text, tokenizer="fse_words@1"):
    """The tokens of one sentence (str, or UTF-8 bytes) by the native tokenizer."""
    cdef int code = _tokenizer_code(tokenizer)
    if isinstance(text, str):
        data = (<str>text).encode("utf-8")
    elif isinstance(text, bytes):
        data = text
    else:
        raise TypeError(f"text must be str or bytes, not {type(text).__name__}")
    cdef const unsigned char *p = data
    cdef int64_t n = len(data)
    cdef unsigned char *cand = <unsigned char *>malloc(2 * n + 8)
    cdef int64_t *starts = <int64_t *>malloc((n + 1) * sizeof(int64_t))
    cdef int64_t *ends = <int64_t *>malloc((n + 1) * sizeof(int64_t))
    cdef fse_tokens out
    cdef int64_t bad
    try:
        if cand == NULL or starts == NULL or ends == NULL:
            raise MemoryError("could not allocate the token buffers")
        out.starts = starts
        out.ends = ends
        with nogil:
            bad = fse_tokenize(code, p, n, cand, &out)
        if bad >= 0:
            _raise_decode_error(data, bad, "")
        return [out.bytes[starts[i]:ends[i]].decode("utf-8") for i in range(out.n_tokens)]
    finally:
        free(cand)
        free(starts)
        free(ends)


# ---------------------------------------------------------------------------------------
# Stages 1 and 2: sentences of text to batches, without the GIL.
# ---------------------------------------------------------------------------------------

# A sentence longer than this is an error: fse reads one sentence per line or string.
cdef Py_ssize_t _MAX_SENTENCE_BYTES = 1 << 30
MAX_SENTENCE_BYTES = _MAX_SENTENCE_BYTES

cdef enum:
    ENCODE_OK = 0
    ENCODE_NO_MEMORY = 1
    ENCODE_BAD_UTF8 = 2
    ENCODE_TOO_LONG = 3


cdef inline int _ensure(_Buffer *b, Py_ssize_t extra) noexcept nogil:
    """Make room for extra more bytes; returns -1 if memory runs out."""
    if extra <= b.capacity - b.size:
        return 0
    return _grow_nogil(b, extra)


cdef class Workspace:
    """The buffers one worker reuses for every batch it encodes from text.

    A Workspace is used by one thread at a time; give every worker its own.
    """

    cdef _Buffer cand, starts, ends, word, tokens, offsets, ngram_offsets, ngrams
    cdef bint busy
    # The tokens the tokenizer gave in the last encode, unknown words included (the
    # sentence length of uSIF, as 1.x counted it).
    cdef readonly int64_t tokens_seen

    def __dealloc__(self):
        free(self.cand.data)
        free(self.starts.data)
        free(self.ends.data)
        free(self.word.data)
        free(self.tokens.data)
        free(self.offsets.data)
        free(self.ngram_offsets.data)
        free(self.ngrams.data)

    def __reduce__(self):
        raise TypeError("a Workspace belongs to one worker and cannot be pickled")


cdef class TextLookup:
    """Stages 1 and 2 for one vocabulary: the tokenizer, the lookup and FastText hashing.

    Immutable and shared by all workers; ``encode`` turns a range of sentences into a
    Batch. Without ``fasttext`` (min_n, max_n, bucket), a token that is no word is dropped;
    with it, it becomes OOV_TOKEN with its first ``max_ngrams`` ngram buckets.
    """

    cdef VocabTable table
    cdef readonly int tokenizer
    cdef bint fasttext
    cdef unsigned int min_n, max_n
    cdef uINT_t bucket
    cdef size_t max_ngrams
    cdef int _state

    def __init__(self, VocabTable table not None, tokenizer, fasttext=None, max_ngrams=0):
        # Built once, as a VocabTable: the state is set before any Python code runs, so a
        # second __init__ (from another thread or from within this one) raises.
        if self._state != TABLE_NEW:
            raise RuntimeError("a TextLookup is built once; make a new one")
        self._state = TABLE_BUILDING
        try:
            self._build(table, tokenizer, fasttext, max_ngrams)
        except BaseException:
            self._state = TABLE_FAILED
            raise
        self._state = TABLE_READY

    cdef _build(self, VocabTable table, tokenizer, fasttext, max_ngrams):
        table._check_ready()
        self.tokenizer = _tokenizer_code(tokenizer)
        self.fasttext = fasttext is not None
        if isinstance(max_ngrams, bool) or not isinstance(max_ngrams, (int, np.integer)):
            raise TypeError(f"max_ngrams must be an int, not {type(max_ngrams).__name__}")
        if self.fasttext:
            if not isinstance(fasttext, tuple) or len(fasttext) != 3 or any(
                isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in fasttext
            ):
                raise TypeError(f"fasttext must be a tuple of three ints (min_n, max_n, bucket), not {fasttext!r}")
            min_n, max_n, bucket = fasttext
            if not 1 <= min_n <= max_n or bucket < 0 or bucket >= 2**32:
                raise ValueError(f"fasttext must be (min_n, max_n, bucket) with 1 <= min_n <= max_n, not {fasttext}")
            if not 0 <= max_ngrams <= MAX_NGRAMS_LIMIT:
                raise ValueError(f"max_ngrams must be between 0 and {MAX_NGRAMS_LIMIT}, not {max_ngrams}")
            self.min_n, self.max_n, self.bucket = min_n, max_n, bucket
            self.max_ngrams = max_ngrams
        elif max_ngrams != 0:
            raise ValueError("max_ngrams needs fasttext")
        self.table = table

    def __reduce__(self):
        raise TypeError("a TextLookup cannot be pickled; build it from the word vectors")

    cdef int _sentence(self, const unsigned char *p, Py_ssize_t n, Workspace ws,
                       Py_ssize_t *bad) noexcept nogil:
        """Append the tokens of the sentence p[0:n] to the batch buffers of ws."""
        cdef fse_tokens out
        cdef Py_ssize_t k, start, length, cap, kept
        cdef bint padded
        cdef uINT_t *ids
        cdef int64_t err
        cdef uINT_t index
        cdef size_t written
        if n > _MAX_SENTENCE_BYTES:
            return ENCODE_TOO_LONG
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
        ws.tokens.size += kept * sizeof(uINT_t)
        if _push_i64_nogil(&ws.offsets, ws.tokens.size // sizeof(uINT_t)):
            return ENCODE_NO_MEMORY
        return ENCODE_OK

    def encode(self, const unsigned char[::1] data, offsets, Py_ssize_t first, Py_ssize_t last,
               first_row, Workspace ws not None):
        """The Batch of one range of sentences of data, which starts at row first_row.

        Line mode (offsets is None): the sentences are the lines that start in the bytes
        [first, last), which the caller aligns to line starts. Item mode: the sentences
        are the strings first to last - 1, delimited by the int64 offsets.
        """
        from fse._contract import Batch

        if self._state != TABLE_READY:
            raise RuntimeError("this TextLookup was not built; make a new one")
        if offsets is not None:
            # An own copy of the offsets of the range, so the caller cannot change them
            # between the check and the reads without the GIL.
            if not isinstance(offsets, np.ndarray) or offsets.dtype != np.int64 or offsets.ndim != 1:
                raise ValueError("offsets must be an int64 vector")
            if not 0 <= first <= last <= len(offsets) - 1:
                raise ValueError(f"range [{first}, {last}) is outside the {len(offsets) - 1} strings")
            offsets = offsets[first : last + 1].copy()
            last -= first
            first = 0
        _check_span(data, offsets, first, last)
        if ws.busy:
            raise RuntimeError("a Workspace is used by one thread at a time")
        cdef Py_ssize_t n = data.shape[0]
        cdef const unsigned char *p = &data[0] if n else NULL
        cdef bint lines = offsets is None
        cdef const np.int64_t[::1] item_offsets
        cdef const np.int64_t *items = NULL
        if not lines:
            item_offsets = offsets
            items = &item_offsets[0]
        cdef Py_ssize_t pos = first, start, end, sentence = 0, bad = -1
        cdef const unsigned char *newline
        cdef int status = ENCODE_OK
        ws.busy = True
        try:
            ws.tokens_seen = 0
            ws.tokens.size = 0
            ws.offsets.size = 0
            ws.ngram_offsets.size = 0
            ws.ngrams.size = 0
            with nogil:
                if _push_i64_nogil(&ws.offsets, 0) or _push_i64_nogil(&ws.ngram_offsets, 0):
                    status = ENCODE_NO_MEMORY
                while status == ENCODE_OK and pos < last:
                    if lines:
                        newline = <const unsigned char *>memchr(p + pos, 10, n - pos)
                        start = pos
                        end = (newline - p) if newline != NULL else n
                        pos = end + 1 if newline != NULL else n
                    else:
                        start = items[pos]
                        end = items[pos + 1]
                        pos += 1
                    status = self._sentence(p + start, end - start, ws, &bad)
                    if status == ENCODE_OK:
                        sentence += 1
            if status == ENCODE_NO_MEMORY:
                raise MemoryError("could not grow the buffers of the Workspace")
            if status == ENCODE_TOO_LONG:
                raise ValueError(
                    f"sentence {sentence} of the range has {end - start} bytes, more than "
                    f"{MAX_SENTENCE_BYTES}; fse reads one sentence per line or string"
                )
            if status == ENCODE_BAD_UTF8:
                _raise_decode_error(
                    bytes(data[start:end]), bad,
                    f" in sentence {sentence} of the range (byte {start + bad} of the source)",
                )
            tokens = _to_array(&ws.tokens, np.uint32)
            batch_offsets = _to_array(&ws.offsets, np.int64)
            if not self.fasttext:
                return Batch(first_row, batch_offsets, tokens)
            return Batch(
                first_row, batch_offsets, tokens,
                _to_array(&ws.ngram_offsets, np.int64), _to_array(&ws.ngrams, np.uint32),
            )
        finally:
            ws.busy = False
