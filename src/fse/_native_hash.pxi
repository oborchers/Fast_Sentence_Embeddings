# Included by _native.pyx: FastText ngram hashing and the limits of a native chunk.

cdef unsigned char _MB_MASK = 0xC0
cdef unsigned char _MB_START = 0x80


cdef inline uINT_t _ft_hash(const unsigned char *bytez, size_t start, size_t stop) noexcept nogil:
    """FNV-1a over bytez[start:stop], with each byte sign-extended as in fastText."""
    cdef uINT_t h = 2166136261
    cdef size_t k
    for k in range(start, stop):
        h = h ^ <uINT_t>(<signed char>bytez[k])
        h = h * 16777619
    return h


cdef size_t ft_ngram_hashes_into(
    const unsigned char *bytez,
    size_t num_bytes,
    unsigned int min_n,
    unsigned int max_n,
    uINT_t bucket,
    uINT_t *out,
    size_t max_out,
) noexcept nogil:
    """Write the bucket of the first max_out ngrams of the UTF-8 word "<word>" into out.

    Returns the number written, min(number of ngrams, max_out). A bucket of 0 means the
    model has no ngram vectors, so no ngram is written.
    """
    cdef size_t i, j, n
    cdef size_t written = 0
    if bucket == 0:
        return 0
    for i in range(num_bytes):
        if bytez[i] & _MB_MASK == _MB_START:
            continue
        j = i
        n = 1
        while j < num_bytes and n <= max_n:
            j += 1
            while j < num_bytes and (bytez[j] & _MB_MASK) == _MB_START:
                j += 1
            if n >= min_n and not (n == 1 and (i == 0 or j == num_bytes)):
                if written == max_out:
                    return written
                out[written] = _ft_hash(bytez, i, j) % bucket
                written += 1
            n += 1
    return written


def ft_ngram_hashes(word, unsigned int min_n, unsigned int max_n, uINT_t bucket, size_t max_ngrams):
    """Return the ngram buckets of word as gensim's ft_ngram_hashes(...)[:max_ngrams] does."""
    cdef bytes utf8_word = ("<%s>" % word).encode("utf-8")
    out = np.empty(max(len(utf8_word) * max(max_n, 1), 1), dtype=np.uint32)
    cdef uINT_t[::1] view = out
    cdef size_t n = ft_ngram_hashes_into(
        utf8_word, len(utf8_word), min_n, max_n, bucket, &view[0], min(max_ngrams, view.shape[0])
    )
    return out[:n].tolist()




# A token of a FastText job that is not in the vocabulary. Its UTF-8 bytes "<token>" are
# stored with the job and hashed into ngram buckets by the worker, without the GIL.
cdef uINT_t OOV_TOKEN = <uINT_t>0xFFFFFFFE


# More ngrams than any word of a real corpus has; a bound that keeps buffer sizes in range.
MAX_NGRAMS_LIMIT = 1 << 24

# Sentences of a native chunk at most, besides its byte size (see scan_spans).
cdef Py_ssize_t CHUNK_MAX_ROWS = 1 << 16
MAX_CHUNK_ROWS = CHUNK_MAX_ROWS
