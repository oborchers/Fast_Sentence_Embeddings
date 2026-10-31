# Included by _native.pyx: FastText ngram hashing and the limits of a native chunk.
# Adapted directly from fastText src/dictionary.cc, Dictionary::hash/computeSubwords.
# Copyright (c) 2016-present, Facebook, Inc. MIT license, see NOTICE.

cdef inline uINT_t _ft_hash(const unsigned char *begin, const unsigned char *end) noexcept nogil:
    """fastText's FNV-1a variant sign-extends bytes for compatibility with its models."""
    cdef uINT_t value = 2166136261
    while begin != end:
        value = (value ^ <uINT_t><signed char>begin[0]) * <uINT_t>16777619
        begin += 1
    return value


cdef size_t ft_ngram_hashes_into(
    const unsigned char *word,
    size_t byte_count,
    unsigned int min_n,
    unsigned int max_n,
    uINT_t bucket,
    uINT_t *out,
    size_t max_out,
) noexcept nogil:
    """Visit UTF-8 character spans in fastText order, omitting lone boundary markers.

    The input includes < and >. Only the first max_out buckets are written; zero
    buckets means the model has no subword table. No buffer allocation needs the GIL.
    """
    cdef size_t start = 0, end, characters, count = 0
    if bucket == 0 or max_out == 0:
        return 0
    while start < byte_count:
        end = start
        characters = 0
        while end < byte_count and characters < max_n:
            # Consume a whole code point before considering this span.
            end += 1
            while end < byte_count and (word[end] & 0xC0) == 0x80:
                end += 1
            characters += 1
            if characters < min_n:
                continue
            if characters == 1 and (start == 0 or end == byte_count):
                continue
            out[count] = _ft_hash(word + start, word + end) % bucket
            count += 1
            if count == max_out:
                return count
        start += 1
        while start < byte_count and (word[start] & 0xC0) == 0x80:
            start += 1
    return count


def ft_ngram_hashes(word, unsigned int min_n, unsigned int max_n, uINT_t bucket, size_t max_ngrams):
    """Return the first max_ngrams subword buckets in fastText's traversal order."""
    cdef bytes bounded = ("<" + word + ">").encode("utf-8")
    out = np.empty(max(1, len(bounded) * max(1, max_n)), dtype=np.uint32)
    cdef uINT_t[::1] view = out
    cdef size_t count = ft_ngram_hashes_into(
        bounded, len(bounded), min_n, max_n, bucket, &view[0], min(max_ngrams, view.shape[0])
    )
    return out[:count].tolist()


# A token of a FastText job that is not in the vocabulary. Its UTF-8 bytes "<token>" are
# stored with the job and hashed into ngram buckets by the worker, without the GIL.
cdef uINT_t OOV_TOKEN = <uINT_t>0xFFFFFFFE


# More ngrams than any word of a real corpus has; a bound that keeps buffer sizes in range.
MAX_NGRAMS_LIMIT = 1 << 24

# Sentences of a native chunk at most, besides its byte size (see scan_spans).
cdef Py_ssize_t CHUNK_MAX_ROWS = 1 << 16
MAX_CHUNK_ROWS = CHUNK_MAX_ROWS
