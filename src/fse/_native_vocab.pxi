# Included by _native.pyx: UTF-8 classes, word hashing and the vocabulary table.

cdef uINT_t NOT_FOUND = <uINT_t>0xFFFFFFFF

# Return codes of the native routines, translated into exceptions with the GIL.
cdef enum:
    NATIVE_OK = 0
    NATIVE_NO_MEMORY = -1
    NATIVE_BAD_UTF8 = -2
    NATIVE_BAD_ROW = -3


cdef inline int _utf8_len(const unsigned char *p, Py_ssize_t i, Py_ssize_t end) noexcept nogil:
    """Length of the valid UTF-8 sequence at p[i], or 0 where CPython's decoder fails."""
    cdef unsigned char b = p[i]
    cdef unsigned char c
    if b < 0x80:
        return 1
    if b < 0xC2:
        return 0
    if b < 0xE0:
        if end - i >= 2 and (p[i + 1] & 0xC0) == 0x80:
            return 2
        return 0
    if b < 0xF0:
        if end - i < 3:
            return 0
        c = p[i + 1]
        if (b == 0xE0 and c < 0xA0) or (b == 0xED and c > 0x9F):
            return 0
        if (c & 0xC0) != 0x80 or (p[i + 2] & 0xC0) != 0x80:
            return 0
        return 3
    if b < 0xF5:
        if end - i < 4:
            return 0
        c = p[i + 1]
        if (b == 0xF0 and c < 0x90) or (b == 0xF4 and c > 0x8F):
            return 0
        if (c & 0xC0) != 0x80 or (p[i + 2] & 0xC0) != 0x80 or (p[i + 3] & 0xC0) != 0x80:
            return 0
        return 4
    return 0


cdef inline bint _is_space(const unsigned char *p, Py_ssize_t i, int n) noexcept nogil:
    """Whether the valid UTF-8 sequence of length n at p[i] satisfies str.isspace()."""
    cdef unsigned char b = p[i]
    cdef unsigned char c, d
    if n == 1:
        return (9 <= b <= 13) or (28 <= b <= 32)
    if n == 2:
        return b == 0xC2 and (p[i + 1] == 0x85 or p[i + 1] == 0xA0)
    if n == 3:
        c = p[i + 1]
        d = p[i + 2]
        if b == 0xE1:
            return c == 0x9A and d == 0x80  # U+1680
        if b == 0xE2:
            if c == 0x80:  # U+2000 to U+200A, U+2028, U+2029, U+202F
                return d <= 0x8A or d == 0xA8 or d == 0xA9 or d == 0xAF
            return c == 0x81 and d == 0x9F  # U+205F
        if b == 0xE3:
            return c == 0x80 and d == 0x80  # U+3000
    return False


# Words of at most this many UTF-8 bytes are stored in their slot, so looking one up
# usually reads a single cache line; longer words are compared in the arena.
cdef enum:
    INLINE_KEY_BYTES = 16


cdef inline np.uint64_t _load_partial(const unsigned char *p, Py_ssize_t k) noexcept nogil:
    """The k <= 8 bytes at p as a little-endian word, zero above them."""
    cdef np.uint64_t w = 0
    cdef Py_ssize_t i
    if k == 8:
        memcpy(&w, p, 8)
        return w
    for i in range(k):
        w |= (<np.uint64_t>p[i]) << (8 * i)
    return w


cdef inline np.uint64_t _mix(np.uint64_t h, np.uint64_t w) noexcept nogil:
    h = (h ^ w) * 0xBF58476D1CE4E5B9ULL
    return h ^ (h >> 29)


cdef inline np.uint64_t _load_masked(const unsigned char *p, Py_ssize_t k) noexcept nogil:
    """The k <= 8 bytes at p as a word, zero above them; reads 8 bytes."""
    cdef np.uint64_t w
    memcpy(&w, p, 8)
    return w if k >= 8 else w & ((1ULL << (8 * k)) - 1)


cdef inline np.uint64_t _hash_word(
    const unsigned char *p, Py_ssize_t n, np.uint64_t *key, bint padded
) noexcept nogil:
    """The slot hash of a word, 8 bytes per step. For a word of at most INLINE_KEY_BYTES
    bytes, key receives its bytes as two zero-padded words, as a slot stores them. With
    padded, 16 bytes from p are readable, so the words are loaded whole and masked.

    Only VocabTable uses it, for building and looking up, so it is no persistence format.
    """
    cdef np.uint64_t h = 0x9E3779B97F4A7C15ULL ^ <np.uint64_t>n
    cdef Py_ssize_t left = n
    if n <= INLINE_KEY_BYTES:
        if padded:
            key[0] = _load_masked(p, n)
            key[1] = _load_masked(p + 8, n - 8) if n > 8 else 0
        else:
            key[0] = _load_partial(p, n if n < 8 else 8)
            key[1] = _load_partial(p + 8, n - 8) if n > 8 else 0
        h = _mix(_mix(h, key[0]), key[1])
    else:
        while left >= 8:
            h = _mix(h, _load_partial(p, 8))
            p += 8
            left -= 8
        h = _mix(h, _load_partial(p, left))
    h *= 0x94D049BB133111EBULL
    return h ^ (h >> 32)


# One slot of the open-addressing table, 32 bytes. value is NOT_FOUND for a free slot.
cdef struct VocabSlot:
    np.uint64_t hash
    uINT_t value
    np.uint32_t length
    # The word itself, zero-padded, if length <= INLINE_KEY_BYTES, else its offset in the
    # arena (int64).
    np.uint64_t key[2]


cdef struct VocabView:
    const VocabSlot *slots
    const unsigned char *arena
    np.uint64_t mask


cdef inline uINT_t _vocab_lookup(
    const VocabView *v, const unsigned char *p, Py_ssize_t n, bint padded=False
) noexcept nogil:
    """Vocabulary index of the n bytes at p, or NOT_FOUND; padded as for _hash_word."""
    cdef np.uint64_t key[2]
    cdef np.uint64_t h = _hash_word(p, n, key, padded)
    cdef np.uint64_t j = h & v.mask
    cdef const VocabSlot *slot
    while True:
        slot = &v.slots[j]
        if slot.value == NOT_FOUND:
            return NOT_FOUND
        if slot.hash == h and slot.length == n:
            if n <= INLINE_KEY_BYTES:
                if slot.key[0] == key[0] and slot.key[1] == key[1]:
                    return slot.value
            elif memcmp(v.arena + <np.int64_t>slot.key[0], p, n) == 0:
                return slot.value
        j = (j + 1) & v.mask


cdef inline np.int64_t _vocab_insert(
    VocabSlot *slots, np.uint64_t mask, const unsigned char *arena, np.int64_t offset,
    Py_ssize_t n, uINT_t index,
) noexcept nogil:
    """Insert the n bytes at arena[offset] as word index; returns -1, or the index of the
    word already stored with the same bytes."""
    cdef const unsigned char *p = arena + offset
    cdef np.uint64_t key[2]
    cdef np.uint64_t h = _hash_word(p, n, key, False)
    cdef np.uint64_t j = h & mask
    while slots[j].value != NOT_FOUND:
        if slots[j].hash == h and slots[j].length == n:
            if n <= INLINE_KEY_BYTES:
                if slots[j].key[0] == key[0] and slots[j].key[1] == key[1]:
                    return slots[j].value
            elif memcmp(arena + <np.int64_t>slots[j].key[0], p, n) == 0:
                return slots[j].value
        j = (j + 1) & mask
    slots[j].hash = h
    slots[j].value = index
    slots[j].length = <np.uint32_t>n
    if n <= INLINE_KEY_BYTES:
        slots[j].key[0] = key[0]
        slots[j].key[1] = key[1]
    else:
        slots[j].key[0] = <np.uint64_t>offset
        slots[j].key[1] = 0
    return -1


cdef class VocabTable:
    """Immutable map from the UTF-8 bytes of a word to its vocabulary index.

    Built without the GIL from the vocabulary of word vectors: ``vocab_bytes`` (uint8)
    holds the words one after another, word i is ``vocab_bytes[vocab_offsets[i]:
    vocab_offsets[i + 1]]``. Open addressing with linear probing, at most half full. Each
    32-byte slot holds the hash, the index and, for words of up to INLINE_KEY_BYTES bytes,
    the word itself, so a lookup usually reads one cache line; longer words are compared
    in the table's own copy of vocab_bytes.
    """

    cdef object _bytes
    cdef object _slots
    cdef Py_ssize_t n_words
    cdef VocabView view

    def __init__(self, vocab_bytes, vocab_offsets):
        if (
            not isinstance(vocab_bytes, np.ndarray) or vocab_bytes.dtype != np.uint8
            or vocab_bytes.ndim != 1 or not vocab_bytes.flags.c_contiguous
        ):
            raise ValueError("vocab_bytes must be a C-contiguous uint8 vector")
        if (
            not isinstance(vocab_offsets, np.ndarray) or vocab_offsets.dtype != np.int64
            or vocab_offsets.ndim != 1 or not vocab_offsets.flags.c_contiguous
        ):
            raise ValueError("vocab_offsets must be a C-contiguous int64 vector")
        # Own copies first, and every check on them: arrays the caller still holds (or a
        # writable base of a read-only view) could change while numpy copies them.
        vocab_offsets = vocab_offsets.copy()
        vocab_bytes = vocab_bytes.copy()
        if (
            len(vocab_offsets) == 0 or vocab_offsets[0] != 0
            or vocab_offsets[len(vocab_offsets) - 1] != len(vocab_bytes)
        ):
            raise ValueError(f"vocab_offsets must run from 0 to {len(vocab_bytes)}, the number of bytes")
        lengths = vocab_offsets[1:] - vocab_offsets[: len(vocab_offsets) - 1]
        if (lengths < 0).any():
            raise ValueError("vocab_offsets must not decrease")
        if len(lengths) and int(lengths.max()) >= 2**32:
            raise ValueError(f"word {int(lengths.argmax())} has 2**32 or more bytes")
        cdef Py_ssize_t n = len(vocab_offsets) - 1
        # Ids from OOV_TOKEN on are reserved.
        if n >= <Py_ssize_t>OOV_TOKEN:
            raise ValueError(f"a vocabulary holds fewer than {OOV_TOKEN} words, not {n}")
        cdef np.uint64_t size = 2
        while size < 2 * <np.uint64_t>n:
            size <<= 1
        # Raw bytes, so the slot layout is the C struct's, whatever numpy would pad.
        self._slots = np.empty(size * sizeof(VocabSlot), dtype=np.uint8)
        self._bytes = vocab_bytes
        self.n_words = n
        cdef VocabSlot *slots = <VocabSlot *>np.PyArray_DATA(self._slots)
        cdef const unsigned char *arena = <const unsigned char *>np.PyArray_DATA(vocab_bytes)
        cdef const np.int64_t[::1] offsets = vocab_offsets
        cdef np.uint64_t mask = size - 1
        cdef np.uint64_t j
        cdef Py_ssize_t i
        cdef np.int64_t dup = -1
        with nogil:
            memset(slots, 0, size * sizeof(VocabSlot))
            for j in range(size):
                slots[j].value = NOT_FOUND
            for i in range(n):
                dup = _vocab_insert(slots, mask, arena, offsets[i], offsets[i + 1] - offsets[i], <uINT_t>i)
                if dup >= 0:
                    break
        if dup >= 0:
            raise ValueError(f"word {i} is also word {dup} of the vocabulary")
        self.view.slots = slots
        self.view.arena = arena
        self.view.mask = mask

    def __reduce__(self):
        raise TypeError("a VocabTable cannot be pickled; build it from the word vectors")

    def __len__(self):
        return self.n_words

    def get(self, word):
        """Vocabulary index of word (str or bytes), or None; for tests."""
        encoded = word if isinstance(word, bytes) else (<str?>word).encode("utf-8")
        cdef uINT_t index = _vocab_lookup(&self.view, encoded, len(encoded))
        return None if index == NOT_FOUND else index
