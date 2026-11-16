# Private direct probes of character masks and every scanner resumption point.
cdef extern from "_native_wordpiece_blocks.h":
    int FSE_WP_SCAN_CAPACITY
    ctypedef struct fse_wp_masks:
        uint64_t slow, space, punct
    ctypedef struct fse_wp_scan:
        int64_t b, open, pend_start, pend_end
        int j, pending, done
    fse_wp_masks fse_wp_classify64(const unsigned char *q, int variant) noexcept nogil
    void fse_wp_scan_init(fse_wp_scan *st, const unsigned char *q, int variant) noexcept nogil
    int fse_wp_scan_words(fse_wp_scan *st, const unsigned char *q, int64_t m,
                          int64_t *starts, int64_t *ends, int capacity, int variant) noexcept nogil
    int fse_wp_scan_next(fse_wp_scan *st, const unsigned char *q, int64_t m,
                         int64_t *start, int64_t *end) noexcept nogil
    const char *fse_wp_simd_name() noexcept nogil


def _wp_scan_capacity():
    """Compiled production arm: one range for the iterator or 64 for the batch."""
    return FSE_WP_SCAN_CAPACITY


def _wp_block_masks(q):
    """Masks of exactly 64 bytes, from every compiled classifier."""
    if not isinstance(q, bytes):
        raise TypeError("q must be bytes")
    if len(q) != 64:
        raise ValueError("q must contain exactly 64 bytes")
    cdef const unsigned char *data = q
    cdef fse_wp_masks masks
    result = {}
    for variant, name in ((1, "byte"), (2, "swar"), (0, fse_wp_simd_name().decode("ascii"))):
        masks = fse_wp_classify64(data, variant)
        result[name] = (masks.slow, masks.space, masks.punct)
    return result


def _wp_scan_words(q, capacity=64, variant="native"):
    """Ranges per call and saved states, including a capacity-one iterator."""
    if not isinstance(q, bytes):
        raise TypeError("q must be bytes")
    q.decode("utf-8")
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("capacity must be an int")
    if not 1 <= capacity <= 64:
        raise ValueError("capacity must be between 1 and 64")
    if not isinstance(variant, str):
        raise TypeError("variant must be a str")
    if variant not in ("native", "byte", "swar"):
        raise ValueError("variant must be native, byte or swar")
    cdef int v = {"native": 0, "byte": 1, "swar": 2}[variant]
    cdef bytes padded = q + b" " * 64
    cdef const unsigned char *data = padded
    cdef fse_wp_scan scanner
    cdef int64_t starts[64]
    cdef int64_t ends[64]
    cdef int size, k
    cdef int64_t start, end
    fse_wp_scan_init(&scanner, data, v)
    calls = []
    while not scanner.done:
        if capacity == 1 and v == 0:
            size = fse_wp_scan_next(&scanner, data, len(q), &start, &end)
            if size:
                starts[0], ends[0] = start, end
        else:
            size = fse_wp_scan_words(&scanner, data, len(q), starts, ends, capacity, v)
        pending = (scanner.pend_start, scanner.pend_end) if scanner.pending else None
        calls.append(([(starts[k], ends[k]) for k in range(size)],
                      (scanner.b, scanner.j, scanner.open, pending, bool(scanner.done))))
    return calls
