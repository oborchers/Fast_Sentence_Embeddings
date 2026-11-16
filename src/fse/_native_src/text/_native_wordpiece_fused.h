/* One read of printable ASCII: lowercase, word boundaries and short keys together.
   The caller owns the added-token split, the output and the token stop. A rejected
   block asks it to replay the WHOLE raw span with the existing normalizer. Nothing
   emitted by this scanner survives that replay. */
#ifndef FSE_NATIVE_WORDPIECE_FUSED_H
#define FSE_NATIVE_WORDPIECE_FUSED_H

#include "_native_wordpiece.h"

typedef struct {
    int64_t pos, available, start, end;
    uint64_t low, other, key[2];
} fse_wp_ascii_scan;

static inline uint64_t fse_wp_low_bytes(uint64_t x, int64_t n) {
    return n >= 8 ? x : x & ((1ull << (8 * n)) - 1);
}

/* Load only the remaining input bytes. Reject the entire block if it contains a
   non-printable byte or '['. SWAR borrows can only flag bytes above the first
   match, so the rejection is exact. Partial blocks mask the zero-filled tail. */
FSE_INLINE int fse_wp_ascii_refill(fse_wp_ascii_scan *s, const uint8_t *p,
                                  int64_t n, uint8_t *q) {
    int64_t k = n - s->pos;
    if (k == 0) return 0;
    if (k > 8) k = 8;
    uint64_t x = 0;
    memcpy(&x, p + s->pos, (size_t)k);
    uint64_t bracket = x ^ (FSE_WP_ONES * '[');
    uint64_t bad = fse_wp_unprintable8(x) |
                   ((bracket - FSE_WP_ONES) & ~bracket & FSE_WP_HIGH);
    if (fse_wp_low_bytes(bad, k)) return -1;
    s->low = fse_wp_lower8(x);
    s->other = fse_wp_not_alnum8(s->low);
    s->available = k;
    memcpy(q + s->pos, &s->low, (size_t)k);
    return 1;
}

static inline void fse_wp_ascii_advance(fse_wp_ascii_scan *s, int64_t k) {
    /* A shift by 64 is undefined in C. */
    s->low = k == 8 ? 0 : s->low >> (8 * k);
    s->other = k == 8 ? 0 : s->other >> (8 * k);
    s->available -= k;
    s->pos += k;
}

/* Add lowered bytes straight from the loaded register to the inline key. Long
   words still write q, but their length-seeded hash must be deferred until end. */
static inline void fse_wp_ascii_key(fse_wp_ascii_scan *s, int64_t k) {
    int64_t used = s->pos - s->start;
    if (used >= 16) return;
    int64_t lane = used / 8, shift = used % 8, first = 8 - shift;
    if (first > k) first = k;
    s->key[lane] |= fse_wp_low_bytes(s->low, first) << (8 * shift);
    if (k > first && lane == 0)
        s->key[1] |= fse_wp_low_bytes(s->low >> (8 * first), k - first);
}

/* 1: a word, 0: end, -1: replay the raw span. Printable ASCII word bytes are
   exactly letters and digits; all other bytes except space are punctuation.
   The existing ASCII initialization checks the alphanumeric classification.
   fse_wp_bytes_below provides the portable lowest-set-bit operation on ARM,
   GCC/Clang and MSVC, without an ISA-specific tzcnt intrinsic or a zero input. */
FSE_INLINE int fse_wp_ascii_next(fse_wp_ascii_scan *s, const uint8_t *p,
                                int64_t n, uint8_t *q) {
    int status;
    for (;;) {
        if (!s->available) {
            status = fse_wp_ascii_refill(s, p, n, q);
            if (status <= 0) return status;
        }
        if ((uint8_t)s->low != ' ') break;
        fse_wp_ascii_advance(s, 1);
    }
    s->start = s->pos;
    s->key[0] = s->key[1] = 0;
    if (s->other & 0x80) {
        s->key[0] = (uint8_t)s->low;
        fse_wp_ascii_advance(s, 1);
    } else {
        for (;;) {
            int64_t k = s->other ? fse_wp_bytes_below(s->other) : 8;
            if (k > s->available) k = s->available;
            fse_wp_ascii_key(s, k);
            fse_wp_ascii_advance(s, k);
            if (s->available) break;
            status = fse_wp_ascii_refill(s, p, n, q);
            if (status < 0) return status;
            if (!status || (s->other & 0x80)) break;
        }
    }
    s->end = s->pos;
    return 1;
}

#endif /* FSE_NATIVE_WORDPIECE_FUSED_H */
