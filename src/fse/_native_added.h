/* What the tokenizers with added tokens share (wordpiece@1, unigram@1): the scan for
   the added tokens of a sentence, its validation as UTF-8 and the character cut of step 1.
   Moved unchanged from _native_wordpiece.h, so the names keep its prefix. */
#ifndef FSE_NATIVE_ADDED_H
#define FSE_NATIVE_ADDED_H

#include <stdint.h>
#include <string.h>

#include "_native_text.h"

/* The index of the longest added token that starts at p[i], with the tokens sorted by
   length (longest first) in bytes[offsets[k]:offsets[k + 1]], or -1. first marks the
   bytes a token starts with. A token is valid UTF-8, so it can only start at the first
   byte of a code point of valid text. */
static inline int64_t fse_wp_added_at(const uint8_t *p, int64_t i, int64_t n,
                                      const uint8_t *first, const uint8_t *bytes,
                                      const int64_t *offsets, int64_t count) {
    if (!first[p[i]]) return -1;
    for (int64_t k = 0; k < count; k++) {
        int64_t len = offsets[k + 1] - offsets[k];
        if (len <= n - i && memcmp(p + i, bytes + offsets[k], (size_t)len) == 0) return k;
    }
    return -1;
}

/* The first byte of p[i:n] an added token may start at, or n: memchr for the one byte
   all tokens start with (single >= 0), else a loop over first. A byte that is not in
   first starts no token, so fse_wp_added_at needs to run only where this stops. */
static inline int64_t fse_wp_next_candidate(const uint8_t *p, int64_t i, int64_t n,
                                            const uint8_t *first, int single) {
    if (single >= 0) {
        const uint8_t *q = (const uint8_t *)memchr(p + i, single, (size_t)(n - i));
        return q ? (int64_t)(q - p) : n;
    }
    while (i < n && !first[p[i]]) i++;
    return i;
}

/* The first invalid UTF-8 byte of p[i:n], or -1. */
static inline int64_t fse_wp_validate(const uint8_t *p, int64_t i, int64_t n) {
    uint32_t cp;
    while (i < n) {
        if (p[i] < 0x80) {
            i++;
            continue;
        }
        int len = fse_utf8_decode(p, i, n, &cp);
        if (!len) return i;
        i += len;
    }
    return -1;
}

/* Step 1: the byte length of the first max_chars code points of p[0:n], which is
   validated whole; -1 - i for the first invalid byte p[i]. */
static inline int64_t fse_wp_char_cut(const uint8_t *p, int64_t n, int64_t max_chars) {
    uint32_t cp;
    int64_t i = 0, chars = 0, cut = n;
    while (i < n) {
        if (chars == max_chars && cut == n) cut = i;
        int len = 1;
        if (p[i] >= 0x80) {
            len = fse_utf8_decode(p, i, n, &cp);
            if (!len) return -1 - i;
        }
        i += len;
        chars++;
    }
    return cut;
}

#endif
