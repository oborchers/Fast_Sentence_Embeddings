/* wordpiece@1 (ARCHITECTURE.md, "wordpiece@1"): the added tokens, the normalization and
   the pre-tokenization of one sentence; _native_wordpiece.pxi holds the greedy match on
   the VocabTable and the sentence around them.

   fse_wp_normalize gives the UTF-8 of steps 3.1 to 3.5 of one piece: every code point
   gives its items from the composed table of _native_unicode_bert.h (its code points
   after cleaning, decomposition, accent removal and lowercase, each with the combining
   class it had before lowercasing), CJK gets spaces and Hangul decomposes by the
   algorithm; items of a nonzero class collect in a run, which is sorted stably by class
   and written when an item of class 0 (or a removed mark of class 0) or the end comes.
   The table guarantees at most 3 bytes out per byte in and that items of a nonzero class
   are single code points.

   Correct first, not fast: 12b profiles and optimizes. fse._reference_wordpiece defines
   the result; tests/test_text_conformance_wordpiece.py compares. */
#ifndef FSE_NATIVE_WORDPIECE_H
#define FSE_NATIVE_WORDPIECE_H

#include <stdint.h>
#include <string.h>

#include "_native_text.h"
#include "_native_unicode_bert.h"

/* Hangul syllables (the algorithm of the standard, section 3.12). */
#define FSE_WP_S_BASE 0xAC00u
#define FSE_WP_L_BASE 0x1100u
#define FSE_WP_V_BASE 0x1161u
#define FSE_WP_T_BASE 0x11A7u
#define FSE_WP_T_COUNT 28u
#define FSE_WP_N_COUNT (21u * 28u)

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

/* Sort the run of r items stably by class (insertion sort; runs are short) and write
   their code points at out + m; returns the new end. */
static inline int64_t fse_wp_flush(uint32_t *run, int64_t r, uint8_t *out, int64_t m) {
    for (int64_t a = 1; a < r; a++) {
        uint32_t item = run[a];
        int64_t b = a;
        while (b > 0 && (run[b - 1] >> 24) > (item >> 24)) {
            run[b] = run[b - 1];
            b--;
        }
        run[b] = item;
    }
    for (int64_t a = 0; a < r; a++) m += fse_utf8_encode(run[a] & 0xFFFFFFu, out + m);
    return m;
}

/* One item: a nonzero class joins the run; class 0 ends it and is written, unless it is
   a removed mark (FSE_BERT_BARRIER), which only ends it. */
static inline int64_t fse_wp_item(uint32_t item, uint32_t *run, int64_t *r, uint8_t *out,
                                  int64_t m) {
    if (item >> 24) {
        run[(*r)++] = item;
        return m;
    }
    m = fse_wp_flush(run, *r, out, m);
    *r = 0;
    if (item != FSE_BERT_BARRIER) m += fse_utf8_encode(item, out + m);
    return m;
}

/* The items of the code point cp with entry e, after its CJK space. */
static inline int64_t fse_wp_items(uint32_t cp, uint32_t e, uint32_t *run, int64_t *r,
                                   uint8_t *out, int64_t m) {
    if (e & FSE_BERT_HANGUL) {
        uint32_t s = cp - FSE_WP_S_BASE;
        m = fse_wp_item(FSE_WP_L_BASE + s / FSE_WP_N_COUNT, run, r, out, m);
        m = fse_wp_item(FSE_WP_V_BASE + (s % FSE_WP_N_COUNT) / FSE_WP_T_COUNT, run, r, out, m);
        if (s % FSE_WP_T_COUNT) m = fse_wp_item(FSE_WP_T_BASE + s % FSE_WP_T_COUNT, run, r, out, m);
        return m;
    }
    if (e & FSE_BERT_EXPAND) {
        const uint32_t *items = fse_bert_pool + (e >> FSE_BERT_SHIFT);
        for (uint32_t k = 1; k <= items[0]; k++) m = fse_wp_item(items[k], run, r, out, m);
        return m;
    }
    return fse_wp_item(cp, run, r, out, m);  /* itself, of class 0 */
}

/* Steps 3.1 to 3.5 of the piece p[0:n] into out, which holds 3 n bytes; run holds 3 n
   items (each is at least one byte of out). Returns the bytes written, or -1 - i for the first invalid UTF-8 at p[i]. */
static int64_t fse_wp_normalize(const uint8_t *p, int64_t n, uint8_t *out, uint32_t *run) {
    int64_t i = 0, m = 0, r = 0;
    while (i < n) {
        uint32_t cp;
        int len = 1;
        if (p[i] < 0x80) {
            cp = p[i];
        } else {
            len = fse_utf8_decode(p, i, n, &cp);
            if (!len) return -1 - i;
        }
        i += len;
        uint32_t e = fse_bert_entry(cp);
        if (e & FSE_BERT_DROP) continue;
        if (e & FSE_BERT_CJK) m = fse_wp_item(0x20, run, &r, out, m);
        m = fse_wp_items(cp, e, run, &r, out, m);
        if (e & FSE_BERT_CJK) m = fse_wp_item(0x20, run, &r, out, m);
    }
    return fse_wp_flush(run, r, out, m);
}

/* The code point at q[i] of normalized (valid) UTF-8 and its length. */
static inline int fse_wp_decode(const uint8_t *q, int64_t i, int64_t m, uint32_t *cp) {
    if (q[i] < 0x80) {
        *cp = q[i];
        return 1;
    }
    int len = fse_utf8_decode(q, i, m, cp);
    return len ? len : 1;  /* never 0: the normalizer writes valid UTF-8 */
}

/* Step 4: the next word of the normalized q[*pos:m] as q[*start:*end]: a run of code
   points that are neither White_Space nor punctuation, or one punctuation code point.
   Returns 0 when none is left. */
static inline int fse_wp_next_word(const uint8_t *q, int64_t m, int64_t *pos, int64_t *start,
                                   int64_t *end) {
    int64_t i = *pos;
    uint32_t cp;
    while (i < m) {
        int len = fse_wp_decode(q, i, m, &cp);
        uint32_t e = fse_bert_entry(cp);
        if (e & FSE_BERT_SPACE) {
            i += len;
            continue;
        }
        *start = i;
        i += len;
        if (!(e & FSE_BERT_PUNCT)) {
            while (i < m) {
                len = fse_wp_decode(q, i, m, &cp);
                if (fse_bert_entry(cp) & (FSE_BERT_SPACE | FSE_BERT_PUNCT)) break;
                i += len;
            }
        }
        *end = *pos = i;
        return 1;
    }
    *pos = m;
    return 0;
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

#endif /* FSE_NATIVE_WORDPIECE_H */
