/* wordpiece@1 (ARCHITECTURE.md, "wordpiece@1"): the added tokens, the normalization and
   the pre-tokenization of one sentence; _native_wordpiece.pxi holds the greedy match on
   the VocabTable and the trie of _native_wordpiece_trie.h, and the sentence around them.

   fse_wp_normalize gives the UTF-8 of steps 3.1 to 3.5 of one piece: every code point
   gives its items from the composed table of _native_unicode_bert.h (its code points
   after cleaning, decomposition, accent removal and lowercase, each with the combining
   class it had before lowercasing), CJK gets spaces and Hangul decomposes by the
   algorithm; items of a nonzero class collect in a run, which is sorted stably by class
   and written when an item of class 0 (or a removed mark of class 0) or the end comes.
   The table guarantees at most 3 bytes out per byte in and that items of a nonzero class
   are single code points.

   The fast paths (the *_ascii functions and fse_wp_next_candidate) give the result of the
   generic ones, which a private switch of WordPieceState keeps for the tests.
   fse._reference_wordpiece defines the result; tests/test_text_conformance_wordpiece.py
   compares. */
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

/* One code point of p at *i, which it advances: its items into the run or out + m.
   Returns the new end, or -1 - i for invalid UTF-8 at p[i]. */
static inline int64_t fse_wp_step(const uint8_t *p, int64_t *i, int64_t n, uint8_t *out,
                                  uint32_t *run, int64_t *r, int64_t m) {
    uint32_t cp;
    int len = 1;
    if (p[*i] < 0x80) {
        cp = p[*i];
    } else {
        len = fse_utf8_decode(p, *i, n, &cp);
        if (!len) return -1 - *i;
    }
    *i += len;
    uint32_t e = fse_bert_entry(cp);
    if (e & FSE_BERT_DROP) return m;
    if (e & FSE_BERT_CJK) m = fse_wp_item(0x20, run, r, out, m);
    m = fse_wp_items(cp, e, run, r, out, m);
    if (e & FSE_BERT_CJK) m = fse_wp_item(0x20, run, r, out, m);
    return m;
}

/* Steps 3.1 to 3.5 of the piece p[0:n] into out, which holds 3 n bytes; run holds 3 n
   items (each is at least one byte of out). Returns the bytes written, or -1 - i for the
   first invalid UTF-8 at p[i]. The generic path: every code point by fse_wp_step. */
static int64_t fse_wp_normalize(const uint8_t *p, int64_t n, uint8_t *out, uint32_t *run) {
    int64_t i = 0, m = 0, r = 0;
    while (i < n) {
        m = fse_wp_step(p, &i, n, out, run, &r, m);
        if (m < 0) return m;
    }
    return fse_wp_flush(run, r, out, m);
}

/* The ASCII tables of the fast paths, derived from fse_bert_entry by fse_wp_ascii_init:
   the normalized byte of an ASCII byte, whether it is kept, and the class of an ASCII
   byte of normalized text (FSE_BERT_SPACE, FSE_BERT_PUNCT or 0 for a word byte). */
static uint8_t fse_wp_ascii_map[128], fse_wp_ascii_keep[128], fse_wp_ascii_class[128];

/* Fills the ASCII tables. Every ASCII entry must have one of three shapes: unchanged,
   removed (FSE_BERT_DROP), or one ASCII item of class 0 (FSE_BERT_EXPAND). Returns 0, or
   1 + b for the first byte b of another shape, which the fast paths cannot take. */
static int fse_wp_ascii_init(void) {
    for (uint32_t b = 0; b < 128; b++) {
        uint32_t e = fse_bert_entry(b);
        fse_wp_ascii_class[b] = (uint8_t)(e & (FSE_BERT_SPACE | FSE_BERT_PUNCT));
        fse_wp_ascii_map[b] = (uint8_t)b;
        fse_wp_ascii_keep[b] = 1;
        if (e & (FSE_BERT_CJK | FSE_BERT_HANGUL)) return 1 + (int)b;
        if (e & FSE_BERT_DROP) {
            fse_wp_ascii_keep[b] = 0;
        } else if (e & FSE_BERT_EXPAND) {
            const uint32_t *items = fse_bert_pool + (e >> FSE_BERT_SHIFT);
            if (items[0] != 1 || items[1] >= 0x80) return 1 + (int)b;  /* class 0 below 0x80 */
            fse_wp_ascii_map[b] = (uint8_t)items[1];
        }
        /* The block step of fse_wp_normalize_ascii keeps printable ASCII and lowercases A to Z. */
        uint32_t lower = b >= 'A' && b <= 'Z' ? b + 0x20 : b;
        if (b >= 0x20 && b < 0x7F && (!fse_wp_ascii_keep[b] || fse_wp_ascii_map[b] != lower))
            return 1 + (int)b;
        /* The block step of fse_wp_next_word_ascii takes a-z and 0-9 as word bytes. */
        if (((b >= 'a' && b <= 'z') || (b >= '0' && b <= '9')) && fse_wp_ascii_class[b])
            return 1 + (int)b;
    }
    return 0;
}

#define FSE_WP_ONES 0x0101010101010101ull
#define FSE_WP_HIGH 0x8080808080808080ull

/* The high bit of each byte of x that is not printable ASCII (0x20 to 0x7E); the lowest
   flagged byte is exact (a borrow can only flag bytes above it). */
static inline uint64_t fse_wp_unprintable8(uint64_t x) {
    uint64_t below = (x - FSE_WP_ONES * 0x20) & ~x;             /* bytes below 0x20 */
    uint64_t del = ((x ^ (FSE_WP_ONES * 0x7F)) - FSE_WP_ONES) & ~(x ^ (FSE_WP_ONES * 0x7F));
    return (x | below | del) & FSE_WP_HIGH;
}

/* x with A to Z lowercased; exact in each byte below the first byte at or above 0x80
   (a carry only moves up, out of a byte at or above 0x80). */
static inline uint64_t fse_wp_lower8(uint64_t x) {
    uint64_t from_a = x + FSE_WP_ONES * (0x80 - 'A');
    uint64_t after_z = x + FSE_WP_ONES * (0x80 - 'Z' - 1);
    return x | (((from_a & ~after_z) & FSE_WP_HIGH) >> 2);
}

/* The number of bytes below the lowest flagged byte of a nonzero mask of high bits. */
static inline int fse_wp_bytes_below(uint64_t mask) {
    uint64_t below = (mask & (~mask + 1)) - 1;  /* all bits below the lowest flagged one */
    return (int)((((below >> 7) & FSE_WP_ONES) * FSE_WP_ONES) >> 56);
}

/* fse_wp_normalize with the ASCII fast path: while the run of marks is empty, at an ASCII
   byte, 8 bytes of printable ASCII are lowercased at once (when at least 8 bytes of the piece are left;
   loads and stores by memcpy, so nothing is read beyond the piece), and any other ASCII
   byte is its table entry. A byte that is not ASCII, and any byte while the run holds
   marks (an ASCII byte ends the run, a removed one does not), takes fse_wp_step. out has
   3 n bytes, so the 8 bytes stored at out + m always fit (m <= 3 i and i + 8 <= n). */
static int64_t fse_wp_normalize_ascii(const uint8_t *p, int64_t n, uint8_t *out,
                                      uint32_t *run) {
    int64_t i = 0, m = 0, r = 0;
    while (i < n) {
        if (r == 0 && p[i] < 0x80) {
            if (n - i >= 8) {
                uint64_t x, low;
                memcpy(&x, p + i, 8);
                uint64_t bad = fse_wp_unprintable8(x);
                low = fse_wp_lower8(x);
                memcpy(out + m, &low, 8);
                if (!bad) {
                    i += 8;
                    m += 8;
                    continue;
                }
                int k = fse_wp_bytes_below(bad);
                i += k;
                m += k;
                if (p[i] >= 0x80) continue;
            }
            uint8_t b = p[i++];
            out[m] = fse_wp_ascii_map[b];
            m += fse_wp_ascii_keep[b];
            continue;
        }
        m = fse_wp_step(p, &i, n, out, run, &r, m);
        if (m < 0) return m;
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

/* The class of the code point at q[i] of normalized UTF-8 (FSE_BERT_SPACE, FSE_BERT_PUNCT
   or 0) and its length: an ASCII byte by its table, the rest by fse_bert_entry. */
static inline uint32_t fse_wp_class_at(const uint8_t *q, int64_t i, int64_t m, int *len) {
    if (q[i] < 0x80) {
        *len = 1;
        return fse_wp_ascii_class[q[i]];
    }
    uint32_t cp;
    *len = fse_wp_decode(q, i, m, &cp);
    return fse_bert_entry(cp) & (FSE_BERT_SPACE | FSE_BERT_PUNCT);
}

/* The high bit of each byte of x that is not a-z or 0-9 (exact in every byte: the bytes
   are compared as 7 bits, so no carry crosses them). */
static inline uint64_t fse_wp_not_alnum8(uint64_t x) {
    uint64_t y = x & ~FSE_WP_HIGH;
    uint64_t digit = (y + FSE_WP_ONES * (0x80 - '0')) & ~(y + FSE_WP_ONES * (0x80 - '9' - 1));
    uint64_t lower = (y + FSE_WP_ONES * (0x80 - 'a')) & ~(y + FSE_WP_ONES * (0x80 - 'z' - 1));
    return ~((digit | lower) & ~x) & FSE_WP_HIGH;
}

/* fse_wp_next_word with the class of ASCII bytes from a table: no decode and no two-stage
   lookup for them. Inside a word, at an ASCII byte, the bytes of a-z and 0-9 (word bytes,
   as module init checks) among the next 8 are skipped at once while at least 8 bytes are
   left. */
static inline int fse_wp_next_word_ascii(const uint8_t *q, int64_t m, int64_t *pos,
                                         int64_t *start, int64_t *end) {
    int64_t i = *pos;
    int len;
    while (i < m) {
        uint32_t c = fse_wp_class_at(q, i, m, &len);
        if (c & FSE_BERT_SPACE) {
            i += len;
            continue;
        }
        *start = i;
        i += len;
        if (!(c & FSE_BERT_PUNCT)) {
            while (i < m) {
                if (q[i] < 0x80) {
                    if (m - i >= 8) {
                        uint64_t x;
                        memcpy(&x, q + i, 8);
                        uint64_t other = fse_wp_not_alnum8(x);
                        if (!other) {
                            i += 8;
                            continue;
                        }
                        i += fse_wp_bytes_below(other);
                        if (q[i] >= 0x80) continue;
                    }
                    if (fse_wp_ascii_class[q[i]]) break;
                    i++;
                    continue;
                }
                if (fse_wp_class_at(q, i, m, &len)) break;
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

/* fse_wp_char_cut 8 bytes at a time: a block of 8 ASCII bytes (when at least 8 are left,
   by memcpy) is 8 code points, and holds the cut at i + (max_chars - chars) when it falls
   inside it; every other byte takes the decoder, so the whole input is still validated. */
static inline int64_t fse_wp_char_cut_ascii(const uint8_t *p, int64_t n, int64_t max_chars) {
    uint32_t cp;
    int64_t i = 0, chars = 0, cut = n;
    while (i < n) {
        if (n - i >= 8) {
            uint64_t x;
            memcpy(&x, p + i, 8);
            if (!(x & FSE_WP_HIGH)) {
                if (cut == n && max_chars - chars < 8) cut = i + (max_chars - chars);
                i += 8;
                chars += 8;
                continue;
            }
        }
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
