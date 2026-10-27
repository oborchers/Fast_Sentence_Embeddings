/* The native tokenizers of fse (ARCHITECTURE.md, "Tokenizers").

   fse_tokenize splits one sentence of UTF-8 bytes into the bytes of its tokens: one pass
   over the code points with one code point of lookahead, which decides whether each is
   a token character (steps 1 to 3), collects runs of them as candidates (step 4) and
   splits every candidate into pieces (step 5). Every rule of step 5 refers to ASCII
   bytes only, so it works on the UTF-8 bytes of the lowercased candidate.

   fse._reference_text defines the result; tests/test_text_conformance.py compares. */
#ifndef FSE_NATIVE_TEXT_H
#define FSE_NATIVE_TEXT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "_native_unicode.h"

#define FSE_TOKENIZER_EXTERNAL 0
#define FSE_TOKENIZER_FSE_WORDS 1

/* Classes of a lowercased code point: an ASCII token character is its own byte value;
   whitespace, separators and other code points (all word characters) get the codes
   below, which no ASCII token character has. */
#define FSE_CLASS_WORD 0x80u  /* a code point from U+0080 that is neither W nor S */
#define FSE_CLASS_SPACE 0xFEu
#define FSE_CLASS_SEPARATOR 0xFFu
#define FSE_CLASS_END 0x100u  /* past the end of the sentence */

/* The class of every ASCII byte after lowercasing. */
static uint16_t fse_ascii_class[128];

static void fse_text_init(void) {
    for (unsigned b = 0; b < 128; b++) {
        unsigned c = b - 'A' < 26u ? b | 0x20u : b;
        if ((b >= 9 && b <= 13) || (b >= 28 && b <= 32)) c = FSE_CLASS_SPACE;
        if (strchr("!\"#$%&()*;<>?@[]`{}", (int)b) && b != 0) c = FSE_CLASS_SEPARATOR;
        fse_ascii_class[b] = (uint16_t)c;
    }
}

/* Whether a non-ASCII code point is whitespace W or a separator S. */
static inline int fse_is_space_cp(uint32_t cp) {
    return cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
static inline int fse_is_separator_cp(uint32_t cp) {
    return cp == 0xAB || cp == 0xBB || (cp >= 0x2012 && cp <= 0x2015) || cp == 0x2018 ||
           cp == 0x2019 || cp == 0x201C || cp == 0x201D || cp == 0x201E;
}

/* Length of the valid UTF-8 sequence at p[i] (as CPython's decoder accepts it) with its
   code point in *cp, or 0 if the bytes there are invalid. */
static inline int fse_utf8_decode(const uint8_t *p, int64_t i, int64_t end, uint32_t *cp) {
    uint8_t b = p[i];
    if (b < 0x80) { *cp = b; return 1; }
    if (b < 0xC2) return 0;
    if (b < 0xE0) {
        if (end - i < 2 || (p[i + 1] & 0xC0) != 0x80) return 0;
        *cp = ((uint32_t)(b & 0x1F) << 6) | (p[i + 1] & 0x3F);
        return 2;
    }
    if (b < 0xF0) {
        if (end - i < 3) return 0;
        uint8_t c = p[i + 1];
        if ((b == 0xE0 && c < 0xA0) || (b == 0xED && c > 0x9F)) return 0;
        if ((c & 0xC0) != 0x80 || (p[i + 2] & 0xC0) != 0x80) return 0;
        *cp = ((uint32_t)(b & 0x0F) << 12) | ((uint32_t)(c & 0x3F) << 6) | (p[i + 2] & 0x3F);
        return 3;
    }
    if (b < 0xF5) {
        if (end - i < 4) return 0;
        uint8_t c = p[i + 1];
        if ((b == 0xF0 && c < 0x90) || (b == 0xF4 && c > 0x8F)) return 0;
        if ((c & 0xC0) != 0x80 || (p[i + 2] & 0xC0) != 0x80 || (p[i + 3] & 0xC0) != 0x80) return 0;
        *cp = ((uint32_t)(b & 0x07) << 18) | ((uint32_t)(c & 0x3F) << 12) |
              ((uint32_t)(p[i + 2] & 0x3F) << 6) | (p[i + 3] & 0x3F);
        return 4;
    }
    return 0;
}

static inline int fse_utf8_encode(uint32_t cp, uint8_t *out) {
    if (cp < 0x80) { out[0] = (uint8_t)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (uint8_t)(0xC0 | (cp >> 6));
        out[1] = (uint8_t)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (uint8_t)(0xE0 | (cp >> 12));
        out[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (uint8_t)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (uint8_t)(0xF0 | (cp >> 18));
    out[1] = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (uint8_t)(0x80 | (cp & 0x3F));
    return 4;
}

/* One lowercased code point: its class, and its UTF-8 bytes. */
typedef struct {
    uint16_t cls;
    uint8_t len;
    uint8_t bytes[4];
} fse_cp;

/* Decode and lowercase the code point at p[i] into *out; returns its input length, or 0
   for invalid UTF-8. */
static inline int fse_next_cp(const uint8_t *p, int64_t i, int64_t end, fse_cp *out) {
    uint8_t b = p[i];
    if (b < 0x80) {
        uint16_t c = fse_ascii_class[b];
        out->cls = c;
        out->len = 1;
        out->bytes[0] = c < 0x80 ? (uint8_t)c : b;
        return 1;
    }
    uint32_t cp;
    int n = fse_utf8_decode(p, i, end, &cp);
    if (!n) return 0;
    cp = fse_simple_lower(cp);
    if (cp < 0x80) {  /* U+212A KELVIN SIGN to k */
        out->cls = fse_ascii_class[cp];
        out->len = 1;
        out->bytes[0] = (uint8_t)cp;
        return n;
    }
    out->cls = fse_is_space_cp(cp) ? FSE_CLASS_SPACE
             : fse_is_separator_cp(cp) ? FSE_CLASS_SEPARATOR : FSE_CLASS_WORD;
    out->len = (uint8_t)fse_utf8_encode(cp, out->bytes);
    return n;
}

static inline int fse_is_word_class(unsigned c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == FSE_CLASS_WORD;
}
static inline int fse_is_alnum_byte(uint8_t b) {
    return (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9');
}

/* The output of one sentence: the bytes of its tokens one after another, and the end
   offset of every token. */
typedef struct {
    uint8_t *bytes;
    int64_t *ends;
    int64_t n_bytes;
    int64_t n_tokens;
} fse_tokens;

static inline void fse_emit(fse_tokens *out, const uint8_t *p, int64_t n) {
    memcpy(out->bytes + out->n_bytes, p, (size_t)n);
    out->n_bytes += n;
    out->ends[out->n_tokens++] = out->n_bytes;
}

/* Step 5c for one part: strip . : ' from both ends, drop it without a-z and 0-9,
   otherwise emit its pieces between the hyphens. */
static inline void fse_emit_part(fse_tokens *out, const uint8_t *p, int64_t n) {
    int64_t s = 0, e = n;
    while (s < e && (p[s] == '.' || p[s] == ':' || p[s] == '\'')) s++;
    while (e > s && (p[e - 1] == '.' || p[e - 1] == ':' || p[e - 1] == '\'')) e--;
    int alnum = 0;
    for (int64_t i = s; i < e; i++)
        if (fse_is_alnum_byte(p[i])) { alnum = 1; break; }
    if (!alnum) return;
    int64_t start = s;
    for (int64_t i = s; i <= e; i++) {
        if (i == e || p[i] == '-') {
            if (i > start) fse_emit(out, p + start, i - start);
            start = i + 1;
        }
    }
}

static inline int fse_equals(const uint8_t *p, int64_t n, const char *word) {
    size_t m = strlen(word);
    return (size_t)n == m && memcmp(p, word, m) == 0;
}
static inline int fse_ends_with(const uint8_t *p, int64_t n, const char *suffix) {
    size_t m = strlen(suffix);
    return (size_t)n >= m && memcmp(p + n - (int64_t)m, suffix, m) == 0;
}

/* The contraction table: (word, length of its first part, offset of its second part). */
static const struct { const char *word; int first; int second; } fse_contractions[] = {
    {"cannot", 3, 3}, {"gimme", 3, 3}, {"gonna", 3, 3}, {"gotta", 3, 3}, {"lemme", 3, 3},
    {"wanna", 3, 3}, {"more'n", 4, 5}, {"d'ye", 1, 2}, {"'tis", 2, 2}, {"'twas", 2, 2},
};
static const char *fse_clitics[] = {"'s", "'m", "'d", "'ll", "'re", "'ve"};

/* Step 5 for one candidate u = p[0:n]. */
static void fse_emit_candidate(fse_tokens *out, const uint8_t *p, int64_t n) {
    int64_t s = 0, e = n;
    while (s < e && (p[s] == '.' || p[s] == ':')) s++;
    while (e > s && (p[e - 1] == '.' || p[e - 1] == ':')) e--;
    p += s;
    n = e - s;
    if (n <= 6) {
        for (size_t k = 0; k < sizeof fse_contractions / sizeof fse_contractions[0]; k++) {
            if (fse_equals(p, n, fse_contractions[k].word)) {
                /* 'tis and 'twas: the first part is "'t", which stripping makes "t". */
                fse_emit_part(out, p, fse_contractions[k].first);
                fse_emit_part(out, p + fse_contractions[k].second, n - fse_contractions[k].second);
                return;
            }
        }
    }
    if (fse_equals(p, n, "n't")) {
        fse_emit_part(out, (const uint8_t *)"not", 3);
        return;
    }
    if (n > 3 && fse_ends_with(p, n, "n't") && p[n - 4] != '\'') {
        fse_emit_part(out, p, n - 3);
        fse_emit_part(out, (const uint8_t *)"not", 3);
        return;
    }
    for (size_t k = 0; k < sizeof fse_clitics / sizeof fse_clitics[0]; k++) {
        int64_t m = (int64_t)strlen(fse_clitics[k]);
        if (n > m && fse_ends_with(p, n, fse_clitics[k]) && p[n - m - 1] != '\'') {
            fse_emit_part(out, p, n - m);
            fse_emit_part(out, p + n - m + 1, m - 1);
            return;
        }
    }
    fse_emit_part(out, p, n);
}

/* Tokenize p[0:n] by fse_words@1 into out. cand must hold 2 * n + 8 bytes, out->bytes
   2 * n + 8 bytes and out->ends n + 1 entries. Returns -1, or the offset of the first
   byte that is not valid UTF-8 (out then holds the tokens before it). */
static int64_t fse_tokenize_fse_words(const uint8_t *p, int64_t n, uint8_t *cand, fse_tokens *out) {
    out->n_bytes = 0;
    out->n_tokens = 0;
    int64_t m = 0;  /* bytes of the current candidate */
    fse_cp cur, next;
    unsigned prev_cls = FSE_CLASS_END;
    int prev_t = 0;
    int64_t i = 0;
    if (n == 0) return -1;
    int len = fse_next_cp(p, 0, n, &cur);
    if (!len) return 0;
    while (1) {
        int64_t j = i + len;
        int next_len = 0;
        unsigned next_cls = FSE_CLASS_END;
        if (j < n) {
            next_len = fse_next_cp(p, j, n, &next);
            if (!next_len) {
                /* The tokens before the bad byte are not complete; the caller raises. */
                return j;
            }
            next_cls = next.cls;
        }
        unsigned c = cur.cls;
        int t;
        if (c == FSE_CLASS_SPACE || c == FSE_CLASS_SEPARATOR) {
            t = 0;
        } else if (c == '.' || c == '-') {
            t = prev_cls != c && next_cls != c;
        } else if (c == ',' || c == ':') {
            t = next_cls >= '0' && next_cls <= '9';
        } else if (c == '\'') {
            int doubled = prev_cls == '\'' || next_cls == '\'';
            int opening = prev_cls != FSE_CLASS_END && prev_t && !fse_is_word_class(prev_cls) &&
                          next_cls != FSE_CLASS_END && fse_is_word_class(next_cls);
            t = !(doubled || opening);
        } else {
            t = 1;
        }
        if (t) {
            memcpy(cand + m, cur.bytes, cur.len);
            m += cur.len;
        } else if (m) {
            fse_emit_candidate(out, cand, m);
            m = 0;
        }
        if (j >= n) break;
        prev_cls = c;
        prev_t = t;
        cur = next;
        len = next_len;
        i = j;
    }
    if (m) fse_emit_candidate(out, cand, m);
    return -1;
}

/* Tokenize p[0:n] by external: split at whitespace W, bytes unchanged. Same buffers and
   result as fse_tokenize_fse_words. */
static int64_t fse_tokenize_external(const uint8_t *p, int64_t n, fse_tokens *out) {
    out->n_bytes = 0;
    out->n_tokens = 0;
    int64_t start = -1, i = 0;
    while (i < n) {
        uint32_t cp;
        int len = fse_utf8_decode(p, i, n, &cp);
        if (!len) return i;
        int space = cp < 0x80 ? fse_ascii_class[cp] == FSE_CLASS_SPACE : fse_is_space_cp(cp);
        if (space) {
            if (start >= 0) fse_emit(out, p + start, i - start);
            start = -1;
        } else if (start < 0) {
            start = i;
        }
        i += len;
    }
    if (start >= 0) fse_emit(out, p + start, n - start);
    return -1;
}

static inline int64_t fse_tokenize(int tokenizer, const uint8_t *p, int64_t n, uint8_t *cand,
                                   fse_tokens *out) {
    return tokenizer == FSE_TOKENIZER_FSE_WORDS ? fse_tokenize_fse_words(p, n, cand, out)
                                                : fse_tokenize_external(p, n, out);
}

#endif /* FSE_NATIVE_TEXT_H */
