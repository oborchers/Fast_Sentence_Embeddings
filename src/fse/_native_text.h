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
static void fse_text_init_kinds(void);

static void fse_text_init(void) {
    for (unsigned b = 0; b < 128; b++) {
        unsigned c = b - 'A' < 26u ? b | 0x20u : b;
        if ((b >= 9 && b <= 13) || (b >= 28 && b <= 32)) c = FSE_CLASS_SPACE;
        if (strchr("!\"#$%&()*;<>?@[]`{}", (int)b) && b != 0) c = FSE_CLASS_SEPARATOR;
        fse_ascii_class[b] = (uint16_t)c;
    }
    fse_text_init_kinds();
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

/* The tokens of one sentence: token k is bytes[starts[k]:ends[k]]. bytes is set by the
   tokenizer: the lowercased candidates of fse_words@1, or the input for external. */
typedef struct {
    const uint8_t *bytes;
    int64_t *starts;
    int64_t *ends;
    int64_t n_tokens;
} fse_tokens;

static inline void fse_emit(fse_tokens *out, int64_t start, int64_t end) {
    out->starts[out->n_tokens] = start;
    out->ends[out->n_tokens++] = end;
}

static inline int fse_is_strip3(uint8_t b) { return b == '.' || b == ':' || b == '\''; }

/* Step 5c for the part c[s:e]: strip . : ' from both ends, drop it without a-z and 0-9,
   otherwise emit its pieces between the hyphens. */
static inline void fse_emit_part(fse_tokens *out, const uint8_t *c, int64_t s, int64_t e) {
    while (s < e && fse_is_strip3(c[s])) s++;
    while (e > s && fse_is_strip3(c[e - 1])) e--;
    int64_t i = s;
    while (i < e && !fse_is_alnum_byte(c[i])) i++;
    if (i == e) return;
    int64_t start = s;
    for (i = s; i < e; i++) {
        if (c[i] == '-') {
            if (i > start) fse_emit(out, start, i);
            start = i + 1;
        }
    }
    if (e > start) fse_emit(out, start, e);
}

/* The contraction table: the word, its length, the end of its first part and the start
   of its second part. */
static const struct { const char *word; int len, first, second; } fse_contractions[] = {
    {"cannot", 6, 3, 3}, {"gimme", 5, 3, 3}, {"gonna", 5, 3, 3}, {"gotta", 5, 3, 3},
    {"lemme", 5, 3, 3}, {"wanna", 5, 3, 3}, {"more'n", 6, 4, 5}, {"d'ye", 4, 1, 2},
    {"'tis", 4, 2, 2}, {"'twas", 5, 2, 2},
};
static const struct { const char *suffix; int len; } fse_clitics[] = {
    {"'s", 2}, {"'m", 2}, {"'d", 2}, {"'ll", 3}, {"'re", 3}, {"'ve", 3},
};

/* Step 5 for the candidate c[s:e]. It may rewrite "n't" to "not" in place. */
static void fse_emit_candidate(fse_tokens *out, uint8_t *c, int64_t s, int64_t e) {
    while (s < e && (c[s] == '.' || c[s] == ':')) s++;
    while (e > s && (c[e - 1] == '.' || c[e - 1] == ':')) e--;
    int64_t n = e - s;
    const uint8_t *u = c + s;
    if (n >= 4 && n <= 6 && (u[0] == 'c' || u[0] == 'g' || u[0] == 'l' || u[0] == 'w' ||
                             u[0] == 'm' || u[0] == 'd' || u[0] == '\'')) {
        for (size_t k = 0; k < sizeof fse_contractions / sizeof fse_contractions[0]; k++) {
            if (n == fse_contractions[k].len && u[0] == (uint8_t)fse_contractions[k].word[0] &&
                memcmp(u, fse_contractions[k].word, (size_t)n) == 0) {
                /* 'tis and 'twas: the first part is "'t", which stripping makes "t". */
                fse_emit_part(out, c, s, s + fse_contractions[k].first);
                fse_emit_part(out, c, s + fse_contractions[k].second, e);
                return;
            }
        }
    }
    if (n >= 3 && u[n - 1] == 't' && u[n - 2] == '\'' && u[n - 3] == 'n') {
        if (n == 3 || u[n - 4] != '\'') {
            c[e - 2] = 'o';  /* "n't" becomes "not" */
            if (n > 3) fse_emit_part(out, c, s, e - 3);
            fse_emit_part(out, c, e - 3, e);
            return;
        }
    }
    /* Every clitic starts with ' two or three bytes before the end. */
    if (n >= 3 && (u[n - 2] == '\'' || u[n - 3] == '\'')) {
        for (size_t k = 0; k < sizeof fse_clitics / sizeof fse_clitics[0]; k++) {
            int64_t m = fse_clitics[k].len;
            if (n > m && memcmp(u + n - m, fse_clitics[k].suffix, (size_t)m) == 0 && u[n - m - 1] != '\'') {
                fse_emit_part(out, c, s, e - m);
                fse_emit_part(out, c, e - m + 1, e);
                return;
            }
        }
    }
    fse_emit_part(out, c, s, e);
}

/* Kinds of ASCII bytes: an ordinary token character, whitespace or a separator, or one
   of the characters whose rule looks at their neighbors. */
#define FSE_KIND_TOKEN 0
#define FSE_KIND_BREAK 1
#define FSE_KIND_CONTEXT 2
static uint8_t fse_ascii_kind[128];

static void fse_text_init_kinds(void) {
    for (unsigned b = 0; b < 128; b++) {
        unsigned c = fse_ascii_class[b];
        fse_ascii_kind[b] = c == FSE_CLASS_SPACE || c == FSE_CLASS_SEPARATOR ? FSE_KIND_BREAK
                          : (c == '.' || c == '-' || c == ',' || c == ':' || c == '\'') ? FSE_KIND_CONTEXT
                          : FSE_KIND_TOKEN;
    }
}

/* The class of the code point at p[i], or FSE_CLASS_END at the end; 0 length marks
   invalid UTF-8. Lowercasing does not change classes of ASCII context characters. */
static inline unsigned fse_peek_class(const uint8_t *p, int64_t i, int64_t n, int *bad) {
    if (i >= n) return FSE_CLASS_END;
    if (p[i] < 0x80) return fse_ascii_class[p[i]];
    fse_cp cp;
    if (!fse_next_cp(p, i, n, &cp)) { *bad = 1; return FSE_CLASS_END; }
    return cp.cls;
}

/* Tokenize p[0:n] by fse_words@1 into out. cand must hold 2 * n + 8 bytes, out->starts
   and out->ends n + 1 entries each. Returns -1, or the offset of the first byte that is
   not valid UTF-8. */
static int64_t fse_tokenize_fse_words(const uint8_t *p, int64_t n, uint8_t *cand, fse_tokens *out) {
    out->bytes = cand;
    out->n_tokens = 0;
    int64_t m = 0, cs = 0;  /* the end of the candidates, the start of the current one */
    unsigned prev_cls = FSE_CLASS_END;
    int prev_t = 0;
    int64_t i = 0;
    while (i < n) {
        uint8_t b = p[i];
        unsigned c;
        int t;
        if (b < 0x80) {
            c = fse_ascii_class[b];
            uint8_t kind = fse_ascii_kind[b];
            if (kind == FSE_KIND_TOKEN) {
                cand[m++] = (uint8_t)c;
                prev_cls = c;
                prev_t = 1;
                i++;
                continue;
            }
            if (kind == FSE_KIND_BREAK) {
                t = 0;
            } else {
                int bad = 0;
                unsigned next = fse_peek_class(p, i + 1, n, &bad);
                if (bad) return i + 1;  /* the code point after the context character */
                if (c == '.' || c == '-') {
                    t = prev_cls != c && next != c;
                } else if (c == ',' || c == ':') {
                    t = next >= '0' && next <= '9';
                } else {
                    int doubled = prev_cls == '\'' || next == '\'';
                    int opening = prev_cls != FSE_CLASS_END && prev_t && !fse_is_word_class(prev_cls) &&
                                  next != FSE_CLASS_END && fse_is_word_class(next);
                    t = !(doubled || opening);
                }
                if (t) cand[m++] = (uint8_t)c;
            }
            i++;
        } else {
            fse_cp cp;
            int len = fse_next_cp(p, i, n, &cp);
            if (!len) return i;
            c = cp.cls;
            /* A word character, or an ASCII letter (U+212A KELVIN SIGN lowercases to k; no
               code point lowercases to an ASCII character with a context rule). */
            t = c == FSE_CLASS_WORD || c < 0x80;
            if (t) {
                memcpy(cand + m, cp.bytes, 4);  /* cand has 8 bytes of slack */
                m += cp.len;
            }
            i += len;
        }
        if (!t && m > cs) {
            fse_emit_candidate(out, cand, cs, m);
            cs = m;
        }
        prev_cls = c;
        prev_t = t;
    }
    if (m > cs) fse_emit_candidate(out, cand, cs, m);
    return -1;
}

/* Tokenize p[0:n] by external: split at whitespace W, bytes unchanged. */
static int64_t fse_tokenize_external(const uint8_t *p, int64_t n, fse_tokens *out) {
    out->bytes = p;
    out->n_tokens = 0;
    int64_t start = -1, i = 0;
    while (i < n) {
        uint8_t b = p[i];
        int space, len = 1;
        if (b < 0x80) {
            space = fse_ascii_class[b] == FSE_CLASS_SPACE;
        } else {
            uint32_t cp;
            len = fse_utf8_decode(p, i, n, &cp);
            if (!len) return i;
            space = fse_is_space_cp(cp);
        }
        if (space) {
            if (start >= 0) fse_emit(out, start, i);
            start = -1;
        } else if (start < 0) {
            start = i;
        }
        i += len;
    }
    if (start >= 0) fse_emit(out, start, n);
    return -1;
}

static inline int64_t fse_tokenize(int tokenizer, const uint8_t *p, int64_t n, uint8_t *cand,
                                   fse_tokens *out) {
    return tokenizer == FSE_TOKENIZER_FSE_WORDS ? fse_tokenize_fse_words(p, n, cand, out)
                                                : fse_tokenize_external(p, n, out);
}

#endif /* FSE_NATIVE_TEXT_H */
