/* unigram@1 (docs/architecture/tokenizers.md, "unigram@1"): the graphemes, the charsmap search and the
   float64 Viterbi; _native_unigram.pxi holds the state, the normalizer steps, the lookups
   of the candidates and the sentence around them. Correct first, not fast.

   The Viterbi is the only arithmetic of the tokenizer. It is binary64 without extended
   precision (checked below), runs in the floating-point environment of the kernels
   (round to nearest, set and restored per call by the wrapper) and only in functions
   marked never to be inlined, so no compiler moves it before the environment is set
   (docs/architecture/numerical-contract.md, "Native kernel"). fse._reference_unigram defines the result;
   tests/test_text_conformance_unigram.py compares. */
#ifndef FSE_NATIVE_UNIGRAM_H
#define FSE_NATIVE_UNIGRAM_H

#include <float.h>
#include <stdint.h>
#include <string.h>

#include "_native_added.h"
#include "_native_kernel_steps.h"
#include "_native_text.h"
#include "_native_unicode_graphemes.h"

#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0 || DBL_MANT_DIG != 53
#error "unigram@1 needs binary64 sums without extended precision (FLT_EVAL_METHOD 0)"
#endif

/* tokenizers' K_UNK_PENALTY: an unknown code point scores this below the lowest piece. */
#define FSE_UNI_UNK_PENALTY 10.0
/* A grapheme of fewer bytes is searched in the charsmap whole (normalizers/precompiled.rs). */
#define FSE_UNI_GRAPHEME_BYTES 6

/* --- Graphemes (UAX #29 of Unicode 17.0.0), as fse._reference_unigram.boundaries ------ */

static inline unsigned fse_gc_break(uint32_t cp) { return fse_gc_props(cp) & FSE_GC_BREAK_MASK; }

static inline unsigned fse_gc_incb(uint32_t cp) {
    return (unsigned)(fse_gc_props(cp) >> FSE_INCB_SHIFT) & 3u;
}

static inline int fse_gc_control(unsigned b) {
    return b == FSE_GC_CONTROL || b == FSE_GC_CR || b == FSE_GC_LF;
}

static inline int fse_gc_hangul(unsigned a, unsigned b) {
    if (a == FSE_GC_L) return b == FSE_GC_L || b == FSE_GC_V || b == FSE_GC_LV || b == FSE_GC_LVT;
    if (a == FSE_GC_LV || a == FSE_GC_V) return b == FSE_GC_V || b == FSE_GC_T;
    return (a == FSE_GC_LVT || a == FSE_GC_T) && b == FSE_GC_T;
}

/* GB9c: a consonant, then linkers and extends with at least one linker, before cps[i]. */
static inline int fse_gc_gb9c(const uint32_t *cps, int64_t i) {
    int64_t linkers = 0, j = i - 1;
    while (j >= 0) {
        unsigned c = fse_gc_incb(cps[j]);
        if (c != FSE_INCB_EXTEND && c != FSE_INCB_LINKER) break;
        linkers += c == FSE_INCB_LINKER;
        j--;
    }
    return linkers > 0 && j >= 0 && fse_gc_incb(cps[j]) == FSE_INCB_CONSONANT;
}

/* GB11: an Extended_Pictographic, extends, a ZWJ at i - 1, and one at i. */
static inline int fse_gc_gb11(const uint32_t *cps, int64_t i) {
    int64_t j = i - 2;
    while (j >= 0 && fse_gc_break(cps[j]) == FSE_GC_EXTEND) j--;
    return j >= 0 && (fse_gc_props(cps[j]) & FSE_GC_EXTENDED_PICTOGRAPHIC);
}

/* Whether no boundary lies before cps[i] (i > 0); regional is the number of regional
   indicators in a row that end at i - 1. */
static inline int fse_gc_joined(const uint32_t *cps, int64_t i, int64_t regional) {
    unsigned a = fse_gc_break(cps[i - 1]), b = fse_gc_break(cps[i]);
    if (a == FSE_GC_CR && b == FSE_GC_LF) return 1;
    if (fse_gc_control(a) || fse_gc_control(b)) return 0;
    if (fse_gc_hangul(a, b)) return 1;
    if (b == FSE_GC_EXTEND || b == FSE_GC_ZWJ || b == FSE_GC_SPACINGMARK || a == FSE_GC_PREPEND)
        return 1;
    if (fse_gc_incb(cps[i]) == FSE_INCB_CONSONANT && fse_gc_gb9c(cps, i)) return 1;
    if (a == FSE_GC_ZWJ && (fse_gc_props(cps[i]) & FSE_GC_EXTENDED_PICTOGRAPHIC) &&
        fse_gc_gb11(cps, i))
        return 1;
    return a == FSE_GC_REGIONAL_INDICATOR && b == FSE_GC_REGIONAL_INDICATOR && regional % 2 == 1;
}

/* The end (exclusive) of the grapheme that starts at cps[i] of cps[0:n]. A grapheme
   starts after a boundary, where a run of regional indicators has an even length, so
   counting them from i gives the parity of the whole run. */
static inline int64_t fse_gc_next(const uint32_t *cps, int64_t i, int64_t n) {
    int64_t j = i + 1, regional = 0;
    while (j < n) {
        regional = fse_gc_break(cps[j - 1]) == FSE_GC_REGIONAL_INDICATOR ? regional + 1 : 0;
        if (!fse_gc_joined(cps, j, regional)) break;
        j++;
    }
    return j;
}

/* --- The charsmap (spm_precompiled 0.1.4), as fse._charsmap.Charsmap.find ------------ */

/* The value of the shortest key that is a prefix of key[0:len] (the search ends at a
   byte 0): 1 and its offset in *value, else 0. Every probe is bounds checked, so even a
   blob that fse._charsmap did not check cannot make it read outside the units. */
static inline int fse_cm_find(const uint32_t *units, int64_t n_units, int64_t root,
                              const uint8_t *key, int64_t len, uint32_t *value) {
    int64_t p = root;
    for (int64_t k = 0; k < len; k++) {
        uint8_t c = key[k];
        if (!c) return 0;
        p ^= c;
        if (p < 0 || p >= n_units) return 0;
        uint32_t unit = units[p];
        if ((unit & ((1u << 31) | 0xFFu)) != c) return 0;
        p ^= (int64_t)((unit >> 10) << ((unit & (1u << 9)) >> 6));
        if ((unit >> 8) & 1u) {
            if (p < 0 || p >= n_units) return 0;
            *value = units[p] & ((1u << 31) - 1u);
            return 1;
        }
    }
    return 0;
}

/* --- The float64 Viterbi -------------------------------------------------------------- */

/* A candidate node: the piece (or the unknown code point) from a start to end, its id
   and score. */
typedef struct {
    int64_t end;
    double score;
    uint32_t id;
} fse_uni_cand;

/* The lowest score minus K_UNK_PENALTY (n >= 1). */
FSE_NOINLINE static double fse_uni_unk_score_body(const double *scores, int64_t n) {
    double low = scores[0];
    for (int64_t i = 1; i < n; i++)
        if (scores[i] < low) low = scores[i];
    return low - FSE_UNI_UNK_PENALTY;
}

static double fse_uni_unk_score(const double *scores, int64_t n) {
    fse_fpenv env = fse_fpenv_enter();
    double out = fse_uni_unk_score_body(scores, n);
    fse_fpenv_leave(env);
    return out;
}

/* The best path ends: the candidates of the k-th start pos[k] (k < count, increasing)
   are cands[first[k]:first[k + 1]], each with start < end <= size; best_start[e] is -1
   until an end is reached. An offer replaces only a strictly lower score, so the earliest
   start wins a tie, as tokenizers' encode_optimized does. Returns -1, or the first start
   that is not reached (an error of the caller). */
FSE_NOINLINE static int64_t fse_uni_viterbi_body(int64_t size, const int64_t *pos,
                                                 const int64_t *first, int64_t count,
                                                 const fse_uni_cand *cands, double *best,
                                                 int64_t *best_start, uint32_t *best_id) {
    best[0] = 0.0;
    best_start[0] = 0;
    for (int64_t e = 1; e <= size; e++) best_start[e] = -1;
    for (int64_t k = 0; k < count; k++) {
        int64_t s = pos[k];
        if (best_start[s] < 0) return s;
        double here = best[s];
        for (int64_t c = first[k]; c < first[k + 1]; c++) {
            int64_t e = cands[c].end;
            double offer = cands[c].score + here;
            if (best_start[e] < 0 || offer > best[e]) {
                best[e] = offer;
                best_start[e] = s;
                best_id[e] = cands[c].id;
            }
        }
    }
    return -1;
}

static int64_t fse_uni_viterbi(int64_t size, const int64_t *pos, const int64_t *first,
                               int64_t count, const fse_uni_cand *cands, double *best,
                               int64_t *best_start, uint32_t *best_id) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t out = fse_uni_viterbi_body(size, pos, first, count, cands, best, best_start, best_id);
    fse_fpenv_leave(env);
    return out;
}

#endif /* FSE_NATIVE_UNIGRAM_H */
