/* The register-blocked batch loop of weighted_mean@1: NEON float32. Included by
   _native_kernel_mean.h after the row kernels and loops.

   A sentence is cut into tiles of at most K terms. A term is one contribution, a row and
   its factor f: a known word gives its row with f = weight (times the scale for int8),
   an unknown FastText word gives its bucket rows in order, each with f = g (times the
   bucket's scale), g = (1/m) * oov_weight, and an unknown word without buckets gives no
   term. For every block of the dimension the accumulators of the block stay in
   registers over all terms of the tile: +0.0 in the first tile, the partial sums the
   previous tile stored in the out row otherwise, then a_i = fma(f, x[i], a_i) for every
   term in order, then a store. Per value this is the sequence of fused operations of the
   row loop (and of fse._reference): the same factors, the same order, the same start;
   only the place of a partial sum between two terms changes, a register within a tile
   and the out row between tiles, and both keep its bits, -0.0 included.

   The builder (fse_fill_tile) keeps these invariants, which the conformance tests and
   their meta-tests (tests/test_conformance_tiles*.py) check:
   - every contribution of the sentence is a term exactly once, in token order, and the
     buckets of an unknown word in bucket order, also when they straddle tiles;
   - k counts the unknown words of the batch, each exactly once, also when it has no
     buckets; j, g1 and g of an unknown word are kept across a tile boundary, so g is
     computed once per word from its full bucket count m;
   - an unknown word without buckets counts for the length n of the sentence (the
     divisor is n = end - begin, the number of tokens, never the number of terms) and
     adds nothing;
   - it prefetches what the row loop prefetches, in the same order: the row of the token
     FSE_PREFETCH_DISTANCE tokens ahead (also across sentences, nothing for an unknown
     token), and inside an unknown word the bucket FSE_PREFETCH_DISTANCE buckets ahead.
     A token's prefetch is issued when the builder reaches the token, so it leads its
     use by the distance plus the rest of the tile.

   The tile rule: K = FSE_TILE_BYTES / row_bytes, at most FSE_TILE_MAX, so the rows of
   one tile stay in the L1 cache while the blocks walk through them. Rows above
   FSE_TILE_BYTES / FSE_TILE_MIN bytes (more than 1024 float32 values) give K below
   FSE_TILE_MIN; they take the row loop, which then loses little to loads and stores of
   the accumulator. No load reads past a row: the blocks are 32 values, then one of 16,
   8 and 4 each at most, then single values. */

#ifndef FSE_NATIVE_KERNEL_TILES_H
#define FSE_NATIVE_KERNEL_TILES_H

#include "_native_kernel_base.h"

#define FSE_TILE_BYTES 16384
#define FSE_TILE_MAX 64
#define FSE_TILE_MIN 4

typedef struct {
    int64_t t, end; /* the next token of the sentence, and its end */
    int64_t j, g1;  /* the next bucket of the current unknown word, and its end */
    int64_t k;      /* the next unknown word of the batch */
    float g;        /* the factor of the buckets of the current unknown word */
} fse_cursor;

/* Append the next terms of the sentence, at most cap; fewer only at its end. */
FSE_INLINE int fse_fill_tile(const fse_params *p, const fse_batch *b, int64_t total,
                             fse_cursor *c, int cap, int scaled, const char **row, float *f) {
    const size_t rb = p->row_bytes;
    const char *rows = (const char *)p->rows;
    const char *ngram_rows = (const char *)p->ngram_rows;
    int n = 0;
    while (n < cap) {
        if (c->j < c->g1) {
            const int64_t j = c->j++;
            if (j + FSE_PREFETCH_DISTANCE < c->g1)
                fse_prefetch_row(ngram_rows + (size_t)b->ngrams[j + FSE_PREFETCH_DISTANCE] * rb, rb);
            const uint32_t bucket = b->ngrams[j];
            row[n] = ngram_rows + (size_t)bucket * rb;
            f[n++] = scaled ? c->g * p->ngram_scales[bucket] : c->g;
            continue;
        }
        if (c->t >= c->end) break;
        const int64_t t = c->t++;
        if (t + FSE_PREFETCH_DISTANCE < total) {
            const uint32_t ahead = b->tokens[t + FSE_PREFETCH_DISTANCE];
            if (ahead != FSE_OOV_TOKEN) fse_prefetch_row(rows + (size_t)ahead * rb, rb);
        }
        const uint32_t w = b->tokens[t];
        if (w == FSE_OOV_TOKEN) {
            const int64_t g0 = b->ngram_offsets[c->k], g1 = b->ngram_offsets[c->k + 1];
            c->k++;
            const int64_t m = g1 - g0;
            if (m == 0) continue; /* counts for the length, adds nothing */
            c->g = (1.0f / (float)m) * p->oov_weight;
            c->j = g0;
            c->g1 = g1;
            continue;
        }
        row[n] = rows + (size_t)w * rb;
        f[n++] = scaled ? p->weights[w] * p->scales[w] : p->weights[w];
    }
    return n;
}

#if defined(FSE_NEON)
/* The accumulators of V vectors (4 values each) at acc + i: +0.0 or loaded. */
#define FSE_NEON_START(a, q) float32x4_t a = first ? zero : vld1q_f32(acc + i + 4 * (q))
#define FSE_NEON_FMA(a, q) a = vfmaq_f32(a, fv, vld1q_f32(x + 4 * (q)))

/* One tile of n terms on the out row acc of dim values. */
FSE_INLINE void fse_tile_f32_neon(float *FSE_RESTRICT acc, int dim, const char *const *row,
                                  const float *f, int n, int first) {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 32 <= dim; i += 32) {
        FSE_NEON_START(a0, 0); FSE_NEON_START(a1, 1); FSE_NEON_START(a2, 2); FSE_NEON_START(a3, 3);
        FSE_NEON_START(a4, 4); FSE_NEON_START(a5, 5); FSE_NEON_START(a6, 6); FSE_NEON_START(a7, 7);
        for (int q = 0; q < n; q++) {
            const float *x = (const float *)row[q] + i;
            const float32x4_t fv = vdupq_n_f32(f[q]);
            FSE_NEON_FMA(a0, 0); FSE_NEON_FMA(a1, 1); FSE_NEON_FMA(a2, 2); FSE_NEON_FMA(a3, 3);
            FSE_NEON_FMA(a4, 4); FSE_NEON_FMA(a5, 5); FSE_NEON_FMA(a6, 6); FSE_NEON_FMA(a7, 7);
        }
        vst1q_f32(acc + i, a0); vst1q_f32(acc + i + 4, a1);
        vst1q_f32(acc + i + 8, a2); vst1q_f32(acc + i + 12, a3);
        vst1q_f32(acc + i + 16, a4); vst1q_f32(acc + i + 20, a5);
        vst1q_f32(acc + i + 24, a6); vst1q_f32(acc + i + 28, a7);
    }
    if (i + 16 <= dim) {
        FSE_NEON_START(a0, 0); FSE_NEON_START(a1, 1); FSE_NEON_START(a2, 2); FSE_NEON_START(a3, 3);
        for (int q = 0; q < n; q++) {
            const float *x = (const float *)row[q] + i;
            const float32x4_t fv = vdupq_n_f32(f[q]);
            FSE_NEON_FMA(a0, 0); FSE_NEON_FMA(a1, 1); FSE_NEON_FMA(a2, 2); FSE_NEON_FMA(a3, 3);
        }
        vst1q_f32(acc + i, a0); vst1q_f32(acc + i + 4, a1);
        vst1q_f32(acc + i + 8, a2); vst1q_f32(acc + i + 12, a3);
        i += 16;
    }
    if (i + 8 <= dim) {
        FSE_NEON_START(a0, 0); FSE_NEON_START(a1, 1);
        for (int q = 0; q < n; q++) {
            const float *x = (const float *)row[q] + i;
            const float32x4_t fv = vdupq_n_f32(f[q]);
            FSE_NEON_FMA(a0, 0); FSE_NEON_FMA(a1, 1);
        }
        vst1q_f32(acc + i, a0); vst1q_f32(acc + i + 4, a1);
        i += 8;
    }
    if (i + 4 <= dim) {
        FSE_NEON_START(a0, 0);
        for (int q = 0; q < n; q++) {
            const float *x = (const float *)row[q] + i;
            const float32x4_t fv = vdupq_n_f32(f[q]);
            FSE_NEON_FMA(a0, 0);
        }
        vst1q_f32(acc + i, a0);
        i += 4;
    }
    for (; i < dim; i++) {
        float a = first ? 0.0f : acc[i];
        for (int q = 0; q < n; q++) a = fmaf(f[q], ((const float *)row[q])[i], a); /* fmadd */
        acc[i] = a;
    }
}
#undef FSE_NEON_START
#undef FSE_NEON_FMA

/* The batch loop of FSE_DEFINE_LOOP (same arguments, same results) with tiles. */
FSE_NOINLINE static int64_t fse_loop_f32_neon(const fse_params *p, const fse_batch *b,
                                              float *out, int64_t *counts, int finish) {
    const int dim = p->dim;
    const size_t cap = FSE_TILE_BYTES / p->row_bytes;
    if (cap < FSE_TILE_MIN) return fse_loop_f32_neon_rows(p, b, out, counts, finish);
    const int K = cap < FSE_TILE_MAX ? (int)cap : FSE_TILE_MAX;
    const char *row[FSE_TILE_MAX];
    float f[FSE_TILE_MAX];
    const int64_t total = b->offsets[b->n_sentences];
    fse_cursor c = {0, 0, 0, 0, 0, 0.0f};
    for (int64_t s = 0; s < b->n_sentences; s++) {
        float *acc = out + (size_t)s * (size_t)dim;
        const int64_t begin = b->offsets[s], end = b->offsets[s + 1];
        c.t = begin;
        c.end = end;
        /* The first tile always runs, so a sentence without terms stores +0.0. */
        int first = 1;
        do {
            const int n = fse_fill_tile(p, b, total, &c, K, 0, row, f);
            if (n || first) fse_tile_f32_neon(acc, dim, row, f, n, first);
            first = 0;
        } while (c.t < c.end || c.j < c.g1);
        const int64_t n = end - begin;
        if (counts) counts[s] = n;
        if (fse_finish_row(acc, dim, finish && n ? 1.0f / (float)n : 1.0f, finish && n))
            return s;
    }
    return -1;
}
#endif

#endif /* FSE_NATIVE_KERNEL_TILES_H */
