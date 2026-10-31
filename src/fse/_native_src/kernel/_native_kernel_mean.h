/* The row kernels and the batch loop of weighted_mean@1, once per instruction set and
   dtype. Included by _native_kernel.h. */

#ifndef FSE_NATIVE_KERNEL_MEAN_H
#define FSE_NATIVE_KERNEL_MEAN_H

#include "_native_kernel_base.h"

/* --- Row kernels: acc[i] = fma(f, x[i], acc[i]) ------------------------------------- */

FSE_INLINE void fse_fma_f32_portable(float *FSE_RESTRICT acc, const void *row, float f, int n) {
    const float *x = (const float *)row;
    for (int i = 0; i < n; i++) acc[i] = fmaf(f, x[i], acc[i]);
}
FSE_INLINE void fse_fma_f16_portable(float *FSE_RESTRICT acc, const void *row, float f, int n) {
    const uint16_t *x = (const uint16_t *)row;
    for (int i = 0; i < n; i++) acc[i] = fmaf(f, fse_half_table[x[i]], acc[i]);
}
FSE_INLINE void fse_fma_i8_portable(float *FSE_RESTRICT acc, const void *row, float f, int n) {
    const int8_t *x = (const int8_t *)row;
    for (int i = 0; i < n; i++) acc[i] = fmaf(f, (float)x[i], acc[i]);
}

#if defined(FSE_X86)
FSE_AVX2_TARGET static inline float fse_fma1_avx2(float f, float x, float a) {
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(f), _mm_set_ss(x), _mm_set_ss(a)));
}
FSE_AVX2_TARGET static inline void fse_fma_f32_avx2(float *acc, const void *row, float f, int n) {
    const float *x = (const float *)row;
    __m256 fv = _mm256_set1_ps(f);
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256 a0 = _mm256_fmadd_ps(fv, _mm256_loadu_ps(x + i), _mm256_loadu_ps(acc + i));
        __m256 a1 = _mm256_fmadd_ps(fv, _mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(acc + i + 8));
        __m256 a2 = _mm256_fmadd_ps(fv, _mm256_loadu_ps(x + i + 16), _mm256_loadu_ps(acc + i + 16));
        __m256 a3 = _mm256_fmadd_ps(fv, _mm256_loadu_ps(x + i + 24), _mm256_loadu_ps(acc + i + 24));
        _mm256_storeu_ps(acc + i, a0);
        _mm256_storeu_ps(acc + i + 8, a1);
        _mm256_storeu_ps(acc + i + 16, a2);
        _mm256_storeu_ps(acc + i + 24, a3);
    }
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(acc + i, _mm256_fmadd_ps(fv, _mm256_loadu_ps(x + i), _mm256_loadu_ps(acc + i)));
    for (; i < n; i++) acc[i] = fse_fma1_avx2(f, x[i], acc[i]);
}
FSE_AVX2_TARGET static inline void fse_fma_f16_avx2(float *acc, const void *row, float f, int n) {
    const uint16_t *x = (const uint16_t *)row;
    __m256 fv = _mm256_set1_ps(f);
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256 a0 = _mm256_fmadd_ps(fv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i))), _mm256_loadu_ps(acc + i));
        __m256 a1 = _mm256_fmadd_ps(fv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i + 8))), _mm256_loadu_ps(acc + i + 8));
        __m256 a2 = _mm256_fmadd_ps(fv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i + 16))), _mm256_loadu_ps(acc + i + 16));
        __m256 a3 = _mm256_fmadd_ps(fv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i + 24))), _mm256_loadu_ps(acc + i + 24));
        _mm256_storeu_ps(acc + i, a0);
        _mm256_storeu_ps(acc + i + 8, a1);
        _mm256_storeu_ps(acc + i + 16, a2);
        _mm256_storeu_ps(acc + i + 24, a3);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i)));
        _mm256_storeu_ps(acc + i, _mm256_fmadd_ps(fv, v, _mm256_loadu_ps(acc + i)));
    }
    for (; i < n; i++) acc[i] = fse_fma1_avx2(f, fse_half_table[x[i]], acc[i]);
}
FSE_AVX2_TARGET static inline void fse_fma_i8_avx2(float *acc, const void *row, float f, int n) {
    const int8_t *x = (const int8_t *)row;
    __m256 fv = _mm256_set1_ps(f);
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i c = _mm256_loadu_si256((const __m256i *)(x + i));
        __m128i lo = _mm256_castsi256_si128(c), hi = _mm256_extracti128_si256(c, 1);
        __m256 v0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
        __m256 v1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
        __m256 v2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
        __m256 v3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
        __m256 a0 = _mm256_fmadd_ps(fv, v0, _mm256_loadu_ps(acc + i));
        __m256 a1 = _mm256_fmadd_ps(fv, v1, _mm256_loadu_ps(acc + i + 8));
        __m256 a2 = _mm256_fmadd_ps(fv, v2, _mm256_loadu_ps(acc + i + 16));
        __m256 a3 = _mm256_fmadd_ps(fv, v3, _mm256_loadu_ps(acc + i + 24));
        _mm256_storeu_ps(acc + i, a0);
        _mm256_storeu_ps(acc + i + 8, a1);
        _mm256_storeu_ps(acc + i + 16, a2);
        _mm256_storeu_ps(acc + i + 24, a3);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(x + i))));
        _mm256_storeu_ps(acc + i, _mm256_fmadd_ps(fv, v, _mm256_loadu_ps(acc + i)));
    }
    for (; i < n; i++) acc[i] = fse_fma1_avx2(f, (float)x[i], acc[i]);
}
#endif

#if defined(FSE_NEON)
static inline void fse_fma_f32_neon(float *acc, const void *row, float f, int n) {
    const float *x = (const float *)row;
    float32x4_t fv = vdupq_n_f32(f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t a0 = vfmaq_f32(vld1q_f32(acc + i), fv, vld1q_f32(x + i));
        float32x4_t a1 = vfmaq_f32(vld1q_f32(acc + i + 4), fv, vld1q_f32(x + i + 4));
        float32x4_t a2 = vfmaq_f32(vld1q_f32(acc + i + 8), fv, vld1q_f32(x + i + 8));
        float32x4_t a3 = vfmaq_f32(vld1q_f32(acc + i + 12), fv, vld1q_f32(x + i + 12));
        vst1q_f32(acc + i, a0);
        vst1q_f32(acc + i + 4, a1);
        vst1q_f32(acc + i + 8, a2);
        vst1q_f32(acc + i + 12, a3);
    }
    for (; i + 4 <= n; i += 4) vst1q_f32(acc + i, vfmaq_f32(vld1q_f32(acc + i), fv, vld1q_f32(x + i)));
    for (; i < n; i++) acc[i] = fmaf(f, x[i], acc[i]);  /* fmadd on aarch64 */
}
static inline void fse_fma_f16_neon(float *acc, const void *row, float f, int n) {
    const uint16_t *x = (const uint16_t *)row;
    float32x4_t fv = vdupq_n_f32(f);
    int i = 0;
    /* 16 values per step, as for float32: one conversion per 4 halves is cheap, but the
       loop overhead of 4 values per step costs more than the arithmetic. */
    for (; i + 16 <= n; i += 16) {
        float16x8_t h0 = vreinterpretq_f16_u16(vld1q_u16(x + i));
        float16x8_t h1 = vreinterpretq_f16_u16(vld1q_u16(x + i + 8));
        float32x4_t a0 = vfmaq_f32(vld1q_f32(acc + i), fv, vcvt_f32_f16(vget_low_f16(h0)));
        float32x4_t a1 = vfmaq_f32(vld1q_f32(acc + i + 4), fv, vcvt_high_f32_f16(h0));
        float32x4_t a2 = vfmaq_f32(vld1q_f32(acc + i + 8), fv, vcvt_f32_f16(vget_low_f16(h1)));
        float32x4_t a3 = vfmaq_f32(vld1q_f32(acc + i + 12), fv, vcvt_high_f32_f16(h1));
        vst1q_f32(acc + i, a0);
        vst1q_f32(acc + i + 4, a1);
        vst1q_f32(acc + i + 8, a2);
        vst1q_f32(acc + i + 12, a3);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(x + i)));
        vst1q_f32(acc + i, vfmaq_f32(vld1q_f32(acc + i), fv, v));
    }
    for (; i < n; i++) acc[i] = fmaf(f, fse_half_table[x[i]], acc[i]);
}
static inline void fse_fma_i8_neon(float *acc, const void *row, float f, int n) {
    const int8_t *x = (const int8_t *)row;
    float32x4_t fv = vdupq_n_f32(f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        int8x16_t c = vld1q_s8(x + i);
        int16x8_t lo = vmovl_s8(vget_low_s8(c)), hi = vmovl_high_s8(c);
        float32x4_t v0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
        float32x4_t v1 = vcvtq_f32_s32(vmovl_high_s16(lo));
        float32x4_t v2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
        float32x4_t v3 = vcvtq_f32_s32(vmovl_high_s16(hi));
        float32x4_t a0 = vfmaq_f32(vld1q_f32(acc + i), fv, v0);
        float32x4_t a1 = vfmaq_f32(vld1q_f32(acc + i + 4), fv, v1);
        float32x4_t a2 = vfmaq_f32(vld1q_f32(acc + i + 8), fv, v2);
        float32x4_t a3 = vfmaq_f32(vld1q_f32(acc + i + 12), fv, v3);
        vst1q_f32(acc + i, a0);
        vst1q_f32(acc + i + 4, a1);
        vst1q_f32(acc + i + 8, a2);
        vst1q_f32(acc + i + 12, a3);
    }
    for (; i + 8 <= n; i += 8) {
        int16x8_t w = vmovl_s8(vld1_s8(x + i));
        float32x4_t lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w)));
        float32x4_t hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w)));
        vst1q_f32(acc + i, vfmaq_f32(vld1q_f32(acc + i), fv, lo));
        vst1q_f32(acc + i + 4, vfmaq_f32(vld1q_f32(acc + i + 4), fv, hi));
    }
    if (i + 4 <= n) { /* 4 bytes through memcpy, so the load never reads past the row */
        int32_t four;
        memcpy(&four, x + i, sizeof four);
        int16x8_t w = vmovl_s8(vreinterpret_s8_s32(vdup_n_s32(four)));
        vst1q_f32(acc + i, vfmaq_f32(vld1q_f32(acc + i), fv, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w)))));
        i += 4;
    }
    for (; i < n; i++) acc[i] = fmaf(f, (float)x[i], acc[i]);
}
#endif

/* Multiply the row by inv if scale is set, and return whether it is not finite. One pass
   without an early exit, so the compiler vectorizes it: a value is not finite exactly when
   its exponent bits are all set. A finite accumulator stays finite times inv <= 1, so
   checking after the scaling gives the same answer as checking before. */
static inline int fse_finish_row(float *FSE_RESTRICT acc, int dim, float inv, int scale) {
    uint32_t bad = 0;
    if (scale) {
        for (int i = 0; i < dim; i++) {
            float v = inv * acc[i];
            uint32_t u;
            memcpy(&u, &v, sizeof u);
            bad |= (uint32_t)((u & 0x7F800000u) == 0x7F800000u);
            acc[i] = v;
        }
    } else {
        for (int i = 0; i < dim; i++) {
            uint32_t u;
            memcpy(&u, &acc[i], sizeof u);
            bad |= (uint32_t)((u & 0x7F800000u) == 0x7F800000u);
        }
    }
    return bad != 0;
}

/* --- The batch loop, once per instruction set and dtype ----------------------------- */

/* Accumulate (finish = 0) or pool (finish = 1) every sentence of the batch into out, one
   row of p->dim floats per sentence; counts receives the tokens per sentence if not NULL.
   Returns -1, or the first sentence whose accumulator is not finite. */
#define FSE_DEFINE_LOOP(NAME, TARGET, FMA, SCALED)                                          \
    FSE_NOINLINE TARGET static int64_t NAME(const fse_params *p, const fse_batch *b, float *out, \
                               int64_t *counts, int finish) {                               \
        const int dim = p->dim;                                                             \
        const size_t rb = p->row_bytes;                                                     \
        const char *rows = (const char *)p->rows;                                           \
        const char *ngram_rows = (const char *)p->ngram_rows;                               \
        int64_t k = 0; /* the next unknown word of the batch */                             \
        const int64_t total = b->offsets[b->n_sentences];                                   \
        for (int64_t s = 0; s < b->n_sentences; s++) {                                      \
            float *acc = out + (size_t)s * (size_t)dim;                                     \
            memset(acc, 0, (size_t)dim * sizeof(float));                                    \
            const int64_t begin = b->offsets[s], end = b->offsets[s + 1];                   \
            for (int64_t t = begin; t < end; t++) {                                         \
                if (t + FSE_PREFETCH_DISTANCE < total) {                                    \
                    uint32_t ahead = b->tokens[t + FSE_PREFETCH_DISTANCE];                  \
                    if (ahead != FSE_OOV_TOKEN) fse_prefetch_row(rows + (size_t)ahead * rb, rb); \
                }                                                                           \
                uint32_t w = b->tokens[t];                                                  \
                if (w == FSE_OOV_TOKEN) {                                                   \
                    const int64_t g0 = b->ngram_offsets[k], g1 = b->ngram_offsets[k + 1];   \
                    k++;                                                                    \
                    const int64_t m = g1 - g0;                                              \
                    if (m == 0) continue; /* counts for the length, adds nothing */         \
                    const float g = (1.0f / (float)m) * p->oov_weight;                      \
                    for (int64_t j = g0; j < g1; j++) {                                     \
                        if (j + FSE_PREFETCH_DISTANCE < g1)                                 \
                            fse_prefetch_row(ngram_rows + (size_t)b->ngrams[j + FSE_PREFETCH_DISTANCE] * rb, rb); \
                        const uint32_t bucket = b->ngrams[j];                               \
                        const float f = SCALED ? g * p->ngram_scales[bucket] : g;          \
                        FMA(acc, ngram_rows + (size_t)bucket * rb, f, dim);                 \
                    }                                                                       \
                } else {                                                                    \
                    const float f = SCALED ? p->weights[w] * p->scales[w] : p->weights[w];  \
                    FMA(acc, rows + (size_t)w * rb, f, dim);                                \
                }                                                                           \
            }                                                                               \
            const int64_t n = end - begin;                                                  \
            if (counts) counts[s] = n;                                                      \
            if (fse_finish_row(acc, dim, finish && n ? 1.0f / (float)n : 1.0f, finish && n)) \
                return s;                                                                   \
        }                                                                                   \
        return -1;                                                                          \
    }

#define FSE_NO_TARGET
FSE_DEFINE_LOOP(fse_loop_f32_portable, FSE_NO_TARGET, fse_fma_f32_portable, 0)
FSE_DEFINE_LOOP(fse_loop_f16_portable, FSE_NO_TARGET, fse_fma_f16_portable, 0)
FSE_DEFINE_LOOP(fse_loop_i8_portable, FSE_NO_TARGET, fse_fma_i8_portable, 1)
#if defined(FSE_X86)
FSE_DEFINE_LOOP(fse_loop_f32_avx2, FSE_AVX2_TARGET, fse_fma_f32_avx2, 0)
FSE_DEFINE_LOOP(fse_loop_f16_avx2, FSE_AVX2_TARGET, fse_fma_f16_avx2, 0)
FSE_DEFINE_LOOP(fse_loop_i8_avx2, FSE_AVX2_TARGET, fse_fma_i8_avx2, 1)
#endif
#if defined(FSE_NEON)
/* The row loop for rows too large for a tile; fse_loop_f32_neon is in _native_kernel_tiles.h. */
FSE_DEFINE_LOOP(fse_loop_f32_neon_rows, FSE_NO_TARGET, fse_fma_f32_neon, 0)
FSE_DEFINE_LOOP(fse_loop_f16_neon, FSE_NO_TARGET, fse_fma_f16_neon, 0)
FSE_DEFINE_LOOP(fse_loop_i8_neon, FSE_NO_TARGET, fse_fma_i8_neon, 1)
#endif

#include "_native_kernel_tiles.h"

#endif /* FSE_NATIVE_KERNEL_MEAN_H */
