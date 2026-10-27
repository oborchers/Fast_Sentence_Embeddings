/* The weighted mean kernel of fse 2.0 (ARCHITECTURE.md, "Numerical contract").

   Author: Oliver Borchers. Licensed under GNU General Public License v3.0.

   Every contribution does acc[i] = fma(f, x[i], acc[i]) in token order, with the factor
   f = weight (times the scale for int8 rows) and the stored row x read exactly; a
   sentence finishes as (1/n) * acc with 1/n rounded first; an empty sentence is the
   zero row. Lanes of a vector are independent elements, so vectorizing over the
   dimension keeps every element's sequence of fused operations: every instruction set
   gives the bits of fse._reference. Nothing else may contract a multiply and an add,
   so this file is built with -ffp-contract=off (or /fp:precise) and refuses fast math. */

#ifndef FSE_NATIVE_KERNEL_H
#define FSE_NATIVE_KERNEL_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__FAST_MATH__)
#error "fse._native must not be built with fast math: it changes the rounding of the kernel"
#endif
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

#if defined(_MSC_VER)
#define FSE_RESTRICT __restrict
#define FSE_INLINE static __forceinline
#define FSE_NOINLINE __declspec(noinline)
#else
#define FSE_RESTRICT __restrict__
#define FSE_INLINE static inline __attribute__((always_inline))
#define FSE_NOINLINE __attribute__((noinline))
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define FSE_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#define FSE_AVX2_TARGET
#else
#include <cpuid.h>
#define FSE_AVX2_TARGET __attribute__((target("avx2,fma,f16c")))
#endif
#endif

#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
#define FSE_NEON 1
#include <arm_neon.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define FSE_PREFETCH(p) __builtin_prefetch((const void *)(p), 0, 3)
#elif defined(FSE_X86)
#define FSE_PREFETCH(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
#else
#define FSE_PREFETCH(p) ((void)(p))
#endif

/* Tokens ahead whose rows are prefetched; 8 was best or within noise of the best on
   Apple M1 and AMD Zen 4 in 1.x for every format and size. */
#define FSE_PREFETCH_DISTANCE 8
#define FSE_OOV_TOKEN 0xFFFFFFFEu

enum { FSE_F32 = 0, FSE_F16 = 1, FSE_I8 = 2 };
enum { FSE_ISA_PORTABLE = 0, FSE_ISA_AVX2 = 1, FSE_ISA_NEON = 2 };

/* Every float16 bit pattern as float32, filled at import: exact widening. */
static float fse_half_table[65536];

typedef struct {
    int dtype;                       /* FSE_F32, FSE_F16 or FSE_I8, for words and ngrams */
    int dim;
    size_t row_bytes;
    const void *rows;                /* word rows */
    const float *scales;             /* int8 only, else NULL */
    const float *weights;
    const void *ngram_rows;          /* FastText only, else NULL */
    const float *ngram_scales;       /* int8 FastText only */
    float oov_weight;
} fse_params;

typedef struct {
    int64_t n_sentences;
    const int64_t *offsets;          /* n_sentences + 1 */
    const uint32_t *tokens;
    const int64_t *ngram_offsets;    /* one range per unknown word, or NULL */
    const uint32_t *ngrams;
} fse_batch;

static inline void fse_prefetch_row(const void *row, size_t bytes) {
    const char *p = (const char *)row;
    if (!bytes) return;
    for (size_t offset = 0; offset < bytes; offset += 64) FSE_PREFETCH(p + offset);
    FSE_PREFETCH(p + bytes - 1);
}

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
    TARGET static int64_t NAME(const fse_params *p, const fse_batch *b, float *out,         \
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
FSE_DEFINE_LOOP(fse_loop_f32_neon, FSE_NO_TARGET, fse_fma_f32_neon, 0)
FSE_DEFINE_LOOP(fse_loop_f16_neon, FSE_NO_TARGET, fse_fma_f16_neon, 0)
FSE_DEFINE_LOOP(fse_loop_i8_neon, FSE_NO_TARGET, fse_fma_i8_neon, 1)
#endif

/* --- Floating point environment ----------------------------------------------------- */

/* Round to nearest and keep subnormals for the duration of a call, whatever another
   library in the process set (a shared object built with fast math sets flush to zero). */
typedef struct { uint64_t saved; } fse_fpenv;

static inline fse_fpenv fse_fpenv_enter(void) {
    fse_fpenv env = {0};
#if defined(FSE_X86)
    unsigned int csr = _mm_getcsr();
    env.saved = csr;
    /* FTZ (bit 15), DAZ (bit 6) off, rounding control (bits 13-14) to nearest. */
    _mm_setcsr(csr & ~(0x8000u | 0x0040u | 0x6000u));
#elif defined(FSE_NEON)
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    env.saved = fpcr;
    /* FZ (bit 24), FZ16 (bit 19) off, RMode (bits 22-23) to nearest. */
    uint64_t want = fpcr & ~((1ull << 24) | (1ull << 19) | (3ull << 22));
    if (want != fpcr) __asm__ __volatile__("msr fpcr, %0" : : "r"(want));
#endif
    return env;
}

static inline void fse_fpenv_leave(fse_fpenv env) {
#if defined(FSE_X86)
    _mm_setcsr((unsigned int)env.saved);
#elif defined(FSE_NEON)
    uint64_t now;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(now));
    if (now != env.saved) __asm__ __volatile__("msr fpcr, %0" : : "r"(env.saved));
#else
    (void)env;
#endif
}

/* --- dot16, postprocessing steps and score ---------------------------------------- */

/* dot16: 16 partial sums, index i into lane i % 16 by fma in index order, then the lanes
   added 8, 4, 2 and 1 apart. Tails update only their own lanes: lanes can hold -0.0, so
   zero padding would change the result. */
static inline float fse_reduce16(float *p) {
    for (int w = 8; w; w >>= 1)
        for (int l = 0; l < w; l++) p[l] = p[l] + p[l + w];
    return p[0];
}

static float fse_dot16_portable(const float *a, const float *b, int n) {
    float p[16] = {0};
    for (int i = 0; i < n; i++) p[i & 15] = fmaf(a[i], b[i], p[i & 15]);
    return fse_reduce16(p);
}

#if defined(FSE_X86)
FSE_AVX2_TARGET static float fse_dot16_avx2(const float *a, const float *b, int n) {
    __m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        lo = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), lo);
        hi = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), hi);
    }
    float p[16];
    _mm256_storeu_ps(p, lo);
    _mm256_storeu_ps(p + 8, hi);
    for (; i < n; i++) p[i & 15] = fse_fma1_avx2(a[i], b[i], p[i & 15]);
    return fse_reduce16(p);
}
#endif

#if defined(FSE_NEON)
static float fse_dot16_neon(const float *a, const float *b, int n) {
    float32x4_t q0 = vdupq_n_f32(0), q1 = q0, q2 = q0, q3 = q0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        q0 = vfmaq_f32(q0, vld1q_f32(a + i), vld1q_f32(b + i));
        q1 = vfmaq_f32(q1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        q2 = vfmaq_f32(q2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        q3 = vfmaq_f32(q3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    float p[16];
    vst1q_f32(p, q0);
    vst1q_f32(p + 4, q1);
    vst1q_f32(p + 8, q2);
    vst1q_f32(p + 12, q3);
    for (; i < n; i++) p[i & 15] = fmaf(a[i], b[i], p[i & 15]);
    return fse_reduce16(p);
}
#endif

/* acc[i] = fma(c, u[i], acc[i]) for float32 rows, by instruction set. */
static void fse_axpy(int isa, float *acc, const float *u, float c, int n) {
#if defined(FSE_X86)
    if (isa == FSE_ISA_AVX2) { fse_fma_f32_avx2(acc, u, c, n); return; }
#endif
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON) { fse_fma_f32_neon(acc, u, c, n); return; }
#endif
    fse_fma_f32_portable(acc, u, c, n);
}

static float fse_dot16(int isa, const float *a, const float *b, int n) {
#if defined(FSE_X86)
    if (isa == FSE_ISA_AVX2) return fse_dot16_avx2(a, b, n);
#endif
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON) return fse_dot16_neon(a, b, n);
#endif
    return fse_dot16_portable(a, b, n);
}

static int fse_finite_row(const float *row, int n) {
    for (int i = 0; i < n; i++)
        if (!isfinite(row[i])) return 0;
    return 1;
}

/* remove_components@1 on every row, in place; scratch holds dim floats. Returns the first
   row that is not finite afterwards, or -1. */
FSE_NOINLINE static int64_t fse_remove_components_body(int isa, float *rows, int64_t n, int dim,
                                     const float *components, int k, const float *weights,
                                     float *scratch) {
    int64_t bad = -1;
    for (int64_t r = 0; r < n; r++) {
        float *row = rows + (size_t)r * (size_t)dim;
        memset(scratch, 0, (size_t)dim * sizeof(float));
        for (int j = 0; j < k; j++) {
            const float *u = components + (size_t)j * (size_t)dim;
            float c = fse_dot16(isa, row, u, dim);  /* the coefficients use the unmodified row */
            if (weights) c = c * weights[j];
            fse_axpy(isa, scratch, u, c, dim);
        }
        for (int i = 0; i < dim; i++) row[i] = row[i] - scratch[i];
        if (bad < 0 && !fse_finite_row(row, dim)) bad = r;
    }
    return bad;
}

/* project@1: out row r holds the k values dot16(row, basis_j). */
FSE_NOINLINE static int64_t fse_project_body(int isa, const float *rows, int64_t n, int dim, const float *basis,
                           int k, float *out) {
    int64_t bad = -1;
    for (int64_t r = 0; r < n; r++) {
        const float *row = rows + (size_t)r * (size_t)dim;
        float *o = out + (size_t)r * (size_t)k;
        for (int j = 0; j < k; j++) o[j] = fse_dot16(isa, row, basis + (size_t)j * (size_t)dim, dim);
        if (bad < 0 && !fse_finite_row(o, k)) bad = r;
    }
    return bad;
}

/* normalize@1 in place. Returns the first nonzero row whose sum of squares underflows to
   0 or overflows in *unnormalizable (else -1), and the first non-finite row, or -1. */
FSE_NOINLINE static int64_t fse_normalize_body(int isa, float *rows, int64_t n, int dim, int64_t *unnormalizable) {
    int64_t bad = -1;
    *unnormalizable = -1;
    for (int64_t r = 0; r < n; r++) {
        float *row = rows + (size_t)r * (size_t)dim;
        float s = fse_dot16(isa, row, row, dim);
        int nonzero = 0;
        for (int i = 0; i < dim; i++)
            if (row[i] != 0.0f) { nonzero = 1; break; }
        if (!nonzero) continue;
        if (s == 0.0f || !isfinite(s)) {
            if (*unnormalizable < 0) *unnormalizable = r;
            continue;
        }
        const float inv = 1.0f / sqrtf(s);
        for (int i = 0; i < dim; i++) row[i] = row[i] * inv;
        if (bad < 0 && !fse_finite_row(row, dim)) bad = r;
    }
    return bad;
}

/* score: dot16(stored row as float32, w) * scale + bias for every stored row; scratch holds
   dim floats. Returns the first row whose score is not finite, or -1. */
FSE_NOINLINE static int64_t fse_score_body(int isa, int dtype, const void *stored, const float *scales, int64_t n,
                         int dim, const float *w, float bias, float *out, float *scratch) {
    int64_t bad = -1;
    const size_t item = dtype == FSE_F32 ? 4 : dtype == FSE_F16 ? 2 : 1;
    for (int64_t r = 0; r < n; r++) {
        const char *row = (const char *)stored + (size_t)r * (size_t)dim * item;
        const float *x = (const float *)row;
        if (dtype == FSE_F16) {
            for (int i = 0; i < dim; i++) scratch[i] = fse_half_table[((const uint16_t *)row)[i]];
            x = scratch;
        } else if (dtype == FSE_I8) {
            for (int i = 0; i < dim; i++) scratch[i] = (float)((const int8_t *)row)[i];
            x = scratch;
        }
        float dot = fse_dot16(isa, x, w, dim);
        if (scales) dot = dot * scales[r];
        out[r] = dot + bias;
        if (bad < 0 && !isfinite(out[r])) bad = r;
    }
    return bad;
}

/* dot16 of rows a[r] and b[r] for r < n into out. */
FSE_NOINLINE static void fse_dot16_rows_body(int isa, const float *a, const float *b, int64_t n,
                                             int dim, float *out) {
    for (int64_t r = 0; r < n; r++)
        out[r] = fse_dot16(isa, a + (size_t)r * (size_t)dim, b + (size_t)r * (size_t)dim, dim);
}

/* The entry points: each sets the floating-point environment, then calls a function that
   is never inlined, so no compiler can move its arithmetic before the environment is set
   (MSVC's /fp:precise assumes an unchanged environment within a function). */
static void fse_dot16_rows(int isa, const float *a, const float *b, int64_t n, int dim,
                           float *out) {
    fse_fpenv env = fse_fpenv_enter();
    fse_dot16_rows_body(isa, a, b, n, dim, out);
    fse_fpenv_leave(env);
}
static int64_t fse_remove_components(int isa, float *rows, int64_t n, int dim,
                                     const float *components, int k, const float *weights,
                                     float *scratch) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t bad = fse_remove_components_body(isa, rows, n, dim, components, k, weights, scratch);
    fse_fpenv_leave(env);
    return bad;
}
static int64_t fse_project(int isa, const float *rows, int64_t n, int dim, const float *basis,
                           int k, float *out) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t bad = fse_project_body(isa, rows, n, dim, basis, k, out);
    fse_fpenv_leave(env);
    return bad;
}
static int64_t fse_normalize(int isa, float *rows, int64_t n, int dim, int64_t *unnormalizable) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t bad = fse_normalize_body(isa, rows, n, dim, unnormalizable);
    fse_fpenv_leave(env);
    return bad;
}
static int64_t fse_score(int isa, int dtype, const void *stored, const float *scales, int64_t n,
                         int dim, const float *w, float bias, float *out, float *scratch) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t bad = fse_score_body(isa, dtype, stored, scales, n, dim, w, bias, out, scratch);
    fse_fpenv_leave(env);
    return bad;
}

/* For the tests: set (on = 1) or clear flush-to-zero and denormals-are-zero (FZ on arm64)
   and return the previous environment, or restore one with fse_test_fpenv_restore. */
static uint64_t fse_test_fpenv_flush(int on) {
    fse_fpenv env = {0};
#if defined(FSE_X86)
    unsigned int csr = _mm_getcsr();
    env.saved = csr;
    _mm_setcsr(on ? (csr | 0x8000u | 0x0040u) : (csr & ~(0x8000u | 0x0040u)));
#elif defined(FSE_NEON)
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    env.saved = fpcr;
    uint64_t want = on ? (fpcr | (1ull << 24)) : (fpcr & ~(1ull << 24));
    __asm__ __volatile__("msr fpcr, %0" : : "r"(want));
#else
    (void)on;
#endif
    return env.saved;
}
static void fse_test_fpenv_restore(uint64_t saved) {
    fse_fpenv env;
    env.saved = saved;
    fse_fpenv_leave(env);
}
static uint64_t fse_test_fpenv_get(void) {
#if defined(FSE_X86)
    return _mm_getcsr();
#elif defined(FSE_NEON)
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr;
#else
    return 0;
#endif
}

/* --- Instruction sets ---------------------------------------------------------------- */

static int fse_has_avx2(void) {
#if defined(FSE_X86)
#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 1);
    int fma = (info[2] >> 12) & 1, osxsave = (info[2] >> 27) & 1, f16c = (info[2] >> 29) & 1;
    if (!(fma && osxsave && f16c)) return 0;
    if ((_xgetbv(0) & 6) != 6) return 0; /* the OS saves the ymm state */
    __cpuidex(info, 7, 0);
    return (info[1] >> 5) & 1;
#else
    __builtin_cpu_init();
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    int f16c = __get_cpuid(1, &eax, &ebx, &ecx, &edx) && ((ecx >> 29) & 1u);
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && f16c;
#endif
#else
    return 0;
#endif
}

static int fse_has_isa(int isa) {
    if (isa == FSE_ISA_PORTABLE) return 1;
    if (isa == FSE_ISA_AVX2) return fse_has_avx2();
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON) return 1;
#endif
    return 0;
}

typedef int64_t (*fse_loop_t)(const fse_params *, const fse_batch *, float *, int64_t *, int);

static fse_loop_t fse_loop_for(int isa, int dtype) {
#if defined(FSE_X86)
    if (isa == FSE_ISA_AVX2)
        return dtype == FSE_F32 ? fse_loop_f32_avx2 : dtype == FSE_F16 ? fse_loop_f16_avx2 : fse_loop_i8_avx2;
#endif
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON)
        return dtype == FSE_F32 ? fse_loop_f32_neon : dtype == FSE_F16 ? fse_loop_f16_neon : fse_loop_i8_neon;
#endif
    return dtype == FSE_F32 ? fse_loop_f32_portable : dtype == FSE_F16 ? fse_loop_f16_portable : fse_loop_i8_portable;
}

static int64_t fse_run(int isa, const fse_params *p, const fse_batch *b, float *out,
                       int64_t *counts, int finish) {
    fse_fpenv env = fse_fpenv_enter();
    /* Called through a pointer, so its arithmetic cannot move before the environment. */
    int64_t bad = fse_loop_for(isa, p->dtype)(p, b, out, counts, finish);
    fse_fpenv_leave(env);
    return bad;
}

#endif /* FSE_NATIVE_KERNEL_H */
