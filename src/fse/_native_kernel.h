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
#else
#define FSE_RESTRICT __restrict__
#define FSE_INLINE static inline __attribute__((always_inline))
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
    for (; i + 8 <= n; i += 8) {
        int16x8_t w = vmovl_s8(vld1_s8(x + i));
        float32x4_t lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w)));
        float32x4_t hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w)));
        vst1q_f32(acc + i, vfmaq_f32(vld1q_f32(acc + i), fv, lo));
        vst1q_f32(acc + i + 4, vfmaq_f32(vld1q_f32(acc + i + 4), fv, hi));
    }
    for (; i < n; i++) acc[i] = fmaf(f, (float)x[i], acc[i]);
}
#endif

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
            for (int i = 0; i < dim; i++)                                                   \
                if (!isfinite(acc[i])) return s;                                            \
            const int64_t n = end - begin;                                                  \
            if (counts) counts[s] = n;                                                      \
            if (finish && n) {                                                              \
                const float inv = 1.0f / (float)n;                                          \
                for (int i = 0; i < dim; i++) acc[i] = inv * acc[i];                        \
            }                                                                               \
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

static int64_t fse_run(int isa, const fse_params *p, const fse_batch *b, float *out,
                       int64_t *counts, int finish) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t bad = fse_loop_for(isa, p->dtype)(p, b, out, counts, finish);
    fse_fpenv_leave(env);
    return bad;
}

#endif /* FSE_NATIVE_KERNEL_H */
