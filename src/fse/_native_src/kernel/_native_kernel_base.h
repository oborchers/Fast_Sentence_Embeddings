/* The build checks, macros, instruction-set headers and shared types of the kernel.
   Included by _native_kernel.h, which documents the contract. */

#ifndef FSE_NATIVE_KERNEL_BASE_H
#define FSE_NATIVE_KERNEL_BASE_H

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

#endif /* FSE_NATIVE_KERNEL_BASE_H */
