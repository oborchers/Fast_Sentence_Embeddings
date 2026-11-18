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
#if defined(FSE_HAVE_AVX512)
#if defined(_MSC_VER)
#define FSE_AVX512_TARGET
#else
#define FSE_AVX512_TARGET __attribute__((target("avx2,fma,f16c,avx512f,avx512dq,avx512bw,avx512vl")))
#endif
#define FSE_AVX512 1
#endif
#endif

/* A remains the default until the story measurement selects a layout. */
#define FSE_AVX512_LAYOUT 0

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
/* One hint per k-th intersecting 64-byte line. Tuning never changes arithmetic. */
#ifndef FSE_PREFETCH_LINE_STRIDE
#define FSE_PREFETCH_LINE_STRIDE 1
#endif
#if FSE_PREFETCH_LINE_STRIDE < 1
#error "FSE_PREFETCH_LINE_STRIDE must be positive"
#endif
#define FSE_OOV_TOKEN 0xFFFFFFFEu

enum { FSE_F32 = 0, FSE_F16 = 1, FSE_I8 = 2 };
enum { FSE_ISA_PORTABLE = 0, FSE_ISA_AVX2 = 1, FSE_ISA_NEON = 2, FSE_ISA_AVX512 = 3 };

/* Per-thread diagnostics exist only in explicitly instrumented test builds. */
#if defined(FSE_TEST_TILES)
#if defined(_MSC_VER)
#define FSE_TEST_TLS __declspec(thread)
#else
#define FSE_TEST_TLS __thread
#endif
static FSE_TEST_TLS unsigned int fse_test_routes;
#define FSE_TEST_ROUTE(bit) (fse_test_routes |= (bit))
static FSE_TEST_TLS int64_t fse_test_tiles;
static FSE_TEST_TLS int fse_test_loop;
static FSE_TEST_TLS int64_t fse_test_builders[2];
static FSE_TEST_TLS uint64_t fse_test_prefetch_count;
static FSE_TEST_TLS uint64_t fse_test_prefetch_digest;
#define FSE_TEST_PREFETCH_RESET() do { fse_test_prefetch_count = 0; fse_test_prefetch_digest = 0; } while (0)
#define FSE_TEST_PREFETCH(address) do { \
    ++fse_test_prefetch_count; \
    fse_test_prefetch_digest = (fse_test_prefetch_digest ^ (uint64_t)(address)) * UINT64_C(1099511628211); \
} while (0)
#define FSE_TEST_BUILDER(known) (++fse_test_builders[known])
#define FSE_TEST_TILE() (++fse_test_tiles)
#define FSE_TEST_ROWS() (fse_test_loop = 0)
#else
#define FSE_TEST_ROUTE(bit) ((void)0)
#define FSE_TEST_BUILDER(known) ((void)0)
#define FSE_TEST_TILE() ((void)0)
#define FSE_TEST_ROWS() ((void)0)
#define FSE_TEST_PREFETCH_RESET() ((void)0)
#define FSE_TEST_PREFETCH(address) ((void)0)
#endif

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
    if (!bytes) return;
    const uintptr_t p = (uintptr_t)row;
    const uintptr_t start = p & ~(uintptr_t)63;
    const size_t last = ((p & 63) + bytes - 1) / 64;
    for (size_t line = 0; line <= last; line += FSE_PREFETCH_LINE_STRIDE) {
        const uintptr_t address = start + line * 64;
        FSE_TEST_PREFETCH(address);
        FSE_PREFETCH((const void *)address);
    }
}

#endif /* FSE_NATIVE_KERNEL_BASE_H */
