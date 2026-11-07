/* The integer sums of the int8 prefilter (docs/architecture/numerical-contract.md, "int8
   prefilter (no new arithmetic)"): for rows of int8 codes and int8 queries,

       D = sum(codes * qcodes)        S = sum(codes * codes)

   in int32. Both are exact for every dimension up to 1024 and every code in [-128, 127]
   (|D| <= 128 * 127 * 1024 and S <= 128 * 128 * 1024 = 2^24), so the order of the additions
   and the width of the vectors change no bit: every variant gives the same integers. The
   float64 arithmetic on them lives in _native_kernel_prefilter.h.

   Variants: portable (plain int32 loops), neon-sdot (vdotq_s32, built when setup.py's
   compile probe finds the instruction, FSE_HAVE_DOTPROD, and run only on a CPU that has it)
   and avx2 (_mm256_cvtepi8_epi16 and _mm256_madd_epi16). A SIMD variant reads whole
   16-byte blocks of the dimension and finishes the rest of a row with scalar code, so it
   never reads past the end of a row. Included by _native_kernel.h. */

#ifndef FSE_NATIVE_KERNEL_INT8_H
#define FSE_NATIVE_KERNEL_INT8_H

#include "_native_kernel_base.h"

#if defined(FSE_NEON) && defined(FSE_HAVE_DOTPROD)
#define FSE_I8_NEON_SDOT 1
#if defined(__clang__)
#define FSE_DOTPROD_TARGET __attribute__((target("dotprod")))
#else
#define FSE_DOTPROD_TARGET __attribute__((target("+dotprod")))
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
#endif
#endif

enum { FSE_I8_PORTABLE = 0, FSE_I8_AVX2 = 1, FSE_I8_NEON = 2 };

/* --- portable ------------------------------------------------------------------------ */

static inline int32_t fse_i8_dot_portable(const int8_t *x, const int8_t *q, int dim, int32_t *s) {
    int32_t d = 0, sq = 0;
    for (int i = 0; i < dim; i++) {
        d += (int32_t)x[i] * (int32_t)q[i];
        sq += (int32_t)x[i] * (int32_t)x[i];
    }
    if (s) *s = sq;
    return d;
}
static inline int32_t fse_i8_sq_portable(const int8_t *x, int dim) {
    int32_t sq = 0;
    for (int i = 0; i < dim; i++) sq += (int32_t)x[i] * (int32_t)x[i];
    return sq;
}
static inline void fse_i8_tile1_portable(const int8_t *x, size_t stride, const int8_t *q0, int dim,
                                         int32_t d[4], int32_t *s) {
    for (int a = 0; a < 4; a++) d[a] = fse_i8_dot_portable(x + (size_t)a * stride, q0, dim, s ? s + a : NULL);
}
static inline void fse_i8_tile2_portable(const int8_t *x, size_t stride, const int8_t *q0,
                                         const int8_t *q1, int dim, int32_t d[4][2], int32_t *s) {
    for (int a = 0; a < 4; a++) {
        const int8_t *row = x + (size_t)a * stride;
        d[a][0] = fse_i8_dot_portable(row, q0, dim, s ? s + a : NULL);
        d[a][1] = fse_i8_dot_portable(row, q1, dim, NULL);
    }
}

/* --- NEON SDOT ----------------------------------------------------------------------- */

#if defined(FSE_I8_NEON_SDOT)
/* Whole 16-byte blocks of the dimension; the tail of a row is scalar. */
#define FSE_I8_BLOCKS(dim) ((dim) & ~15)

FSE_DOTPROD_TARGET static inline int32_t fse_i8_dot_neon(const int8_t *x, const int8_t *q, int dim,
                                                          int32_t *s) {
    int32x4_t acc = vdupq_n_s32(0), sq = vdupq_n_s32(0);
    const int blocks = FSE_I8_BLOCKS(dim);
    for (int i = 0; i < blocks; i += 16) {
        int8x16_t v = vld1q_s8(x + i);
        acc = vdotq_s32(acc, v, vld1q_s8(q + i));
        sq = vdotq_s32(sq, v, v);
    }
    int32_t d = vaddvq_s32(acc), t = vaddvq_s32(sq);
    for (int i = blocks; i < dim; i++) {
        d += (int32_t)x[i] * (int32_t)q[i];
        t += (int32_t)x[i] * (int32_t)x[i];
    }
    if (s) *s = t;
    return d;
}
FSE_DOTPROD_TARGET static inline int32_t fse_i8_sq_neon(const int8_t *x, int dim) {
    int32x4_t sq = vdupq_n_s32(0);
    const int blocks = FSE_I8_BLOCKS(dim);
    for (int i = 0; i < blocks; i += 16) {
        int8x16_t v = vld1q_s8(x + i);
        sq = vdotq_s32(sq, v, v);
    }
    int32_t t = vaddvq_s32(sq);
    for (int i = blocks; i < dim; i++) t += (int32_t)x[i] * (int32_t)x[i];
    return t;
}

/* Four rows by one or two queries, the accumulators in registers over the whole row; the
   sums of squares only when s is given. */
FSE_DOTPROD_TARGET static inline void fse_i8_tile1_neon(const int8_t *x, size_t stride,
                                                        const int8_t *q0, int dim, int32_t d[4],
                                                        int32_t *s) {
    const int8_t *r[4] = {x, x + stride, x + 2 * stride, x + 3 * stride};
    int32x4_t acc[4], sq[4];
    for (int a = 0; a < 4; a++) acc[a] = sq[a] = vdupq_n_s32(0);
    const int blocks = FSE_I8_BLOCKS(dim);
    for (int i = 0; i < blocks; i += 16) {
        int8x16_t y = vld1q_s8(q0 + i);
        for (int a = 0; a < 4; a++) {
            int8x16_t v = vld1q_s8(r[a] + i);
            acc[a] = vdotq_s32(acc[a], v, y);
            if (s) sq[a] = vdotq_s32(sq[a], v, v);
        }
    }
    for (int a = 0; a < 4; a++) {
        d[a] = vaddvq_s32(acc[a]);
        if (s) s[a] = vaddvq_s32(sq[a]);
        for (int i = blocks; i < dim; i++) {
            d[a] += (int32_t)r[a][i] * (int32_t)q0[i];
            if (s) s[a] += (int32_t)r[a][i] * (int32_t)r[a][i];
        }
    }
}
FSE_DOTPROD_TARGET static inline void fse_i8_tile2_neon(const int8_t *x, size_t stride,
                                                        const int8_t *q0, const int8_t *q1,
                                                        int dim, int32_t d[4][2], int32_t *s) {
    const int8_t *r[4] = {x, x + stride, x + 2 * stride, x + 3 * stride};
    int32x4_t acc[4][2], sq[4];
    for (int a = 0; a < 4; a++) acc[a][0] = acc[a][1] = sq[a] = vdupq_n_s32(0);
    const int blocks = FSE_I8_BLOCKS(dim);
    for (int i = 0; i < blocks; i += 16) {
        int8x16_t y0 = vld1q_s8(q0 + i), y1 = vld1q_s8(q1 + i);
        for (int a = 0; a < 4; a++) {
            int8x16_t v = vld1q_s8(r[a] + i);
            acc[a][0] = vdotq_s32(acc[a][0], v, y0);
            acc[a][1] = vdotq_s32(acc[a][1], v, y1);
            if (s) sq[a] = vdotq_s32(sq[a], v, v);
        }
    }
    for (int a = 0; a < 4; a++) {
        d[a][0] = vaddvq_s32(acc[a][0]);
        d[a][1] = vaddvq_s32(acc[a][1]);
        if (s) s[a] = vaddvq_s32(sq[a]);
        for (int i = blocks; i < dim; i++) {
            d[a][0] += (int32_t)r[a][i] * (int32_t)q0[i];
            d[a][1] += (int32_t)r[a][i] * (int32_t)q1[i];
            if (s) s[a] += (int32_t)r[a][i] * (int32_t)r[a][i];
        }
    }
}

/* Linux: the kernel reports the extension in AT_HWCAP (glibc and musl); macOS: sysctl. */
static int fse_has_dotprod(void) {
#if defined(__APPLE__)
    int value = 0;
    size_t size = sizeof value;
    return sysctlbyname("hw.optional.arm.FEAT_DotProd", &value, &size, NULL, 0) == 0 && value;
#elif defined(__linux__)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    return 0;
#endif
}
#else
static int fse_has_dotprod(void) { return 0; }
#endif /* FSE_I8_NEON_SDOT */

/* --- AVX2 ---------------------------------------------------------------------------- */

#if defined(FSE_X86)
/* 16 codes widen to int16 and madd adds neighbouring products into 8 int32 lanes: at most
   2 * 128 * 128 per lane and block, so the lanes cannot overflow before the reduction. */
FSE_AVX2_TARGET static inline int32_t fse_i8_hsum(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}
FSE_AVX2_TARGET static inline __m256i fse_i8_wide(const int8_t *p) {
    return _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)p));
}

FSE_AVX2_TARGET static inline int32_t fse_i8_dot_avx2(const int8_t *x, const int8_t *q, int dim,
                                                       int32_t *s) {
    __m256i acc = _mm256_setzero_si256(), sq = _mm256_setzero_si256();
    const int blocks = dim & ~15;
    for (int i = 0; i < blocks; i += 16) {
        __m256i v = fse_i8_wide(x + i);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(v, fse_i8_wide(q + i)));
        sq = _mm256_add_epi32(sq, _mm256_madd_epi16(v, v));
    }
    int32_t d = fse_i8_hsum(acc), t = fse_i8_hsum(sq);
    for (int i = blocks; i < dim; i++) {
        d += (int32_t)x[i] * (int32_t)q[i];
        t += (int32_t)x[i] * (int32_t)x[i];
    }
    if (s) *s = t;
    return d;
}
FSE_AVX2_TARGET static inline int32_t fse_i8_sq_avx2(const int8_t *x, int dim) {
    __m256i sq = _mm256_setzero_si256();
    const int blocks = dim & ~15;
    for (int i = 0; i < blocks; i += 16) {
        __m256i v = fse_i8_wide(x + i);
        sq = _mm256_add_epi32(sq, _mm256_madd_epi16(v, v));
    }
    int32_t t = fse_i8_hsum(sq);
    for (int i = blocks; i < dim; i++) t += (int32_t)x[i] * (int32_t)x[i];
    return t;
}
FSE_AVX2_TARGET static inline void fse_i8_tile1_avx2(const int8_t *x, size_t stride,
                                                     const int8_t *q0, int dim, int32_t d[4],
                                                     int32_t *s) {
    const int8_t *r[4] = {x, x + stride, x + 2 * stride, x + 3 * stride};
    __m256i acc[4], sq[4];
    for (int a = 0; a < 4; a++) acc[a] = sq[a] = _mm256_setzero_si256();
    const int blocks = dim & ~15;
    for (int i = 0; i < blocks; i += 16) {
        __m256i y = fse_i8_wide(q0 + i);
        for (int a = 0; a < 4; a++) {
            __m256i v = fse_i8_wide(r[a] + i);
            acc[a] = _mm256_add_epi32(acc[a], _mm256_madd_epi16(v, y));
            if (s) sq[a] = _mm256_add_epi32(sq[a], _mm256_madd_epi16(v, v));
        }
    }
    for (int a = 0; a < 4; a++) {
        d[a] = fse_i8_hsum(acc[a]);
        if (s) s[a] = fse_i8_hsum(sq[a]);
        for (int i = blocks; i < dim; i++) {
            d[a] += (int32_t)r[a][i] * (int32_t)q0[i];
            if (s) s[a] += (int32_t)r[a][i] * (int32_t)r[a][i];
        }
    }
}
FSE_AVX2_TARGET static inline void fse_i8_tile2_avx2(const int8_t *x, size_t stride,
                                                     const int8_t *q0, const int8_t *q1, int dim,
                                                     int32_t d[4][2], int32_t *s) {
    const int8_t *r[4] = {x, x + stride, x + 2 * stride, x + 3 * stride};
    __m256i acc[4][2], sq[4];
    for (int a = 0; a < 4; a++) acc[a][0] = acc[a][1] = sq[a] = _mm256_setzero_si256();
    const int blocks = dim & ~15;
    for (int i = 0; i < blocks; i += 16) {
        __m256i y0 = fse_i8_wide(q0 + i), y1 = fse_i8_wide(q1 + i);
        for (int a = 0; a < 4; a++) {
            __m256i v = fse_i8_wide(r[a] + i);
            acc[a][0] = _mm256_add_epi32(acc[a][0], _mm256_madd_epi16(v, y0));
            acc[a][1] = _mm256_add_epi32(acc[a][1], _mm256_madd_epi16(v, y1));
            if (s) sq[a] = _mm256_add_epi32(sq[a], _mm256_madd_epi16(v, v));
        }
    }
    for (int a = 0; a < 4; a++) {
        d[a][0] = fse_i8_hsum(acc[a][0]);
        d[a][1] = fse_i8_hsum(acc[a][1]);
        if (s) s[a] = fse_i8_hsum(sq[a]);
        for (int i = blocks; i < dim; i++) {
            d[a][0] += (int32_t)r[a][i] * (int32_t)q0[i];
            d[a][1] += (int32_t)r[a][i] * (int32_t)q1[i];
            if (s) s[a] += (int32_t)r[a][i] * (int32_t)r[a][i];
        }
    }
}
#endif /* FSE_X86 */

#endif /* FSE_NATIVE_KERNEL_INT8_H */
