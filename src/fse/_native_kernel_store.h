/* Stage 5, the storage of finished rows (quantize_rows@1): float16 and int8 with a scale
   per row, as fse._reference.quantize_rows. Included by _native_kernel.h. */

#ifndef FSE_NATIVE_KERNEL_STORE_H
#define FSE_NATIVE_KERNEL_STORE_H

#include <float.h>

#include "_native_kernel_steps.h"

#define FSE_FLOAT16_MAX 65504.0f

/* float32 to float16 rounded to nearest even, subnormals included, as numpy converts
   (npy_floatbits_to_halfbits); only finite values within +-65504 come here. */
FSE_INLINE uint16_t fse_float_to_half(float value) {
    uint32_t f;
    memcpy(&f, &value, sizeof f);
    const uint16_t sign = (uint16_t)((f & 0x80000000u) >> 16);
    uint32_t exp = f & 0x7f800000u;
    if (exp <= 0x38000000u) { /* a subnormal or zero half */
        if (exp < 0x33000000u) return sign;
        exp >>= 23;
        uint32_t sig = 0x00800000u + (f & 0x007fffffu);
        sig >>= (113 - exp);
        /* Ties to even: the shift may have dropped up to 11 bits, so check them too. */
        if (((sig & 0x00003fffu) != 0x00001000u) || (f & 0x000007ffu)) sig += 0x00001000u;
        return (uint16_t)(sign + (uint16_t)(sig >> 13));
    }
    const uint16_t hexp = (uint16_t)((exp - 0x38000000u) >> 13);
    uint32_t sig = f & 0x007fffffu;
    if ((sig & 0x00003fffu) != 0x00001000u) sig += 0x00001000u;
    /* A carry of the rounding runs into the exponent, as it must. */
    return (uint16_t)(sign + (uint16_t)((sig >> 13) + hexp));
}

/* One row: whether every value is finite, with the largest absolute value in *absmax;
   the int8 codes clip(rint(x / divisor), -127, 127); the float16 codes. The vector
   versions use only exact operations: IEEE division, rounding to nearest even (the
   kernel's environment), comparisons and the hardware float16 conversion. */
static int fse_absmax_portable(const float *row, int dim, float *absmax) {
    float m = 0.0f;
    int finite = 1;
    for (int i = 0; i < dim; i++) {
        const float a = fabsf(row[i]);
        finite &= a <= FLT_MAX; /* false for NaN and infinity */
        m = a > m ? a : m;
    }
    *absmax = m;
    return finite;
}
FSE_INLINE int8_t fse_code_i8(float x, float divisor) {
    float q = rintf(x / divisor);
    q = q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q;
    return (int8_t)q;
}
static void fse_codes_i8_portable(const float *row, int dim, float divisor, int8_t *out) {
    for (int i = 0; i < dim; i++) out[i] = fse_code_i8(row[i], divisor);
}
static void fse_codes_f16_portable(const float *row, int dim, uint16_t *out) {
    for (int i = 0; i < dim; i++) out[i] = fse_float_to_half(row[i]);
}

#if defined(FSE_NEON)
static int fse_absmax_neon(const float *row, int dim, float *absmax) {
    float32x4_t m = vdupq_n_f32(0.0f);
    uint32x4_t ok = vdupq_n_u32(0xffffffffu);
    const float32x4_t top = vdupq_n_f32(FLT_MAX);
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        const float32x4_t a = vabsq_f32(vld1q_f32(row + i));
        ok = vandq_u32(ok, vcleq_f32(a, top));
        m = vmaxq_f32(m, a);
    }
    float tail;
    int finite = fse_absmax_portable(row + i, dim - i, &tail) && vminvq_u32(ok) != 0;
    const float head = vmaxvq_f32(m);
    *absmax = head > tail ? head : tail;
    return finite;
}
static void fse_codes_i8_neon(const float *row, int dim, float divisor, int8_t *out) {
    const float32x4_t d = vdupq_n_f32(divisor), hi = vdupq_n_f32(127.0f), lo = vdupq_n_f32(-127.0f);
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        float32x4_t a = vrndnq_f32(vdivq_f32(vld1q_f32(row + i), d));
        float32x4_t b = vrndnq_f32(vdivq_f32(vld1q_f32(row + i + 4), d));
        a = vmaxq_f32(vminq_f32(a, hi), lo);
        b = vmaxq_f32(vminq_f32(b, hi), lo);
        const int16x8_t w = vcombine_s16(vmovn_s32(vcvtq_s32_f32(a)), vmovn_s32(vcvtq_s32_f32(b)));
        vst1_s8(out + i, vmovn_s16(w));
    }
    for (; i < dim; i++) out[i] = fse_code_i8(row[i], divisor);
}
static void fse_codes_f16_neon(const float *row, int dim, uint16_t *out) {
    int i = 0;
    for (; i + 4 <= dim; i += 4)
        vst1_u16(out + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(row + i))));
    for (; i < dim; i++) out[i] = fse_float_to_half(row[i]);
}
#endif

#if defined(FSE_X86)
FSE_AVX2_TARGET static int fse_absmax_avx2(const float *row, int dim, float *absmax) {
    const __m256 sign = _mm256_set1_ps(-0.0f), top = _mm256_set1_ps(FLT_MAX);
    __m256 m = _mm256_setzero_ps(), ok = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 a = _mm256_andnot_ps(sign, _mm256_loadu_ps(row + i));
        ok = _mm256_and_ps(ok, _mm256_cmp_ps(a, top, _CMP_LE_OQ));
        m = _mm256_max_ps(m, a);
    }
    float lanes[8], tail;
    _mm256_storeu_ps(lanes, m);
    int finite = fse_absmax_portable(row + i, dim - i, &tail) && _mm256_movemask_ps(ok) == 0xff;
    for (int l = 0; l < 8; l++) tail = lanes[l] > tail ? lanes[l] : tail;
    *absmax = tail;
    return finite;
}
FSE_AVX2_TARGET static void fse_codes_i8_avx2(const float *row, int dim, float divisor, int8_t *out) {
    const __m256 d = _mm256_set1_ps(divisor), hi = _mm256_set1_ps(127.0f), lo = _mm256_set1_ps(-127.0f);
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 q = _mm256_round_ps(_mm256_div_ps(_mm256_loadu_ps(row + i), d),
                                   _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        q = _mm256_max_ps(_mm256_min_ps(q, hi), lo);
        const __m256i v = _mm256_cvtps_epi32(q);
        const __m128i w = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        _mm_storel_epi64((__m128i *)(out + i), _mm_packs_epi16(w, w));
    }
    for (; i < dim; i++) out[i] = fse_code_i8(row[i], divisor);
}
FSE_AVX2_TARGET static void fse_codes_f16_avx2(const float *row, int dim, uint16_t *out) {
    int i = 0;
    for (; i + 8 <= dim; i += 8)
        _mm_storeu_si128((__m128i *)(out + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(row + i), _MM_FROUND_TO_NEAREST_INT));
    for (; i < dim; i++) out[i] = fse_float_to_half(row[i]);
}
#endif

typedef int (*fse_absmax_fn)(const float *, int, float *);
typedef void (*fse_codes_i8_fn)(const float *, int, float, int8_t *);
typedef void (*fse_codes_f16_fn)(const float *, int, uint16_t *);

static fse_absmax_fn fse_absmax_of(int isa) {
#if defined(FSE_X86)
    if (isa == FSE_ISA_AVX2) return fse_absmax_avx2;
#endif
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON) return fse_absmax_neon;
#endif
    (void)isa;
    return fse_absmax_portable;
}
static fse_codes_i8_fn fse_codes_i8_of(int isa) {
#if defined(FSE_X86)
    if (isa == FSE_ISA_AVX2) return fse_codes_i8_avx2;
#endif
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON) return fse_codes_i8_neon;
#endif
    (void)isa;
    return fse_codes_i8_portable;
}
static fse_codes_f16_fn fse_codes_f16_of(int isa) {
#if defined(FSE_X86)
    if (isa == FSE_ISA_AVX2) return fse_codes_f16_avx2;
#endif
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON) return fse_codes_f16_neon;
#endif
    (void)isa;
    return fse_codes_f16_portable;
}

/* The first row that is not finite in *bad, else the first row beyond +-65504 in
   *beyond (each -1 if there is none); every other row is stored. */
FSE_NOINLINE static void fse_store_f16_body(int isa, const float *rows, int64_t n, int dim,
                                            uint16_t *codes, int64_t *bad, int64_t *beyond) {
    const fse_absmax_fn absmax = fse_absmax_of(isa);
    const fse_codes_f16_fn convert = fse_codes_f16_of(isa);
    *bad = -1;
    *beyond = -1;
    for (int64_t r = 0; r < n; r++) {
        const float *row = rows + (size_t)r * (size_t)dim;
        float m;
        if (!absmax(row, dim, &m)) {
            if (*bad < 0) *bad = r;
        } else if (m > FSE_FLOAT16_MAX) {
            if (*beyond < 0) *beyond = r;
        } else {
            convert(row, dim, codes + (size_t)r * (size_t)dim);
        }
    }
}

/* int8 per row: s = absmax / 127, stepped down once if 127 * s overflows; codes
   clip(rint(x / s), -127, 127), all 0 for s = 0. The first row that is not finite in
   *bad, or -1. */
FSE_NOINLINE static void fse_store_i8_body(int isa, const float *rows, int64_t n, int dim,
                                           int8_t *codes, float *scales, int64_t *bad) {
    const fse_absmax_fn absmax = fse_absmax_of(isa);
    const fse_codes_i8_fn convert = fse_codes_i8_of(isa);
    *bad = -1;
    for (int64_t r = 0; r < n; r++) {
        const float *row = rows + (size_t)r * (size_t)dim;
        float m;
        if (!absmax(row, dim, &m)) {
            if (*bad < 0) *bad = r;
            continue;
        }
        float s = m / 127.0f;
        if (!isfinite(s * 127.0f)) s = nextafterf(s, 0.0f);
        scales[r] = s;
        convert(row, dim, s > 0.0f ? s : 1.0f, codes + (size_t)r * (size_t)dim);
    }
}

static void fse_store_f16(int isa, const float *rows, int64_t n, int dim, uint16_t *codes,
                          int64_t *bad, int64_t *beyond) {
    fse_fpenv env = fse_fpenv_enter();
    fse_store_f16_body(isa, rows, n, dim, codes, bad, beyond);
    fse_fpenv_leave(env);
}
static void fse_store_i8(int isa, const float *rows, int64_t n, int dim, int8_t *codes,
                         float *scales, int64_t *bad) {
    fse_fpenv env = fse_fpenv_enter();
    fse_store_i8_body(isa, rows, n, dim, codes, scales, bad);
    fse_fpenv_leave(env);
}

#endif /* FSE_NATIVE_KERNEL_STORE_H */
