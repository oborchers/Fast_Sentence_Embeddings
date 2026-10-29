/* Linear heads: the bias of folded logits, exp@1, the links and the labels
   (ARCHITECTURE.md, "Numerical contract"; fse._reference_head is the reference).
   Included by _native_kernel.h, after _native_kernel_steps.h, whose floating-point
   environment it uses.

   Every rule is scalar float32 arithmetic with one rounding per operation: the adds,
   products and divisions of IEEE 754, fmaf for the fused steps, rintf (round to nearest
   even in the environment the entry points set) and powers of two built from their
   bits, so every instruction set gives the bits of the reference. */

#ifndef FSE_NATIVE_KERNEL_HEAD_H
#define FSE_NATIVE_KERNEL_HEAD_H

#include "_native_kernel_base.h"
#include "_native_kernel_steps.h"

enum { FSE_LINK_SIGMOID = 0, FSE_LINK_SOFTMAX = 1, FSE_LINK_OVR = 2 };

/* The constants of exp@1 (the hex floats of fse._reference_head), written as the exact
   decimal values of those float32 numbers, which every compiler converts exactly. */
#define FSE_EXP_LOG2E 1.44269502162933349609375f                       /* 0x1.715476p+0 */
#define FSE_EXP_LN2_HI 0.693147182464599609375f                         /* 0x1.62e430p-1 */
#define FSE_EXP_LN2_LO (-1.9046542121259335544891655445098876953125e-9f) /* -0x1.05c610p-29 */
#define FSE_EXP_C7 0.000198412701138295233249664306640625f               /* 0x1.a01a02p-13 */
#define FSE_EXP_C6 0.001388888922519981861114501953125f                  /* 0x1.6c16c2p-10 */
#define FSE_EXP_C5 0.008333333767950534820556640625f                     /* 0x1.111112p-7 */
#define FSE_EXP_C4 0.0416666679084300994873046875f                       /* 0x1.555556p-5 */
#define FSE_EXP_C3 0.16666667163372039794921875f                         /* 0x1.555556p-3 */
#define FSE_EXP_LOW (-104.0f)

/* 2**e for -126 <= e <= 127, exact. */
static inline float fse_pow2(int e) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)(e + 127) << 23;
    return v.f;
}

/* exp@1 of x <= 0: k = rint(x * log2e), r = x - k ln2 in two fused steps, the polynomial
   of degree 7 by Horner with fused multiply-adds, then (p * 2**(k + 64)) * 2**-64. */
static inline float fse_exp1(float x) {
    if (!(x >= FSE_EXP_LOW)) return 0.0f;
    const float k = rintf(x * FSE_EXP_LOG2E);
    float r = fmaf(-k, FSE_EXP_LN2_HI, x);
    r = fmaf(-k, FSE_EXP_LN2_LO, r);
    float p = FSE_EXP_C7;
    p = fmaf(p, r, FSE_EXP_C6);
    p = fmaf(p, r, FSE_EXP_C5);
    p = fmaf(p, r, FSE_EXP_C4);
    p = fmaf(p, r, FSE_EXP_C3);
    p = fmaf(p, r, 0.5f);
    p = fmaf(p, r, 1.0f);
    p = fmaf(p, r, 1.0f);
    const float scaled = p * fse_pow2((int)k + 64);
    return scaled * fse_pow2(-64);
}

/* sigmoid@1: the probability of the positive class. */
static inline float fse_sigmoid1(float z) {
    const float e = fse_exp1(-fabsf(z));
    const float d = 1.0f + e;
    return z >= 0.0f ? 1.0f / d : e / d;
}

/* --- Blocks of values: exp@1 and sigmoid@1 by 4 (NEON) or 8 (AVX2) lanes ------------------

   Every lane runs the operations of the scalar functions above, each one IEEE operation
   with one rounding (round to nearest even for the integer k, fused multiply-adds, the
   division), so a lane gives the bits of fse_exp1 and fse_sigmoid1; a lane below
   FSE_EXP_LOW computes something and is replaced by +0.0. Tails run the scalar code. */

#if defined(FSE_NEON)
static inline float32x4_t fse_exp1_neon(float32x4_t x) {
    const uint32x4_t ok = vcgeq_f32(x, vdupq_n_f32(FSE_EXP_LOW)); /* false for NaN too */
    const float32x4_t k = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(FSE_EXP_LOG2E)));
    const float32x4_t nk = vnegq_f32(k);
    float32x4_t r = vfmaq_f32(x, nk, vdupq_n_f32(FSE_EXP_LN2_HI)); /* x + (-k) ln2_hi */
    r = vfmaq_f32(r, nk, vdupq_n_f32(FSE_EXP_LN2_LO));
    float32x4_t p = vdupq_n_f32(FSE_EXP_C7);
    p = vfmaq_f32(vdupq_n_f32(FSE_EXP_C6), p, r); /* c6 + p r */
    p = vfmaq_f32(vdupq_n_f32(FSE_EXP_C5), p, r);
    p = vfmaq_f32(vdupq_n_f32(FSE_EXP_C4), p, r);
    p = vfmaq_f32(vdupq_n_f32(FSE_EXP_C3), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    const int32x4_t e = vaddq_s32(vcvtq_s32_f32(k), vdupq_n_s32(64 + 127));
    const float32x4_t scale = vreinterpretq_f32_s32(vshlq_n_s32(e, 23));
    const float32x4_t out = vmulq_f32(vmulq_f32(p, scale), vdupq_n_f32(fse_pow2(-64)));
    return vbslq_f32(ok, out, vdupq_n_f32(0.0f));
}
static inline float32x4_t fse_sigmoid1_neon(float32x4_t z) {
    const float32x4_t e = fse_exp1_neon(vnegq_f32(vabsq_f32(z)));
    const float32x4_t d = vaddq_f32(vdupq_n_f32(1.0f), e);
    const uint32x4_t up = vcgeq_f32(z, vdupq_n_f32(0.0f)); /* -0.0 >= 0 */
    return vbslq_f32(up, vdivq_f32(vdupq_n_f32(1.0f), d), vdivq_f32(e, d));
}
#endif

#if defined(FSE_X86)
FSE_AVX2_TARGET static inline __m256 fse_exp1_avx2(__m256 x) {
    const __m256 ok = _mm256_cmp_ps(x, _mm256_set1_ps(FSE_EXP_LOW), _CMP_GE_OQ);
    const __m256 k = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(FSE_EXP_LOG2E)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 nk = _mm256_xor_ps(k, _mm256_set1_ps(-0.0f)); /* -k, exactly */
    __m256 r = _mm256_fmadd_ps(nk, _mm256_set1_ps(FSE_EXP_LN2_HI), x);
    r = _mm256_fmadd_ps(nk, _mm256_set1_ps(FSE_EXP_LN2_LO), r);
    __m256 p = _mm256_set1_ps(FSE_EXP_C7);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(FSE_EXP_C6));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(FSE_EXP_C5));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(FSE_EXP_C4));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(FSE_EXP_C3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    const __m256i e = _mm256_add_epi32(_mm256_cvttps_epi32(k), _mm256_set1_epi32(64 + 127));
    const __m256 scale = _mm256_castsi256_ps(_mm256_slli_epi32(e, 23));
    const __m256 out = _mm256_mul_ps(_mm256_mul_ps(p, scale), _mm256_set1_ps(fse_pow2(-64)));
    return _mm256_and_ps(out, ok); /* +0.0 where the input is below the threshold */
}
FSE_AVX2_TARGET static inline __m256 fse_sigmoid1_avx2(__m256 z) {
    const __m256 e = fse_exp1_avx2(_mm256_or_ps(z, _mm256_set1_ps(-0.0f))); /* exp(-|z|) */
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 d = _mm256_add_ps(one, e);
    const __m256 up = _mm256_cmp_ps(z, _mm256_setzero_ps(), _CMP_GE_OQ);
    return _mm256_blendv_ps(_mm256_div_ps(e, d), _mm256_div_ps(one, d), up);
}
FSE_AVX2_TARGET static void fse_exp1_block_avx2(const float *x, float shift, int n, float *out) {
    int i = 0;
    const __m256 m = _mm256_set1_ps(shift);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(out + i, fse_exp1_avx2(_mm256_sub_ps(_mm256_loadu_ps(x + i), m)));
    for (; i < n; i++) out[i] = fse_exp1(x[i] - shift);
}
FSE_AVX2_TARGET static void fse_sigmoid1_block_avx2(const float *z, int n, float *out) {
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, fse_sigmoid1_avx2(_mm256_loadu_ps(z + i)));
    for (; i < n; i++) out[i] = fse_sigmoid1(z[i]);
}
FSE_AVX2_TARGET static void fse_divide_block_avx2(float *o, float s, int n) {
    int i = 0;
    const __m256 sv = _mm256_set1_ps(s);
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(o + i, _mm256_div_ps(_mm256_loadu_ps(o + i), sv));
    for (; i < n; i++) o[i] = o[i] / s;
}
#endif

/* out[i] = exp@1(x[i] - shift); with shift +0.0 the inputs themselves (-0.0 kept). */
static void fse_exp1_block(int isa, const float *x, float shift, int n, float *out) {
    int i = 0;
#if defined(FSE_X86)
    if (isa == FSE_ISA_AVX2) {
        fse_exp1_block_avx2(x, shift, n, out);
        return;
    }
#endif
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON) {
        const float32x4_t m = vdupq_n_f32(shift);
        for (; i + 4 <= n; i += 4) vst1q_f32(out + i, fse_exp1_neon(vsubq_f32(vld1q_f32(x + i), m)));
    }
#endif
    for (; i < n; i++) out[i] = fse_exp1(x[i] - shift);
}

static void fse_sigmoid1_block(int isa, const float *z, int n, float *out) {
    int i = 0;
#if defined(FSE_X86)
    if (isa == FSE_ISA_AVX2) {
        fse_sigmoid1_block_avx2(z, n, out);
        return;
    }
#endif
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON)
        for (; i + 4 <= n; i += 4) vst1q_f32(out + i, fse_sigmoid1_neon(vld1q_f32(z + i)));
#endif
    for (; i < n; i++) out[i] = fse_sigmoid1(z[i]);
}

/* o[i] = o[i] / s, each division correctly rounded. */
static void fse_divide_block(int isa, float *o, float s, int n) {
    int i = 0;
#if defined(FSE_X86)
    if (isa == FSE_ISA_AVX2) {
        fse_divide_block_avx2(o, s, n);
        return;
    }
#endif
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON) {
        const float32x4_t sv = vdupq_n_f32(s);
        for (; i + 4 <= n; i += 4) vst1q_f32(o + i, vdivq_f32(vld1q_f32(o + i), sv));
    }
#endif
    for (; i < n; i++) o[i] = o[i] / s;
}

/* Whether all k values are finite, by lanes where the ISA has them (no arithmetic). */
static int fse_finite_values(int isa, const float *z, int k) {
    int j = 0;
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON) {
        const float32x4_t inf = vdupq_n_f32(INFINITY);
        uint32x4_t all = vdupq_n_u32(0xFFFFFFFFu);
        for (; j + 4 <= k; j += 4) all = vandq_u32(all, vcltq_f32(vabsq_f32(vld1q_f32(z + j)), inf));
        if (vminvq_u32(all) == 0) return 0;
    }
#endif
    (void)isa;
    for (; j < k; j++)
        if (!isfinite(z[j])) return 0;
    return 1;
}

/* The largest of k finite values: any order gives the same value, up to the sign of a
   zero, which changes no z - max that exp@1 sees (exp@1 of -0.0 and +0.0 is 1). */
static float fse_max_values(int isa, const float *z, int k) {
    float m = z[0];
    int j = 1;
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON && k >= 8) {
        float32x4_t mv = vld1q_f32(z);
        for (j = 4; j + 4 <= k; j += 4) mv = vmaxq_f32(mv, vld1q_f32(z + j));
        m = vmaxvq_f32(mv);
    }
#endif
    (void)isa;
    for (; j < k; j++)
        if (z[j] > m) m = z[j];
    return m;
}

/* out = rows + bias, one rounding per value, for n rows of k folded logits; rows and out
   may be the same. Returns the first row whose logits are not finite, or -1. */
FSE_NOINLINE static int64_t fse_head_bias_body(int isa, const float *rows, int64_t n, int k,
                                               const float *bias, float *out) {
    int64_t bad = -1;
    for (int64_t r = 0; r < n; r++) {
        const float *z = rows + (size_t)r * (size_t)k;
        float *o = out + (size_t)r * (size_t)k;
        for (int j = 0; j < k; j++) o[j] = z[j] + bias[j];
        if (bad < 0 && !fse_finite_values(isa, o, k)) bad = r;
    }
    return bad;
}

/* The sums run one value at a time in index order, from +0.0, as the contract fixes them. */
static float fse_index_sum(const float *o, int k) {
    float s = 0.0f;
    for (int j = 0; j < k; j++) s = s + o[j];
    return s;
}

static void fse_softmax_row(int isa, const float *z, int k, float *o) {
    const float m = fse_max_values(isa, z, k);
    fse_exp1_block(isa, z, m, k, o);
    fse_divide_block(isa, o, fse_index_sum(o, k), k);
}

static void fse_ovr_row(int isa, const float *z, int k, float *o) {
    fse_sigmoid1_block(isa, z, k, o);
    const float s = fse_index_sum(o, k);
    if (s == 0.0f) {
        const float uniform = 1.0f / (float)k;
        for (int j = 0; j < k; j++) o[j] = uniform;
        return;
    }
    fse_divide_block(isa, o, s, k);
}

/* The probabilities of n rows of k finished logits: [1 - p, p] (2 values) for sigmoid
   (k = 1), else k values of softmax or ovr. Returns the first row whose logits are not
   finite (its outputs are undefined), or -1. */
FSE_NOINLINE static int64_t fse_head_proba_body(int isa, int link, const float *logits, int64_t n,
                                                int k, float *out) {
    int64_t bad = -1;
    const int width = link == FSE_LINK_SIGMOID ? 2 : k;
    for (int64_t r = 0; r < n; r++) {
        const float *z = logits + (size_t)r * (size_t)k;
        float *o = out + (size_t)r * (size_t)width;
        if (!fse_finite_values(isa, z, k)) {
            if (bad < 0) bad = r;
            continue;
        }
        if (link == FSE_LINK_SIGMOID) {
            const float p = fse_sigmoid1(z[0]);
            o[0] = 1.0f - p;
            o[1] = p;
        } else if (link == FSE_LINK_SOFTMAX) {
            fse_softmax_row(isa, z, k, o);
        } else {
            fse_ovr_row(isa, z, k, o);
        }
    }
    return bad;
}

/* The label index of every row: z > 0 for k = 1, else the first largest logit. Returns the
   first row whose logits are not finite, or -1. */
FSE_NOINLINE static int64_t fse_head_labels_body(const float *logits, int64_t n, int k,
                                                 int64_t *out) {
    int64_t bad = -1;
    for (int64_t r = 0; r < n; r++) {
        const float *z = logits + (size_t)r * (size_t)k;
        if (!fse_finite_values(FSE_ISA_PORTABLE, z, k)) {
            if (bad < 0) bad = r;
            out[r] = 0;
            continue;
        }
        int64_t best = 0;
        if (k == 1) {
            best = z[0] > 0.0f;
        } else {
            for (int j = 1; j < k; j++)
                if (z[j] > z[best]) best = j;
        }
        out[r] = best;
    }
    return bad;
}

FSE_NOINLINE static void fse_exp1_rows_body(int isa, const float *x, int64_t n, float *out) {
    for (int64_t i = 0; i < n; i += 1 << 20) {
        const int64_t left = n - i;
        fse_exp1_block(isa, x + i, 0.0f, left < (1 << 20) ? (int)left : 1 << 20, out + i);
    }
}

/* The entry points set the floating-point environment around a body that is never inlined
   (see _native_kernel_steps.h). */
static int64_t fse_head_bias(int isa, const float *rows, int64_t n, int k, const float *bias,
                             float *out) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t bad = fse_head_bias_body(isa, rows, n, k, bias, out);
    fse_fpenv_leave(env);
    return bad;
}
static int64_t fse_head_proba(int isa, int link, const float *logits, int64_t n, int k,
                              float *out) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t bad = fse_head_proba_body(isa, link, logits, n, k, out);
    fse_fpenv_leave(env);
    return bad;
}
static int64_t fse_head_labels(const float *logits, int64_t n, int k, int64_t *out) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t bad = fse_head_labels_body(logits, n, k, out);
    fse_fpenv_leave(env);
    return bad;
}
static void fse_exp1_rows(int isa, const float *x, int64_t n, float *out) {
    fse_fpenv env = fse_fpenv_enter();
    fse_exp1_rows_body(isa, x, n, out);
    fse_fpenv_leave(env);
}

#endif /* FSE_NATIVE_KERNEL_HEAD_H */
