/* The floating-point environment, dot16, normalize@1 and score.
   Included by _native_kernel.h. */

#ifndef FSE_NATIVE_KERNEL_STEPS_H
#define FSE_NATIVE_KERNEL_STEPS_H

#include "_native_kernel_base.h"

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

/* --- dot16, normalize@1 and score ------------------------------------------------ */

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

/* Independent rows share no accumulators. Keep each lane's FMA order and the
   dot16 reduction tree, including unpadded dimension tails. */
static void fse_norm2_portable(const float *a, const float *b, int n, float *s) {
    float p[16] = {0}, q[16] = {0};
    for (int i = 0; i < n; i++) {
        p[i & 15] = fmaf(a[i], a[i], p[i & 15]);
        q[i & 15] = fmaf(b[i], b[i], q[i & 15]);
    }
    s[0] = fse_reduce16(p);
    s[1] = fse_reduce16(q);
}

#if defined(FSE_X86)
FSE_AVX2_TARGET static void fse_norm2_avx2(const float *a, const float *b, int n, float *s) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 xa0 = _mm256_loadu_ps(a + i + 0);
        a0 = _mm256_fmadd_ps(xa0, xa0, a0);
        __m256 xb0 = _mm256_loadu_ps(b + i + 0);
        b0 = _mm256_fmadd_ps(xb0, xb0, b0);
        __m256 xa1 = _mm256_loadu_ps(a + i + 8);
        a1 = _mm256_fmadd_ps(xa1, xa1, a1);
        __m256 xb1 = _mm256_loadu_ps(b + i + 8);
        b1 = _mm256_fmadd_ps(xb1, xb1, b1);
    }
    float p[16], q[16];
    _mm256_storeu_ps(p + 0, a0);
    _mm256_storeu_ps(p + 8, a1);
    _mm256_storeu_ps(q + 0, b0);
    _mm256_storeu_ps(q + 8, b1);
    for (; i < n; i++) {
        p[i & 15] = fse_fma1_avx2(a[i], a[i], p[i & 15]);
        q[i & 15] = fse_fma1_avx2(b[i], b[i], q[i & 15]);
    }
    s[0] = fse_reduce16(p);
    s[1] = fse_reduce16(q);
}
#endif

#if defined(FSE_NEON)
static void fse_norm2_neon(const float *a, const float *b, int n, float *s) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    float32x4_t b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0), b2 = vdupq_n_f32(0), b3 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t xa0 = vld1q_f32(a + i + 0);
        a0 = vfmaq_f32(a0, xa0, xa0);
        float32x4_t xb0 = vld1q_f32(b + i + 0);
        b0 = vfmaq_f32(b0, xb0, xb0);
        float32x4_t xa1 = vld1q_f32(a + i + 4);
        a1 = vfmaq_f32(a1, xa1, xa1);
        float32x4_t xb1 = vld1q_f32(b + i + 4);
        b1 = vfmaq_f32(b1, xb1, xb1);
        float32x4_t xa2 = vld1q_f32(a + i + 8);
        a2 = vfmaq_f32(a2, xa2, xa2);
        float32x4_t xb2 = vld1q_f32(b + i + 8);
        b2 = vfmaq_f32(b2, xb2, xb2);
        float32x4_t xa3 = vld1q_f32(a + i + 12);
        a3 = vfmaq_f32(a3, xa3, xa3);
        float32x4_t xb3 = vld1q_f32(b + i + 12);
        b3 = vfmaq_f32(b3, xb3, xb3);
    }
    float p[16], q[16];
    vst1q_f32(p + 0, a0);
    vst1q_f32(p + 4, a1);
    vst1q_f32(p + 8, a2);
    vst1q_f32(p + 12, a3);
    vst1q_f32(q + 0, b0);
    vst1q_f32(q + 4, b1);
    vst1q_f32(q + 8, b2);
    vst1q_f32(q + 12, b3);
    for (; i < n; i++) {
        p[i & 15] = fmaf(a[i], a[i], p[i & 15]);
        q[i & 15] = fmaf(b[i], b[i], q[i & 15]);
    }
    s[0] = fse_reduce16(p);
    s[1] = fse_reduce16(q);
}
#endif

static void fse_norm2(int isa, const float *a, const float *b, int n, float *s) {
#if defined(FSE_X86)
    if (isa == FSE_ISA_AVX2) { fse_norm2_avx2(a, b, n, s); return; }
#endif
#if defined(FSE_NEON)
    if (isa == FSE_ISA_NEON) { fse_norm2_neon(a, b, n, s); return; }
#endif
    fse_norm2_portable(a, b, n, s);
}

/* normalize@1 in place. Returns the first nonzero row whose sum of squares underflows to
   0 or overflows in *unnormalizable (else -1), and the first non-finite row, or -1. */
FSE_NOINLINE static int64_t fse_normalize_body(int isa, float *rows, int64_t n, int dim, int64_t *unnormalizable) {
    int64_t bad = -1;
    *unnormalizable = -1;
    float sums[2];
    for (int64_t r = 0; r < n; r++) {
        float *row = rows + (size_t)r * (size_t)dim;
        if ((r & 1) == 0) {
            if (r + 1 < n) fse_norm2(isa, row, row + dim, dim, sums);
            else sums[0] = fse_dot16(isa, row, row, dim);
        }
        float s = sums[r & 1];
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

#endif /* FSE_NATIVE_KERNEL_STEPS_H */
