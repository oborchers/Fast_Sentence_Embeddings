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

static int fse_finite_values(const float *z, int k) {
    for (int j = 0; j < k; j++)
        if (!isfinite(z[j])) return 0;
    return 1;
}

/* out = rows + bias, one rounding per value, for n rows of k folded logits; rows and out
   may be the same. Returns the first row whose logits are not finite, or -1. */
FSE_NOINLINE static int64_t fse_head_bias_body(const float *rows, int64_t n, int k,
                                               const float *bias, float *out) {
    int64_t bad = -1;
    for (int64_t r = 0; r < n; r++) {
        const float *z = rows + (size_t)r * (size_t)k;
        float *o = out + (size_t)r * (size_t)k;
        for (int j = 0; j < k; j++) o[j] = z[j] + bias[j];
        if (bad < 0 && !fse_finite_values(o, k)) bad = r;
    }
    return bad;
}

static void fse_softmax_row(const float *z, int k, float *o) {
    float m = z[0];
    for (int j = 1; j < k; j++)
        if (z[j] > m) m = z[j];
    float s = 0.0f;
    for (int j = 0; j < k; j++) {
        o[j] = fse_exp1(z[j] - m);
        s = s + o[j];
    }
    for (int j = 0; j < k; j++) o[j] = o[j] / s;
}

static void fse_ovr_row(const float *z, int k, float *o) {
    float s = 0.0f;
    for (int j = 0; j < k; j++) {
        o[j] = fse_sigmoid1(z[j]);
        s = s + o[j];
    }
    if (s == 0.0f) {
        const float uniform = 1.0f / (float)k;
        for (int j = 0; j < k; j++) o[j] = uniform;
        return;
    }
    for (int j = 0; j < k; j++) o[j] = o[j] / s;
}

/* The probabilities of n rows of k finished logits: [1 - p, p] (2 values) for sigmoid
   (k = 1), else k values of softmax or ovr. Returns the first row whose logits are not
   finite (its outputs are undefined), or -1. */
FSE_NOINLINE static int64_t fse_head_proba_body(int link, const float *logits, int64_t n, int k,
                                                float *out) {
    int64_t bad = -1;
    const int width = link == FSE_LINK_SIGMOID ? 2 : k;
    for (int64_t r = 0; r < n; r++) {
        const float *z = logits + (size_t)r * (size_t)k;
        float *o = out + (size_t)r * (size_t)width;
        if (!fse_finite_values(z, k)) {
            if (bad < 0) bad = r;
            continue;
        }
        if (link == FSE_LINK_SIGMOID) {
            const float p = fse_sigmoid1(z[0]);
            o[0] = 1.0f - p;
            o[1] = p;
        } else if (link == FSE_LINK_SOFTMAX) {
            fse_softmax_row(z, k, o);
        } else {
            fse_ovr_row(z, k, o);
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
        if (!fse_finite_values(z, k)) {
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

FSE_NOINLINE static void fse_exp1_rows_body(const float *x, int64_t n, float *out) {
    for (int64_t i = 0; i < n; i++) out[i] = fse_exp1(x[i]);
}

/* The entry points set the floating-point environment around a body that is never inlined
   (see _native_kernel_steps.h). */
static int64_t fse_head_bias(const float *rows, int64_t n, int k, const float *bias, float *out) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t bad = fse_head_bias_body(rows, n, k, bias, out);
    fse_fpenv_leave(env);
    return bad;
}
static int64_t fse_head_proba(int link, const float *logits, int64_t n, int k, float *out) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t bad = fse_head_proba_body(link, logits, n, k, out);
    fse_fpenv_leave(env);
    return bad;
}
static int64_t fse_head_labels(const float *logits, int64_t n, int k, int64_t *out) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t bad = fse_head_labels_body(logits, n, k, out);
    fse_fpenv_leave(env);
    return bad;
}
static void fse_exp1_rows(const float *x, int64_t n, float *out) {
    fse_fpenv env = fse_fpenv_enter();
    fse_exp1_rows_body(x, n, out);
    fse_fpenv_leave(env);
}

#endif /* FSE_NATIVE_KERNEL_HEAD_H */
