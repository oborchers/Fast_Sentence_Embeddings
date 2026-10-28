/* The search of the numerical contract: inverse norms, the cosine of every stored row with
   many queries and the linear score with many weight vectors. Included by
   _native_kernel.h, after _native_kernel_steps.h, whose dot16 and floating-point
   environment it uses. */

#ifndef FSE_NATIVE_KERNEL_SEARCH_H
#define FSE_NATIVE_KERNEL_SEARCH_H

#include "_native_kernel_base.h"
#include "_native_kernel_steps.h"

/* Stored row r (float32, float16 or int8) as float32, exactly: the row itself, or its
   values decoded into scratch. */
static inline const float *fse_stored_row(int dtype, const void *stored, int64_t r, int dim,
                                          float *scratch) {
    const size_t item = dtype == FSE_F32 ? 4 : dtype == FSE_F16 ? 2 : 1;
    const char *row = (const char *)stored + (size_t)r * (size_t)dim * item;
    if (dtype == FSE_F32) return (const float *)row;
    if (dtype == FSE_F16)
        for (int i = 0; i < dim; i++) scratch[i] = fse_half_table[((const uint16_t *)row)[i]];
    else
        for (int i = 0; i < dim; i++) scratch[i] = (float)((const int8_t *)row)[i];
    return scratch;
}

/* float32(1 / sqrt(s)) per row, s the float64 sum of squares in index order; 0 for a zero
   row. Returns the first row whose inverse norm is beyond float32, or -1. */
FSE_NOINLINE static int64_t fse_inverse_norms_body(int dtype, const void *stored, int64_t n,
                                                   int dim, float *out, float *scratch) {
    int64_t bad = -1;
    for (int64_t r = 0; r < n; r++) {
        const float *x = fse_stored_row(dtype, stored, r, dim, scratch);
        double s = 0.0;
        for (int i = 0; i < dim; i++) s += (double)x[i] * (double)x[i];
        out[r] = s > 0.0 ? (float)(1.0 / sqrt(s)) : 0.0f;
        if (bad < 0 && !isfinite(out[r])) bad = r;
    }
    return bad;
}

/* out[r * q + j] = dot16(row r, units[j]) * inverse[r], for unit queries (normalized by
   normalize@1 before). Returns the first non-finite pair in (query, row) order as
   query * n + row, or -1. */
FSE_NOINLINE static int64_t fse_cosines_body(int isa, int dtype, const void *stored,
                                             const float *inverse, int64_t n, int dim,
                                             const float *units, int q, float *out,
                                             float *scratch) {
    int64_t bad = -1;
    for (int64_t r = 0; r < n; r++) {
        const float *x = fse_stored_row(dtype, stored, r, dim, scratch);
        float *o = out + (size_t)r * (size_t)q;
        for (int j = 0; j < q; j++) {
            o[j] = fse_dot16(isa, x, units + (size_t)j * (size_t)dim, dim) * inverse[r];
            if (!isfinite(o[j]) && (bad < 0 || (int64_t)j * n + r < bad)) bad = (int64_t)j * n + r;
        }
    }
    return bad;
}

/* out[r * m + j] = dot16(row r, weights[j]) * scale[r] + biases[j], two roundings, the scale
   only for int8. Returns the first non-finite pair in (vector, row) order as j * n + r, or
   -1. */
FSE_NOINLINE static int64_t fse_scores_body(int isa, int dtype, const void *stored,
                                            const float *scales, int64_t n, int dim,
                                            const float *weights, const float *biases, int m,
                                            float *out, float *scratch) {
    int64_t bad = -1;
    for (int64_t r = 0; r < n; r++) {
        const float *x = fse_stored_row(dtype, stored, r, dim, scratch);
        float *o = out + (size_t)r * (size_t)m;
        for (int j = 0; j < m; j++) {
            float dot = fse_dot16(isa, x, weights + (size_t)j * (size_t)dim, dim);
            if (scales) dot = dot * scales[r];
            o[j] = dot + biases[j];
            if (!isfinite(o[j]) && (bad < 0 || (int64_t)j * n + r < bad)) bad = (int64_t)j * n + r;
        }
    }
    return bad;
}

/* The entry points set the floating-point environment around a body that is never inlined
   (see _native_kernel_steps.h). */
static int64_t fse_inverse_norms(int dtype, const void *stored, int64_t n, int dim, float *out,
                                 float *scratch) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t bad = fse_inverse_norms_body(dtype, stored, n, dim, out, scratch);
    fse_fpenv_leave(env);
    return bad;
}
static int64_t fse_cosines(int isa, int dtype, const void *stored, const float *inverse, int64_t n,
                           int dim, const float *units, int q, float *out, float *scratch) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t bad = fse_cosines_body(isa, dtype, stored, inverse, n, dim, units, q, out, scratch);
    fse_fpenv_leave(env);
    return bad;
}
static int64_t fse_scores(int isa, int dtype, const void *stored, const float *scales, int64_t n,
                          int dim, const float *weights, const float *biases, int m, float *out,
                          float *scratch) {
    fse_fpenv env = fse_fpenv_enter();
    int64_t bad = fse_scores_body(isa, dtype, stored, scales, n, dim, weights, biases, m, out, scratch);
    fse_fpenv_leave(env);
    return bad;
}

#endif /* FSE_NATIVE_KERNEL_SEARCH_H */
