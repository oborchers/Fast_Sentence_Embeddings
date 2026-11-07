/* The int8 prefilter (docs/architecture/numerical-contract.md, "int8 prefilter (no new
   arithmetic)"), as fse._reference_prefilter: the slack of a query and one pass over a block
   of stored int8 rows. Included by _native_kernel.h, after _native_kernel_int8.h (the integer
   sums) and _native_kernel_steps.h (the floating-point environment).

   Every float64 operation is written in the grouping of the reference and runs in the
   environment of fse_fpenv_enter (round to nearest, no flush to zero, no contraction), so the
   bits are those of the reference on every variant. */

#ifndef FSE_NATIVE_KERNEL_PREFILTER_H
#define FSE_NATIVE_KERNEL_PREFILTER_H

#include "_native_kernel_base.h"
#include "_native_kernel_int8.h"
#include "_native_kernel_steps.h"

enum { FSE_PF_OK = 0, FSE_PF_OVERFLOW = 1, FSE_PF_UNSAFE = 2 };

/* The fixed constants of the proof; only the maintainer changes them. */
#define FSE_PF_SLACK 1.003
#define FSE_PF_ABSOLUTE (1.0 / 16384.0)               /* 2^-14 */
#define FSE_PF_QUERY_NORM2_MAX 1.002
#define FSE_PF_ROW_NORM2_MAX (1.0 + 1.0 / 1048576.0)  /* 1 + 2^-20 */

/* E = 1.003 * eps + 2^-14 and whether ||u||^2 is finite and at most 1.002, per query; n2
   and eps are float64 sums in index order, s * code is exact and the difference rounds once. */
FSE_NOINLINE static void fse_int8_bounds_body(int64_t q, int dim, const float *units,
                                              const int8_t *qcodes, const float *qscales,
                                              double *slack, int32_t *ok) {
    for (int64_t j = 0; j < q; j++) {
        const float *u = units + (size_t)j * (size_t)dim;
        const int8_t *c = qcodes + (size_t)j * (size_t)dim;
        const double scale = (double)qscales[j];
        double n2 = 0.0, squares = 0.0;
        for (int i = 0; i < dim; i++) {
            const double x = (double)u[i];
            const double e = x - scale * (double)c[i];
            n2 += x * x;
            squares += e * e;
        }
        slack[j] = FSE_PF_SLACK * sqrt(squares) + FSE_PF_ABSOLUTE;
        ok[j] = isfinite(n2) && n2 <= FSE_PF_QUERY_NORM2_MAX;
    }
}
static void fse_int8_bounds(int64_t q, int dim, const float *units, const int8_t *qcodes,
                            const float *qscales, double *slack, int32_t *ok) {
    fse_fpenv env = fse_fpenv_enter();
    fse_int8_bounds_body(q, dim, units, qcodes, qscales, slack, ok);
    fse_fpenv_leave(env);
}

/* The state of one pass. The caller owns every array (nothing here allocates):
   heap holds q min-heaps of k estimates, heap_n their sizes, thr the floor
   fl(max(T_in, heap minimum) - E) of each query, rows and upper the kept candidates of each
   query (cap per query), count how many. */
typedef struct {
    int q, k;
    int64_t cap;
    const float *qscales;
    const double *slack, *floor_in;
    double *heap;
    int32_t *heap_n;
    double *thr;
    int64_t *rows, *count;
    double *upper;
    int overflow;
} fse_pf;

static inline int fse_pf_row_safe(float inverse, int32_t squares) {
    const double inv = (double)inverse;
    return isfinite(inv) && inv >= 0.0 && (inv * inv) * (double)squares <= FSE_PF_ROW_NORM2_MAX;
}

static inline void fse_pf_sift_down(double *h, int n, int i, double a) {
    for (;;) {
        const int64_t c0 = 2 * (int64_t)i + 1; /* wide: k may exceed 2^30 */
        if (c0 >= n) break;
        int c = (int)c0;
        if (c + 1 < n && h[c + 1] < h[c]) c++;
        if (h[c] >= a) break;
        h[i] = h[c];
        i = c;
    }
    h[i] = a;
}

/* Keep what stays against the current floor, in order. */
static inline void fse_pf_compact(fse_pf *st, int j) {
    int64_t *rows = st->rows + (size_t)j * (size_t)st->cap;
    double *upper = st->upper + (size_t)j * (size_t)st->cap;
    int64_t m = 0;
    for (int64_t c = 0; c < st->count[j]; c++)
        if (upper[c] >= st->thr[j]) {
            rows[m] = rows[c];
            upper[m++] = upper[c];
        }
    st->count[j] = m;
}

/* The estimate a of one row for query j: it enters the heap of the k largest, then the
   row is kept if fl(a + E) reaches the floor; a full list drops what the floor now cuts, and
   if it is still full the pass overflows. */
static inline void fse_pf_see(fse_pf *st, int j, int64_t row, double a) {
    double *h = st->heap + (size_t)j * (size_t)st->k;
    const double slack = st->slack[j];
    int32_t n = st->heap_n[j];
    if (n < st->k) {
        int i = n++;
        while (i > 0 && h[(i - 1) / 2] > a) {
            h[i] = h[(i - 1) / 2];
            i = (i - 1) / 2;
        }
        h[i] = a;
        st->heap_n[j] = n;
    } else if (a > h[0]) {
        fse_pf_sift_down(h, n, 0, a);
    }
    if (n == st->k) {
        const double running = h[0] > st->floor_in[j] ? h[0] : st->floor_in[j];
        st->thr[j] = running - slack;
    }
    const double top = a + slack;
    if (top < st->thr[j]) return;
    if (st->count[j] == st->cap) {
        fse_pf_compact(st, j);
        if (st->count[j] == st->cap) {
            st->overflow = 1;
            return;
        }
    }
    st->rows[(size_t)j * (size_t)st->cap + (size_t)st->count[j]] = row;
    st->upper[(size_t)j * (size_t)st->cap + (size_t)st->count[j]++] = top;
}

/* The heaps of the k largest, descending, padded with -inf (heapsort of a min-heap). */
static void fse_pf_finish(fse_pf *st) {
    for (int j = 0; j < st->q; j++) {
        double *h = st->heap + (size_t)j * (size_t)st->k;
        for (int end = st->heap_n[j] - 1; end > 0; end--) {
            const double last = h[end];
            h[end] = h[0];
            fse_pf_sift_down(h, end, 0, last);
        }
        for (int i = st->heap_n[j]; i < st->k; i++) h[i] = -INFINITY;
    }
}

/* a = (s * D) * inv. The sum with +0.0 turns -0.0 into +0.0 (D negative with a scale or an
   inverse norm of 0): the sign of a zero estimate carries nothing, and with it every tie of
   equal values has equal bits, so no heap layout can show in the output. */
static inline double fse_pf_estimate(int32_t d, float qscale, float inverse) {
    return ((double)d * (double)qscale) * (double)inverse + 0.0;
}

/* The pass over rows [0, n) of one variant. The first query tile of a group of four rows
   also gives their sums of squares, and a group is checked before any of its rows is seen,
   so the first unsafe row of the block is found whatever else happens. After an overflow the
   rest of the rows are only checked for safety. Returns FSE_PF_OK, FSE_PF_OVERFLOW or
   FSE_PF_UNSAFE (with the row). */
#define FSE_I8_DRIVER(NAME, ATTR, V)                                                             \
    ATTR FSE_NOINLINE static int NAME(fse_pf *st, const int8_t *codes, const float *inv,         \
                                      int64_t n, int dim, const int8_t *qcodes,                  \
                                      int64_t *unsafe_row) {                                     \
        const size_t stride = (size_t)dim;                                                       \
        const int q = st->q;                                                                     \
        for (int j = 0; j < q; j++) {                                                            \
            st->heap_n[j] = 0;                                                                   \
            st->count[j] = 0;                                                                    \
            st->thr[j] = st->floor_in[j] - st->slack[j];                                         \
        }                                                                                        \
        int64_t r = 0;                                                                           \
        for (; r + 4 <= n && !st->overflow; r += 4) {                                            \
            const int8_t *x = codes + (size_t)r * stride;                                        \
            int32_t d[4][2], d1[4], s[4];                                                        \
            for (int j = 0; j < q; j += 2) {                                                     \
                const int8_t *q0 = qcodes + (size_t)j * stride;                                  \
                const int w = j + 2 <= q ? 2 : 1;                                                \
                if (w == 2) {                                                                    \
                    fse_i8_tile2_##V(x, stride, q0, q0 + stride, dim, d, j ? NULL : s);          \
                } else {                                                                         \
                    fse_i8_tile1_##V(x, stride, q0, dim, d1, j ? NULL : s);                      \
                    for (int a = 0; a < 4; a++) d[a][0] = d1[a];                                 \
                }                                                                                \
                if (!j)                                                                          \
                    for (int a = 0; a < 4; a++)                                                  \
                        if (!fse_pf_row_safe(inv[r + a], s[a])) {                                \
                            *unsafe_row = r + a;                                                 \
                            return FSE_PF_UNSAFE;                                                \
                        }                                                                        \
                for (int a = 0; a < 4; a++)                                                      \
                    for (int b = 0; b < w; b++)                                                  \
                        fse_pf_see(st, j + b, r + a,                                             \
                                   fse_pf_estimate(d[a][b], st->qscales[j + b], inv[r + a]));    \
            }                                                                                    \
        }                                                                                        \
        for (; r < n && !st->overflow; r++) {                                                    \
            int32_t s = 0;                                                                       \
            const int8_t *x = codes + (size_t)r * stride;                                        \
            for (int j = 0; j < q; j++) {                                                        \
                const int32_t d = fse_i8_dot_##V(x, qcodes + (size_t)j * stride, dim, j ? NULL : &s); \
                if (!j && !fse_pf_row_safe(inv[r], s)) {                                         \
                    *unsafe_row = r;                                                             \
                    return FSE_PF_UNSAFE;                                                        \
                }                                                                                \
                fse_pf_see(st, j, r, fse_pf_estimate(d, st->qscales[j], inv[r]));                \
            }                                                                                    \
        }                                                                                        \
        for (; r < n; r++)                                                                       \
            if (!fse_pf_row_safe(inv[r], fse_i8_sq_##V(codes + (size_t)r * stride, dim))) {      \
                *unsafe_row = r;                                                                 \
                return FSE_PF_UNSAFE;                                                            \
            }                                                                                    \
        return st->overflow ? FSE_PF_OVERFLOW : FSE_PF_OK;                                       \
    }

FSE_I8_DRIVER(fse_int8_pass_portable, , portable)
#if defined(FSE_X86)
FSE_I8_DRIVER(fse_int8_pass_avx2, FSE_AVX2_TARGET, avx2)
#endif
#if defined(FSE_I8_NEON_SDOT)
FSE_I8_DRIVER(fse_int8_pass_neon, FSE_DOTPROD_TARGET, neon)
#endif

/* One pass over a block of n rows of dim codes, for q queries (codes qcodes, scales
   qscales, slack E and incoming threshold floor_in per query), keeping the k largest
   estimates per query. Returns FSE_PF_OK (heap, rows, upper and count are the result),
   FSE_PF_OVERFLOW or FSE_PF_UNSAFE (*unsafe_row is the first unsafe row; nothing else is
   valid), or -1 for a variant that is not built. */
static int fse_int8_pass(int variant, const int8_t *codes, const float *inverse, int64_t n,
                         int dim, const int8_t *qcodes, const float *qscales, int q, int k,
                         const double *slack, const double *floor_in, double *heap,
                         int32_t *heap_n, double *thr, int64_t *rows, double *upper, int64_t cap,
                         int64_t *count, int64_t *unsafe_row) {
    fse_pf st = {q, k, cap, qscales, slack, floor_in, heap, heap_n, thr, rows, count, upper, 0};
    int status = -1;
    fse_fpenv env = fse_fpenv_enter();
    if (variant == FSE_I8_PORTABLE)
        status = fse_int8_pass_portable(&st, codes, inverse, n, dim, qcodes, unsafe_row);
#if defined(FSE_X86)
    else if (variant == FSE_I8_AVX2)
        status = fse_int8_pass_avx2(&st, codes, inverse, n, dim, qcodes, unsafe_row);
#endif
#if defined(FSE_I8_NEON_SDOT)
    else if (variant == FSE_I8_NEON)
        status = fse_int8_pass_neon(&st, codes, inverse, n, dim, qcodes, unsafe_row);
#endif
    if (status == FSE_PF_OK) fse_pf_finish(&st);
    fse_fpenv_leave(env);
    return status;
}

#endif /* FSE_NATIVE_KERNEL_PREFILTER_H */
