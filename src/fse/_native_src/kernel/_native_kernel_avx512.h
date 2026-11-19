/* AVX-512 float32 pooling. Every active lane keeps its ordered explicit FMA chain.
   Both layouts reuse the existing builders and prefetch helper without modification. */
#ifndef FSE_NATIVE_KERNEL_AVX512_H
#define FSE_NATIVE_KERNEL_AVX512_H
#if defined(FSE_AVX512)

FSE_INLINE FSE_AVX512_TARGET void fse_fma_f32_avx512(float *acc, const void *row, float f, int dim) {
    const float *x = (const float *)row;
    const __m512 fv = _mm512_set1_ps(f);
    int i = 0;
    for (; i + 16 <= dim; i += 16)
        _mm512_storeu_ps(acc + i, _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + i), _mm512_loadu_ps(acc + i)));
    if (i < dim) {
        const __mmask16 m = (__mmask16)((1u << (dim - i)) - 1u);
        __m512 a = _mm512_maskz_loadu_ps(m, acc + i);
        a = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m, x + i), a);
        _mm512_mask_storeu_ps(acc + i, m, a);
    }
}
FSE_DEFINE_LOOP(fse_loop_f32_avx512_rows, FSE_AVX512_TARGET, fse_fma_f32_avx512, 0)

#if FSE_AVX512_LAYOUT == 0
/* Layout A: eight registers per 128 values, then 64/32/16 and a zero-masked tail. */
FSE_INLINE FSE_AVX512_TARGET void fse_tile_f32_avx512(float *acc, int dim,
                              const char *const *row, const float *f, int n, int first) {
    int i = 0;
    for (; i + 128 <= dim; i += 128) {
        __m512 a0 = first ? _mm512_setzero_ps() : _mm512_loadu_ps(acc + i + 0);
        __m512 a1 = first ? _mm512_setzero_ps() : _mm512_loadu_ps(acc + i + 16);
        __m512 a2 = first ? _mm512_setzero_ps() : _mm512_loadu_ps(acc + i + 32);
        __m512 a3 = first ? _mm512_setzero_ps() : _mm512_loadu_ps(acc + i + 48);
        __m512 a4 = first ? _mm512_setzero_ps() : _mm512_loadu_ps(acc + i + 64);
        __m512 a5 = first ? _mm512_setzero_ps() : _mm512_loadu_ps(acc + i + 80);
        __m512 a6 = first ? _mm512_setzero_ps() : _mm512_loadu_ps(acc + i + 96);
        __m512 a7 = first ? _mm512_setzero_ps() : _mm512_loadu_ps(acc + i + 112);
        for (int q = 0; q < n; q++) {
            const float *x = (const float *)row[q] + i;
            const __m512 fv = _mm512_set1_ps(f[q]);
            a0 = _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + 0), a0);
            a1 = _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + 16), a1);
            a2 = _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + 32), a2);
            a3 = _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + 48), a3);
            a4 = _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + 64), a4);
            a5 = _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + 80), a5);
            a6 = _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + 96), a6);
            a7 = _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + 112), a7);
        }
        _mm512_storeu_ps(acc + i + 0, a0);
        _mm512_storeu_ps(acc + i + 16, a1);
        _mm512_storeu_ps(acc + i + 32, a2);
        _mm512_storeu_ps(acc + i + 48, a3);
        _mm512_storeu_ps(acc + i + 64, a4);
        _mm512_storeu_ps(acc + i + 80, a5);
        _mm512_storeu_ps(acc + i + 96, a6);
        _mm512_storeu_ps(acc + i + 112, a7);
    }
    if ( i + 64 <= dim) {
        __m512 a0 = first ? _mm512_setzero_ps() : _mm512_loadu_ps(acc + i + 0);
        __m512 a1 = first ? _mm512_setzero_ps() : _mm512_loadu_ps(acc + i + 16);
        __m512 a2 = first ? _mm512_setzero_ps() : _mm512_loadu_ps(acc + i + 32);
        __m512 a3 = first ? _mm512_setzero_ps() : _mm512_loadu_ps(acc + i + 48);
        for (int q = 0; q < n; q++) {
            const float *x = (const float *)row[q] + i;
            const __m512 fv = _mm512_set1_ps(f[q]);
            a0 = _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + 0), a0);
            a1 = _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + 16), a1);
            a2 = _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + 32), a2);
            a3 = _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + 48), a3);
        }
        _mm512_storeu_ps(acc + i + 0, a0);
        _mm512_storeu_ps(acc + i + 16, a1);
        _mm512_storeu_ps(acc + i + 32, a2);
        _mm512_storeu_ps(acc + i + 48, a3);
        i += 64;
    }
    if ( i + 32 <= dim) {
        __m512 a0 = first ? _mm512_setzero_ps() : _mm512_loadu_ps(acc + i + 0);
        __m512 a1 = first ? _mm512_setzero_ps() : _mm512_loadu_ps(acc + i + 16);
        for (int q = 0; q < n; q++) {
            const float *x = (const float *)row[q] + i;
            const __m512 fv = _mm512_set1_ps(f[q]);
            a0 = _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + 0), a0);
            a1 = _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + 16), a1);
        }
        _mm512_storeu_ps(acc + i + 0, a0);
        _mm512_storeu_ps(acc + i + 16, a1);
        i += 32;
    }
    if ( i + 16 <= dim) {
        __m512 a0 = first ? _mm512_setzero_ps() : _mm512_loadu_ps(acc + i + 0);
        for (int q = 0; q < n; q++) {
            const float *x = (const float *)row[q] + i;
            const __m512 fv = _mm512_set1_ps(f[q]);
            a0 = _mm512_fmadd_ps(fv, _mm512_loadu_ps(x + 0), a0);
        }
        _mm512_storeu_ps(acc + i + 0, a0);
        i += 16;
    }
    if (i < dim) {
        const __mmask16 m = (__mmask16)((1u << (dim - i)) - 1u);
        __m512 a = first ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m, acc + i);
        for (int q = 0; q < n; q++)
            a = _mm512_fmadd_ps(_mm512_set1_ps(f[q]),
                               _mm512_maskz_loadu_ps(m, (const float *)row[q] + i), a);
        _mm512_mask_storeu_ps(acc + i, m, a);
    }
}
#elif FSE_AVX512_LAYOUT == 1
/* Layout B: sixteen accumulators per segment, with masked partial vectors. A zero mask
   skips the address expression too: no pointer is formed outside the source row. */
FSE_INLINE FSE_AVX512_TARGET void fse_tile_f32_avx512(float *acc, int dim,
                              const char *const *row, const float *f, int n, int first) {
    for (int i = 0; i < dim; i += 256) {
        const int left0 = dim - i - 0;
        const __mmask16 m0 = left0 >= 16 ? 0xffff : left0 > 0 ? (__mmask16)((1u << left0) - 1u) : 0;
        __m512 a0 = first || !m0 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m0, acc + i + 0);
        const int left1 = dim - i - 16;
        const __mmask16 m1 = left1 >= 16 ? 0xffff : left1 > 0 ? (__mmask16)((1u << left1) - 1u) : 0;
        __m512 a1 = first || !m1 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m1, acc + i + 16);
        const int left2 = dim - i - 32;
        const __mmask16 m2 = left2 >= 16 ? 0xffff : left2 > 0 ? (__mmask16)((1u << left2) - 1u) : 0;
        __m512 a2 = first || !m2 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m2, acc + i + 32);
        const int left3 = dim - i - 48;
        const __mmask16 m3 = left3 >= 16 ? 0xffff : left3 > 0 ? (__mmask16)((1u << left3) - 1u) : 0;
        __m512 a3 = first || !m3 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m3, acc + i + 48);
        const int left4 = dim - i - 64;
        const __mmask16 m4 = left4 >= 16 ? 0xffff : left4 > 0 ? (__mmask16)((1u << left4) - 1u) : 0;
        __m512 a4 = first || !m4 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m4, acc + i + 64);
        const int left5 = dim - i - 80;
        const __mmask16 m5 = left5 >= 16 ? 0xffff : left5 > 0 ? (__mmask16)((1u << left5) - 1u) : 0;
        __m512 a5 = first || !m5 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m5, acc + i + 80);
        const int left6 = dim - i - 96;
        const __mmask16 m6 = left6 >= 16 ? 0xffff : left6 > 0 ? (__mmask16)((1u << left6) - 1u) : 0;
        __m512 a6 = first || !m6 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m6, acc + i + 96);
        const int left7 = dim - i - 112;
        const __mmask16 m7 = left7 >= 16 ? 0xffff : left7 > 0 ? (__mmask16)((1u << left7) - 1u) : 0;
        __m512 a7 = first || !m7 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m7, acc + i + 112);
        const int left8 = dim - i - 128;
        const __mmask16 m8 = left8 >= 16 ? 0xffff : left8 > 0 ? (__mmask16)((1u << left8) - 1u) : 0;
        __m512 a8 = first || !m8 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m8, acc + i + 128);
        const int left9 = dim - i - 144;
        const __mmask16 m9 = left9 >= 16 ? 0xffff : left9 > 0 ? (__mmask16)((1u << left9) - 1u) : 0;
        __m512 a9 = first || !m9 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m9, acc + i + 144);
        const int left10 = dim - i - 160;
        const __mmask16 m10 = left10 >= 16 ? 0xffff : left10 > 0 ? (__mmask16)((1u << left10) - 1u) : 0;
        __m512 a10 = first || !m10 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m10, acc + i + 160);
        const int left11 = dim - i - 176;
        const __mmask16 m11 = left11 >= 16 ? 0xffff : left11 > 0 ? (__mmask16)((1u << left11) - 1u) : 0;
        __m512 a11 = first || !m11 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m11, acc + i + 176);
        const int left12 = dim - i - 192;
        const __mmask16 m12 = left12 >= 16 ? 0xffff : left12 > 0 ? (__mmask16)((1u << left12) - 1u) : 0;
        __m512 a12 = first || !m12 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m12, acc + i + 192);
        const int left13 = dim - i - 208;
        const __mmask16 m13 = left13 >= 16 ? 0xffff : left13 > 0 ? (__mmask16)((1u << left13) - 1u) : 0;
        __m512 a13 = first || !m13 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m13, acc + i + 208);
        const int left14 = dim - i - 224;
        const __mmask16 m14 = left14 >= 16 ? 0xffff : left14 > 0 ? (__mmask16)((1u << left14) - 1u) : 0;
        __m512 a14 = first || !m14 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m14, acc + i + 224);
        const int left15 = dim - i - 240;
        const __mmask16 m15 = left15 >= 16 ? 0xffff : left15 > 0 ? (__mmask16)((1u << left15) - 1u) : 0;
        __m512 a15 = first || !m15 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(m15, acc + i + 240);
        for (int q = 0; q < n; q++) {
            const float *x = (const float *)row[q] + i;
            const __m512 fv = _mm512_set1_ps(f[q]);
            if (m0) a0 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m0, x + 0), a0);
            if (m1) a1 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m1, x + 16), a1);
            if (m2) a2 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m2, x + 32), a2);
            if (m3) a3 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m3, x + 48), a3);
            if (m4) a4 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m4, x + 64), a4);
            if (m5) a5 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m5, x + 80), a5);
            if (m6) a6 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m6, x + 96), a6);
            if (m7) a7 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m7, x + 112), a7);
            if (m8) a8 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m8, x + 128), a8);
            if (m9) a9 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m9, x + 144), a9);
            if (m10) a10 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m10, x + 160), a10);
            if (m11) a11 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m11, x + 176), a11);
            if (m12) a12 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m12, x + 192), a12);
            if (m13) a13 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m13, x + 208), a13);
            if (m14) a14 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m14, x + 224), a14);
            if (m15) a15 = _mm512_fmadd_ps(fv, _mm512_maskz_loadu_ps(m15, x + 240), a15);
        }
        if (m0) _mm512_mask_storeu_ps(acc + i + 0, m0, a0);
        if (m1) _mm512_mask_storeu_ps(acc + i + 16, m1, a1);
        if (m2) _mm512_mask_storeu_ps(acc + i + 32, m2, a2);
        if (m3) _mm512_mask_storeu_ps(acc + i + 48, m3, a3);
        if (m4) _mm512_mask_storeu_ps(acc + i + 64, m4, a4);
        if (m5) _mm512_mask_storeu_ps(acc + i + 80, m5, a5);
        if (m6) _mm512_mask_storeu_ps(acc + i + 96, m6, a6);
        if (m7) _mm512_mask_storeu_ps(acc + i + 112, m7, a7);
        if (m8) _mm512_mask_storeu_ps(acc + i + 128, m8, a8);
        if (m9) _mm512_mask_storeu_ps(acc + i + 144, m9, a9);
        if (m10) _mm512_mask_storeu_ps(acc + i + 160, m10, a10);
        if (m11) _mm512_mask_storeu_ps(acc + i + 176, m11, a11);
        if (m12) _mm512_mask_storeu_ps(acc + i + 192, m12, a12);
        if (m13) _mm512_mask_storeu_ps(acc + i + 208, m13, a13);
        if (m14) _mm512_mask_storeu_ps(acc + i + 224, m14, a14);
        if (m15) _mm512_mask_storeu_ps(acc + i + 240, m15, a15);
    }
}
#else
#error "unknown AVX-512 tile layout"
#endif

FSE_INLINE FSE_AVX512_TARGET int64_t fse_loop_f32_avx512_tiles(const fse_params *p, const fse_batch *b,
                         float *out, int64_t *counts, int finish, size_t cap, int known_only) {
    const int dim = p->dim;
    const int K = cap < FSE_TILE_MAX ? (int)cap : FSE_TILE_MAX;
    const char *row[FSE_TILE_MAX];
    float f[FSE_TILE_MAX];
    const int64_t total = b->offsets[b->n_sentences];
    fse_cursor c = {0, 0, 0, 0, 0, 0.0f};
    for (int64_t s = 0; s < b->n_sentences; s++) {
        float *acc = out + (size_t)s * (size_t)dim;
        const int64_t begin = b->offsets[s], end = b->offsets[s + 1];
        c.t = begin;
        c.end = end;
        /* The first tile always runs, so a sentence without terms stores +0.0. */
        int first = 1;
        do {
            const int n = known_only ? fse_fill_tile_known(p, b, total, &c, K, row, f)
                                     : fse_fill_tile(p, b, total, &c, K, 0, row, f);
            if (n || first) {
                FSE_TEST_TILE();
                fse_tile_f32_avx512(acc, dim, row, f, n, first);
            }
            first = 0;
        } while (c.t < c.end || c.j < c.g1);
        const int64_t n = end - begin;
        if (counts) counts[s] = n;
        if (fse_finish_row(acc, dim, finish && n ? 1.0f / (float)n : 1.0f, finish && n))
            return s;
    }
    return -1;
}

/* Literal mode arguments remove FastText branches from the known-only batch body. */
FSE_NOINLINE FSE_AVX512_TARGET static int64_t fse_loop_f32_avx512(const fse_params *p, const fse_batch *b,
                                              float *out, int64_t *counts, int finish) {
    const size_t cap = FSE_TILE_BYTES / p->row_bytes;
    if (cap < FSE_TILE_MIN) {
        FSE_TEST_ROWS();
        return fse_loop_f32_avx512_rows(p, b, out, counts, finish);
    }
    if (b->ngram_offsets == NULL)
        return fse_loop_f32_avx512_tiles(p, b, out, counts, finish, cap, 1);
    return fse_loop_f32_avx512_tiles(p, b, out, counts, finish, cap, 0);
}
#endif /* FSE_AVX512 */
#endif /* FSE_NATIVE_KERNEL_AVX512_H */
