#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The Metal source of the pooling kernel of the mlx backend (ARCHITECTURE.md, "MLX").

``weighted_mean@1`` in the order of the cpu backend: one thread per ``VEC`` values of a
row (four when the dimension is a multiple of four, else one), ``acc = fma(f, row, acc)``
in token order with the ngram rows of an unknown FastText word in bucket order, then
``(1 / n) * acc`` with ``1 / n`` rounded first. Only explicit ``fma``; contraction is off
(``safe`` alone would fuse ``x * y + z``) and fast math refuses to compile.

The GPU flushes subnormal values to zero. The backend refuses, at prepare, inputs whose
factors or row values are or may be subnormal (``_kernels.subnormal_input``), so a flush
can only meet a product or a partial sum below 2**-126, which the absolute term of the
tolerance contract covers. The kernel flags per sentence what the host must compute: an
accumulator that is not finite (the cpu's error), a count of 2**24 or more that would
round as a float, and for ``normalize`` a value of 2**60 or more (its squares may
overflow) and all values below 2**-50 (its sum of squares may be below 2**-100, or the
row is zero), which is ANDed over the row. Each SIMD group of 32 threads lies inside one
sentence and writes its flags.
"""

from __future__ import annotations

OOV_TOKEN_HEX = "0xFFFFFFFEu"

NONFINITE = 2
LONG = 4
HUGE = 8
TINY = 16

HEADER = """
#pragma METAL fp contract(off)
#if defined(__FAST_MATH__)
#error "the fse kernels need math_mode safe"
#endif
constant constexpr uint FSE_NONFINITE = 2u;
constant constexpr uint FSE_LONG = 4u;
constant constexpr uint FSE_HUGE = 8u;
constant constexpr uint FSE_TINY = 16u;
constant constexpr uint FSE_EXACT_COUNTS = 16777216u;

inline uint fse_magnitude(float x) { return as_type<uint>(x) & 0x7fffffffu; }
inline uint4 fse_magnitude(float4 x) { return as_type<uint4>(x) & 0x7fffffffu; }
inline bool fse_nonfinite(float x) { return fse_magnitude(x) >= 0x7f800000u; }
inline bool fse_nonfinite(float4 x) { return any(fse_magnitude(x) >= 0x7f800000u); }
// Whether a value is 2**60 or more, and whether all are below 2**-50 (on the bits).
inline bool fse_huge(float x) { return fse_magnitude(x) >= 0x5D800000u; }
inline bool fse_huge(float4 x) { return any(fse_magnitude(x) >= 0x5D800000u); }
inline bool fse_tiny(float x) { return fse_magnitude(x) < 0x26800000u; }
inline bool fse_tiny(float4 x) { return all(fse_magnitude(x) < 0x26800000u); }
// Row ``i`` of VEC values of a table in either address space (MLX passes small arrays as
// constant): rows start at multiples of VEC values, so the vector loads are aligned.
template <uint V, typename T> struct fse_vec { typedef vec<T, V> type; };
template <typename T> struct fse_vec<1, T> { typedef T type; };
template <uint V, typename T>
inline typename fse_vec<V, T>::type fse_load(const device T *p, ulong i) {
    return ((const device typename fse_vec<V, T>::type *)p)[i];
}
template <uint V, typename T>
inline typename fse_vec<V, T>::type fse_load(const constant T *p, ulong i) {
    return ((const constant typename fse_vec<V, T>::type *)p)[i];
}
"""

_TYPES = {4: "typedef float4 acc_t;", 1: "typedef float acc_t;"}

_POOL = (
    """
    const uint q = thread_position_in_grid.x;
    const uint s = thread_position_in_grid.y;
    constexpr uint ROW = DIM / VEC;
    uint flag = 0u;
    bool tiny = true;
    if (q < ROW) {
        acc_t acc = 0.0f;
        uint k = FASTTEXT ? unknown[s] : 0u;
        const uint begin = offsets[s];
        const uint end = offsets[s + 1];
        for (uint t = begin; t < end; ++t) {
            const uint w = tokens[t];
            if (FASTTEXT && w == """
    + OOV_TOKEN_HEX
    + """) {
                const uint b0 = ngram_offsets[k];
                const uint b1 = ngram_offsets[k + 1];
                ++k;
                if (b1 == b0) {
                    continue;
                }
                if (b1 - b0 >= FSE_EXACT_COUNTS) {
                    flag |= FSE_LONG;
                }
                const float shared = metal::precise::divide(1.0f, float(b1 - b0)) * oov_weight[0];
                for (uint b = b0; b < b1; ++b) {
                    const uint g = ngrams[b];
                    const float f = INT8 ? shared * ngram_scales[g] : shared;
                    const acc_t x = acc_t(fse_load<VEC>(ngram_vectors, (ulong)g * ROW + q));
                    acc = metal::fma(acc_t(f), x, acc);
                }
                continue;
            }
            const float f = INT8 ? weights[w] * scales[w] : weights[w];
            const acc_t x = acc_t(fse_load<VEC>(vectors, (ulong)w * ROW + q));
            acc = metal::fma(acc_t(f), x, acc);
        }
        if (fse_nonfinite(acc)) {
            flag |= FSE_NONFINITE;
        }
        const uint n = end - begin;
        if (n >= FSE_EXACT_COUNTS) {
            flag |= FSE_LONG;
        }
        const float inverse = metal::precise::divide(1.0f, float(n));
        const acc_t row = n ? acc_t(inverse) * acc : acc_t(0.0f);
        ((device acc_t *)out)[(ulong)s * ROW + q] = row;
        flag |= fse_huge(row) ? FSE_HUGE : 0u;
        tiny = fse_tiny(row);
    }
    const uint any_flag = simd_or(flag) | (simd_all(tiny) ? FSE_TINY : 0u);
    if (thread_index_in_simdgroup == 0u) {
        flags[(ulong)s * FLAG_WORDS + q / 32u] = any_flag;
    }
"""
)


def pool(vec: int) -> str:
    """The body of the pooling kernel with ``vec`` values (4 or 1) per thread."""
    return "\n    " + _TYPES[vec] + "\n    constexpr uint VEC = " + str(vec) + "u;" + _POOL


POOL_INPUTS = (
    "vectors",
    "weights",
    "scales",
    "ngram_vectors",
    "ngram_scales",
    "oov_weight",
    "tokens",
    "offsets",
    "unknown",
    "ngram_offsets",
    "ngrams",
)
POOL_OUTPUTS = ("out", "flags")
