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
can only meet a product or a partial sum below 2**-126, which the tolerance contract's
bound covers. The kernel flags per sentence an accumulator that is not finite (the cpu's
error) and a count of 2**24 or more that would round as a float; each SIMD group of 32
threads lies inside one sentence and writes the OR of its flags. With ``BOUND`` it also
gives per sentence ``sum |f| ||x||`` (``norms``: the norms of the stored rows, rounded
up), from which the host bounds the row's error (``_bounds``), and with ``NORMALIZE`` the
sum of squares of the row before it normalizes.
"""

from __future__ import annotations

OOV_TOKEN_HEX = "0xFFFFFFFEu"

NONFINITE = 2
LONG = 4

HEADER = """
#pragma METAL fp contract(off)
#if defined(__FAST_MATH__)
#error "the fse kernels need math_mode safe"
#endif
constant constexpr uint FSE_NONFINITE = 2u;
constant constexpr uint FSE_LONG = 4u;
constant constexpr uint FSE_EXACT_COUNTS = 16777216u;

inline uint fse_magnitude(float x) { return as_type<uint>(x) & 0x7fffffffu; }
inline uint4 fse_magnitude(float4 x) { return as_type<uint4>(x) & 0x7fffffffu; }
inline void fse_put(threadgroup float *p, uint i, float x) { p[i] = x; }
inline void fse_put(threadgroup float *p, uint i, float4 x) {
    p[i] = x.x;
    p[i + 1] = x.y;
    p[i + 2] = x.z;
    p[i + 3] = x.w;
}
inline bool fse_nonfinite(float x) { return fse_magnitude(x) >= 0x7f800000u; }
inline bool fse_nonfinite(float4 x) { return any(fse_magnitude(x) >= 0x7f800000u); }
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
    acc_t row = acc_t(0.0f);
    float magnitude = 0.0f;  // with BOUND: sum |f| ||x||, the same in every thread of a row
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
                    if (BOUND) {
                        magnitude = metal::fma(metal::fabs(f), ngram_norms[g], magnitude);
                    }
                }
                continue;
            }
            const float f = INT8 ? weights[w] * scales[w] : weights[w];
            const acc_t x = acc_t(fse_load<VEC>(vectors, (ulong)w * ROW + q));
            acc = metal::fma(acc_t(f), x, acc);
            if (BOUND) {
                magnitude = metal::fma(metal::fabs(f), norms[w], magnitude);
            }
        }
        if (fse_nonfinite(acc)) {
            flag |= FSE_NONFINITE;
        }
        const uint n = end - begin;
        if (n >= FSE_EXACT_COUNTS) {
            flag |= FSE_LONG;
        }
        const float inverse = metal::precise::divide(1.0f, float(n));
        row = n ? acc_t(inverse) * acc : acc_t(0.0f);
        if (!NORMALIZE) {
            ((device acc_t *)out)[(ulong)s * ROW + q] = row;
        }
        if (BOUND && q == 0u) {
            bounds[(ulong)s * 2u] = magnitude;
        }
    }
    if (NORMALIZE) {
        // normalize@1 as the cpu computes it (the rows of a threadgroup are whole here):
        // dot16 of the row with itself (16 lanes by fma in index order, then added 8, 4, 2
        // and 1 lanes apart), 1 / sqrt correctly rounded, one product per value. The sum
        // of squares goes to ``bounds`` for the host's check (a row near a limit of
        // normalize comes from the host); a zero or nonfinite one leaves the row as is.
        threadgroup float values[1024];
        threadgroup float lanes[16 * 8];
        threadgroup float inverses[8];
        const uint ly = thread_position_in_threadgroup.y;
        if (q < ROW) {
            fse_put(values, ly * DIM + q * VEC, row);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q < 16u) {
            float lane = 0.0f;
            for (uint i = q; i < DIM; i += 16u) {
                lane = metal::fma(values[ly * DIM + i], values[ly * DIM + i], lane);
            }
            lanes[ly * 16u + q] = lane;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q == 0u) {
            float l[16];
            for (uint j = 0u; j < 16u; ++j) {
                l[j] = lanes[ly * 16u + j];
            }
            for (uint width = 8u; width > 0u; width >>= 1u) {
                for (uint j = 0u; j < width; ++j) {
                    l[j] = l[j] + l[j + width];
                }
            }
            bounds[(ulong)s * 2u + 1u] = l[0];
            const bool usable = fse_magnitude(l[0]) != 0u && !fse_nonfinite(l[0]);
            inverses[ly] = usable ? metal::precise::divide(1.0f, metal::precise::sqrt(l[0])) : 1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q < ROW) {
            ((device acc_t *)out)[(ulong)s * ROW + q] = row * acc_t(inverses[ly]);
        }
    }
    const uint any_flag = simd_or(flag);
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
    "norms",
    "ngram_norms",
)
POOL_OUTPUTS = ("out", "flags", "bounds")
