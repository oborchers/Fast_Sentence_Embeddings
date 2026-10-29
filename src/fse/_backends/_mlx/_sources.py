#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The Metal sources of the mlx backend (ARCHITECTURE.md, "MLX").

``weighted_mean@1`` in the order of the cpu backend: one thread per ``VEC`` values of a
row (four when the dimension is a multiple of four, else one), ``acc = fma(f, row, acc)``
in token order with the ngram rows of an unknown FastText word in bucket order, then
``(1 / n) * acc`` with ``1 / n`` rounded first. Only explicit ``fma``; contraction is off
(``safe`` alone would fuse ``x * y + z``) and fast math refuses to compile.

The GPU flushes subnormal inputs and results to zero, which the cpu backend does not.
The kernel detects every place a flush could change a value, conservatively, with integer
tests on bit patterns (which nothing flushes), and the host computes a flagged task again:

- ``PREPARE`` makes the factor of every word (``weight``, or ``weight * scale`` for int8)
  once per table and gives a word NaN as its factor when its weight, scale or a value of
  its row is subnormal, when ``weight * scale`` may underflow (both nonzero and their
  biased exponents summing to less than 128), or when a product of the factor and a
  nonzero value of the row may (the factor's exponent plus the row's smallest one below
  128). A NaN accumulator flags the sentence as nonfinite. For the ngram rows it gives the
  smallest exponent of each row, or 0 for a row with a subnormal value, which the pooling
  kernel tests once per bucket against the bucket's factor.
- After every fma the result's magnitude bits are at most those of 2**-126 (a subnormal,
  flushed or exactly that boundary) while the result or the accumulator before is
  nonzero; an exact cancellation to zero is a false alarm the host computes.
- The factors made in the kernel (``(1 / m) * oov_weight``, times the ngram scale) and
  ``(1 / n) * acc`` flag a subnormal operand or a product that may underflow.

A sentence of 2**24 tokens or more, or an unknown word of as many buckets, whose count
would round as a float, and an accumulator that is not finite set flags too. Each SIMD
group of 32 threads lies inside one sentence and writes the OR of its flags.
"""

from __future__ import annotations

OOV_TOKEN_HEX = "0xFFFFFFFEu"

SUBNORMAL = 1
NONFINITE = 2
LONG = 4
# For normalize: a value of 2**60 or more (its squares may overflow), and all values of a
# SIMD group's part of the row below 2**-60 (ANDed over the row: its squares may be
# subnormal, or it is zero).
HUGE = 8
TINY = 16

# The smallest exponent field of a row without nonzero values: no product can underflow.
NO_EXPONENT = 512

HEADER = """
#pragma METAL fp contract(off)
#if defined(__FAST_MATH__)
#error "the fse kernels need math_mode safe"
#endif
constant constexpr uint FSE_SUBNORMAL = 1u;
constant constexpr uint FSE_NONFINITE = 2u;
constant constexpr uint FSE_LONG = 4u;
constant constexpr uint FSE_HUGE = 8u;
constant constexpr uint FSE_TINY = 16u;
constant constexpr uint FSE_EXACT_COUNTS = 16777216u;
constant constexpr uint FSE_NO_EXPONENT = 512u;

inline uint fse_exponent(float x) { return (as_type<uint>(x) >> 23) & 0xffu; }
inline bool fse_nonzero(float x) { return (as_type<uint>(x) & 0x7fffffffu) != 0u; }
inline bool fse_subnormal(float x) {
    const uint bits = as_type<uint>(x) & 0x7fffffffu;
    return bits != 0u && bits < 0x00800000u;
}
inline bool fse_small_product(float a, float b) {
    return fse_nonzero(a) && fse_nonzero(b) && fse_exponent(a) + fse_exponent(b) < 128u;
}
inline float fse_mul(float a, float b, thread uint &flag) {
    if (fse_subnormal(a) || fse_subnormal(b) || fse_small_product(a, b)) {
        flag |= FSE_SUBNORMAL;
    }
    return a * b;
}
inline float4 fse_mul(float a, float4 b, thread uint &flag) {
    return float4(fse_mul(a, b.x, flag), fse_mul(a, b.y, flag), fse_mul(a, b.z, flag),
                  fse_mul(a, b.w, flag));
}
// Whether a product of the factor f and a nonzero value of exponent fields from ``low``
// on may be subnormal; ``low`` 0 (a subnormal value) flags any factor.
inline bool fse_small_factor(float f, uint low) {
    return low == 0u || (fse_nonzero(f) && fse_exponent(f) + low < 128u);
}
// Whether the product of two normal values is exact: the product of their significands
// has at most 24 significant bits.
inline bool fse_exact_product(float a, float b) {
    const ulong p = ulong((as_type<uint>(a) & 0x7fffffu) | 0x800000u)
        * ulong((as_type<uint>(b) & 0x7fffffu) | 0x800000u);
    return (p >> ctz(p)) < (1ul << 24);
}
inline uint fse_magnitude(float x) { return as_type<uint>(x) & 0x7fffffffu; }
inline uint4 fse_magnitude(float4 x) { return as_type<uint4>(x) & 0x7fffffffu; }
// The flag of ``r = fma(f, x, before)``: r is subnormal, flushed or exactly 2**-126 (its
// magnitude bits from 1 to those of 2**-126), or zero from a nonzero ``before`` unless
// the product is exact and equals ``-before`` (an exact cancellation, +0.0 in IEEE
// arithmetic too; a subnormal IEEE result the GPU flushed fails the test).
inline uint fse_cancelled(float f, float x, float before, float r) {
    if (fse_magnitude(r) != 0u || fse_magnitude(before) == 0u) {
        return 0u;
    }
    return fse_exact_product(f, x) && f * x == -before ? 0u : FSE_SUBNORMAL;
}
inline uint fse_step(float f, float x, float before, float r) {
    return fse_magnitude(r) - 1u < 0x00800000u ? FSE_SUBNORMAL : fse_cancelled(f, x, before, r);
}
inline uint fse_step(float f, float4 x, float4 before, float4 r) {
    const uint4 bits = fse_magnitude(r);
    if (any(bits - 1u < 0x00800000u)) {
        return FSE_SUBNORMAL;
    }
    if (!any(bits == 0u)) {
        return 0u;
    }
    return fse_cancelled(f, x.x, before.x, r.x) | fse_cancelled(f, x.y, before.y, r.y)
        | fse_cancelled(f, x.z, before.z, r.z) | fse_cancelled(f, x.w, before.w, r.w);
}
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
// Whether a value is 2**60 or more, and whether all are below 2**-60 (on the bits).
inline bool fse_huge(float x) { return fse_magnitude(x) >= 0x5D800000u; }
inline bool fse_huge(float4 x) { return any(fse_magnitude(x) >= 0x5D800000u); }
inline bool fse_tiny(float x) { return fse_magnitude(x) < 0x21800000u; }
inline bool fse_tiny(float4 x) { return all(fse_magnitude(x) < 0x21800000u); }
inline bool fse_nonfinite(uint bits) { return bits >= 0x7f800000u; }
inline bool fse_nonfinite(uint4 bits) { return any(bits >= 0x7f800000u); }
"""

# One SIMD group per row: the bits of the factor of word w (NaN when flagged) or, for the
# ngram rows, the smallest exponent field of the row's nonzero values (0 with a subnormal
# value). As uint32, so that no copy of MLX moves them as floats.
PREPARE = """
    const uint lane = thread_position_in_grid.x;
    const uint w = thread_position_in_grid.y;
    uint low = FSE_NO_EXPONENT;
    for (uint d = lane; d < DIM; d += 32u) {
        const uint bits = fse_magnitude(float(vectors[(ulong)w * DIM + d]));
        low = min(low, bits == 0u ? FSE_NO_EXPONENT : (bits < 0x00800000u ? 0u : bits >> 23));
    }
    low = simd_min(low);
    if (lane == 0u) {
        if (WORDS) {
            const float weight = weights[w];
            const float scale = INT8 ? scales[w] : 1.0f;
            uint flag = 0u;
            const float f = INT8 ? fse_mul(weight, scale, flag) : weight;
            const bool flagged = flag != 0u || fse_subnormal(weight) || fse_small_factor(f, low);
            factors[w] = as_type<uint>(flagged ? NAN : f);
        } else {
            factors[w] = low;
        }
    }
"""

PREPARE_INPUTS = ("vectors", "weights", "scales")
PREPARE_OUTPUTS = ("factors",)

_TYPES = {
    4: "typedef float4 acc_t;",
    1: "typedef float acc_t;",
}

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
                const float inverse = metal::precise::divide(1.0f, float(b1 - b0));
                const float shared = fse_mul(inverse, oov_weight[0], flag);
                for (uint b = b0; b < b1; ++b) {
                    const uint g = ngrams[b];
                    const float f = INT8 ? fse_mul(shared, ngram_scales[g], flag) : shared;
                    if (fse_small_factor(f, ngram_low[g])) {
                        flag |= FSE_SUBNORMAL;
                    }
                    const acc_t x = acc_t(fse_load<VEC>(ngram_vectors, (ulong)g * ROW + q));
                    const acc_t before = acc;
                    acc = metal::fma(acc_t(f), x, acc);
                    flag |= fse_step(f, x, before, acc);
                }
                continue;
            }
            const float f = as_type<float>(factors[w]);
            const acc_t x = acc_t(fse_load<VEC>(vectors, (ulong)w * ROW + q));
            const acc_t before = acc;
            acc = metal::fma(acc_t(f), x, acc);
            flag |= fse_step(f, x, before, acc);
        }
        if (fse_nonfinite(fse_magnitude(acc))) {
            flag |= FSE_NONFINITE;
        }
        const uint n = end - begin;
        if (n >= FSE_EXACT_COUNTS) {
            flag |= FSE_LONG;
        }
        const float inverse = metal::precise::divide(1.0f, float(n));
        const acc_t row = n ? fse_mul(inverse, acc, flag) : acc_t(0.0f);
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
    "factors",
    "ngram_vectors",
    "ngram_scales",
    "ngram_low",
    "oov_weight",
    "tokens",
    "offsets",
    "unknown",
    "ngram_offsets",
    "ngrams",
)
POOL_OUTPUTS = ("out", "flags")
