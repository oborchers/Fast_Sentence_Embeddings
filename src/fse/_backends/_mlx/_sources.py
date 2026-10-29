#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The Metal source of the pooling kernel of the mlx backend (ARCHITECTURE.md, "MLX").

``weighted_mean@1`` exactly as ``fse._reference.pool_weighted_mean``: one thread per
value of a row (dimension ``d`` of sentence ``s``), ``acc = fma(factor, row[d], acc)`` in
token order with the ngram rows of an unknown FastText word in bucket order, then
``(1 / n) * acc`` with ``1 / n`` rounded first. Only explicit ``fma``; contraction is off
(``safe`` alone would fuse ``x * y + z``) and fast math refuses to compile.

The GPU flushes subnormal inputs and results to zero, which IEEE arithmetic and the
contract do not. Every multiply and fma therefore goes through a wrapper that sets a flag
whenever an operand is subnormal or the result may be (integer tests on the bit patterns,
which nothing flushes), and the host computes a flagged task again on the CPU:

- ``a * b`` may underflow only if both are nonzero and their biased exponents sum to less
  than 128 (then ``|a * b| < 2**-126`` is possible; from 128 on the product is normal);
- ``fma(a, b, c)``: the same for the product, or a subnormal ``c``, or a result whose
  exponent field is 0 although an operand was nonzero (a subnormal or a flushed result),
  unless the sum cancels exactly: an exact product (its significand fits into 24 bits,
  tested on the integers) that equals ``-c`` gives +0.0 on the GPU as in IEEE arithmetic;
- a sentence of 2**24 tokens or more, or an unknown word of as many buckets, whose count
  would round as a float;
- an accumulator that is not finite, which the CPU reports as the error of its row.

Each SIMD group of 32 threads lies inside one sentence and writes the OR of its flags.
"""

from __future__ import annotations

OOV_TOKEN_HEX = "0xFFFFFFFEu"

SUBNORMAL = 1
NONFINITE = 2
LONG = 4

HEADER = """
#pragma METAL fp contract(off)
#if defined(__FAST_MATH__)
#error "the fse kernels need math_mode safe"
#endif
constant constexpr uint FSE_SUBNORMAL = 1u;
constant constexpr uint FSE_NONFINITE = 2u;
constant constexpr uint FSE_LONG = 4u;
constant constexpr long FSE_EXACT_COUNTS = 16777216L;

inline uint fse_exponent(float x) { return (as_type<uint>(x) >> 23) & 0xffu; }
inline bool fse_nonzero(float x) { return (as_type<uint>(x) & 0x7fffffffu) != 0u; }
inline bool fse_subnormal(float x) {
    const uint bits = as_type<uint>(x) & 0x7fffffffu;
    return bits != 0u && bits < 0x00800000u;
}
inline bool fse_small_product(float a, float b) {
    return fse_nonzero(a) && fse_nonzero(b) && fse_exponent(a) + fse_exponent(b) < 128u;
}
// Whether the product of two normal values is exact: the product of their significands
// has at most 24 significant bits.
inline bool fse_exact_product(float a, float b) {
    const ulong p = ulong((as_type<uint>(a) & 0x7fffffu) | 0x800000u)
        * ulong((as_type<uint>(b) & 0x7fffffu) | 0x800000u);
    return (p >> ctz(p)) < (1ul << 24);
}
inline float fse_mul(float a, float b, thread uint &flag) {
    if (fse_subnormal(a) || fse_subnormal(b) || fse_small_product(a, b)) {
        flag |= FSE_SUBNORMAL;
    }
    return a * b;
}
inline float fse_fma(float a, float b, float c, thread uint &flag) {
    if (fse_subnormal(a) || fse_subnormal(b) || fse_subnormal(c) || fse_small_product(a, b)) {
        flag |= FSE_SUBNORMAL;
    }
    const float r = metal::fma(a, b, c);
    if (fse_exponent(r) == 0u && ((fse_nonzero(a) && fse_nonzero(b)) || fse_nonzero(c))) {
        const bool cancels = r == 0.0f && fse_nonzero(c) && a * b == -c && fse_exact_product(a, b);
        if (!cancels) {
            flag |= FSE_SUBNORMAL;
        }
    }
    return r;
}
"""

POOL = (
    """
    const uint d = thread_position_in_grid.x;
    const uint s = thread_position_in_grid.y;
    uint flag = 0u;
    if (d < DIM) {
        float acc = 0.0f;
        long k = FASTTEXT ? unknown[s] : 0;
        const long begin = offsets[s];
        const long end = offsets[s + 1];
        for (long t = begin; t < end; ++t) {
            const uint w = tokens[t];
            if (FASTTEXT && w == """
    + OOV_TOKEN_HEX
    + """) {
                const long b0 = ngram_offsets[k];
                const long b1 = ngram_offsets[k + 1];
                ++k;
                if (b1 == b0) {
                    continue;
                }
                if (b1 - b0 >= FSE_EXACT_COUNTS) {
                    flag |= FSE_LONG;
                }
                const float inverse = metal::precise::divide(1.0f, float(b1 - b0));
                const float shared = fse_mul(inverse, oov_weight[0], flag);
                for (long b = b0; b < b1; ++b) {
                    const uint q = ngrams[b];
                    const float f = INT8 ? fse_mul(shared, ngram_scales[q], flag) : shared;
                    acc = fse_fma(f, float(ngram_vectors[(ulong)q * DIM + d]), acc, flag);
                }
                continue;
            }
            const float f = INT8 ? fse_mul(weights[w], scales[w], flag) : weights[w];
            acc = fse_fma(f, float(vectors[(ulong)w * DIM + d]), acc, flag);
        }
        if (fse_exponent(acc) == 255u) {
            flag |= FSE_NONFINITE;
        }
        const long n = end - begin;
        if (n >= FSE_EXACT_COUNTS) {
            flag |= FSE_LONG;
        }
        const float inverse = metal::precise::divide(1.0f, float(n));
        out[(ulong)s * DIM + d] = n ? fse_mul(inverse, acc, flag) : 0.0f;
    }
    if (!CHECKS) {
        flag &= FSE_NONFINITE;
    }
    const uint any = simd_or(flag);
    if (thread_index_in_simdgroup == 0u) {
        flags[(ulong)s * WORDS + d / 32u] = any;
    }
"""
)

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
