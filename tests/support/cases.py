"""The cases of the conformance suite of the weighted mean (tests/test_conformance.py and
tests/test_conformance_device.py): every dtype, dimensions across the vector widths, empty,
long and unknown-word sentences, subnormal values and exact constructions."""

from __future__ import annotations

import numpy as np

from fse._contract import OOV_TOKEN
from support.conformance import BIG, F32, TINY, Case, _rng


def random_case(name, dtype, seed, *, words=40, dim=13, n=25, length=12, empty=0.15, ft=False):
    rng = _rng(seed)
    base = rng.standard_normal((words, dim)).astype(F32)
    weights = rng.random(words).astype(F32)
    sentences, ngrams = [], []
    for _ in range(n):
        size = 0 if rng.random() < empty else int(rng.integers(1, length + 1))
        sentence = []
        for _ in range(size):
            if ft and rng.random() < 0.3:
                sentence.append(OOV_TOKEN)
                ngrams.append(tuple(int(b) for b in rng.integers(0, 17, rng.integers(0, 6))))
            else:
                sentence.append(int(rng.integers(0, words)))
        sentences.append(tuple(sentence))
    if not ft:
        return Case(name, dtype, base, weights, tuple(sentences))
    ngram_base = rng.standard_normal((17, dim)).astype(F32)
    return Case(name, dtype, base, weights, tuple(sentences), tuple(ngrams), ngram_base, 0.75)


SEEDS = {"float32": 32, "float16": 16, "int8": 8}


def _cases():
    out = []
    for dtype in ("float32", "float16", "int8"):
        rng = _rng(SEEDS[dtype])  # fixed: hash() of a str changes with PYTHONHASHSEED
        base = rng.standard_normal((30, 9)).astype(F32)
        w = rng.random(30).astype(F32)
        out += [
            Case("no sentences", dtype, base, w, ()),
            Case("all empty", dtype, base, w, ((), (), ())),
            Case("empty first middle last", dtype, base, w, ((), (1, 2), (), (3,), ())),
            Case("single tokens", dtype, base, w, tuple((i,) for i in range(30))),
            Case(
                "1000 tokens", dtype, base, w, (tuple(int(i) for i in rng.integers(0, 30, 1000)),)
            ),
            Case(
                "5000 tokens",
                dtype,
                base,
                w,
                (tuple(int(i) for i in rng.integers(0, 30, 5000)), (1,)),
            ),
            Case("one word repeated", dtype, base, w, ((7,) * 1000, (7,) * 3)),
        ]
        for dim in (
            1,
            3,
            4,
            7,
            8,
            9,
            12,
            15,
            16,
            17,
            20,
            28,
            31,
            32,
            33,
            36,
            63,
            64,
            65,
            100,
            300,
            301,
        ):
            out.append(random_case(f"dim {dim}", dtype, dim, dim=dim, n=6))
        out.append(random_case("fasttext", dtype, 11, ft=True))
        out.append(
            Case(
                "only unknown words",
                dtype,
                base,
                w,
                ((OOV_TOKEN,), (OOV_TOKEN, OOV_TOKEN), (OOV_TOKEN,) * 3),
                ((), (0,), (5, 5, 5), (0, 16), (16,), tuple(range(17)) * 12),
                rng.standard_normal((17, 9)).astype(F32),
                0.5,
            )
        )
        out.append(
            Case(
                "fasttext without unknown words",
                dtype,
                base,
                w,
                ((1, 2), ()),
                (),
                rng.standard_normal((17, 9)).astype(F32),
                0.5,
            )
        )
        sub = np.full((4, 9), TINY, dtype=F32) * np.arange(1, 37, dtype=F32).reshape(4, 9)
        out.append(
            Case("subnormal rows", dtype, sub, np.full(4, 0.5, F32), ((0, 1, 2, 3), (0, 0), (3,)))
        )
        small = rng.standard_normal((5, 9)).astype(F32) * F32(1e-30)
        out.append(
            Case("subnormal products", dtype, small, np.full(5, 1e-10, F32), ((0, 1, 2), (4,) * 50))
        )
        cross = np.vstack([np.full(9, TINY * 64, F32), np.full(9, F32(1e-37), F32)])
        out.append(Case("subnormal to normal", dtype, cross, np.ones(2, F32), ((0,) * 9 + (1,),)))
        signed = np.array([[-0.0, 1.0, -2.0], [0.0, -1.0, 2.0]], dtype=F32)
        out.append(
            Case("signed zeros", dtype, signed, np.array([-0.0, 1.0], F32), ((0,), (0, 1), (1, 0)))
        )
        out.append(Case("large first row", dtype, base, w, ((1, 2), (3,)), first_row=2**40))
        for seed in range(12):
            out.append(
                random_case(
                    f"random {seed}",
                    dtype,
                    100 + seed,
                    ft=seed % 3 == 0,
                    dim=int(_rng(seed).integers(1, 40)),
                )
            )
    # float32 only: exact constructions.
    f, x = F32(1 + 2**-12), F32(1 + 2**-12)
    fused = np.array([[x], [-(f * x)]], dtype=F32)
    out.append(
        # The rounded negative product first, so the fma of word 0 keeps the rounding error.
        Case("fused, not rounded products", "float32", fused, np.array([f, 1], F32), ((1, 0),))
    )
    v = F32(1 + 2**-23)
    out.append(
        Case(
            "inverse length, not division",
            "float32",
            np.array([[v]], F32),
            np.ones(1, F32),
            ((0, 0, 0),),
        )
    )
    tiny_scale = np.array([[1e-38, 2e-38, 0.0], [127.0, -127.0, 1.0]], dtype=F32)
    out.append(
        Case("int8 tiny scales", "int8", tiny_scale, np.array([1e-3, 0.5], F32), ((0, 1), (0,)))
    )
    zero_row = np.zeros((2, 5), dtype=F32)
    out.append(Case("int8 zero rows", "int8", zero_row, np.ones(2, F32), ((0, 1),)))
    half = np.array([[65504.0, -65504.0, 6e-8, 1.0]], dtype=F32)
    out.append(Case("float16 extremes", "float16", half, np.ones(1, F32) * F32(0.25), ((0, 0),)))
    # Subnormal values only in the middle of the arithmetic, which a GPU that flushes them
    # gets wrong: a tiny oov_weight / m, a product in the subnormal range whose sum is
    # normal, and normal rows whose sum cancels to a subnormal value.
    unknown = rng.standard_normal((3, 4)).astype(F32)
    out.append(
        Case(
            "tiny oov weight",
            "float32",
            np.ones((1, 4), F32),
            np.ones(1, F32),
            ((OOV_TOKEN, 0), (OOV_TOKEN,)),
            ((0, 1, 2), (2,)),
            unknown,
            1e-38,
        )
    )
    tiny = np.array([[2.0**-70], [2.0**-125]], F32)
    out.append(Case("subnormal product", "float32", tiny, np.array([2.0**-70, 1], F32), ((1, 0),)))
    # A subnormal weight of a large value: a normal product the GPU would flush away.
    large = np.array([[2.0**100, -1.0]], F32)
    out.append(
        Case(
            "subnormal weight, large row",
            "float32",
            large,
            np.full(1, TINY * 8, F32),
            ((0,), (0, 0)),
        )
    )
    # A subnormal int8 scale with a large weight: normal factors from a subnormal scale.
    small = np.array([[TINY * 1016, -TINY * 508]], F32)
    out.append(
        Case("subnormal scale, large weight", "int8", small, np.full(1, 2.0**100, F32), ((0,),))
    )
    # Normal inputs whose factors are subnormal: an int8 weight 2**-63 times the scale
    # 2**-64, and an oov weight 2**-126 halved over two buckets of rows 2**126.
    codes = np.array([[127 * 2.0**-64, -(2.0**-64)]], F32)
    out.append(Case("subnormal int8 factor", "int8", codes, np.full(1, 2.0**-63, F32), ((0,),)))
    huge = np.full((2, 2), 2.0**126, F32)
    out.append(
        Case(
            "subnormal ngram factor",
            "float32",
            np.ones((1, 2), F32),
            np.ones(1, F32),
            ((OOV_TOKEN,), (0, OOV_TOKEN)),
            ((0, 1), (1,)),
            huge,
            2.0**-126,
        )
    )
    # A flush that turns a later rounding (review of PR #121): 2**-63 * 2**-64 flushed,
    # then f * f with f = 1 + 2**-12, -2**-11 and 1,021 times -3 * 2**-26.
    turn = np.array([[2.0**-64], [1 + 2.0**-12], [-(2.0**-11)], [-3 * 2.0**-26]], F32)
    out.append(
        Case(
            "a flush turns a rounding",
            "float32",
            turn,
            np.array([2.0**-63, 1 + 2.0**-12, 1, 1], F32),
            ((0, 1, 2, *(3,) * 1021),),
        )
    )
    cancel = np.array([[1.25 * 2.0**-126], [-(2.0**-126)]], F32)
    out.append(Case("cancels to subnormal", "float32", cancel, np.ones(2, F32), ((0, 1),)))
    return out


CASES = _cases()
OVERFLOW = Case(
    "overflow", "float32", np.array([[BIG, 1.0]], F32), np.ones(1, F32), ((0,), (0, 0)), first_row=7
)
