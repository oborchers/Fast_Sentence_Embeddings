"""Sweep exp@1 over every float32 input it takes and report its error against float64 exp.

    uv run python tools/check_exp.py [--chunk-bits 24]

exp@1 (docs/architecture/numerical-contract.md, "Numerical contract (`transform_version` 2)") takes x <= 0; below -104 it is +0.0. The
sweep runs the kernel (``fse._native.exp1``, bit-identical to ``fse._reference_head.exp``
by tests/test_native_head.py) on every float32 from -0.0 down to -104, compares it with
numpy's float64 exp, and prints the largest error in float32 ulps of
the exact value (for a subnormal result the ulp is the smallest subnormal), the input
where it occurs, and how many inputs are off by more than 0.5 and 1 ulp. It also checks
that every instruction set of this machine (portable, NEON or AVX2 lanes) gives the same
bits on every input. The contract asks for at most 2 ulps; the tool exits with 1 if that
or the bits of an instruction set fail.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from fse import _native

F32 = np.float32
SMALLEST = float(np.finfo(F32).smallest_subnormal)
LOW = int(np.array([-104.0], F32).view(np.uint32)[0])  # the bits of -104
NEGATIVE_ZERO = 0x80000000


def ulps(x: np.ndarray) -> np.ndarray:
    got = _native.exp1(x, isa="portable").astype(np.float64)
    want = np.exp(x.astype(np.float64))
    step = np.spacing(np.abs(want.astype(F32))).astype(np.float64)
    step = np.where(step == 0, SMALLEST, step)
    return np.abs(got - want) / step


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunk-bits", type=int, default=24)
    args = parser.parse_args(argv)
    step = 1 << args.chunk_bits
    start = time.perf_counter()
    worst, where, over_half, over_one, total, mismatched = 0.0, 0.0, 0, 0, 0, 0
    for first in range(NEGATIVE_ZERO, LOW + 1, step):
        last = min(first + step, LOW + 1)
        x = np.arange(first, last, dtype=np.uint64).astype(np.uint32).view(F32)
        error = ulps(x)
        portable = _native.exp1(x, isa="portable").view(np.uint32)
        for isa in _native.available_isas()[1:]:
            mismatched += int(
                np.count_nonzero(_native.exp1(x, isa=isa).view(np.uint32) != portable)
            )
        i = int(np.argmax(error))
        if error[i] > worst:
            worst, where = float(error[i]), float(x[i])
        over_half += int(np.count_nonzero(error > 0.5))
        over_one += int(np.count_nonzero(error > 1))
        total += len(x)
    seconds = time.perf_counter() - start
    print(f"exp@1 on {total:,} float32 inputs from -0.0 to -104 ({seconds:.1f} s)")
    print(f"largest error: {worst:.4f} ulp at x = {where!r} ({where.hex()})")
    print(f"inputs off by more than 0.5 ulp: {over_half:,}; by more than 1 ulp: {over_one:,}")
    isas = ", ".join(_native.available_isas())
    print(f"inputs where an instruction set ({isas}) differs from portable: {mismatched:,}")
    return 0 if worst <= 2 and not mismatched else 1


if __name__ == "__main__":
    sys.exit(main())
