"""A short check of an installed fse wheel, for the Windows wheels that run no full suite.

    python -I tools/wheel_smoke.py

cibuildwheel runs it (``[[tool.cibuildwheel.overrides]]`` in pyproject.toml) on the Windows
wheels of the Python versions whose full suite does not run on Windows (DR-344: only one
version runs the suite there; Unix runs it on all). It needs no network, no test data and
nothing but numpy: it imports ``fse`` and its native extension, makes a small encoder from
random word vectors, encodes with the native kernel and with the numpy reference of the
contract, and requires the same bits, for float32, float16 and int8 tables and for one and
two workers. It exits with 1 on the first difference. It takes a second or two.
"""

from __future__ import annotations

import sys

import numpy as np

import fse
from fse import _native


def _encoder(dtype: str):
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((60, 12)).astype(np.float32)
    table = fse.WordVectors.from_arrays(vectors, words=[f"w{i}" for i in range(60)])
    if dtype != "float32":
        table = table.quantize(dtype)
    return fse.Encoder.from_word_vectors(table)


def _texts() -> list[str]:
    rng = np.random.default_rng(1)
    texts = [""]
    for size in (1, 2, 5, 17, 40):
        texts.append(" ".join(f"w{t}" if t < 60 else f"zz{t}" for t in rng.integers(0, 70, size)))
    return texts * 4


def _check(dtype: str, workers: int) -> None:
    encoder, texts = _encoder(dtype), _texts()
    native = encoder.encode(texts, workers=workers)
    reference = encoder.encode(texts, workers=workers, backend="numpy")
    if native.shape != (len(texts), 12) or native.dtype != np.float32:
        raise SystemExit(f"{dtype}, {workers} workers: unexpected result {native.shape}")
    if not np.array_equal(native.view(np.uint32), reference.view(np.uint32)):
        raise SystemExit(f"{dtype}, {workers} workers: the kernel differs from the reference")


def main() -> int:
    print(f"fse {fse.__version__}, python {sys.version.split()[0]}, isa {_native.available_isas()}")
    for dtype in ("float32", "float16", "int8"):
        for workers in (1, 2):
            _check(dtype, workers)
    print("fse wheel smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
