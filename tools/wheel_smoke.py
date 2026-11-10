"""A short check of an installed fse wheel, for the Windows wheels that run no full suite.

    python -I tools/wheel_smoke.py

cibuildwheel runs it (``[[tool.cibuildwheel.overrides]]`` in pyproject.toml) on the Windows
wheels of the Python versions whose full suite does not run on Windows (DR-344: only one
version runs the suite there; Unix runs it on all). It needs no network, no test data and
nothing but numpy: it imports ``fse`` and its native extension, makes a small encoder from
random word vectors, encodes with the native kernel and with the numpy reference of the
contract, and requires the same bits, for float32, float16 and int8 tables and for one and
two workers (the text is larger than the limit under which the tasks run in the calling
thread). It exits with 1 on the first difference. It takes a few seconds.
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
    """Empty, short and long sentences with unknown words: more than 64 KiB of text, so a
    call with two workers runs its tasks in worker threads and not in the calling thread."""
    rng = np.random.default_rng(1)
    texts = [""]
    for _ in range(1200):
        size = int(rng.integers(1, 41))
        texts.append(" ".join(f"w{t}" if t < 60 else f"zz{t}" for t in rng.integers(0, 70, size)))
    return texts


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
