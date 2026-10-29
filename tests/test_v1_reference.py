"""The 1.1.0 reference fixture: the 2.0 reference stays close to what 1.1.0 computed.

The fixture in tests/test_data/v1_reference was written by fse 1.1.0 from PyPI with
benchmarks/baseline/make_v1_fixture.py. 2.0 pools only, so the pooling configurations
(Average, of plain and FastText vectors) are compared; the SIF and uSIF rows of the
fixture also removed components, which fse 2.0 no longer does at encode time.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fse._contract import Batch
from fse._reference import pool_weighted_mean

FIXTURE = Path(__file__).parent / "test_data" / "v1_reference"
CONFIGS = ["Average", "AverageFT"]
DTYPES = ["float32", "float16", "int8"]


@pytest.fixture(scope="module")
def ref():
    with np.load(FIXTURE / "outputs.npz") as data:
        arrays = dict(data)
    sentences = json.loads((FIXTURE / "sentences.json").read_text(encoding="utf-8"))
    words = json.loads((FIXTURE / "words.json").read_text(encoding="utf-8"))
    return arrays, sentences, words


def test_fixture_is_intact():
    manifest = json.loads((FIXTURE / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["fse"] == "1.1.0"
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("dtype", ["float16", "int8"])
def test_storage_rules_reproduce_the_codes_of_1_1_0(ref, dtype):
    """The 2.0 quantization equals the word vector codes 1.1.0 computed."""
    from fse._reference import quantize_rows

    arrays, _, _ = ref
    codes, scales = quantize_rows(arrays["kv.vectors"], dtype)
    assert codes.tobytes() == arrays[f"Average.{dtype}.rows"].tobytes()
    if dtype == "int8":
        assert scales.tobytes() == arrays[f"Average.{dtype}.scales"].tobytes()


def assert_rows_close(got, expected, tolerance=1e-5):
    """Equal within ``tolerance`` of each row's largest value, also for tiny rows."""
    got = np.asarray(got, dtype=np.float32)
    scale = np.abs(expected).max(axis=1, keepdims=True)
    zero = (scale == 0).ravel()
    np.testing.assert_array_equal(got[zero], 0)
    relative = np.abs(got - expected)[~zero] / scale[~zero]
    assert relative.max(initial=0) < tolerance, float(relative.max())


def _batch(arrays, kind):
    keys = ["offsets", "tokens", "ngram_offsets", "ngrams"]
    parts = {k: arrays[f"batch.{kind}.{k}"] for k in keys if f"batch.{kind}.{k}" in arrays}
    return Batch(0, **parts)


def v2_rows(arrays, name, dtype):
    """The 2.0 reference pooling of the rows and weights of this configuration."""
    prefix = f"{name}.{dtype}"
    fasttext = name == "AverageFT"
    if dtype == "float32":
        rows = arrays["ft.vectors_vocab" if fasttext else "kv.vectors"]
        ngram_rows = arrays["ft.vectors_ngrams"] if fasttext else None
    else:
        rows = arrays[f"{prefix}.rows"]
        ngram_rows = arrays.get(f"{prefix}.ngram_rows")
    weights = arrays[f"{prefix}.weights"]
    return pool_weighted_mean(
        _batch(arrays, "ft" if fasttext else "kv"),
        rows,
        weights,
        arrays.get(f"{prefix}.scales"),
        ngram_rows,
        arrays.get(f"{prefix}.ngram_scales"),
        oov_weight=float(weights.max()) if fasttext else None,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", CONFIGS)
def test_v2_reference_is_close_to_1_1_0_infer(ref, name, dtype):
    arrays, _, _ = ref
    expected = arrays[f"{name}.{dtype}.infer"]
    got = v2_rows(arrays, name, dtype)
    # Measured on Apple M1 (Accelerate): averaging is bit-identical. Other BLAS libraries
    # may not fuse saxpy, which adds a rounding per token.
    assert_rows_close(got, expected)
    # Empty sentences are zero rows in both; train in 1.x fills them with EPS instead.
    empty = np.flatnonzero(
        np.diff(arrays[f"batch.{'ft' if name == 'AverageFT' else 'kv'}.offsets"]) == 0
    )
    assert len(empty)
    np.testing.assert_array_equal(got[empty], 0)
