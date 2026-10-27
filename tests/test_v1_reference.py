"""The 1.1.0 reference fixture: 1.x keeps its results, and the 2.0 reference stays close.

The fixture in tests/test_data/v1_reference was written by fse 1.1.0 from PyPI with
benchmarks/baseline/make_v1_fixture.py.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fse._contract import Batch, RemoveComponents
from fse._reference import pool_weighted_mean, postprocess

FIXTURE = Path(__file__).parent / "test_data" / "v1_reference"
CONFIGS = ["Average", "SIF", "uSIF", "AverageFT"]
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


def _model(name, dtype, arrays, words):
    from gensim.models import FastText, KeyedVectors

    from fse import SIF, Average, uSIF

    if name == "AverageFT":
        ft = FastText(vector_size=12, min_count=1, bucket=500, seed=0)
        ft.build_vocab([words["ft"]])
        wv = ft.wv
        order = [wv.key_to_index[w] for w in words["ft"]]
        wv.vectors_vocab[order] = arrays["ft.vectors_vocab"]
        wv.vectors_ngrams[:] = arrays["ft.vectors_ngrams"]
        counts = arrays["ft.counts"]
    else:
        wv = KeyedVectors(12)
        wv.add_vectors(words["kv"], arrays["kv.vectors"])
        counts = arrays["kv.counts"]
    for word, count in zip(words["ft" if name == "AverageFT" else "kv"], counts, strict=True):
        wv.set_vecattr(word, "count", int(count))
    kwargs = {"SIF": {"components": 2}, "uSIF": {"components": 2, "length": 11}}.get(name, {})
    cls = {"SIF": SIF, "uSIF": uSIF}.get(name, Average)
    return cls(wv, wv_dtype=dtype, **kwargs)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", CONFIGS)
def test_1x_reproduces_1_1_0(ref, name, dtype):
    from fse import IndexedList

    arrays, sentences, words = ref
    model = _model(name, dtype, arrays, words)
    data = IndexedList(sentences)
    model.train(data)
    prefix = f"{name}.{dtype}"
    np.testing.assert_allclose(model.sv.vectors, arrays[f"{prefix}.train"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model.infer(data), arrays[f"{prefix}.infer"], rtol=1e-5, atol=1e-6)


def _batch(arrays, kind):
    keys = ["offsets", "tokens", "ngram_offsets", "ngrams"]
    parts = {k: arrays[f"batch.{kind}.{k}"] for k in keys if f"batch.{kind}.{k}" in arrays}
    return Batch(0, **parts)


def v2_rows(arrays, name, dtype):
    """The 2.0 reference applied to what 1.1.0 learned for this configuration."""
    prefix = f"{name}.{dtype}"
    fasttext = name == "AverageFT"
    if dtype == "float32":
        rows = arrays["ft.vectors_vocab" if fasttext else "kv.vectors"]
        ngram_rows = arrays["ft.vectors_ngrams"] if fasttext else None
    else:
        rows = arrays[f"{prefix}.rows"]
        ngram_rows = arrays.get(f"{prefix}.ngram_rows")
    weights = arrays[f"{prefix}.weights"]
    out = pool_weighted_mean(
        _batch(arrays, "ft" if fasttext else "kv"),
        rows,
        weights,
        arrays.get(f"{prefix}.scales"),
        ngram_rows,
        arrays.get(f"{prefix}.ngram_scales"),
        oov_weight=float(weights.max()) if fasttext else None,
    )
    if f"{prefix}.components" in arrays:
        step = RemoveComponents(
            np.ascontiguousarray(arrays[f"{prefix}.components"]),
            arrays.get(f"{prefix}.component_weights"),
        )
        out = postprocess(out, [step])
    return out


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", CONFIGS)
def test_v2_reference_is_close_to_1_1_0_infer(ref, name, dtype):
    arrays, _, _ = ref
    expected = arrays[f"{name}.{dtype}.infer"]
    got = v2_rows(arrays, name, dtype)
    # Measured on Apple M1 (Accelerate): averaging is bit-identical, the removal of
    # components (BLAS in 1.x, dot16 in 2.0) differs by at most 3e-7 of the largest value
    # of a row. Other BLAS libraries may not fuse saxpy, which adds a rounding per token.
    scale = np.maximum(np.abs(expected).max(axis=1, keepdims=True), np.finfo(np.float32).tiny)
    relative = float((np.abs(got - expected) / scale).max())
    assert relative < 1e-5, relative
    # Empty sentences are zero rows in both; train in 1.x fills them with EPS instead.
    empty = np.flatnonzero(
        np.diff(arrays[f"batch.{'ft' if name == 'AverageFT' else 'kv'}.offsets"]) == 0
    )
    assert len(empty)
    np.testing.assert_array_equal(got[empty], 0)
