"""The runtime gives the bits of the reference pipeline for every backend.

Every source kind, dtype, step list, FastText, worker count and small chunks: the
reference tokenizes with fse._reference_text, pools with fse._reference.pool_weighted_mean
and runs fse._reference.postprocess, all in one batch.
"""

import functools

import numpy as np
import pytest

from fse import _native
from fse._contract import Normalize
from fse._runtime import EncodeSpec, Pipeline, _executor, _pipeline, _tasks
from fse._text import StringSource
from support.conformance import require_mlx
from support.runtime import expected, source, texts, word_vectors

F32 = np.float32


STEPS = {"none": (), "normalize": (Normalize(),)}


@functools.cache
def _spec(dtype, fasttext, steps, tokenizer):
    wv = word_vectors(dtype, fasttext)
    weights = np.random.default_rng(3).random(len(wv)).astype(F32)
    extra = {"oov_weight": 0.75, "max_ngrams": 12} if fasttext else {}
    return EncodeSpec(wv, weights, STEPS[steps], tokenizer, **extra)


@functools.cache
def _expected(dtype, fasttext, steps, tokenizer, n):
    return expected(_spec(dtype, fasttext, steps, tokenizer), texts(n))


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("kind", ["strings", "file", "tokens", "stream"])
@pytest.mark.parametrize("fasttext", [False, True], ids=["plain", "fasttext"])
@pytest.mark.parametrize("steps", list(STEPS))
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("backend", ["cpu", "numpy", "mlx"])
def test_runtime_gives_the_reference_bits(
    tmp_path, monkeypatch, backend, dtype, steps, fasttext, kind, workers
):
    if backend == "mlx":
        require_mlx()
    tokenizer = "external" if kind == "tokens" else "fse_words@1"
    n = 120 if backend == "numpy" else 500
    sentences = texts(n)
    spec = _spec(dtype, fasttext, steps, tokenizer)
    want = _expected(dtype, fasttext, steps, tokenizer, n)
    src = source(kind, sentences, tmp_path)
    pools = []
    if workers > 1:  # threads, also for inputs small enough to run inline
        monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 0)
        real = _executor.ThreadPoolExecutor
        monkeypatch.setattr(
            _executor, "ThreadPoolExecutor", lambda **kw: pools.append(1) or real(**kw)
        )
    got = Pipeline(spec, backend).infer(src, workers=workers, chunk_bytes=2048)
    assert pools == ([1] if workers > 1 else [])
    assert got.shape == want.shape
    assert got.view(np.uint32).tobytes() == want.view(np.uint32).tobytes()


@pytest.mark.parametrize("isa", _native.available_isas())
def test_every_isa(isa):
    spec = _spec("int8", True, "normalize", "fse_words@1")
    sentences = texts(800)
    got = Pipeline(spec, "cpu", isa=isa).infer(StringSource(sentences), workers=4, chunk_bytes=1024)
    assert got.tobytes() == expected(spec, sentences).tobytes()


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
@pytest.mark.parametrize("env", [{"flush": True}, {"round_up": True}])
def test_workers_ignore_their_floating_point_environment(monkeypatch, env, backend):
    """Worker threads that start with flush-to-zero or rounding up give the same bits."""
    if backend == "mlx":
        require_mlx()
    spec = _spec("float32", False, "normalize", "fse_words@1")
    sentences = texts(2000)
    want = expected(spec, sentences)
    made = _pipeline.new_workspace
    saved = []

    def hooked():
        saved.append(_native._test_fpenv_set(**env))  # in the worker thread, which then ends
        return made()

    monkeypatch.setattr(_pipeline, "new_workspace", hooked)
    before = _native._test_fpenv_get()
    got = Pipeline(spec, backend).infer(StringSource(sentences), workers=3, chunk_bytes=1024)
    assert _native._test_fpenv_get() == before
    assert len(saved) >= 2
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("backend", ["cpu", "mlx"])
def test_a_subnormal_oov_weight_survives_a_foreign_environment(backend):
    """oov_weight is converted by the kernel in its own environment, not by EncodeSpec."""
    if backend == "mlx":
        require_mlx()
    sentences = texts(300)
    wv = word_vectors("float32", True)
    weights = np.ones(len(wv), F32)
    tiny = np.float32(np.finfo(F32).smallest_subnormal) * np.float32(3)
    want = expected(EncodeSpec(wv, weights, oov_weight=tiny, max_ngrams=12), sentences)
    saved = _native._test_fpenv_set(flush=True)
    try:
        spec = EncodeSpec(wv, weights, oov_weight=tiny, max_ngrams=12)
        pipeline = Pipeline(spec, backend)
    finally:
        _native._test_fpenv_restore(saved)
    got = pipeline.infer(StringSource(sentences), workers=1)
    assert got.tobytes() == want.tobytes()
    assert np.any(want != 0)
