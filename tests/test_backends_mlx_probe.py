"""The MLX experiment (tests/support/mlx_probe.py); skipped without MLX and Metal.

Run it with: uv run --with mlx pytest -s tests/test_backends_mlx_probe.py
"""

import numpy as np
import pytest

from fse import _reference
from fse._backends import PoolInput, UnsupportedByBackendError
from fse._contract import Batch, Normalize
from fse._runtime import EncodeSpec, Pipeline, _pipeline
from fse._text import StringSource
from support.conformance import TINY
from support.mlx_probe import MlxProbe
from support.runtime import expected, texts, word_vectors

F32 = np.float32


@pytest.fixture
def probe():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():  # pragma: no cover - only on Apple silicon
        pytest.skip("Metal is not available")
    return MlxProbe()


def _cases():
    rng = np.random.default_rng(0)
    lengths = rng.integers(0, 30, 400)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    tokens = rng.integers(0, 500, offsets[-1]).astype(np.uint32)
    batch = Batch(0, offsets, tokens)
    normal = rng.standard_normal((500, 37)).astype(F32)
    subnormal = (normal * TINY * F32(64)).astype(F32)
    return {"normal": (batch, normal), "subnormal": (batch, subnormal)}


def test_report_where_metal_keeps_the_bits(probe):
    """The finding for PR 9: normal rows must match bit for bit; subnormal ones are reported."""
    weights = np.random.default_rng(1).random(500).astype(F32)
    same = {}
    for name, (batch, rows) in _cases().items():
        prepared = probe.prepare(PoolInput(rows, weights), ())
        out = np.empty((batch.n_sentences, 37), F32)
        probe.rows(batch, prepared, out)
        want = _reference.pool_weighted_mean(batch, rows, weights)
        same[name] = out.view(np.uint32).tobytes() == want.view(np.uint32).tobytes()
    print("mlx-probe bit-equal to the reference:", same)
    assert same["normal"]


def test_the_runtime_drives_the_probe(probe, monkeypatch):
    real = _pipeline.get_backend
    monkeypatch.setattr(
        _pipeline,
        "get_backend",
        lambda name, **o: probe if name == "mlx-probe" else real(name, **o),
    )
    with pytest.raises(UnsupportedByBackendError, match="mlx-probe backend cannot run the step"):
        Pipeline(EncodeSpec(word_vectors(), np.ones(200, F32), (Normalize(),)), "mlx-probe")
    spec = EncodeSpec(word_vectors(), np.random.default_rng(2).random(200).astype(F32))
    sentences = texts(2000)
    got = Pipeline(spec, "mlx-probe").infer(StringSource(sentences), workers=4, chunk_bytes=2048)
    want = expected(spec, sentences)
    print("mlx-probe runtime rows bit-equal:", got.tobytes() == want.tobytes())
    assert got.tobytes() == want.tobytes()
