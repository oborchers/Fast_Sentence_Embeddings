"""Numerical witnesses for the measurement oracle, including preserved conversion rules."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fse
from fse._encoder._format import Source
from fse._tokenizer_spec import TokenizerSpec, WordPieceConfig

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
pytestmark = pytest.mark.skipif(
    not (BENCHMARKS / "mlx_gate_parity.py").exists() or sys.platform == "win32",
    reason="gate benchmark sources and POSIX process observation required",
)


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("mlx_gate_parity")


@pytest.fixture
def data(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("mlx_gate_data")


@pytest.fixture
def model():
    wv = fse.WordVectors.from_arrays(
        np.array([[0, 0], [1, 0.5], [-0.5, 1]], np.float32), words=["[UNK]", "hello", "world"]
    )
    tokenizer = TokenizerSpec("wordpiece@1", WordPieceConfig((), 0), "drop", 2, 10)
    return fse.Encoder._converted(
        wv,
        tokenizer,
        normalize=True,
        source=Source("model2vec", "example/table", "abc"),
        license="mit",
    )


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("mode", ["raw", "norm"])
def test_real_encoder_variant_uses_public_quantization_and_keeps_model_rules(
    data, model, monkeypatch, dtype, mode
):
    original, calls = fse.Encoder.quantize, []

    def quantize(self, kind):
        calls.append(kind)
        return original(self, kind)

    monkeypatch.setattr(fse.Encoder, "quantize", quantize)
    enc = data.variant(model, mode, dtype)
    assert calls == ([] if dtype == "float32" else [dtype])
    assert enc.word_vectors.dtype == dtype
    assert bool(enc.steps) == (mode == "norm")
    assert enc._content.tokenizer == model._content.tokenizer
    assert enc._content.source == model._content.source
    assert enc._content.license == "mit"
    rows = enc.encode(["hello unknown", "", "world"], workers=1)
    assert rows.dtype == np.float32
    assert rows.shape == (3, 2)
    assert not rows[1].view(np.uint32).any()


def test_full_row_audit_checks_empty_rows_and_counts_model_tokens(gate, data, model):
    enc = data.variant(model, "raw", "int8")
    texts = ["hello", "", "world"]
    rows = enc.encode(texts, workers=1)
    guard = SimpleNamespace(check=lambda *_a: None)
    assert gate.audit(enc, texts, rows, rows.copy(), guard) == {
        "bit_equal": True,
        "differing_rows": 0,
        "differing_values": 0,
        "max_error_over_bound": 0.0,
        "tokens": 2,
        "empty_rows": 1,
        "contract_passed": True,
    }
    wrong = rows.copy()
    wrong[1, 0] = -0.0
    with pytest.raises(AssertionError, match="positive zero"):
        gate.audit(enc, texts, rows, wrong, guard)


def test_a_one_ulp_difference_passes_but_a_dropped_token_fails(gate, data, model):
    enc = data.variant(model, "norm", "float16")
    texts = ["hello"]
    cpu = enc.encode(texts, workers=1)
    guard = SimpleNamespace(check=lambda *_a: None)
    other = np.nextafter(cpu, np.full_like(cpu, np.inf))
    result = gate.audit(enc, texts, cpu, other, guard)
    assert not result["bit_equal"]
    assert 0 < result["max_error_over_bound"] < 1
    with pytest.raises(AssertionError, match="exceeds the contract"):
        gate.audit(enc, texts, cpu, np.zeros_like(cpu), guard)


def test_bound_includes_long_sum_rounding_and_absolute_flush_term(gate):
    k, dim, m = 1024, 256, 2.0
    bound, exact = gate.acceptance(k, dim, m, None)
    assert not exact
    assert bound > 1e-5 * m
    assert bound == (1e-5 + 2 * (k * 2**-24 / (1 - k * 2**-24))) * m + 2**-125 * k * dim**0.5
    flush, _ = gate.acceptance(1, 256, 0, None)
    assert flush == 2**-121
    for n in (0, 2**24):
        assert gate.acceptance(n, 256, 1, None) == (0.0, True)


def test_tiny_or_cancelled_normalization_requires_exact_bits(gate):
    assert gate.acceptance(1, 2, 1, 0)[1]
    assert gate.acceptance(1, 2, 0, 2**-60)[1]
    got, want = np.array([0.0], np.float32), np.array([-0.0], np.float32)
    with pytest.raises(AssertionError, match="bit-identical"):
        gate.check_bound(got, want, 1, True)
    with pytest.raises(AssertionError, match="finiteness"):
        gate.equal(np.array([np.nan], np.float32), got)


def test_launch_observation_is_untimed_and_restores_after_error(gate, monkeypatch):
    from fse._backends._mlx import _kernels

    def original(*_a, **_k):
        return "launched"

    monkeypatch.setattr(_kernels, "launch", original)

    def fail():
        with gate.launches() as seen:
            assert _kernels.launch() == "launched"
            assert seen == [1]
            raise ValueError("stop")

    with pytest.raises(ValueError, match="stop"):
        fail()
    assert _kernels.launch is original


def test_wrong_snapshot_revision_or_layout_is_rejected(data, tmp_path):
    revision = "1" * 40
    path = tmp_path / "models--example--table" / "snapshots" / revision
    path.mkdir(parents=True)
    receipt = [{"repo": "example/table", "type": "model", "revision": revision, "path": str(path)}]
    assert data.snapshot(receipt, "example/table", revision, "model") == path
    with pytest.raises(ValueError, match="does not pin"):
        data.snapshot(receipt, "example/table", "2" * 40, "model")
    receipt[0]["path"] = str(tmp_path)
    with pytest.raises(ValueError, match="unexpected snapshot"):
        data.snapshot(receipt, "example/table", revision, "model")


def test_dataset_selection_ignores_other_cached_revisions(data, tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    repo, revision, _ = data.DATASETS["wikitext"]
    monkeypatch.setitem(data.DATASETS, "wikitext", (repo, revision, 2))
    root = tmp_path / "datasets--Salesforce--wikitext" / "snapshots"
    path = root / revision
    (path / "wikitext-103-raw-v1").mkdir(parents=True)
    for split, text in (("validation", "a"), ("test", "b")):
        pq.write_table(
            pa.table({"text": [" ", text]}), path / "wikitext-103-raw-v1" / f"{split}-00000.parquet"
        )
    (root / "unrelated").mkdir()
    receipt = [{"repo": repo, "type": "dataset", "revision": revision, "path": str(path)}]
    cell = data.matrix()[0]
    texts, metadata = data.corpus(cell, receipt)
    assert texts == ["a", "b"]
    assert all(revision in file for file in metadata["files"])
    assert metadata["checksum"].startswith("xxh3-128:")
