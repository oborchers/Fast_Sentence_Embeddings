"""Offline classic preparation: frequency arithmetic, folded tables and provenance."""

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import fse

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
pytestmark = pytest.mark.skipif(
    not (BENCHMARKS / "prepare_hub_classics.py").exists(), reason="benchmarks absent"
)


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return (
        importlib.import_module("hub_classic_math"),
        importlib.import_module("prepare_hub_classics"),
        importlib.import_module("verify_hub_classics"),
    )


def test_wordfreq_weights_match_the_historical_float32_frequency_arithmetic(modules, monkeypatch):
    math, _, _ = modules
    frequencies = {"frequent": 0.1, "rare": 1e-7}

    def get_frequency_dict(language, wordlist):
        assert (language, wordlist) == ("en", "best")
        return frequencies

    monkeypatch.setitem(
        sys.modules, "wordfreq", types.SimpleNamespace(get_frequency_dict=get_frequency_dict)
    )
    actual = math.wordfreq_weights(["rare", "missing", "frequent"])
    counts = np.array([214, 21, 214748364], dtype=np.int64)
    probabilities = counts.astype(np.float32)
    probabilities /= counts.sum()
    expected = (0.001 / (0.001 + probabilities)).astype(np.float32)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_folded_table_removes_the_same_directions_as_sentence_projection(modules):
    pytest.importorskip("scipy")
    math, _, _ = modules
    rng = np.random.default_rng(0)
    table = rng.normal(size=(37, 12)).astype(np.float32)
    basis = math.components(table.T @ table)
    folded = np.empty_like(table)
    math.fold_table(table, basis, folded)
    np.testing.assert_allclose(
        folded.mean(axis=0), math.remove(table.mean(axis=0), basis), atol=2e-7
    )


def test_variants_roundtrip_and_keep_original_source_and_license(modules, tmp_path):
    _, prepare, _ = modules
    encoder = fse.Encoder.from_arrays(np.eye(3, dtype=np.float32), words=["a", "b", "c"])
    records = prepare.write_variants(encoder, "glove-wiki-gigaword-300", "raw", tmp_path)
    assert [r["dtype"] for r in records] == ["float32", "float16", "int8"]
    for record in records:
        folder = tmp_path / record["repo"].split("/")[1]
        loaded = fse.Encoder.from_pretrained(folder)
        assert loaded._content.source.repo == "fse/glove-wiki-gigaword-300"
        assert loaded._content.source.revision == prepare.SOURCES["glove-wiki-gigaword-300"]
        assert loaded._content.license == "pddl"
        assert loaded.fingerprint == record["fingerprint"]
        assert json.loads((folder / "config.json").read_text())["fingerprint"] == loaded.fingerprint
    with pytest.raises(FileExistsError):
        prepare.write_variants(encoder, "glove-wiki-gigaword-300", "raw", tmp_path)


def test_retrieval_excludes_self_and_has_stable_tie_order(modules):
    _, _, verify = modules
    rows = np.ones((15, 3), dtype=np.float32)
    actual = verify.neighbors(rows)
    assert actual.shape == (15, 10)
    np.testing.assert_array_equal(actual[0], np.arange(1, 11))
    np.testing.assert_array_equal(actual[14], np.arange(10))
    assert verify.recall(actual, actual) == 1.0
    assert all(index not in neighbors for index, neighbors in enumerate(actual))


def test_fasttext_sif_weights_unknown_words_like_the_largest_known_weight(modules, tmp_path):
    gensim = pytest.importorskip("gensim")
    _, prepare, _ = modules
    kv = gensim.models.FastText(
        sentences=[["known", "word"]],
        vector_size=3,
        min_count=1,
        workers=1,
        epochs=1,
        bucket=32,
        min_n=2,
        max_n=3,
    ).wv
    weights = np.linspace(0.2, 0.7, len(kv), dtype=np.float32)
    actual = prepare.weighted_encoder(kv, weights, tmp_path)
    source = prepare.from_gensim(kv)
    expected = fse.Encoder.from_arrays(
        source.stored_vectors * weights[:, None],
        words=kv.index_to_key,
        fasttext=source.fasttext,
        ngrams=source.stored_ngrams,
        oov_weight=float(weights.max()),
        tokenizer="external",
        normalize=False,
    )
    assert actual._content.oov_weight == weights.max()
    tokens = [["unseen"], [kv.index_to_key[0], "unseen"]]
    np.testing.assert_array_equal(
        actual.encode(tokens).view(np.uint32), expected.encode(tokens).view(np.uint32)
    )
