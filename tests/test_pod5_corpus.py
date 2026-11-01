"""Byte aliases require proof of identical paragraph and token sequences."""

import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def corpus(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/quality"))
    return importlib.import_module("pod5_corpus")


def reference(corpus, path):
    proof = corpus.corpus_hashes(path)
    fit = {"tokenized_corpus_xxh3": proof["reference_raw_xxh3"], "paragraphs": 2}
    return {"models": [{"artifact": {"preparation_record": {"fit": fit}}}]}, proof


def test_whitespace_alias_preserves_original_metadata(corpus, tmp_path):
    path = tmp_path / "historical.txt"
    path.write_text("one  two\nthree\t four \n")
    historical, proof = reference(corpus, path)
    original = copy.deepcopy(historical)
    artifacts = [
        {"preparation_record": {"fit": {"tokenized_corpus_xxh3": proof["consumed_tokens_xxh3"]}}}
    ]
    compared, checked = corpus.verified_reference(historical, artifacts, path)
    assert historical == original
    assert compared != historical
    assert checked["changed_lines"] == 2
    assert checked["paragraphs"] == 2
    assert checked["canonical_bytes"] == len(b"one two\nthree four\n")
    with pytest.raises(ValueError, match="actual historical corpus"):
        corpus.verified_reference(historical, artifacts, None)


@pytest.mark.parametrize(
    "changed", ["two one\nthree four\n", "one two three\nfour\n", "one two\nthree five\n"]
)
def test_token_order_paragraphs_and_words_cannot_be_aliased(corpus, tmp_path, changed):
    path = tmp_path / "historical.txt"
    path.write_text("one  two\nthree four\n")
    historical, _ = reference(corpus, path)
    other = tmp_path / "other.txt"
    other.write_text(changed)
    observed = corpus.corpus_hashes(other)["consumed_tokens_xxh3"]
    artifacts = [{"preparation_record": {"fit": {"tokenized_corpus_xxh3": observed}}}]
    with pytest.raises(ValueError, match="consumed token sequences differ"):
        corpus.verified_reference(historical, artifacts, path)
    with pytest.raises(ValueError, match="recorded GloVe hash"):
        corpus.verified_reference(historical, artifacts, other)


def test_alias_matches_actual_consumers(corpus, tmp_path, monkeypatch):
    readers = importlib.import_module("classic_corpus")
    monkeypatch.setattr(readers, "FIT_LINES", 3)
    historical = tmp_path / "old.txt"
    canonical = tmp_path / "new.txt"
    historical.write_text("one  two\n \t\nthree four \n")
    canonical.write_text("one two\n\nthree four\n")
    expected = [[["one", "two"], []], [["three", "four"]]]
    assert list(readers.batches(historical, size=2)) == expected
    assert list(readers.batches(canonical, size=2)) == expected
    assert readers.frequencies(historical) == readers.frequencies(canonical)
    assert (
        corpus.corpus_hashes(historical)["consumed_tokens_xxh3"]
        == corpus.corpus_hashes(canonical)["reference_raw_xxh3"]
    )
