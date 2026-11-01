"""Reject measurements and cached resources that no longer identify the same run."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import fse

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
pytestmark = pytest.mark.skipif(
    not (BENCHMARKS / "hub_classic_evidence.py").exists(), reason="benchmarks absent"
)


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("hub_classic_evidence")


@pytest.mark.parametrize(
    "field", ["encoder_fingerprint", "state_xxh3", "dataset", "mteb_version", "tokenizer"]
)
def test_measurements_from_another_run_or_evaluator_are_refused(evidence, field):
    expected = {
        "encoder_fingerprint": "a",
        "state_xxh3": "b",
        "dataset": {"revision": "r"},
        "mteb_version": "2.26.25",
        "tokenizer": {"checksum": "c"},
    }
    row = {**expected, "main_score": 0.4}
    assert evidence.validate_score(row, expected, "main_score", 0, 1) == 0.4
    row[field] = "different"
    with pytest.raises(ValueError, match="identity mismatch"):
        evidence.validate_score(row, expected, "main_score", 0, 1)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_nonfinite_or_invalid_retrieval_scores_cannot_select_a_corpus(evidence, score):
    with pytest.raises(ValueError, match="main_score"):
        evidence.validate_score({"main_score": score}, {}, "main_score", 0, 1)


def test_changed_fit_state_or_candidate_is_refused(evidence, monkeypatch, tmp_path):
    monkeypatch.setattr(evidence, "tokenizer_identity", lambda: {})
    state = tmp_path / "glove-wikitext-wordfreq-fit.npz"
    state.write_bytes(b"initial fit")
    model = fse.Encoder.from_arrays(np.eye(2, dtype=np.float32), words=["a", "b"])
    model.save(tmp_path / "glove-wikitext-wordfreq.safetensors")
    receipt = {
        "state_xxh3": evidence.digest(state),
        "candidate_fingerprint": model.fingerprint,
        "tokenizer": {},
        "source_repo": "fse/glove-wiki-gigaword-300",
        "source_revision": "4e5ad402664236b85be45a0d107b95259ba8a909",
    }
    (tmp_path / "glove-wikitext-fit.json").write_text(json.dumps(receipt))
    assert (
        evidence.candidate_evidence(tmp_path, "wikitext")["encoder_fingerprint"]
        == model.fingerprint
    )
    state.write_bytes(b"changed fit")
    with pytest.raises(ValueError, match="fitted-state checksum"):
        evidence.candidate_evidence(tmp_path, "wikitext")


@pytest.mark.usefixtures("evidence")
def test_changed_wikitext_cache_cannot_claim_the_pinned_revision(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    corpus = importlib.import_module("hub_classic_corpus")
    path = tmp_path / "wikitext-103-b08601e0.txt"
    path.write_text("changed corpus\n")
    with pytest.raises(ValueError, match="WikiText tokenized corpus checksum"):
        corpus.fit_corpus(tmp_path)


def test_card_measurements_must_match_the_model_fingerprint(evidence, tmp_path):
    model = fse.Encoder.from_arrays(np.eye(2, dtype=np.float32), words=["a", "b"])
    folder = tmp_path / "model"
    model.save_pretrained(folder)
    record = {"fingerprint": model.fingerprint, "files": {}}
    with pytest.raises(ValueError, match="fingerprints do not match"):
        evidence.validate_build(folder, record, {"fingerprint": "different"}, None)


def test_fineweb_cache_requires_matching_sampling_and_tokenizer_evidence(
    evidence, monkeypatch, tmp_path
):
    fineweb = importlib.import_module("hub_classic_fineweb")
    monkeypatch.setattr(fineweb, "tokenizer_identity", lambda: {})
    target = tmp_path / "fineweb-seed0-859346.txt"
    target.write_text("prepared tokens\n")
    receipt = {
        "repo": fineweb.REPO,
        "revision": fineweb.REVISION,
        "seed": 0,
        "texts": fineweb.FIT_LINES,
        "tokenizer": {},
        "corpus_xxh3": evidence.digest(target),
        "sample": "default_rng(0).choice(total, 859346, replace=False), sorted source order",
        "unit": "nonempty original text line with an ASCII letter or digit, excluding headings beginning =",
    }
    path = tmp_path / "fineweb-sample.json"
    path.write_text(json.dumps(receipt))
    assert fineweb.checked_sample(target, tmp_path) == target
    receipt["seed"] = 1
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="pinned sample manifest"):
        fineweb.checked_sample(target, tmp_path)


@pytest.mark.usefixtures("evidence")
def test_existing_comparison_evidence_is_refused_before_replacing_receipts(tmp_path):
    pytest.importorskip("scipy")
    compare = importlib.import_module("compare_hub_fit")
    state = tmp_path / "glove-wikitext-wordfreq-fit.npz"
    state.write_bytes(b"existing fit")
    with pytest.raises(FileExistsError, match="no evidence was replaced"):
        compare.preflight(tmp_path, ["wikitext"])
    assert state.read_bytes() == b"existing fit"


def test_preprocessing_changes_invalidate_identity_even_with_identical_nltk_resources(
    evidence, monkeypatch, tmp_path
):
    import types

    corpus = importlib.import_module("hub_classic_corpus")
    (tmp_path / "resource.txt").write_text("fixed tokenizer resource")
    pins = {"resource.txt": evidence.digest(tmp_path / "resource.txt")}
    fake_nltk = types.SimpleNamespace(data=types.SimpleNamespace(find=lambda _name: str(tmp_path)))
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "nltk", fake_nltk)
        patch.setattr(evidence, "PUNKT", pins)
        patch.setattr(evidence, "version", lambda _name: "fixture")
        evidence.tokenizer_identity.cache_clear()
        try:
            before = evidence.tokenizer_identity()
            patch.setattr(corpus, "tokenize", lambda _text: ["changed preprocessing"])
            evidence.tokenizer_identity.cache_clear()
            after = evidence.tokenizer_identity()
            assert before["nltk_version"] == after["nltk_version"]
            assert before["punkt_tab_english"] == after["punkt_tab_english"]
            assert before["preprocessing_code_xxh3"] != after["preprocessing_code_xxh3"]
        finally:
            evidence.tokenizer_identity.cache_clear()
