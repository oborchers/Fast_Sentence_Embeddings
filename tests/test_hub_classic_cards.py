"""Classic cards retain the core template and disclose source terms and offline fit."""

import importlib
from pathlib import Path

import pytest

from support.hub_classics import delivery_fixture

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
pytestmark = pytest.mark.skipif(
    not (BENCHMARKS / "hub_classic_cards.py").exists(), reason="benchmarks absent"
)
hub = pytest.importorskip("huggingface_hub")


@pytest.mark.parametrize(
    "name",
    [
        "glove-wiki-gigaword-300",
        "word2vec-google-news-300",
        "paranmt-300",
        "paragram-300-sl999",
        "fasttext-crawl-subwords-300",
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_source_terms_fit_metadata_and_quantized_relation(monkeypatch, tmp_path, name, dtype):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    cards = importlib.import_module("hub_classic_cards")
    prepare = importlib.import_module("prepare_hub_classics")
    models, records, verification, fit, _ = delivery_fixture(monkeypatch, tmp_path, name)
    record = next(r for r in records if r["method"] == "sif10" and r["dtype"] == dtype)
    folder = models / record["repo"].split("/")[1]
    text = cards.render(folder, record, verification, fit)
    (tmp_path / "README.md").write_text(text)
    metadata = hub.metadata_load(tmp_path / "README.md")
    assert metadata["license"] == prepare.LICENSES[name]
    assert metadata["base_model"] == f"fse/{name}"
    assert metadata.get("base_model_relation") == ("quantized" if dtype == "int8" else None)
    assert metadata["datasets"] == ["Salesforce/wikitext"]
    assert {
        "fse",
        "static-embeddings",
        "gensim",
        "word-embeddings",
        cards.SOURCE_TAGS[name],
    } <= set(metadata["tags"])
    assert fit["state_xxh3"] in text
    assert "wordfreq version: 3.1.1" in text
    assert "never on the evaluation texts" in text
    assert "untouched holdout" in text
    if name.startswith("word2vec"):
        assert "released for research purposes only" in text
        assert metadata["license_link"] == "https://code.google.com/archive/p/word2vec/"
    if name.startswith("fasttext"):
        assert "Facebook" in text
        assert "ShareAlike" in text
        assert "Changes:" in text
    if name.startswith(("paranmt", "paragram")):
        assert "did not state a license" in text
        assert "request removal" in text
