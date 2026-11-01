"""Classic cards retain the core template and disclose source terms and offline fit."""

import importlib
from pathlib import Path

import numpy as np
import pytest

import fse

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
pytestmark = pytest.mark.skipif(
    not (BENCHMARKS / "hub_classic_cards.py").exists(), reason="benchmarks absent"
)
hub = pytest.importorskip("huggingface_hub")


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("hub_classic_cards"), importlib.import_module(
        "prepare_hub_classics"
    )


def fit_receipt():
    return {
        "corpus": "Salesforce/wikitext",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "wordfreq_version": "3.1.1",
        "paragraphs": 859346,
        "seed": 0,
        "corpus_xxh3": "xxh3-128:" + "1" * 32,
        "state_xxh3": "xxh3-128:" + "2" * 32,
    }


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
def test_source_terms_fit_metadata_and_quantized_relation(modules, tmp_path, name, dtype):
    cards, prepare = modules
    encoder = prepare.provenance(
        fse.Encoder.from_arrays(np.eye(3, dtype=np.float32), words=["a", "b", "c"]), name
    )
    if dtype != "float32":
        encoder = encoder.quantize(dtype)
    folder = tmp_path / "model"
    encoder.save_pretrained(folder)
    repo = f"fse/{name}-sif10" + ("" if dtype == "float32" else f"-{dtype}")
    state = tmp_path / f"{name}-fit.npz"
    state.write_bytes(b"fixture fitted state")
    receipt = fit_receipt()
    receipt["state_xxh3"] = prepare.digest(state)
    record = {
        "fingerprint": encoder.fingerprint,
        "source": f"fse/{name}",
        "repo": repo,
        "method": "sif10",
        "files": {
            p.name: {"bytes": p.stat().st_size, "xxh3": prepare.digest(p)} for p in folder.iterdir()
        },
    }
    rows = [
        dict(
            repo=repo if method == "sif10" and stored == dtype else f"fixture/{method}-{stored}",
            method=method,
            dtype=stored,
            fingerprint=encoder.fingerprint,
            model_xxh3=prepare.digest(folder / "model.safetensors"),
            state_xxh3=receipt["state_xxh3"],
            pearson=50.0,
            difference_float32_points=0.0,
            recall_at_10=1.0,
            default_tokenizer_pearson=50.0,
        )
        for method in ("raw", "sif10")
        for stored in ("float32", "float16", "int8")
    ]
    text = cards.render(folder, record, {"rows": rows}, receipt)
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
    assert receipt["state_xxh3"] in text
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
