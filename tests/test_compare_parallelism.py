"""Model2vec's tokenizer setting and its optional batch pool are distinct controls."""

import importlib
from pathlib import Path


def test_wikitext_default_does_not_claim_ten_tokenizer_threads(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    settings = importlib.import_module("compare_parallelism").settings
    default = settings("model2vec", 10, 5352)
    assert default["threads"] is None
    assert default["tokenizer_thread_count"] is None
    assert default["tokenizer_parallelism"] == "on"
    assert default["parallel_batch_pool"] == "disabled"
    assert settings("model2vec", 1, 5352)["tokenizer_parallelism"] == "off"
    assert settings("fse", 10, 5352)["fse_workers"] == 10


def test_large_default_batch_uses_pool_and_disables_tokenizer_parallelism(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    settings = importlib.import_module("compare_parallelism").settings
    default = settings("model2vec", 10, 151674)
    assert default["parallel_batch_pool"] == "default"
    assert default["tokenizer_parallelism"] == "off"
