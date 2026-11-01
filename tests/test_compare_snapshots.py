"""Pinned inference lookup must not require unused exports or evaluation YAML."""

import fnmatch
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "repo", ["minishlab/potion-multilingual-128M", "sentence-transformers/all-MiniLM-L6-v2"]
)
def test_pinned_lookup_uses_all_required_cached_inference_files(monkeypatch, tmp_path, repo):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    module = importlib.import_module("compare_snapshots")
    cached = {"model.safetensors", "tokenizer.json", "config.json", "modules.json"}
    if repo.startswith("sentence-transformers/"):
        cached |= {
            "1_Pooling/config.json",
            "config_sentence_transformers.json",
            "sentence_bert_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        }
    missing = {"onnx/model.onnx", "pytorch_model.bin", ".eval_results/task.yaml"}
    if repo.startswith("minishlab/"):
        missing |= {"tokenizer_config.json", "special_tokens_map.json", "vocab.txt"}

    def download(name, *, revision, local_files_only, allow_patterns=None):
        selected = {
            p
            for p in cached | missing
            if allow_patterns is None
            or any(fnmatch.fnmatch(p, pattern) for pattern in allow_patterns)
        }
        if not selected <= cached:
            raise FileNotFoundError("unused repository files are not cached")
        assert cached <= selected
        assert (name, revision, local_files_only) == (repo, "immutable-revision", True)
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    assert module.cached_snapshot(repo, "immutable-revision") == tmp_path


def test_static_embedding_selection_includes_nested_weights_and_tokenizer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    module = importlib.import_module("compare_snapshots")
    patterns = module.inference_files("sentence-transformers/static-retrieval-mrl-en-v1")
    for name in (
        "0_StaticEmbedding/model.safetensors",
        "0_StaticEmbedding/tokenizer.json",
        "modules.json",
        "config_sentence_transformers.json",
    ):
        assert any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
