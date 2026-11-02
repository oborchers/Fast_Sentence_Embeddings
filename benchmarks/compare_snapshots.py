"""Resolve only the pinned inference files used by the selected loader."""

from __future__ import annotations

from pathlib import Path

MINILM_FILES = [
    "model.safetensors",
    "config.json",
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "1_Pooling/config.json",
]


def inference_files(repo: str) -> list[str]:
    if repo.startswith("minishlab/"):
        return ["model.safetensors", "tokenizer.json", "config.json", "modules.json"]
    if repo == "sentence-transformers/all-MiniLM-L6-v2":
        return MINILM_FILES.copy()
    return ["*.safetensors", "*.json", "*.txt", "*.model"]


def cached_snapshot(repo: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo, revision=revision, local_files_only=True, allow_patterns=inference_files(repo)
        )
    )
