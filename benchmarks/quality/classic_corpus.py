"""The complete cached WikiText-103 training fit corpus, never evaluation texts."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from quality.classic_sources import FIT_LINES, FIT_REPO, FIT_REVISION


def tokenize(text: str) -> list[str]:
    from nltk import word_tokenize

    result = []
    for token in word_tokenize(text):
        if re.match(".*[A-Za-z0-9].*", token):
            token = token.lower().strip("';.:()").strip('"')
            result.extend(re.split("[-]", "not" if token == "n't" else token))
    return [token for token in result if token]


def fit_corpus(cache: Path) -> Path:
    import nltk
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    target = cache / "wikitext-103-b08601e0.txt"
    if target.exists():
        return target
    folder = Path(
        snapshot_download(
            FIT_REPO,
            repo_type="dataset",
            revision=FIT_REVISION,
            allow_patterns=["wikitext-103-raw-v1/train-*.parquet"],
            max_workers=2,
        )
    )
    nltk.download("punkt_tab", quiet=True, raise_on_error=True)
    cache.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".preparing")
    count = 0
    with temporary.open("x", encoding="utf-8") as stream:
        for file in sorted((folder / "wikitext-103-raw-v1").glob("train-*.parquet")):
            for batch in pq.ParquetFile(file).iter_batches(columns=["text"], batch_size=4096):
                for row in batch.column(0).to_pylist():
                    text = row.strip()
                    if text and not text.startswith("=") and re.search("[A-Za-z0-9]", text):
                        stream.write(" ".join(tokenize(text)) + "\n")
                        count += 1
    if count != FIT_LINES:
        raise ValueError(f"fit corpus has {count} paragraphs, expected {FIT_LINES}")
    temporary.rename(target)
    return target


def batches(path: Path, size: int = 16384):
    batch = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            batch.append(line.split())
            if len(batch) == size:
                yield batch
                batch = []
    if batch:
        yield batch


def frequencies(path: Path) -> tuple[Counter, int]:
    counts, rows = Counter(), 0
    for batch in batches(path):
        for tokens in batch:
            counts.update(tokens)
        rows += len(batch)
    if rows != FIT_LINES:
        raise ValueError(f"fit corpus has {rows} paragraphs, expected {FIT_LINES}")
    return counts, rows
