"""The exact recipe 00 punctuation/min-five-words rule, on pinned raw train rows."""

import re
from pathlib import Path

REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
RECIPE_SOURCE_COMMIT = "aeae3bb36b9b56efa2bf502925049e0616242754"


def split_sentences(text):
    text = " ".join(text.split())
    if text and not text.startswith("="):
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if len(sentence.split()) >= 5:
                yield sentence


def chunks():
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    folder = Path(
        snapshot_download(
            "Salesforce/wikitext",
            repo_type="dataset",
            revision=REVISION,
            allow_patterns=["wikitext-103-raw-v1/train-*.parquet"],
            max_workers=2,
        )
    )
    for part in range(2):
        source = folder / f"wikitext-103-raw-v1/train-{part:05d}-of-00002.parquet"
        for batch in pq.ParquetFile(source).iter_batches(batch_size=4096, columns=["text"]):
            for text in batch.column(0).to_pylist():
                yield from split_sentences(text)
