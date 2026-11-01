"""A fixed-size FineWeb paragraph sample for the offline component-fit comparison."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from hub_classic_corpus import tokenize
from hub_classic_sources import FIT_LINES

REPO = "HuggingFaceFW/fineweb-edu-llama3-annotations"
REVISION = "72df4c92fb1b48beceb16016e8f695ec40a6c3a5"


def paragraphs(cache: Path):
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    folder = Path(
        snapshot_download(
            REPO,
            repo_type="dataset",
            revision=REVISION,
            token=False,
            cache_dir=cache / "fineweb" / "hub",
            local_files_only=True,
            allow_patterns=["data/*.parquet"],
        )
    )
    for number in range(8):
        path = folder / f"data/train-{number:05d}-of-00008.parquet"
        for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=1024):
            for text in batch.column(0).to_pylist():
                yield from eligible_lines(text)


def eligible_lines(text):
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if paragraph and not paragraph.startswith("=") and re.search("[A-Za-z0-9]", paragraph):
            yield paragraph


def sampled_corpus(cache: Path, output: Path) -> Path:
    target = output / "fineweb-seed0-859346.txt"
    if target.exists():
        return target
    total = sum(1 for _ in paragraphs(cache))
    if total < FIT_LINES:
        raise ValueError(f"FineWeb has only {total} eligible paragraphs")
    print(json.dumps({"fineweb_eligible_paragraphs": total}), flush=True)
    indices = np.sort(np.random.default_rng(0).choice(total, FIT_LINES, replace=False))
    cursor = 0
    temporary = target.with_suffix(".preparing")
    with temporary.open("x", encoding="utf-8") as stream:
        for index, paragraph in enumerate(paragraphs(cache)):
            if index == indices[cursor]:
                stream.write(" ".join(tokenize(paragraph)) + "\n")
                cursor += 1
                if cursor % 50000 == 0:
                    print(json.dumps({"fineweb_tokenized": cursor}), flush=True)
                if cursor == len(indices):
                    break
    if cursor != FIT_LINES:
        raise ValueError(f"sample has only {cursor} paragraphs")
    temporary.rename(target)
    (output / "fineweb-sample.json").write_text(
        json.dumps(
            {
                "repo": REPO,
                "revision": REVISION,
                "seed": 0,
                "texts": FIT_LINES,
                "eligible_paragraphs": total,
                "sample": "default_rng(0).choice(total, 859346, replace=False), sorted source order",
                "unit": "nonempty original text line with an ASCII letter or digit, excluding headings beginning =",
            },
            indent=2,
        )
        + "\n"
    )
    return target
