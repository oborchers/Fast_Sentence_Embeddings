"""Read original STS-B rows from pinned history, with a verified download fallback."""

from __future__ import annotations

import csv
import io
import subprocess
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

from fse._container._hash import checksum

COMMIT = "fc3cbfb0509f56dfe2374c4c94fd68024f8810d6"
CHECKSUMS = {
    "train": "xxh3-128:12051520f759418df14205fc3d94e0e4",
    "dev": "xxh3-128:11e6cf0bd5f34dca9a63fd7151d2caab",
    "test": "xxh3-128:139bba1402050737ae635ecf8498280b",
}


@lru_cache(maxsize=3)
def read_rows(split="test"):
    """Preserve the original row order, sentence bytes and unrounded labels."""
    if split not in CHECKSUMS:
        raise ValueError("STS-B split must be train, dev or test")
    relative = f"evaluation/sts-{split}.csv"
    try:
        data = subprocess.check_output(
            ["git", "show", f"{COMMIT}:{relative}"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        url = f"https://raw.githubusercontent.com/oborchers/Fast_Sentence_Embeddings/{COMMIT}/{relative}"
        with urlopen(url, timeout=30) as response:
            data = response.read(2**21)
    if checksum(data) != CHECKSUMS[split]:
        raise ValueError(f"STS-B {split} fixture checksum differs")
    return tuple(
        tuple(row)
        for row in csv.reader(
            io.StringIO(data.decode("utf-8"), newline=""), delimiter="\t", quoting=csv.QUOTE_NONE
        )
    )
