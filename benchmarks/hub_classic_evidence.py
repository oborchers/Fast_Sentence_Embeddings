"""Pinned benchmark resources and checks binding measurements to prepared tables."""

from __future__ import annotations

import inspect
import subprocess
import textwrap
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from fse._container._hash import Hasher, checksum

STS_COMMIT = "fc3cbfb0509f56dfe2374c4c94fd68024f8810d6"
STS_CHECKSUM = "xxh3-128:139bba1402050737ae635ecf8498280b"
WIKI_CHECKSUM = "xxh3-128:c89c1a2929270add9040835399b8bea4"
PUNKT = {
    "abbrev_types.txt": "xxh3-128:0acdbf18bd21d71f604380094bf25402",
    "collocations.tab": "xxh3-128:234e0b2209c617091029602a9025d12c",
    "ortho_context.tab": "xxh3-128:6ef2c67cab33775cdfe9a537e1c6c4ed",
    "sent_starters.txt": "xxh3-128:faa8bab2de114ffc95373f90d0fa7da0",
}


def implementation_digest(function) -> str:
    return checksum(textwrap.dedent(inspect.getsource(function)).strip().encode("utf-8"))


def digest(path: Path) -> str:
    hasher = Hasher()
    with path.open("rb") as stream:
        while block := stream.read(8 * 2**20):
            hasher.update(block)
    return hasher.checksum()


@lru_cache(maxsize=1)
def tokenizer_identity():
    import nltk
    from hub_classic_corpus import tokenize

    path = Path(str(nltk.data.find("tokenizers/punkt_tab/english")))
    actual = {name: digest(path / name) for name in PUNKT}
    if actual != PUNKT:
        raise ValueError("NLTK English punkt_tab resources do not match the pinned checksums")
    return {
        "nltk_version": version("nltk"),
        "punkt_tab_english": actual,
        "preprocessing_recipe": "fse_1x_nltk@1",
        "preprocessing_code_xxh3": implementation_digest(tokenize),
    }


def benchmark_bytes():
    data = subprocess.check_output(["git", "show", f"{STS_COMMIT}:evaluation/sts-test.csv"])
    if checksum(data) != STS_CHECKSUM:
        raise ValueError("STS-B fixture checksum mismatch")
    return data


def benchmark_identity():
    from hub_classic_metrics import normalized, score

    return {
        "fixture_commit": STS_COMMIT,
        "fixture_xxh3": STS_CHECKSUM,
        "cosine_code_xxh3": implementation_digest(normalized),
        "pearson_code_xxh3": implementation_digest(score),
        **tokenizer_identity(),
    }
