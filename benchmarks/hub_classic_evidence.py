"""Pinned benchmark resources and checks binding measurements to prepared tables."""

from __future__ import annotations

import json
import math
import subprocess
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import fse
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
MTEB_VERSION = "2.26.25"


def digest(path: Path) -> str:
    hasher = Hasher()
    with path.open("rb") as stream:
        while block := stream.read(8 * 2**20):
            hasher.update(block)
    return hasher.checksum()


@lru_cache(maxsize=1)
def tokenizer_identity():
    import nltk

    path = Path(str(nltk.data.find("tokenizers/punkt_tab/english")))
    actual = {name: digest(path / name) for name in PUNKT}
    if actual != PUNKT:
        raise ValueError("NLTK English punkt_tab resources do not match the pinned checksums")
    return {"nltk_version": version("nltk"), "punkt_tab_english": actual}


def benchmark_bytes():
    data = subprocess.check_output(["git", "show", f"{STS_COMMIT}:evaluation/sts-test.csv"])
    if checksum(data) != STS_CHECKSUM:
        raise ValueError("STS-B fixture checksum mismatch")
    return data


def benchmark_identity():
    return {"fixture_commit": STS_COMMIT, "fixture_xxh3": STS_CHECKSUM, **tokenizer_identity()}


def validate_score(row, expected, key, lower, upper):
    for name, value in expected.items():
        if row.get(name) != value:
            raise ValueError(f"measurement identity mismatch for {name}")
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    if not lower <= value <= upper:
        raise ValueError(f"{key} outside [{lower}, {upper}]")
    return value


def candidate_evidence(output, kind):
    fit = json.loads((output / f"glove-{kind}-fit.json").read_text())
    state = output / f"glove-{kind}-wordfreq-fit.npz"
    if digest(state) != fit["state_xxh3"]:
        raise ValueError("candidate fitted-state checksum mismatch")
    model = fse.Encoder.load(output / f"glove-{kind}-wordfreq.safetensors", mmap=True)
    if model.fingerprint != fit["candidate_fingerprint"]:
        raise ValueError("candidate fingerprint does not match its fit receipt")
    if fit.get("tokenizer") != tokenizer_identity():
        raise ValueError("candidate tokenizer resources differ from its fit receipt")
    return {
        "corpus": kind,
        "encoder_fingerprint": model.fingerprint,
        "state_xxh3": fit["state_xxh3"],
        "tokenizer": tokenizer_identity(),
    }


def validate_build(folder, record, result, fit):
    metadata = fse.Encoder.from_pretrained(folder, mmap=True)
    if (
        metadata.fingerprint != record["fingerprint"]
        or result.get("fingerprint") != record["fingerprint"]
    ):
        raise ValueError("model, build and verification fingerprints do not match")
    for name, spec in record["files"].items():
        path = folder / name
        if path.stat().st_size != spec["bytes"] or digest(path) != spec["xxh3"]:
            raise ValueError(f"build file changed: {path.name}")
    if result.get("model_xxh3") != record["files"]["model.safetensors"]["xxh3"]:
        raise ValueError("verification does not describe the recorded model file")
    if fit is not None:
        state = folder.parent / f"{record['source'].split('/')[1]}-fit.npz"
        if digest(state) != fit["state_xxh3"] or result.get("state_xxh3") != fit["state_xxh3"]:
            raise ValueError("verification and fitted state do not match")
