"""Prove training-input identity across harmless cached whitespace differences."""

from __future__ import annotations

import copy
from pathlib import Path


def corpus_hashes(path: Path) -> dict:
    from fse._container._hash import Hasher

    before = path.stat()
    raw, canonical = Hasher(), Hasher()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            raw.update(block)
    rows = canonical_bytes = changed_lines = 0
    # This is exactly classic_corpus.batches' per-paragraph consumption rule.
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            normalized = " ".join(line.split()) + "\n"
            encoded = normalized.encode("utf-8")
            canonical.update(encoded)
            rows += 1
            canonical_bytes += len(encoded)
            changed_lines += normalized != line
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("historical corpus changed during identity verification")
    return {
        "reference_raw_xxh3": raw.checksum(),
        "consumed_tokens_xxh3": canonical.checksum(),
        "paragraphs": rows,
        "reference_bytes": before.st_size,
        "canonical_bytes": canonical_bytes,
        "changed_lines": changed_lines,
        "rule": "one paragraph per line; line.split(); one-space serialization",
    }


def verified_reference(historical: dict, artifacts: list, path: Path | None):
    reference = historical["models"][0]["artifact"]["preparation_record"]["fit"]
    expected = reference["tokenized_corpus_xxh3"]
    observed = {entry["preparation_record"]["fit"]["tokenized_corpus_xxh3"] for entry in artifacts}
    if observed == {expected}:
        return historical, {"status": "identical serialized corpus", "reference_raw_xxh3": expected}
    if path is None:
        raise ValueError(
            "corpus hashes differ; provide the actual historical corpus for a token-identity proof"
        )
    proof = corpus_hashes(path)
    if proof["reference_raw_xxh3"] != expected:
        raise ValueError("historical corpus file does not match the recorded GloVe hash")
    if proof["paragraphs"] != reference["paragraphs"] or observed != {
        proof["consumed_tokens_xxh3"]
    }:
        raise ValueError("training paragraphs or consumed token sequences differ")
    compared = copy.deepcopy(historical)
    compared["models"][0]["artifact"]["preparation_record"]["fit"]["tokenized_corpus_xxh3"] = proof[
        "consumed_tokens_xxh3"
    ]
    return compared, {"status": "identical consumed tokens; serialized whitespace differs", **proof}
