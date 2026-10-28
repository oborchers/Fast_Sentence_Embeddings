"""Texts, words and helpers the tests of the text sources share."""

import numpy as np

from fse import _native
from fse._wordvectors import FastTextParams, WordVectors

F32 = np.float32
TEXTS = [
    "I don't know.",
    "",
    "The cat sat; the dog didn't.",
    "caf\u00e9 \u4e2d\u56fd \u0130stanbul",
    "   ",
    "well-known U.S. facts, 1,000 of them",
    "x",
]
WORDS = [
    "i",
    "do",
    "not",
    "know",
    "the",
    "cat",
    "sat",
    "dog",
    "did",
    "caf\u00e9",
    "istanbul",
    "well",
    "known",
    "u.s",
    "1,000",
    "of",
    "x",
]


def vectors(fasttext=False):
    if not fasttext:
        return WordVectors.from_arrays(np.ones((len(WORDS), 3), F32), words=WORDS)
    return WordVectors.from_arrays(
        np.ones((len(WORDS), 3), F32),
        words=WORDS,
        fasttext=FastTextParams(3, 5, 50),
        ngrams=np.ones((50, 3), F32),
    )


def ids():
    return {w.encode(): i for i, w in enumerate(WORDS)}


def encode_all(source, lookup, workers=1, chunk_bytes=None):
    scan = source.scan(workers=workers, chunk_bytes=chunk_bytes)
    ws = _native.Workspace()
    return scan, [source.encode(chunk, lookup, ws) for chunk in scan.chunks]


def concat(batches):
    tokens = np.concatenate([b.tokens for b in batches]) if batches else np.zeros(0, np.uint32)
    offsets = [0]
    for b in batches:
        base = offsets[-1]
        offsets.extend(base + int(o) for o in b.offsets[1:])
    return tokens, np.array(offsets, np.int64)
