"""Encoders made from their content directly, for the tests before and besides fit."""

import numpy as np

from fse._contract import Normalize, Project, RemoveComponents
from fse._encoder import Encoder
from fse._encoder._format import EncoderContent
from support.runtime import word_vectors

F32 = np.float32


def basis(k, d, seed=0):
    q = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, k)))[0]
    return np.ascontiguousarray(q.T, dtype=F32)


STEPS = {
    "none": (),
    "normalize": (Normalize(),),
    "all": (
        RemoveComponents(basis(2, 12, 1), np.array([1.0, 0.5], F32)),
        Project(basis(6, 12, 2)),
        Normalize(),
    ),
}


def unfolded(dtype="float32", fasttext=False, steps="all", weights=True, tokenizer="fse_words@1"):
    wv = word_vectors(dtype, fasttext)
    w = np.random.default_rng(3).random(len(wv)).astype(F32) if weights else None
    content = EncoderContent(
        tokenizer=tokenizer,
        steps=STEPS[steps],
        input_dim=wv.dim,
        weights=w,
        oov_weight=np.array(0.75, F32) if fasttext else None,
        max_ngrams=12 if fasttext else None,
        word_vectors=wv.fingerprint,
        vocabulary=wv.vocabulary_fingerprint,
        recipe={"name": "test", "version": 1},
    )
    return Encoder._new(content, wv)


def folded(dtype="float32"):
    """A folded encoder: removal and projection folded into its own word vectors."""
    from fse._wordvectors import WordVectors

    original = word_vectors("float32")
    steps = (RemoveComponents(basis(1, 12, 4)), Project(basis(6, 12, 5)))
    rows = original.vectors.astype(np.float64)
    u = steps[0].components.astype(np.float64)
    rows = (rows - (rows @ u.T) @ u) @ steps[1].basis.astype(np.float64).T
    words = [
        original.vocab_bytes[a:b].tobytes().decode()
        for a, b in zip(original.vocab_offsets[:-1], original.vocab_offsets[1:], strict=True)
    ]
    wv = WordVectors.from_arrays(rows.astype(F32), words=words)
    wv = wv if dtype == "float32" else wv.quantize(dtype)
    content = EncoderContent(
        tokenizer="fse_words@1",
        steps=(Normalize(),),
        input_dim=6,
        weights=np.random.default_rng(3).random(len(wv)).astype(F32),
        oov_weight=None,
        max_ngrams=None,
        word_vectors=wv.fingerprint,
        vocabulary=wv.vocabulary_fingerprint,
        recipe={"name": "test", "version": 1},
        folded=steps,
        folded_from=original.fingerprint,
    )
    return Encoder._new(content, wv)
