"""Finished encoders for the tests, made with Encoder.from_word_vectors."""

from fse._encoder import Encoder
from support.runtime import word_vectors


def finished(
    dtype="float32", fasttext=False, normalize=True, tokenizer="fse_words@1", dim=6, seed=0
):
    """An encoder of a table of 200 words (``w0`` to ``w199``) of ``dim`` values."""
    wv = word_vectors(dtype, fasttext, dim=dim, seed=seed)
    extra = {"max_ngrams": 12, "oov_weight": 0.75} if fasttext else {}
    return Encoder.from_word_vectors(wv, tokenizer=tokenizer, normalize=normalize, **extra)
