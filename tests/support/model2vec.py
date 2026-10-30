"""Tiny Model2Vec models written by model2vec itself.

Their rows are compared with model2vec's within the bound of tests/support/converted.py
(model2vec divides by ``norm + 1e-32``, which that bound covers).
"""

import numpy as np

from support import wordpiece as wp

DIM = 16


def rows(n_tokens: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n_tokens, DIM)).astype(np.float32)


def save_model(folder, table=None, *, normalize=True, max_length=512, tokens=None):
    """A Model2Vec model saved by model2vec 0.10.0 into ``folder``."""
    from model2vec import StaticModel

    tokens = wp.vocabulary() if tokens is None else tokens
    tokenizer = wp.hf_tokenizer(tokens, {t: tokens.index(t) for t in wp.SPECIAL if t in tokens})
    table = rows(len(tokens)) if table is None else table
    StaticModel(table, tokenizer, normalize=normalize, max_length=max_length).save_pretrained(
        folder
    )
    return folder


def save_unigram_model(folder, *, normalize=True, max_length=512, vocab=None):
    """A Model2Vec model with a Unigram tokenizer (the normalizer chain of
    potion-multilingual-128M, tests/support/unigram.py) saved by model2vec 0.10.0."""
    from model2vec import StaticModel

    from support import unigram

    vocab = unigram.pieces_and_scores() if vocab is None else vocab
    table = rows(len(vocab), seed=1)
    tokenizer = unigram.hf_tokenizer(vocab)
    StaticModel(table, tokenizer, normalize=normalize, max_length=max_length).save_pretrained(
        folder
    )
    return folder
