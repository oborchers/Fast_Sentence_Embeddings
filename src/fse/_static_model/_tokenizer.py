#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The read-only tokenizer of a ``StaticModel`` (ARCHITECTURE.md, "model2vec compatibility").

``model2vec.StaticModel.tokenizer`` is a ``tokenizers.Tokenizer`` with truncation at the
model's ``max_length`` and padding off. This facade offers what that object does for the
callers of a static model, with the same ids: ``encode`` and ``encode_batch`` (an object
with ``ids``), the vocabulary and the lookup of a token by id and back. The
unknown id is kept, as in ``tokenizers``. fse has no post-processor (no ``[CLS]`` and
``[SEP]``) and no offsets, so ``add_special_tokens=True``, the default of ``tokenizers``,
raises ``ValueError``: pass ``False``, as model2vec does itself. An encoding has no
``tokens`` either: ``tokenizers`` names an unknown piece of a Unigram text by the characters
it matched, which only offsets could give back. Nothing can be changed on it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Encoding:
    """The ids of one text (``tokenizers.Encoding`` has more: no tokens, offsets, masks or
    type ids here)."""

    ids: list[int]


def _no_special_tokens(add_special_tokens: object) -> None:
    if add_special_tokens is not False:
        raise ValueError(
            f"add_special_tokens={add_special_tokens!r}: this tokenizer adds no special tokens "
            "([CLS], [SEP]), as the one of model2vec's StaticModel is used; pass "
            "add_special_tokens=False"
        )


class Tokenizer:
    """The tokenizer of a ``StaticModel``: its ids, with the model's current ``max_length``
    as the cut in tokens and the unknown id kept. ``ids`` gives the lists of ids of a
    list of texts; ``tokens`` is the vocabulary in the order of the ids."""

    __slots__ = ("_by_token", "_ids", "_tokens")

    def __init__(self, ids: Callable[[Any], list[list[int]]], tokens: tuple[str, ...]):
        self._ids = ids
        self._tokens = tokens
        self._by_token: dict[str, int] | None = None

    def encode(self, sequence: str, add_special_tokens: bool = True) -> Encoding:
        """The ids of ``sequence`` (``add_special_tokens=False`` only)."""
        _no_special_tokens(add_special_tokens)
        if not isinstance(sequence, str):
            raise TypeError(f"sequence must be a str, not {type(sequence).__name__}")
        return self._encodings([sequence])[0]

    def encode_batch(self, input: Any, add_special_tokens: bool = True) -> list[Encoding]:
        """As ``encode``, for a list, tuple or numpy array of str."""
        _no_special_tokens(add_special_tokens)
        return self._encodings(input)

    encode_batch_fast = encode_batch

    def _encodings(self, texts: Any) -> list[Encoding]:
        return [Encoding(ids) for ids in self._ids(texts)]

    def get_vocab(self, with_added_tokens: bool = True) -> dict[str, int]:
        """The token to id mapping (a new dict; ``with_added_tokens=True`` only: the
        vocabulary does not tell added tokens apart)."""
        return dict(self._mapping(with_added_tokens))

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        """The number of tokens."""
        return len(self._mapping(with_added_tokens))

    def token_to_id(self, token: str) -> int | None:
        """The id of ``token``, or None."""
        if not isinstance(token, str):
            raise TypeError(f"token must be a str, not {type(token).__name__}")
        return self._mapping(True).get(token)

    def id_to_token(self, id: int) -> str | None:
        """The token of ``id``, or None when there is no such id (``tokenizers`` takes a u32
        and raises ``OverflowError`` outside it; a bool is refused here, where it counts as 0
        or 1 there)."""
        if isinstance(id, bool) or not isinstance(id, int):
            raise TypeError(f"id must be an int, not {type(id).__name__}")
        if not 0 <= id < 2**32:
            raise OverflowError(f"id must fit an unsigned 32-bit integer, not {id}")
        return self._tokens[id] if id < len(self._tokens) else None

    def _mapping(self, with_added_tokens: object) -> dict[str, int]:
        if with_added_tokens is not True:
            raise ValueError(
                f"with_added_tokens={with_added_tokens!r}: the vocabulary does not tell the "
                "added tokens apart; use with_added_tokens=True"
            )
        mapping = self._by_token
        if mapping is None:
            mapping = self._by_token = {token: i for i, token in enumerate(self._tokens)}
        return mapping

    def __reduce__(self):
        raise TypeError("a Tokenizer cannot be pickled; use StaticModel.from_pretrained")

    def __repr__(self) -> str:
        return f"Tokenizer({len(self._tokens)} tokens)"
