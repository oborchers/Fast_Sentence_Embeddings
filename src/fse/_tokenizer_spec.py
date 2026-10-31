#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""A tokenizer with its parameters, as an encoder hands it to the text lookup.

docs/architecture/tokenizers.md, "wordpiece@1" and docs/architecture/formats.md, "encoder 1". ``fse_words@1`` and ``external`` have no
parameters; ``wordpiece@1`` has a ``WordPieceConfig`` (its added tokens, unknown id,
continuation prefix, word limit and the flags of its normalizer), a rule for its unknown
id (``drop`` or ``keep``) and an optional length rule (``max_tokens``, ``max_chars``). Both
types are immutable, so a pipeline built from one cannot see a change. Values this fse
does not support raise UnsupportedArtifactError, malformed ones ValueError.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar

import numpy as np
from numpy import ndarray

from fse._contract import Tokenizer, UnsupportedArtifactError, check_tokenizer
from fse._tokenizer_unigram import UnigramConfig

# What an unknown id does: model2vec drops it, sentence-transformers pools its row.
UNKNOWN_RULES = ("drop", "keep")
WORDPIECE = Tokenizer.WORDPIECE.id
UNIGRAM = Tokenizer.UNIGRAM.id
_FLAGS = ("clean_text", "handle_chinese_chars", "strip_accents", "lowercase")
# Integers of the parameters stay within int64, as the native state stores them.
_INT_LIMIT = 2**62


def _exact_int(value: object, name: str, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, not {type(value).__name__}")
    if not low <= value < _INT_LIMIT:
        raise ValueError(f"{name} must be between {low} and 2**62 - 1, not {value}")
    return value


def _added_tokens(value: object) -> tuple[tuple[int, str], ...]:
    if not isinstance(value, tuple) or not all(
        isinstance(pair, tuple) and len(pair) == 2 for pair in value
    ):
        raise ValueError("added_tokens must be a tuple of (id, content) pairs")
    ids = [_exact_int(i, "the id of an added token", 0) for i, _ in value]
    contents = [content for _, content in value]
    for content in contents:
        if not isinstance(content, str) or not content:
            raise ValueError(
                f"the content of an added token must be a non-empty str, not {content!r}"
            )
        content.encode("utf-8")  # a lone surrogate raises
    if ids != sorted(set(ids)):
        raise ValueError(f"added_tokens must be sorted by id, each id once, not {ids}")
    if len(set(contents)) != len(contents):
        raise ValueError("the contents of the added tokens must be unique")
    return value


@dataclass(frozen=True)
class WordPieceConfig:
    """The parameters of ``wordpiece@1`` (docs/architecture/tokenizers.md, "wordpiece@1")."""

    identifier: ClassVar[str] = Tokenizer.WORDPIECE.id

    added_tokens: tuple[tuple[int, str], ...]
    unk_id: int
    continuing_subword_prefix: str = "##"
    max_input_chars_per_word: int = 100
    clean_text: bool = True
    handle_chinese_chars: bool = True
    strip_accents: bool = True
    lowercase: bool = True

    def __post_init__(self):
        _added_tokens(self.added_tokens)
        _exact_int(self.unk_id, "unk_id", 0)
        prefix = self.continuing_subword_prefix
        if not isinstance(prefix, str) or not prefix:
            raise ValueError(f"continuing_subword_prefix must be a non-empty str, not {prefix!r}")
        prefix.encode("utf-8")
        _exact_int(self.max_input_chars_per_word, "max_input_chars_per_word", 1)
        flags = (self.clean_text, self.handle_chinese_chars, self.strip_accents, self.lowercase)
        for flag, value in zip(_FLAGS, flags, strict=True):
            if not isinstance(value, bool):
                raise ValueError(f"{flag} must be a bool, not {type(value).__name__}")
            if not value:
                raise UnsupportedArtifactError(
                    f"wordpiece@1 with {flag}=false is not supported by this fse; its four "
                    f"flags {', '.join(_FLAGS)} must be true"
                )

    def to_json(self) -> dict[str, Any]:
        """The parameters as a JSON object (``fse.tokenizer_config``)."""
        out: dict[str, Any] = {f.name: getattr(self, f.name) for f in fields(self)}
        out["added_tokens"] = [list(pair) for pair in self.added_tokens]
        return out

    @classmethod
    def from_json(cls, value: object) -> WordPieceConfig:
        """The parameters of a JSON object with exactly the keys of ``to_json``."""
        if not isinstance(value, dict):
            raise ValueError(f"the parameters of wordpiece@1 must be a JSON object, not {value!r}")
        names = {f.name for f in fields(cls)}
        if set(value) != names:
            missing, extra = sorted(names - set(value)), sorted(set(value) - names)
            raise ValueError(
                f"the parameters of wordpiece@1 lack {missing} and hold unknown {extra}"
            )
        added = value["added_tokens"]
        if not isinstance(added, list) or not all(
            isinstance(pair, list) and len(pair) == 2 for pair in added
        ):
            raise ValueError("added_tokens must be a list of [id, content] pairs")
        return cls(**{**value, "added_tokens": tuple(tuple(pair) for pair in added)})

    def check_vocabulary(self, vocab_bytes: ndarray, vocab_offsets: ndarray) -> None:
        """Every id is a token of the vocabulary (``0 <= id < n``), and every added token
        is the vocabulary string of its id."""
        n = len(vocab_offsets) - 1
        for name, i in [("unk_id", self.unk_id)] + [
            ("an added token", i) for i, _ in self.added_tokens
        ]:
            if not 0 <= i < n:
                raise ValueError(
                    f"{name} is {i}, but the vocabulary holds {n} tokens (ids 0 to {n - 1})"
                )
        data = np.asarray(vocab_bytes, dtype=np.uint8)
        for i, content in self.added_tokens:
            start, end = int(vocab_offsets[i]), int(vocab_offsets[i + 1])
            if not 0 <= start <= end <= len(data):
                raise ValueError(f"the offsets of token {i} lie outside the vocabulary")
            word = data[start:end].tobytes()
            if word != content.encode("utf-8"):
                raise ValueError(f"the added token {content!r} has id {i}, whose token is {word!r}")


@dataclass(frozen=True)
class TokenizerSpec:
    """A tokenizer (``name``, an identifier or ``external``) with its parameters: for
    ``wordpiece@1`` its ``config``, the rule ``unknown`` for its unknown id, and optionally
    ``max_tokens`` (keep the first ids of a sentence) and ``max_chars`` (cut the sentence to
    its first code points before, only with ``max_tokens``). Other tokenizers take none."""

    name: str
    config: WordPieceConfig | UnigramConfig | None = None
    unknown: str | None = None
    max_tokens: int | None = None
    max_chars: int | None = None

    def __post_init__(self):
        name = check_tokenizer(self.name)
        object.__setattr__(self, "name", name)
        parameters = _PARAMETERS.get(name)
        if parameters is None:
            given = [f.name for f in fields(self)[1:] if getattr(self, f.name) is not None]
            if given:
                raise ValueError(f"the tokenizer {name} has no parameters, but {given} are given")
            return
        if not isinstance(self.config, parameters):
            raise ValueError(
                f"{name} needs its parameters (a {parameters.__name__}); its encoders are "
                "converted from models that have a tokenizer"
            )
        if self.unknown not in UNKNOWN_RULES:
            raise UnsupportedArtifactError(
                f"the rule for unknown ids must be one of {', '.join(UNKNOWN_RULES)}, not {self.unknown!r}"
            )
        if self.max_tokens is not None:
            _exact_int(self.max_tokens, "max_tokens", 1)
        if self.max_chars is not None:
            if self.max_tokens is None:
                raise ValueError("max_chars is given only with max_tokens")
            _exact_int(self.max_chars, "max_chars", 1)

    @classmethod
    def of(cls, tokenizer: object) -> TokenizerSpec:
        """``tokenizer`` itself, or the spec of a tokenizer name without parameters."""
        if isinstance(tokenizer, TokenizerSpec):
            return tokenizer
        return cls(check_tokenizer(tokenizer))

    def check_vocabulary(self, vocab_bytes: ndarray, vocab_offsets: ndarray) -> None:
        """The ids of the parameters are tokens of the vocabulary (nothing to check for a
        tokenizer without parameters)."""
        if self.config is not None:
            self.config.check_vocabulary(vocab_bytes, vocab_offsets)

    def native_arguments(self) -> tuple:
        """The arguments of the native state of the tokenizer after its table and
        vocabulary (WordPieceState or UnigramState), from its parameters and rules."""
        cuts = (self.max_tokens or 0, self.max_chars or 0, self.unknown == "drop")
        config = self.config
        if isinstance(config, UnigramConfig):
            return (*config.native_arguments(), *cuts)
        if config is None:
            raise ValueError(f"the tokenizer {self.name} has no native state of its own")
        added = tuple(i for i, _ in config.added_tokens)
        prefix, chars = config.continuing_subword_prefix, config.max_input_chars_per_word
        return (config.unk_id, prefix, chars, added, *cuts)

    def describe(self) -> str:
        """The tokenizer and its rules in words, for ``repr``."""
        if self.config is None:
            return self.name
        rules = "unknown ids dropped" if self.unknown == "drop" else "unknown ids kept"
        if self.max_tokens is not None:
            rules += f", the first {self.max_tokens} tokens"
        if self.max_chars is not None:
            rules += f" of the first {self.max_chars} characters"
        return f"{self.name} ({rules})"


# The tokenizers with parameters and the type of their parameters.
_PARAMETERS: dict[str, type] = {WORDPIECE: WordPieceConfig, UNIGRAM: UnigramConfig}
