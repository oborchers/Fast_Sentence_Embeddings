#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""A model read under the rules of model2vec 0.10.0 itself (ARCHITECTURE.md, "model2vec
compatibility").

``Encoder.from_pretrained`` reads a sentence-transformers repository with the rules of
sentence-transformers. ``model2vec.StaticModel.from_pretrained`` reads the same repository
with its own: it takes the first of three folder layouts whose three files exist
(``config.json`` with ``model.safetensors`` and ``tokenizer.json``;
``config_sentence_transformers.json`` with ``model.safetensors`` and ``tokenizer.json``;
the same two files in ``0_StaticEmbedding/``), uses that config as the config, ignores
``modules.json`` and prompts, drops the unknown id, cuts to 512 tokens and to 512 times
the median token length in characters, overrides the truncation of the tokenizer.json and
normalizes only if the config says so (default false). This module reads the files that way, for
``fse.StaticModel``, and gives the parts of an encoder: the table, the parameters of the
tokenizer, the length rule and whether to normalize. It adds two refusals model2vec does
not have: a module other than a StaticEmbedding or a Normalize (a Dense projection that
model2vec would drop silently), and every file that is not read the way the encoder
needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._tokenizer_json import MAX_BYTES as MAX_TOKENIZER_BYTES
from fse._tokenizer_json import tokenizer_of
from fse._tokenizer_spec import WordPieceConfig
from fse._tokenizer_unigram import UnigramConfig
from fse._wordvectors import WordVectors

from ._files import MAX_CONFIG_BYTES, Files, parse_config, read_table
from ._format import Source
from ._front_matter import front_matter, strings
from ._model2vec import DEFAULT_MAX_LENGTH, check_config, median_length, model_table
from ._pretrained import MAX_CARD_BYTES, LocalFiles, _check_modules, card_license


class Unset:
    """The value of an argument that was not given (None means "no limit" there)."""

    def __repr__(self) -> str:
        return "<unset>"


UNSET = Unset()


@dataclass(frozen=True)
class Layout:
    """The three files of one folder layout of model2vec, relative to the model's folder."""

    config: str
    embeddings: str
    tokenizer: str
    sentence_transformers: bool

    def complete(self, files: Files | LocalFiles) -> bool:
        return all(files.has(name) for name in (self.config, self.embeddings, self.tokenizer))


# In the order model2vec tries them (persistence/datamodels.py, FOLDER_LAYOUTS).
LAYOUTS = (
    Layout("config.json", "model.safetensors", "tokenizer.json", False),
    Layout("config_sentence_transformers.json", "model.safetensors", "tokenizer.json", True),
    Layout(
        "config_sentence_transformers.json",
        "0_StaticEmbedding/model.safetensors",
        "0_StaticEmbedding/tokenizer.json",
        True,
    ),
)
_ST_TABLE = frozenset({"embedding.weight"})
_FOLDER = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
# The value of a model card field: a repository id or a language code.
_PLAIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+-]*")


@dataclass(frozen=True, eq=False)
class Rules:
    """A model read under the rules of model2vec: the table with its vocabulary, the
    parameters of the tokenizer (the unknown id is always dropped), the ``config`` as
    stored, the resolved ``normalize`` and ``max_length`` (None: no cut), the median token
    length, the fields of the model card and where the model came from."""

    word_vectors: WordVectors
    tokens: tuple[str, ...]
    parameters: WordPieceConfig | UnigramConfig
    config: dict[str, Any]
    normalize: bool
    max_length: int | None
    median_token_length: int
    base_model_name: str | list[str] | None
    language: str | list[str] | None
    source: Source | None
    license: str | tuple[str, ...] | None


class Prefixed:
    """The files of ``files`` below the folder ``prefix`` (a subfolder of the model)."""

    def __init__(self, files: Files | LocalFiles, subfolder: str):
        self._files = files
        self._prefix = subfolder + "/"
        self.source = files.source
        self.revision = files.revision

    def has(self, name: str) -> bool:
        return self._files.has(self._prefix + name)

    def where(self, name: str) -> Path:
        return self._files.where(self._prefix + name)

    def read(
        self, name: str, *, lfs: bool = False, limit: int | None = None, inspect: Any = None
    ) -> bytes:
        return self._files.read(self._prefix + name, lfs=lfs, limit=limit, inspect=inspect)


def subfolder_of(files: Files | LocalFiles, subfolder: str | None) -> Files | LocalFiles:
    """``files`` below ``subfolder`` (folders of plain names, joined by ``/``), or itself."""
    if subfolder is None:
        return files
    if not all(_FOLDER.fullmatch(part) and part != ".." for part in subfolder.split("/")):
        raise ValueError(
            f"subfolder must be folders of plain names joined by '/', not {subfolder!r}"
        )
    return Prefixed(files, subfolder)


def find_rules(
    files: Files | LocalFiles,
    *,
    normalize: bool | None,
    max_length: int | Unset | None,
    dimensionality: int | None,
) -> Rules | None:
    """The model in ``files`` read under model2vec's rules, or None for an fse model (a
    config.json of ``model_type`` fse), which holds its own."""
    layout = next((layout for layout in LAYOUTS if layout.complete(files)), None)
    if layout is None:
        if _is_fse(files):
            return None
        raise _no_layout()
    config = _config(files, layout)
    if files.has("modules.json"):
        _check_modules(files, None if layout.sentence_transformers else config.get("model_type"))
    tokens, parameters = tokenizer_of(
        files.read(layout.tokenizer, limit=MAX_TOKENIZER_BYTES), files.where(layout.tokenizer)
    )
    table = _table(files, layout, config, len(tokens))
    width = table.shape[1]
    table, kept = reduced(table, dimensionality)
    words = WordVectors._from_arrays(table, tokens, None, None, None, {}, copy=False)
    card = files.read("README.md", limit=MAX_CARD_BYTES) if files.has("README.md") else None
    fields = card_fields(card, files.where("README.md"))
    return Rules(
        words,
        tuple(tokens),
        parameters,
        config,
        config.get("normalize", False) if normalize is None else normalize,
        config.get("max_length", DEFAULT_MAX_LENGTH)
        if isinstance(max_length, Unset)
        else max_length,
        median_length(tokens),
        fields["base_model"],
        fields["language"],
        Source("model2vec", files.source, files.revision, None if kept is None else (width, kept)),
        card_license(card, files.where("README.md")),
    )


def _is_fse(files: Files | LocalFiles) -> bool:
    """A config.json of model_type fse (a folder ``Encoder.save_pretrained`` wrote)."""
    if not files.has("config.json"):
        return False
    where = files.where("config.json")
    config = parse_config(files.read("config.json", limit=MAX_CONFIG_BYTES), where)
    return isinstance(config, dict) and config.get("model_type") == "fse"


def _no_layout() -> UnsupportedArtifactError:
    """No layout of model2vec is complete: the error names the three."""
    tried = "; ".join(
        f"{layout.config} with {layout.embeddings} and {layout.tokenizer}" for layout in LAYOUTS
    )
    return UnsupportedArtifactError(
        f"the model has none of the folder layouts model2vec reads ({tried}); a "
        "Model2Vec or static sentence-transformers model has one of them"
    )


def _config(files: Files | LocalFiles, layout: Layout) -> dict[str, Any]:
    """The config of the layout. A Model2Vec config is checked as ``Encoder`` checks it
    (every key known); the config_sentence_transformers.json of the other layouts is the
    config as it is (model2vec reads only normalize and max_length from it), and those two
    keys are checked."""
    where = files.where(layout.config)
    config = parse_config(files.read(layout.config, limit=MAX_CONFIG_BYTES), where)
    if not layout.sentence_transformers:
        return check_config(config, where)
    if not isinstance(config, dict):
        raise CorruptArtifactError(f"{where} must be a JSON object")
    if not isinstance(config.get("normalize", False), bool):
        raise CorruptArtifactError(f"{where}: normalize must be true or false")
    limit = config.get("max_length", DEFAULT_MAX_LENGTH)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise CorruptArtifactError(f"{where}: max_length must be a positive int or null")
    return config


def _table(files: Files | LocalFiles, layout: Layout, config: dict, n_tokens: int) -> np.ndarray:
    """The float32 table of the layout: ``embeddings`` of a Model2Vec model, else
    ``embedding.weight`` (model2vec reads exactly that name)."""
    if not layout.sentence_transformers:
        return model_table(files, config, n_tokens)
    return read_table(files, layout.embeddings, _ST_TABLE, n_tokens)


def reduced(table: np.ndarray, dimensionality: int | None) -> tuple[np.ndarray, int | None]:
    """The first ``dimensionality`` columns (a contiguous copy) and how many were kept
    (None: all, nothing copied)."""
    if dimensionality is None or dimensionality == table.shape[1]:
        return table, None
    if dimensionality > table.shape[1]:
        raise ValueError(
            f"dimensionality {dimensionality} is greater than the {table.shape[1]} "
            "dimensions of the model"
        )
    kept = np.ascontiguousarray(table[:, :dimensionality])
    kept.setflags(write=False)
    return kept, dimensionality


# --- the model card: base_model and language -------------------------------------------


def card_fields(data: bytes | None, path: Path) -> dict[str, str | list[str] | None]:
    """``base_model`` and ``language`` of the YAML front matter of a model card as model2vec
    reads them: a string, a list of strings, or None (no card, no key, null). Another form
    raises."""
    keys = ("base_model", "language")
    if data is None:
        return dict.fromkeys(keys)
    values = front_matter(data, path, keys)
    return {key: strings(values[key], key, path, _PLAIN) for key in keys}
