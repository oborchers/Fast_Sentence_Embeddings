#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""sentence-transformers models of one StaticEmbedding as the parts of an encoder.

ARCHITECTURE.md, "sentence-transformers models". The rules are those of
sentence-transformers 6.2.0 (``StaticEmbedding``, ``SentenceTransformer.encode``):
``modules.json`` lists exactly one StaticEmbedding; its folder holds ``model.safetensors``
(one float32 tensor, ``embedding.weight`` or model2vec's ``embeddings``) and a BERT
``tokenizer.json``, encoded without special tokens and every id pooled, ``[UNK]``
included (``fse.unknown=keep``), with no length rule but the plain truncation of the
tokenizer.json. ``truncate_dim`` (the argument, else the config's) keeps the first
dimensions of the table, which is ST's truncation of the mean. A Normalize module (ST
divides by ``max(norm, 1e-12)``, which ``normalize@1`` does not reproduce for tiny rows),
a Dense module (a projection with or without an activation, refused unread), any other module and non-empty prompts are
refused; ``normalize=True`` applies fse's own ``normalize@1`` after the truncation.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from fse import _log
from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._tokenizer_json import MAX_BYTES as MAX_TOKENIZER_BYTES
from fse._tokenizer_json import tokenizer_and_truncation
from fse._tokenizer_spec import TokenizerSpec
from fse._wordvectors import WordVectors

from ._files import (
    DENSE,
    NORMALIZE,
    STATIC_EMBEDDING,
    Converted,
    Files,
    read_config,
    read_table,
)

MODEL_RULE = "model"
CONFIG = "config_sentence_transformers.json"
_MODULE_KEYS = {"idx", "name", "path", "type"}
_FOLDER = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
_IGNORED = {"__version__", "similarity_fn_name", "requirements"}
_CONFIG_KEYS = _IGNORED | {"model_type", "prompts", "default_prompt_name", "truncate_dim"}
_TABLE_NAMES = frozenset({"embedding.weight", "embeddings"})


def read_sentence_transformers(
    files: Files,
    *,
    kind: Any = None,
    truncate: bool = True,
    max_length: Any = MODEL_RULE,
    truncate_dim: Any = MODEL_RULE,
    normalize: Any = MODEL_RULE,
) -> Converted:
    """The encoder parts of a sentence-transformers model in ``files`` (``kind``, the
    model_type of a config.json next to it, only names the model in errors)."""
    folder = static_folder(files, kind)
    stored_dim = _config(files)
    prefix = "" if folder in ("", ".") else folder + "/"
    name = prefix + "tokenizer.json"
    tokens, parameters, limit = tokenizer_and_truncation(
        files.read(name, limit=MAX_TOKENIZER_BYTES), files.where(name)
    )
    table = _table(files, prefix, len(tokens))
    width = table.shape[1]
    kept = _kept(truncate_dim, stored_dim, width, files.where(CONFIG))
    if kept is not None:
        table = np.ascontiguousarray(table[:, :kept])
        table.setflags(write=False)
    # The table is a read-only view of the bytes that were read once and checked (or its
    # own copy of their first columns), which nothing else holds.
    words = WordVectors._from_arrays(table, tokens, None, None, None, {}, copy=False)
    spec = TokenizerSpec("wordpiece@1", parameters, "keep", _length(truncate, max_length, limit))
    return Converted(
        words,
        spec,
        False if normalize == MODEL_RULE else normalize,
        None if kept is None else (width, kept),
    )


def static_folder(files: Files, kind: Any = None) -> str:
    """The folder of the StaticEmbedding of ``modules.json``, which must list it alone."""
    where = files.where("modules.json")
    modules = read_config(files, "modules.json")
    if (
        not isinstance(modules, list)
        or not modules
        or not all(isinstance(m, dict) and isinstance(m.get("type"), str) for m in modules)
    ):
        raise CorruptArtifactError(
            f"{where} must be a list of modules (a non-empty JSON list of objects with a type)"
        )
    types = [m["type"] for m in modules]
    if types[0] not in STATIC_EMBEDDING or len(types) > 1:
        _refuse(types, kind)
    module = modules[0]
    if not _MODULE_KEYS <= set(module) <= _MODULE_KEYS | {"kwargs"}:
        raise CorruptArtifactError(
            f"{where}: a module holds exactly {sorted(_MODULE_KEYS)} and optionally kwargs, "
            f"not {sorted(module)}"
        )
    kwargs = module.get("kwargs", [])  # absent is [], as sentence-transformers reads it
    if not isinstance(kwargs, list):
        raise CorruptArtifactError(
            f"{where}: the kwargs of a module must be a list, not {kwargs!r}"
        )
    if kwargs:
        raise UnsupportedArtifactError(
            f"{where}: the StaticEmbedding has kwargs {kwargs!r}, which this fse does not "
            "pass to anything"
        )
    _name(module["name"], where)
    return _folder(module["path"], where)


def _refuse(types: list[str], kind: Any) -> None:
    """A model of other modules than one StaticEmbedding raises, naming them."""
    static = types[0] in STATIC_EMBEDDING and len(types) == 2
    if static and types[1] in NORMALIZE:
        raise UnsupportedArtifactError(
            "the model normalizes by a Normalize module after its StaticEmbedding, which "
            "divides by max(norm, 1e-12); this fse does not read it, as normalize@1 "
            "(without that floor) would give other rows for tiny ones"
        )
    if types[0] in STATIC_EMBEDDING and types[1:2] and types[1] in DENSE:
        raise UnsupportedArtifactError(
            "the model has a Dense module (a projection, with or without an activation) "
            "after its StaticEmbedding, which this fse does not read"
        )
    what = "" if kind is None else f" (model_type {kind!r})"
    raise UnsupportedArtifactError(
        f"the model{what} is no static embedding model: its modules are {types}; fse reads "
        "a sentence-transformers model of one StaticEmbedding"
    )


def _name(name: Any, where: Any) -> None:
    if not isinstance(name, str) or not name or "." in name:
        raise CorruptArtifactError(
            f"{where}: the name of a module must be a non-empty str without '.', not {name!r}"
        )


def _folder(path: Any, where: Any) -> str:
    """The folder of a module: the root (``""`` or ``"."``) or one folder below it."""
    if not isinstance(path, str) or not (
        path in ("", ".") or (_FOLDER.fullmatch(path) and path != "..")
    ):
        raise CorruptArtifactError(
            f"{where}: the path of a module must be the model's folder or one folder in "
            f"it, not {path!r}"
        )
    return path


def _config(files: Files) -> int | None:
    """The truncate_dim of config_sentence_transformers.json (None without one); every
    key is known, and prompts must be empty."""
    if not files.has(CONFIG):
        return None
    where = files.where(CONFIG)
    config = read_config(files, CONFIG)
    if not isinstance(config, dict):
        raise CorruptArtifactError(f"{where} must be a JSON object")
    unknown = set(config) - _CONFIG_KEYS
    if unknown:
        raise UnsupportedArtifactError(
            f"{where} holds keys this fse does not know: {sorted(unknown)}; they could change "
            "how the model encodes"
        )
    ignored = sorted(_IGNORED & set(config))  # fse's own names of these keys
    if ignored:
        _log.emit(
            "artifact.ignored",
            "fse artifact: ignored the keys %s of a sentence-transformers config",
            ignored,
            source="sentence_transformers",
            keys=ignored,
        )
    if config.get("model_type", "SentenceTransformer") != "SentenceTransformer":
        raise UnsupportedArtifactError(
            f"{where}: the model is a {config['model_type']!r}, not a SentenceTransformer"
        )
    _prompts(config, where)
    dim = config.get("truncate_dim")
    if dim is not None and (isinstance(dim, bool) or not isinstance(dim, int) or dim < 1):
        raise CorruptArtifactError(f"{where}: truncate_dim must be a positive int or null")
    return dim


def _prompts(config: dict, where: Any) -> None:
    prompts = config.get("prompts", {})
    if not isinstance(prompts, dict) or not all(isinstance(p, str) for p in prompts.values()):
        raise CorruptArtifactError(f"{where}: prompts must map names to strings")
    given = sorted(name for name, prompt in prompts.items() if prompt)
    if given:
        raise UnsupportedArtifactError(
            f"{where} has the prompts {given}; this fse encodes texts as they are, without "
            "prompts (prepend them to the texts)"
        )
    default = config.get("default_prompt_name")
    if default is not None and default not in prompts:
        raise CorruptArtifactError(f"{where}: default_prompt_name {default!r} is no prompt")


def _table(files: Files, prefix: str, n_tokens: int) -> np.ndarray:
    name = prefix + "model.safetensors"
    if not files.has(name):
        if files.has(prefix + "pytorch_model.bin"):
            raise UnsupportedArtifactError(
                f"{files.where(prefix + 'pytorch_model.bin')} is a pickle file; this fse "
                "reads the table only from model.safetensors"
            )
        raise CorruptArtifactError(f"the model has no file {name}")
    return read_table(files, name, _TABLE_NAMES, n_tokens)


def _kept(truncate_dim: Any, stored: int | None, width: int, where: Any) -> int | None:
    """The dimensions kept, or None for all of them."""
    kept = stored if truncate_dim == MODEL_RULE else truncate_dim
    if kept is None or kept == width:
        return None
    if kept > width:
        source = f"{where} states" if truncate_dim == MODEL_RULE else "the argument is"
        raise ValueError(f"truncate_dim {kept}: {source} more than the {width} dimensions")
    return kept


def _length(truncate: bool, max_length: Any, limit: int | None) -> int | None:
    """The token cut: the plain truncation of the tokenizer.json, or none."""
    if max_length != MODEL_RULE:
        raise ValueError(
            "max_length is a length rule of Model2Vec models; a sentence-transformers model "
            "reads texts whole or by the truncation of its tokenizer.json"
        )
    return limit if truncate else None
