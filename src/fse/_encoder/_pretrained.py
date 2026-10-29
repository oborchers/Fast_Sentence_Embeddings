#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Where ``Encoder.from_pretrained`` finds a model, and what kind it is.

ARCHITECTURE.md, "from_pretrained". A local ``.safetensors`` file is an fse encoder; a
local folder or a Hub repository holds a model whose ``config.json`` says what it is:
``"model_type": "fse"`` names the file of an fse encoder; a Model2Vec model has
``config.json``, ``model.safetensors`` and ``tokenizer.json`` (as model2vec finds it, with
any or no model_type of model2vec); a sentence-transformers model is not supported yet;
anything else is no static embedding model and raises, naming what it is. The files come
through ``Files`` (``fse._encoder._model2vec``), each read once into bytes that are checked
and then parsed: a folder, whose links are followed only into the blobs of a Hugging Face
cache, or a Hub snapshot checked against its commit (``fse._hub``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fse._container import CorruptArtifactError, read_regular, safetensors_header, strict_json
from fse._contract import UnsupportedArtifactError
from fse._hub import REPO_ID, HubFiles, resolve_link

from ._format import Source
from ._model2vec import (
    FILES,
    MAX_CONFIG_BYTES,
    MODEL_RULE,
    Files,
    Model2Vec,
    parse_config,
    read_model2vec,
)

# The config.json of an fse encoder on the Hub: exactly these keys (provisional, fixed
# with the first upload of fse models).
FSE_CONFIG = {"model_type", "fse_kind", "fse_format_version", "encoder"}
_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors")
MAX_CARD_BYTES = 16 * 2**20
_STATIC_MODULES = {
    "sentence_transformers.models.StaticEmbedding",
    "sentence_transformers.models.Normalize",
}


@dataclass(frozen=True, eq=False)
class Found:
    """A model: a local fse encoder file (``path``), an fse encoder of the Hub as its
    checked bytes and name (``data``), or the parts of a Model2Vec model."""

    path: Path | None = None
    data: tuple[bytes, Path] | None = None
    model2vec: Model2Vec | None = None
    source: Source | None = None
    license: str | tuple[str, ...] | None = None


class LocalFiles:
    """The files of a folder; a link is followed only into the blobs of the cache of a
    Hugging Face snapshot (``<repo>/snapshots/<commit>/`` next to ``<repo>/blobs``)."""

    source: str | None = None
    revision: str | None = None

    def __init__(self, folder: Path):
        self.folder = folder

    def has(self, name: str) -> bool:
        return os.path.lexists(self.folder / name)

    def where(self, name: str) -> Path:
        return self.folder / name

    def path_of(self, name: str) -> Path:
        """The regular file of ``name``: itself, or the blob a cache link points to."""
        path = self.folder / name
        if path.is_symlink():
            return resolve_link(path, self.folder.parent.parent)[0]
        return path

    def read(
        self, name: str, *, lfs: bool = False, limit: int | None = None, inspect: Any = None
    ) -> bytes:
        del lfs  # nothing to check against: the files are the caller's
        return read_regular(self.path_of(name), limit=limit, inspect=inspect)


def files(
    repo_or_path: Any,
    *,
    revision: str | None,
    token: Any,
    cache_dir: Any,
    local_files_only: bool,
) -> Files | LocalFiles | Path:
    """The files of a local file or folder, or of a Hub repository."""
    path = Path(repo_or_path)
    if path.exists():
        if revision is not None or token is not None or cache_dir is not None or local_files_only:
            raise ValueError("revision, token, cache_dir and local_files_only are for the Hub")
        if path.is_dir():
            return LocalFiles(path)
        if path.suffix != ".safetensors":
            raise ValueError(f"{path} is no .safetensors file of an fse encoder, nor a folder")
        return path
    if isinstance(repo_or_path, str) and REPO_ID.fullmatch(repo_or_path):
        return HubFiles(
            repo_or_path,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    raise FileNotFoundError(
        f"{repo_or_path} is no local file or folder, nor a Hub repository id (namespace/name)"
    )


def find(
    files: Files | LocalFiles | Path,
    *,
    truncate: bool = True,
    max_length: Any = MODEL_RULE,
    mmap: bool = False,
) -> Found:
    """The model in ``files``: an fse encoder, or the parts of a Model2Vec model."""
    if isinstance(files, Path):  # a local encoder file
        _no_length_rule(truncate, max_length)
        return Found(path=files)
    if not files.has("config.json"):
        _refuse_sentence_transformers(files)
        raise UnsupportedArtifactError("the model has no config.json: it is no model fse reads")
    where = files.where("config.json")
    config = parse_config(files.read("config.json", limit=MAX_CONFIG_BYTES), where)
    if not isinstance(config, dict):
        raise CorruptArtifactError(f"{where} must be a JSON object")
    if config.get("model_type") == "fse":
        _no_length_rule(truncate, max_length)
        return _fse_encoder(files, config, where, mmap)
    _model2vec_files(files, config.get("model_type"))
    if mmap:
        raise ValueError("mmap is for fse encoders; a Model2Vec model is converted in memory")
    model = read_model2vec(files, config, truncate=truncate, max_length=max_length)
    source = Source("model2vec", files.source, files.revision)
    card = files.read("README.md", limit=MAX_CARD_BYTES) if files.has("README.md") else None
    return Found(
        model2vec=model, source=source, license=card_license(card, files.where("README.md"))
    )


def _no_length_rule(truncate: bool, max_length: Any) -> None:
    if not truncate or max_length != MODEL_RULE:
        raise ValueError(
            "an fse encoder holds its length rule; truncate and max_length are for converted models"
        )


_SENTENCE_TRANSFORMERS = (
    "the model is a sentence-transformers model; its StaticEmbedding is not supported yet "
    "(a later fse reads it)"
)


def _refuse_sentence_transformers(files: Files | LocalFiles) -> None:
    if files.has("config_sentence_transformers.json"):
        raise UnsupportedArtifactError(_SENTENCE_TRANSFORMERS)


def _model2vec_files(files: Files | LocalFiles, kind: Any) -> None:
    """The model is a Model2Vec model; anything else raises, naming what it is."""
    _check_modules(files, kind)
    present = [name for name in FILES if files.has(name)]
    if kind not in (None, "model2vec") or len(present) < len(FILES):
        _refuse_sentence_transformers(files)
        raise UnsupportedArtifactError(
            f"the model (model_type {kind!r}, files {present}) is no static embedding "
            f"model fse reads: a Model2Vec model has {', '.join(FILES)}"
        )


def _fse_encoder(files: Files | LocalFiles, config: dict, path: Path, mmap: bool) -> Found:
    """The encoder file named by the config.json of an fse model."""
    if set(config) != FSE_CONFIG:
        raise UnsupportedArtifactError(f"{path} of an fse model holds exactly {sorted(FSE_CONFIG)}")
    if config["fse_kind"] != "encoder" or config["fse_format_version"] != 1:
        raise UnsupportedArtifactError(
            f"{path} names a {config['fse_kind']!r} of format version "
            f"{config['fse_format_version']!r}; this fse reads encoders of version 1"
        )
    name = config["encoder"]
    if not isinstance(name, str) or not _FILE_NAME.fullmatch(name):
        raise CorruptArtifactError(f"{path}: encoder must name a .safetensors file, not {name!r}")
    if not files.has(name):
        raise CorruptArtifactError(f"the model has no file {name}, which {path} names")
    if isinstance(files, LocalFiles):
        return Found(path=files.path_of(name))
    if mmap:
        raise ValueError(
            "mmap is for local fse encoders: a file of the Hub is checked as one copy in "
            "memory; save the encoder and load it with mmap"
        )
    where = files.where(name)

    def header(handle, size: int) -> None:  # before the data are read
        safetensors_header(handle, where, size)

    return Found(data=(files.read(name, lfs=True, inspect=header), where))


def _check_modules(files: Files | LocalFiles, kind: Any) -> None:
    """A modules.json of sentence-transformers lists only static modules."""
    if not files.has("modules.json"):
        return
    where = files.where("modules.json")
    modules = strict_json(files.read("modules.json", limit=MAX_CONFIG_BYTES), where)
    if not isinstance(modules, list) or not all(isinstance(m, dict) for m in modules):
        raise CorruptArtifactError(f"{where} must be a list of modules")
    types = [m.get("type") for m in modules]
    if not set(types) <= _STATIC_MODULES:
        raise UnsupportedArtifactError(
            f"the model (model_type {kind!r}) is no static embedding model: its modules are {types}"
        )


# A license id as the Hub writes it (an SPDX id or "other"), and the YAML values that are
# none (null) or no string (bools and numbers of YAML 1.1 and 1.2).
_LICENSE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_NULL = {"", "~", "null", "Null", "NULL"}
_NOT_A_STRING = re.compile(
    r"(?i:true|false|yes|no|on|off|y|n|[-+]?\.inf|\.nan)"
    r"|[-+]?(\d[\d_]*(\.\d*)?|\.\d+)([eE][-+]?\d+)?|0x[0-9a-fA-F]+|0o[0-7]+"
)


def card_license(data: bytes | None, path: Path) -> str | tuple[str, ...] | None:
    """The ``license`` of the YAML front matter of a model card: one license id, a tuple of
    the ids of a list (block or flow), or None for a card without front matter, without
    the key or with null. Anything else (another YAML value, an empty list, the key twice)
    raises."""
    if data is None:
        return None
    lines = _front_matter(data.decode("utf-8", errors="strict"))
    found = [k for k, line in enumerate(lines) if line.startswith("license:")]
    if len(found) > 1:
        raise CorruptArtifactError(f"{path} states license {len(found)} times")
    if not found:
        return None
    k = found[0]
    return _licenses(
        _without_comment(lines[k][len("license:") :]), _block_items(lines[k + 1 :]), path
    )


def _licenses(value: str, items: list[str], path: Path) -> str | tuple[str, ...] | None:
    """The license of the value of the key, or of the block list that follows it."""
    if value in _NULL and not items:
        return None
    if items:
        if value:
            raise CorruptArtifactError(f"{path} states license as a value and a list")
        return tuple(_license_id(item, path) for item in items)
    if value.startswith("["):
        if not value.endswith("]") or not value[1:-1].strip():
            raise CorruptArtifactError(f"{path} states a license list fse cannot read: {value!r}")
        return tuple(_license_id(item, path) for item in value[1:-1].split(","))
    return _license_id(value, path)


def _front_matter(text: str) -> list[str]:
    """The lines between the first two lines ``---`` of a card, or none."""
    lines = text.splitlines()
    if not lines or lines[0].rstrip() != "---":
        return []
    for k, line in enumerate(lines[1:], start=1):
        if line.rstrip() in ("---", "..."):
            return lines[1:k]
    return []


def _without_comment(text: str) -> str:
    """A YAML value without a trailing comment (`` #`` outside quotes), stripped."""
    text = text.strip()
    if text[:1] in ("'", '"'):
        end = text.find(text[0], 1)
        rest = text[end + 1 :].strip() if end > 0 else "?"
        return text[: end + 1] if not rest or rest.startswith("#") else text
    return text.split(" #", 1)[0].strip()


def _block_items(lines: list[str]) -> list[str]:
    """The items of a block list (``- id`` lines, indented or not) that follows a key."""
    items = []
    for line in lines:
        match = re.match(r"\s*- (.*)$", line)
        if match is None:
            break
        items.append(_without_comment(match.group(1)))
    return items


def _license_id(text: str, path: Path) -> str:
    """One license id: a plain or quoted YAML string of an id, never null, a bool or a
    number."""
    value = text.strip()
    quoted = len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"')
    if quoted:
        value = value[1:-1]
    elif value in _NULL or _NOT_A_STRING.fullmatch(value):
        raise CorruptArtifactError(f"{path} states a license that is no string: {text!r}")
    if not _LICENSE_ID.fullmatch(value):
        raise CorruptArtifactError(f"{path} states a license fse cannot read: {text!r}")
    return value
