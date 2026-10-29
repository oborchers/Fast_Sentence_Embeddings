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
through ``Files``: a folder, whose links are followed only into the blobs of a Hugging
Face cache, or a verified Hub snapshot (``fse._hub``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fse._container import CorruptArtifactError, strict_json
from fse._contract import UnsupportedArtifactError
from fse._hub import REPO_ID, HubFiles, resolve_link

from ._format import Source
from ._model2vec import FILES, MODEL_RULE, Model2Vec, read_model2vec

# The config.json of an fse encoder on the Hub: exactly these keys.
FSE_CONFIG = {"model_type", "fse_kind", "fse_format_version", "encoder"}
_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors")
_SMALL_BYTES = 2**20
_STATIC_MODULES = {
    "sentence_transformers.models.StaticEmbedding",
    "sentence_transformers.models.Normalize",
}


class Files(Protocol):
    """The files of a model: ``get`` gives the path of a regular file, checked (for the
    Hub against its commit; ``lfs`` for files that must be LFS files there), or None if
    the model has no such file. ``source`` says where it came from."""

    source: str | None
    revision: str | None

    def get(self, name: str, *, lfs: bool = False) -> Path | None: ...


@dataclass(frozen=True, eq=False)
class Found:
    """A model: an fse encoder file (``path``) or the parts of a Model2Vec model."""

    path: Path | None = None
    model2vec: Model2Vec | None = None
    source: Source | None = None
    license: str | None = None


class LocalFiles:
    """The files of a folder; a link is followed only into the blobs of the cache of a
    Hugging Face snapshot (``<repo>/snapshots/<commit>/`` next to ``<repo>/blobs``)."""

    source: str | None = None
    revision: str | None = None

    def __init__(self, folder: Path):
        self.folder = folder

    def get(self, name: str, *, lfs: bool = False) -> Path | None:
        del lfs  # nothing to check against: the files are the caller's
        path = self.folder / name
        if not os.path.lexists(path):
            return None
        if path.is_symlink():
            return resolve_link(path, self.folder.parent.parent / "blobs")
        return path


def files(
    repo_or_path: Any,
    *,
    revision: str | None,
    token: Any,
    cache_dir: Any,
    local_files_only: bool,
) -> Files | Path:
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
    files: Files | Path, *, truncate: bool = True, max_length: Any = MODEL_RULE, mmap: bool = False
) -> Found:
    """The model in ``files``: an fse encoder file, or the parts of a Model2Vec model."""
    if isinstance(files, Path):  # a local encoder file
        _no_length_rule(truncate, max_length)
        return Found(path=files)
    config_path = files.get("config.json")
    if config_path is None:
        _refuse_sentence_transformers(files)
        raise UnsupportedArtifactError("the model has no config.json: it is no model fse reads")
    config = _json(config_path)
    if config.get("model_type") == "fse":
        _no_length_rule(truncate, max_length)
        return Found(path=_fse_file(files, config, config_path))
    paths = _model2vec_files(files, config.get("model_type"))
    if mmap:
        raise ValueError("mmap is for fse encoders; a Model2Vec model is converted in memory")
    model = read_model2vec(paths, truncate=truncate, max_length=max_length)
    source = Source("model2vec", files.source, files.revision)
    return Found(model2vec=model, source=source, license=card_license(files.get("README.md")))


def _no_length_rule(truncate: bool, max_length: Any) -> None:
    if not truncate or max_length != MODEL_RULE:
        raise ValueError(
            "an fse encoder holds its length rule; truncate and max_length are for converted models"
        )


def _refuse_sentence_transformers(files: Files) -> None:
    if files.get("config_sentence_transformers.json") is not None:
        raise UnsupportedArtifactError(_SENTENCE_TRANSFORMERS)


def _model2vec_files(files: Files, kind: Any) -> dict[str, Path]:
    """The files of a Model2Vec model; anything else raises, naming what it is."""
    _check_modules(files, kind)
    found = {name: files.get(name, lfs=name.endswith(".safetensors")) for name in FILES}
    paths = {name: path for name, path in found.items() if path is not None}
    if kind not in (None, "model2vec") or len(paths) < len(FILES):
        _refuse_sentence_transformers(files)
        raise UnsupportedArtifactError(
            f"the model (model_type {kind!r}, files {sorted(paths)}) is no static embedding "
            f"model fse reads: a Model2Vec model has {', '.join(FILES)}"
        )
    return paths


_SENTENCE_TRANSFORMERS = (
    "the model is a sentence-transformers model; its StaticEmbedding is not supported yet "
    "(a later fse reads it)"
)


def _json(path: Path) -> dict:
    if path.stat().st_size > _SMALL_BYTES:
        raise CorruptArtifactError(f"{path} has more than {_SMALL_BYTES} bytes")
    value = strict_json(path.read_bytes(), path)
    if not isinstance(value, dict):
        raise CorruptArtifactError(f"{path} must be a JSON object")
    return value


def _fse_file(files: Files, config: dict, path: Path) -> Path:
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
    file = files.get(name, lfs=True)
    if file is None:
        raise CorruptArtifactError(f"the model has no file {name}, which {path} names")
    return file


def _check_modules(files: Files, kind: Any) -> None:
    """A modules.json of sentence-transformers lists only static modules."""
    path = files.get("modules.json")
    if path is None:
        return
    modules = strict_json(path.read_bytes(), path) if path.stat().st_size <= _SMALL_BYTES else None
    if not isinstance(modules, list) or not all(isinstance(m, dict) for m in modules):
        raise CorruptArtifactError(f"{path} must be a list of modules")
    types = [m.get("type") for m in modules]
    if not set(types) <= _STATIC_MODULES:
        raise UnsupportedArtifactError(
            f"the model (model_type {kind!r}) is no static embedding model: its modules are {types}"
        )


def card_license(path: Path | None) -> str | None:
    """The ``license`` of the YAML front matter of a model card, if it states one as a
    plain value; a card without one gives None, a license of another form raises."""
    if path is None or path.stat().st_size > 16 * _SMALL_BYTES:
        return None
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            return None
        match = re.match(r"license:(.*)$", line)
        if match:
            value = match.group(1).strip().strip("'\"")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", value):
                raise CorruptArtifactError(f"{path} states a license fse cannot read: {line!r}")
            return value
    return None
