#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Where ``Encoder.from_pretrained`` finds a model, and what kind it is.

ARCHITECTURE.md, "from_pretrained". A local ``.safetensors`` file is an fse encoder; a
local folder or a Hub repository holds a model that says what it is: a ``config.json``
with ``"model_type": "fse"`` names the file of an fse encoder; a Model2Vec model has
``config.json``, ``model.safetensors`` and ``tokenizer.json`` (as model2vec finds it, with
any or no model_type of model2vec); otherwise a ``modules.json`` makes it a
sentence-transformers model (``fse._encoder._sentence_transformers``); anything else is no
static embedding model and raises, naming what it is. The files come through ``Files``
(``fse._encoder._files``), each read once into bytes that are checked and then parsed: a
folder, whose links are followed only into the blobs of a Hugging Face cache, or a Hub
snapshot checked against its commit (``fse._hub``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fse._container import CorruptArtifactError, read_regular, safetensors_header
from fse._contract import UnsupportedArtifactError
from fse._hub import REPO_ID, HubFiles, check_folders, resolve_link

from . import _folder
from ._files import MAX_CONFIG_BYTES, NORMALIZE, STATIC_EMBEDDING, Converted, Files, parse_config
from ._format import Source
from ._front_matter import front_matter, strings
from ._model2vec import FILES, MODEL_RULE, read_model2vec
from ._sentence_transformers import read_sentence_transformers

MAX_CARD_BYTES = 16 * 2**20


@dataclass(frozen=True, eq=False)
class Found:
    """A model: a local fse model file (``path``), an fse model of the Hub as its checked
    bytes and name (``data``), or the parts of a converted model; ``config`` is the checked
    config.json of an fse folder, which the loaded model must match."""

    path: Path | None = None
    config: dict | None = None
    config_path: Path | None = None
    data: tuple[bytes, Path] | None = None
    converted: Converted | None = None
    source: Source | None = None
    license: str | tuple[str, ...] | None = None


class LocalFiles:
    """The files of a folder; a link is followed only into the blobs of the cache of a
    Hugging Face snapshot (``<repo>/snapshots/<commit>/`` next to ``<repo>/blobs``), and
    no folder below the model's folder may be a link."""

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
        check_folders(self.folder, name)
        if path.is_symlink():
            return resolve_link(path, name)[0]
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
    truncate_dim: Any = MODEL_RULE,
    normalize: Any = MODEL_RULE,
    mmap: bool = False,
) -> Found:
    """The model in ``files``: an fse encoder, or the parts of a Model2Vec or a
    sentence-transformers model."""
    rules = {"truncate": truncate, "max_length": max_length}
    st_rules = {"truncate_dim": truncate_dim, "normalize": normalize}
    if isinstance(files, Path):  # a local encoder file
        _model_rules(rules | st_rules)
        return Found(path=files)
    config = _config(files)
    kind = None if config is None else config.get("model_type")
    if config is not None and kind == "fse":
        _model_rules(rules | st_rules)
        return fse_model(files, config, "encoder", mmap)
    if mmap:
        raise ValueError("mmap is for fse encoders; a converted model is converted in memory")
    if config is not None and _is_model2vec(files, kind):
        _model_rules(st_rules)
        return _found(files, "model2vec", read_model2vec(files, config, **rules))
    if files.has("modules.json"):
        model = read_sentence_transformers(files, kind=kind, **rules, **st_rules)
        return _found(files, "sentence-transformers", model)
    raise _refusal(files, kind)


def _config(files: Files | LocalFiles) -> dict | None:
    """The parsed config.json, or None without one."""
    if not files.has("config.json"):
        return None
    where = files.where("config.json")
    config = parse_config(files.read("config.json", limit=MAX_CONFIG_BYTES), where)
    if not isinstance(config, dict):
        raise CorruptArtifactError(f"{where} must be a JSON object")
    return config


def _found(files: Files | LocalFiles, kind: str, model: Converted) -> Found:
    """A converted model with its provenance: where it came from, and its license."""
    source = Source(kind, files.source, files.revision, model.dims)
    card = files.read("README.md", limit=MAX_CARD_BYTES) if files.has("README.md") else None
    return Found(
        converted=model, source=source, license=card_license(card, files.where("README.md"))
    )


_DEFAULTS = {"truncate": True, "max_length": MODEL_RULE}
_DEFAULTS |= {"truncate_dim": MODEL_RULE, "normalize": MODEL_RULE}


def _model_rules(given: dict[str, Any]) -> None:
    """No rule of a converted model is given for a model that holds its own."""
    changed = sorted(name for name, value in given.items() if value != _DEFAULTS[name])
    if changed:
        raise ValueError(
            f"{' and '.join(changed)}: this model holds its own rules (truncate and "
            "max_length are for Model2Vec and sentence-transformers models, truncate_dim "
            "and normalize for sentence-transformers models)"
        )


def _is_model2vec(files: Files | LocalFiles, kind: Any) -> bool:
    """A config.json with model_type model2vec or none, and the files of a Model2Vec
    model, is one; its modules.json, if any, lists only static modules. Without those
    files, a modules.json makes it a sentence-transformers model (which ignores the
    config.json); without one, a missing file raises."""
    if kind not in (None, "model2vec"):
        return False
    if not all(files.has(name) for name in FILES):
        if files.has("modules.json"):
            return False
        raise _refusal(files, kind)
    _check_modules(files, kind)
    return True


def _refusal(files: Files | LocalFiles, kind: Any) -> UnsupportedArtifactError:
    """Neither an fse, a Model2Vec nor a sentence-transformers model: the error that
    names what the model is."""
    if files.has("config_sentence_transformers.json"):
        return UnsupportedArtifactError(
            "the model has config_sentence_transformers.json but no modules.json; "
            "sentence-transformers builds a Transformer and Pooling for it, which is no "
            "static embedding model"
        )
    if not files.has("config.json"):
        return UnsupportedArtifactError(
            "the model has no config.json and no modules.json: it is no model fse reads"
        )
    present = [name for name in FILES if files.has(name)]
    return UnsupportedArtifactError(
        f"the model (model_type {kind!r}, files {present}) is no static embedding "
        f"model fse reads: a Model2Vec model has {', '.join(FILES)}, a "
        "sentence-transformers model a modules.json"
    )


def find_fse(files: Files | LocalFiles | Path, kind: str, mmap: bool) -> Found:
    """The fse model of ``kind`` in ``files``: a local ``.safetensors`` file, or a folder
    with the config.json of an fse model; anything else raises."""
    if isinstance(files, Path):
        return Found(path=files)
    config = _config(files)
    if config is None or config.get("model_type") != "fse":
        raise UnsupportedArtifactError(
            f"the model is no fse folder (a config.json with model_type fse); a {kind} is "
            "read only from fse models"
        )
    return fse_model(files, config, kind, mmap)


def fse_model(files: Files | LocalFiles, config: dict, kind: str, mmap: bool) -> Found:
    """The model file named by the config.json of an fse folder, of ``kind``."""
    where = files.where("config.json")
    _folder.parse_config(config, where)
    if config["fse_kind"] != kind:
        if kind == "encoder":
            raise UnsupportedArtifactError(
                f"{where} names a linear head; open it with fse.Classifier.from_pretrained or "
                "fse.Regressor.from_pretrained"
            )
        raise UnsupportedArtifactError(
            f"{where} names an {config['fse_kind']}; open it with fse.Encoder.from_pretrained"
        )
    name = config["model"]
    if not files.has(name):
        raise CorruptArtifactError(f"the model has no file {name}, which {where} names")
    if isinstance(files, LocalFiles):
        return Found(path=files.path_of(name), config=config, config_path=where)
    if mmap:
        raise ValueError(
            "mmap is for local fse models: a file of the Hub is checked as one copy in "
            "memory; save the model and load it with mmap"
        )
    path = files.where(name)

    def header(handle, size: int) -> None:  # before the data are read
        safetensors_header(handle, path, size)

    data = files.read(name, lfs=True, inspect=header)
    return Found(data=(data, path), config=config, config_path=where)


def check_loaded(found: Found, fingerprint: str | None, kind: str) -> None:
    """The model loaded from an fse folder is the one its config.json names, compared
    with the metadata actually loaded, so a file swapped after a first look cannot pass."""
    if found.config is not None:
        assert found.config_path is not None
        _folder.check_loaded(found.config, fingerprint, kind, found.config_path)


def _check_modules(files: Files | LocalFiles, kind: Any) -> None:
    """A modules.json of sentence-transformers lists only static modules."""
    if not files.has("modules.json"):
        return
    where = files.where("modules.json")
    modules = parse_config(files.read("modules.json", limit=MAX_CONFIG_BYTES), where)
    if not isinstance(modules, list) or not all(isinstance(m, dict) for m in modules):
        raise CorruptArtifactError(f"{where} must be a list of modules")
    types = [m.get("type") for m in modules]
    if not set(types) <= STATIC_EMBEDDING | NORMALIZE:
        raise UnsupportedArtifactError(
            f"the model (model_type {kind!r}) is no static embedding model: its modules are {types}"
        )


# A license id as the Hub writes it: an SPDX id or "other".
_LICENSE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


def card_license(data: bytes | None, path: Path) -> str | tuple[str, ...] | None:
    """The ``license`` of the YAML front matter of a model card: one license id, a tuple of
    the ids of a list, or None for a card without front matter, without the key or with
    null. Anything else (another YAML value, an empty list, the key twice, a YAML error in
    the front matter) raises."""
    if data is None:
        return None
    value = strings(front_matter(data, path, ("license",))["license"], "license", path, _LICENSE_ID)
    return tuple(value) if isinstance(value, list) else value
