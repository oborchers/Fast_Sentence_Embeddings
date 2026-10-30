#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The Hugging Face folder of an fse model: ``config.json``, ``model.safetensors`` and a
``README.md`` model card, written atomically, and pushed to the Hub.

ARCHITECTURE.md, "Hub folders". ``config.json`` is the query file by which the Hub counts
downloads; it names the model file and repeats its kind, format version and fingerprint,
which a reader compares with the metadata it actually loaded. The model file is exactly
what ``save`` writes; config and card are made from its header. A folder is written into
a temporary sibling and published by an exclusive rename, so ``path`` is either complete
or absent, never replaced; extra files (LICENSE, NOTICE) are checked and copied as the
bytes that were read. ``push`` saves into a new subfolder of a temporary directory and
hands it to ``fse._hub_push``.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fse import _hub_push
from fse._container import (
    CorruptArtifactError,
    PublishedNotDurableError,
    discard_temporary,
    publish_directory,
    read_regular,
    safetensors_header,
)
from fse._container._common import _open_regular, _sync_directory
from fse._contract import UnsupportedArtifactError

from . import _card

CONFIG = "config.json"
MODEL = "model.safetensors"
CARD = "README.md"
CONFIG_VERSION = 1
CONFIG_KEYS = frozenset(
    {"model_type", "fse_config_version", "fse_kind", "fse_format_version", "model", "fingerprint"}
)
KINDS = ("encoder", "linear_head")
# The provisional config of a pre-release: its file under the key "encoder", no version.
_PROVISIONAL = frozenset({"model_type", "fse_kind", "fse_format_version", "encoder"})
# A plain file name of the repository: no folder, no pattern characters.
_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
MAX_EXTRA_BYTES = 2**20


def config_bytes(metadata: dict[str, str]) -> bytes:
    """The ``config.json`` of the model file with ``metadata`` (its header)."""
    config = {
        "fingerprint": metadata["fse.fingerprint"],
        "fse_config_version": CONFIG_VERSION,
        "fse_format_version": int(metadata["fse.format_version"]),
        "fse_kind": metadata["fse.kind"],
        "model": MODEL,
        "model_type": "fse",
    }
    return (json.dumps(config, sort_keys=True, indent=2) + "\n").encode("utf-8")


def parse_config(config: dict, where: Path) -> dict:
    """The checked ``config.json`` of an fse folder (``"model_type": "fse"``)."""
    if set(config) == _PROVISIONAL:
        raise UnsupportedArtifactError(
            f"{where} was written by a pre-release of 2.0 (its model file under the key "
            "encoder); save the model again"
        )
    if set(config) != CONFIG_KEYS:
        missing, extra = sorted(CONFIG_KEYS - set(config)), sorted(set(config) - CONFIG_KEYS)
        raise UnsupportedArtifactError(
            f"{where} of an fse model holds exactly {sorted(CONFIG_KEYS)}; it lacks {missing} "
            f"and holds {extra}"
        )
    if not _is_int(config["fse_config_version"]) or config["fse_config_version"] != CONFIG_VERSION:
        raise UnsupportedArtifactError(
            f"{where} has fse_config_version {config['fse_config_version']!r}; this fse reads "
            f"version {CONFIG_VERSION} (a newer fse wrote it)"
        )
    version = config["fse_format_version"]
    if config["fse_kind"] not in KINDS or not _is_int(version) or version != 1:
        raise UnsupportedArtifactError(
            f"{where} names a {config['fse_kind']!r} of format version "
            f"{config['fse_format_version']!r}; this fse reads encoders and linear heads of "
            "version 1"
        )
    if config["model"] != MODEL:
        raise CorruptArtifactError(f"{where}: model must be {MODEL!r}, not {config['model']!r}")
    if not isinstance(config["fingerprint"], str):
        raise CorruptArtifactError(f"{where}: fingerprint must be a str")
    return config


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def check_loaded(config: dict, metadata_fingerprint: str | None, kind: str, where: Path) -> None:
    """The model that was loaded is the one the config names: its kind, and the
    fingerprint of the metadata actually loaded (not of an earlier look at the file)."""
    if config["fse_kind"] != kind:
        raise CorruptArtifactError(
            f"{where} names the kind {config['fse_kind']}, the file holds a {kind}"
        )
    if metadata_fingerprint != config["fingerprint"]:
        raise CorruptArtifactError(
            f"{where} names the fingerprint {config['fingerprint']}, but the model file that "
            f"was loaded has {metadata_fingerprint}; the folder's files do not belong together"
        )


def extra_files(files: Mapping[str, Any] | None, reserved: frozenset[str]) -> dict[str, bytes]:
    """The checked extra files (repository name to local path): plain names that are not
    the model's own, regular UTF-8 text files of at most 1 MiB, each read once."""
    out: dict[str, bytes] = {}
    for name, source in (files or {}).items():
        if not isinstance(name, str) or not _FILE_NAME.fullmatch(name) or name == ".gitattributes":
            raise ValueError(f"{name!r} is no plain file name for an extra file")
        if name in reserved:
            raise ValueError(f"{name} is a file of the model itself, not an extra file")
        data = read_regular(Path(source), limit=MAX_EXTRA_BYTES)
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"{source} is no UTF-8 text; extra files are texts") from None
        out[name] = data
    return out


def save(
    path: Any,
    write_model: Callable[[Path], None],
    *,
    license: Any = None,
    language: list[str] | None = None,
    model_name: str | None = None,
    extra: Mapping[str, Any] | None = None,
    location: str | None = None,
) -> tuple[Path, Any]:
    """Write the folder of a model at the new ``path``: ``write_model(file)`` writes its
    ``.safetensors`` file, config and card are made from that file's header. Returns the
    published path and the license of the card."""
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(f"{path} exists; a model folder is never written into or replaced")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"{path.parent} is no directory")
    extras = extra_files(extra, frozenset({CONFIG, MODEL, CARD}))
    tmp = Path(tempfile.mkdtemp(prefix=f".{path.name}.writing-", dir=path.parent))
    try:
        write_model(tmp / MODEL)
        metadata, shapes = _header(tmp / MODEL)
        chosen = _card.resolve_license(metadata, license)
        card = _card.card(
            metadata,
            shapes,
            license=chosen,
            language=language,
            model_name=path.name if model_name is None else model_name,
            location=location or str(path),
        )
        for name, data in {CONFIG: config_bytes(metadata), CARD: card, **extras}.items():
            _write_new(tmp / name, data)
        _sync_directory(tmp)
    except BaseException as error:
        discard_temporary(tmp, error)
        raise
    try:
        publish_directory(tmp, path)
    except PublishedNotDurableError:
        raise  # published and complete: path stays, tmp is gone
    except FileExistsError:
        exists = FileExistsError(f"{path} exists; another writer published it first")
        discard_temporary(tmp, exists)
        raise exists from None
    except BaseException as error:
        discard_temporary(tmp, error)  # nothing published, or tmp is gone already
        raise
    return path, chosen


def _header(file: Path) -> tuple[dict[str, str], dict[str, tuple[int, ...]]]:
    """The metadata and tensor shapes of the model file this call wrote."""
    with _open_regular(file) as handle:
        _, entries, metadata = safetensors_header(handle, file)
    return dict(metadata), {name: tuple(spec[1]) for name, spec in entries.items()}


def _write_new(path: Path, data: bytes) -> None:
    with open(path, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def push(
    repo_id: str,
    write_model: Callable[[Path], None],
    what: str,
    hub: dict[str, Any],
    card: dict[str, Any],
) -> str:
    """Push the folder of a model to ``repo_id`` in one checked commit; returns its URL.
    ``hub`` holds the options of ``push_folder`` (token, private, revision, create_pr,
    commit_message, replace), ``card`` those of ``save`` (license, language, model_name,
    extra). A model without a license (recorded or given) is refused before anything is
    asked of the Hub."""
    if not isinstance(repo_id, str) or not _hub_push.REPO_ID.fullmatch(repo_id):
        raise ValueError(f"{repo_id!r} is no Hub repository id of the form namespace/name")
    with tempfile.TemporaryDirectory(prefix="fse-push-") as tmp:
        # A new folder: save never writes into an existing one.
        folder, chosen = save(Path(tmp) / "model", write_model, location=repo_id, **card)
        if chosen is None:
            raise ValueError(
                "the model records no license; pass license= (the license of the model) to "
                "push it, as a public repository without one cannot be used by others"
            )
        message = hub.pop("commit_message") or f"Upload {what} with fse"
        return _hub_push.push_folder(folder, repo_id, commit_message=message, **hub)
