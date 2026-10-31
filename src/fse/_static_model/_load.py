#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Where ``StaticModel.from_pretrained`` finds a model, and under which rules.

A Model2Vec model and a sentence-transformers static repository are read under the rules of
model2vec (``fse._encoder._model2vec_rules``); an fse encoder (a local ``.safetensors``
file, or a folder or Hub repository whose ``config.json`` says ``model_type`` fse) is
loaded as an ``Encoder`` and must fit the closed profile of ``fse.StaticModel``
(``fse._static_model._profile``).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from fse._encoder import Encoder
from fse._encoder._model2vec_rules import Rules, Unset, card_fields, find_rules, subfolder_of
from fse._encoder._pretrained import MAX_CARD_BYTES, LocalFiles, files

from ._profile import profile_rules


def load_rules(
    path: Any,
    *,
    revision: str | None,
    token: Any,
    cache_dir: Any,
    local_files_only: bool,
    subfolder: str | None,
    normalize: bool | None,
    max_length: int | Unset | None,
    dimensionality: int | None,
) -> Rules:
    """The rules of the model at ``path`` (a local folder or file, or a Hub repository)."""
    found = files(
        path, revision=revision, token=token, cache_dir=cache_dir, local_files_only=local_files_only
    )
    if not isinstance(found, Path):
        rules = find_rules(
            subfolder_of(found, subfolder),
            normalize=normalize,
            max_length=max_length,
            dimensionality=dimensionality,
        )
        if rules is not None:
            return rules
    if subfolder is not None:
        raise ValueError("subfolder is for the folders of Model2Vec models, not for an fse model")
    encoder = _encoder(path, found, token, cache_dir, local_files_only)
    rules = profile_rules(
        encoder, normalize=normalize, max_length=max_length, dimensionality=dimensionality
    )
    return replace(rules, language=_card_language(found))


def _card_language(found: Any) -> str | list[str] | None:
    """The ``language`` of the model card of an fse folder (None: no card, or a file)."""
    if isinstance(found, Path) or not found.has("README.md"):
        return None
    card = found.read("README.md", limit=MAX_CARD_BYTES)
    return card_fields(card, found.where("README.md"))["language"]


def _encoder(path: Any, found: Any, token: Any, cache_dir: Any, local_files_only: bool) -> Encoder:
    """The fse encoder at ``path``; a Hub repository at the commit that was looked at."""
    if isinstance(found, Path | LocalFiles):
        return Encoder.from_pretrained(path)
    return Encoder.from_pretrained(
        path,
        revision=found.revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
