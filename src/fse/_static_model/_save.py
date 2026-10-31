#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``StaticModel.save_pretrained`` and ``push_to_hub``: the fse folder of the model's encoder
(docs/architecture/hub.md, "model2vec compatibility (fse.StaticModel)" (Persistence and its compatibility limit)).

The signatures are model2vec's, but the folder is not: it is the folder of
``Encoder.save_pretrained`` (``config.json`` of ``model_type`` fse, ``model.safetensors``,
a model card), which ``fse.StaticModel.from_pretrained`` and ``fse.Encoder.from_pretrained``
read and model2vec cannot. Nothing is added to the format: the ``config`` of a reloaded
model is rebuilt from what determines its rows (``normalize``, ``max_length``), its
``base_model_name`` is the Hub repository the table was converted from (``fse.source``), its
``language`` that of the model card.
"""

from __future__ import annotations

from pathlib import Path

from fse._encoder import Encoder


def language_list(language: str | list[str] | None) -> list[str] | None:
    """The ``language`` of a model as the list the model card of fse takes."""
    if language is None:
        return None
    return [language] if isinstance(language, str) else list(language)


def refuse_subfolder(subfolder: str | None) -> None:
    """fse writes one folder at ``path``; model2vec's ``subfolder`` has no counterpart."""
    if subfolder is not None:
        raise ValueError(
            f"subfolder={subfolder!r} is not supported: fse writes the model into the folder "
            "path itself; pass the folder you want (path/subfolder) as the path"
        )


def save(
    encoder: Encoder,
    language: str | list[str] | None,
    path: Path,
    model_name: str | None,
    subfolder: str | None,
) -> Path:
    """Write the folder of a model (its ``encoder`` with the current ``normalize`` and
    ``max_length``, and its ``language``) at the new ``path``."""
    refuse_subfolder(subfolder)
    return encoder.save_pretrained(path, language=language_list(language), model_name=model_name)


def push(
    encoder: Encoder,
    language: str | list[str] | None,
    repo_id: str,
    private: bool,
    token: str | bool | None,
    subfolder: str | None,
) -> str:
    """Publish the folder of ``model`` to ``repo_id`` in one commit and return its URL.
    A model whose source records no license is refused before the Hub is asked."""
    refuse_subfolder(subfolder)
    if encoder._content.license is None:
        raise ValueError(
            "the model records no license (its model card names none), and StaticModel."
            "push_to_hub has no license argument: use model.encoder.push_to_hub(repo_id, "
            "license=...) with the license of the model, as a repository without one cannot "
            "be used by others"
        )
    return encoder.push_to_hub(
        repo_id, token=token, private=private, language=language_list(language)
    )
