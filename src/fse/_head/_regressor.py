#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``fse.Regressor``: an encoder and a linear regression head (``predict``)."""

from __future__ import annotations

from typing import Any, Self

from fse._encoder import _pretrained
from fse._validation import PathLike, validate_arguments

from . import _model2vec_files, _sklearn
from ._model import LinearModel
from ._regressor_predict import _RegressorPredicting


class Regressor(_RegressorPredicting, LinearModel):
    """Predicts values of texts with an encoder and a linear head ``z = A x + b`` on its
    rows, one value per target. Make one with ``Regressor.from_sklearn``,
    ``Regressor.from_arrays`` or ``Regressor.load``; fse trains nothing. A head trained on
    raw means folds into the table.
    """

    __slots__ = ()
    TASK = "regression"

    @classmethod
    @validate_arguments
    def from_sklearn(cls, encoder: Any, model: Any, *, normalized: bool | None = None) -> Self:
        """The head of a fitted scikit-learn linear regressor (or a Pipeline of an optional
        ``StandardScaler`` and the regressor) on the rows of
        ``encoder``; reads its attributes and imports nothing of scikit-learn."""
        found = _sklearn.read(model, "regression")
        return cls._build(
            encoder,
            found.weight,
            found.bias,
            normalized=normalized,
            source=found.source,
            flat=found.flat,
        )

    @classmethod
    @validate_arguments
    def from_arrays(
        cls, encoder: Any, weight: Any, bias: Any = None, *, normalized: bool | None = None
    ) -> Self:
        """The head ``weight`` (targets, d) or (d,) and ``bias`` (targets, a scalar or None)
        on the rows of ``encoder``; ``normalized`` as for ``Classifier.from_arrays``."""
        return cls._build(encoder, weight, bias, normalized=normalized)

    @classmethod
    @validate_arguments
    def load(cls, path: PathLike, mmap: bool = False) -> Self:
        """Open a regressor saved with ``save``, complete or standalone.

        Without ``mmap`` the arrays are read into memory and every checksum is checked.
        With ``mmap=True`` they are mapped read-only and only the structure is checked, so
        the file must stay unchanged while the regressor is in use (as for ``WordVectors.load``)."""
        return cls._load(path, mmap)

    @classmethod
    @validate_arguments
    def from_pretrained(
        cls,
        repo_or_path: PathLike,
        *,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: PathLike | None = None,
        local_files_only: bool = False,
        mmap: bool = False,
    ) -> Self:
        """The regressor of a local ``.safetensors`` file or of an fse folder written by
        ``save_pretrained``, local or a Hugging Face Hub repository (``namespace/name``) at
        ``revision`` (default branch if None), checked against its commit; ``mmap`` as for
        ``load``, for local files."""
        files = _pretrained.files(
            repo_or_path,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        _model2vec_files.refuse_for_regressor(files)
        return cls._from_files(files, mmap)
