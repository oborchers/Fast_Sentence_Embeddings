#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""``fse.StaticModel``: the API of ``model2vec.StaticModel`` on fse's engine.

ARCHITECTURE.md, "model2vec compatibility". A ``StaticModel`` is a thin wrapper around fse
``Encoder`` objects of one table: the encoder of its own ``max_length`` and ``normalize``
(``encoder``), and, when ``encode`` is given other values per call, the encoders of those
(``fse._static_model._variants``). It adds no arithmetic: its rows are those of the
``Encoder``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import ndarray

from fse._encoder import Encoder
from fse._encoder._model2vec_rules import UNSET, Rules, Unset
from fse._validation import PathLike, PositiveInt, validate_arguments

from . import _arguments as arguments
from ._load import load_rules
from ._variants import Variants


class StaticModel:
    """A static embedding model that encodes as ``model2vec.StaticModel`` does: the mean
    of the rows of a text's tokens, optionally normalized, the unknown token dropped, a
    text cut to ``max_length`` tokens. Load one with ``StaticModel.from_pretrained``.

    Where model2vec ignores an argument, fse raises when it would change what the caller
    gets: an unknown keyword is a ``TypeError``, ``quantize_to`` other than float32 and
    ``vocabulary_quantization`` a ``ValueError``. The tables are float32, ``embedding`` is
    read-only, and ``encoder`` is the ``fse.Encoder`` of the model's own settings.
    """

    __slots__ = ("_rules", "_state", "_variants")
    _rules: Rules
    _state: tuple[int | None, bool, Encoder]
    _variants: Variants

    def __init__(self, *_args: Any, **_kwargs: Any):
        raise TypeError("make a StaticModel with StaticModel.from_pretrained")

    @classmethod
    def _new(cls, rules: Rules) -> StaticModel:
        out = object.__new__(cls)
        out._rules = rules
        out._variants = Variants(rules)
        out._state = (
            rules.max_length,
            rules.normalize,
            out._variants.own(rules.max_length, rules.normalize),
        )
        return out

    @classmethod
    @validate_arguments
    def from_pretrained(
        cls,
        path: PathLike,
        token: str | bool | None = None,
        normalize: bool | None = None,
        subfolder: str | None = None,
        quantize_to: Any = None,
        dimensionality: PositiveInt | None = None,
        vocabulary_quantization: PositiveInt | None = None,
        max_length: PositiveInt | Unset | None = UNSET,
        force_download: bool = True,
        *,
        revision: str | None = None,
        cache_dir: PathLike | None = None,
        local_files_only: bool = False,
    ):  # -> StaticModel
        """Load a model from a local folder or a Hugging Face Hub repository (``namespace/name``,
        at ``revision``, checked against its commit), read with the rules of model2vec 0.10.0.

        ``normalize`` and ``max_length`` default to the model's config (else False and 512;
        ``max_length=None`` cuts nothing). ``dimensionality`` keeps the first columns of the
        table. ``subfolder`` is a folder inside the model. ``quantize_to`` may be
        ``"float32"`` only, ``vocabulary_quantization`` None only. ``force_download`` is
        accepted and changes nothing: fse checks every file against the commit it reads.
        An fse encoder (a ``.safetensors`` file or an fse folder) loads if it fits
        model2vec's rules (float32, a WordPiece or Unigram tokenizer, unknown ids dropped).
        """
        del force_download  # a cache policy: no row depends on it
        arguments.loading_arguments(quantize_to, vocabulary_quantization)
        rules = load_rules(
            path,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            subfolder=subfolder,
            normalize=normalize,
            max_length=max_length,
            dimensionality=dimensionality,
        )
        return cls._new(rules)

    # --- encoding ---------------------------------------------------------------------

    def encode(
        self,
        sentences: Any,
        *,
        show_progress_bar: bool = False,
        max_length: int | Unset | None = UNSET,
        normalize: bool | None = None,
        batch_size: int = 1024,
        use_multiprocessing: bool = True,
        multiprocessing_threshold: int = 10_000,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | None = None,
    ) -> ndarray:
        """Encode a sentence (a vector of ``dim`` float32 values) or a list, tuple or numpy
        array of sentences (a matrix, one row each).

        ``max_length`` (tokens; None: no cut) and ``normalize`` default to the model's.
        ``show_progress_bar``, ``batch_size``, ``use_multiprocessing`` and
        ``multiprocessing_threshold`` change no row (fse sizes its own tasks;
        ``use_multiprocessing=False`` encodes in one thread); ``convert_to_numpy``,
        ``convert_to_tensor`` and ``device`` are accepted only with the values that
        change nothing. Streams and files go to ``encoder.encode``.
        """
        workers = arguments.encode_arguments(
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            use_multiprocessing=use_multiprocessing,
            multiprocessing_threshold=multiprocessing_threshold,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
        )
        own_length, own_normalize, encoder = self._state
        cut = (
            own_length
            if isinstance(max_length, Unset)
            else arguments.length("max_length", max_length)
        )
        norm = own_normalize if normalize is None else arguments.flag("normalize", normalize)
        if cut != own_length or norm != own_normalize:
            encoder = self._variants.get(cut, norm)  # type: ignore[arg-type]
        if isinstance(sentences, str):
            rows = encoder.encode([sentences], workers=workers)
            return rows[0]  # type: ignore[index]
        if not isinstance(sentences, list | tuple | np.ndarray):
            raise TypeError(
                f"sentences must be a str or a list, tuple or numpy array of str, not "
                f"{type(sentences).__name__}; streams and files go to StaticModel.encoder.encode"
            )
        return encoder.encode(sentences, workers=workers)  # type: ignore[return-value]

    # --- attributes -------------------------------------------------------------------

    @property
    def encoder(self) -> Encoder:
        """The ``fse.Encoder`` of the model's own ``max_length`` and ``normalize``, to use
        with sinks, stores and heads. Another setting gives another encoder."""
        return self._state[2]

    @property
    def dim(self) -> int:
        """The dimension of the rows."""
        return self._rules.word_vectors.dim

    @property
    def normalize(self) -> bool:
        """Whether rows are scaled to unit length (the default of ``encode``)."""
        return self._state[1]

    @normalize.setter
    @validate_arguments
    def normalize(self, value: bool) -> None:
        max_length = self._state[0]
        self._state = (max_length, value, self._variants.own(max_length, value))

    @property
    def max_length(self) -> int | None:
        """The tokens a text is cut to by default (None: not cut)."""
        return self._state[0]

    @max_length.setter
    @validate_arguments
    def max_length(self, value: PositiveInt | None) -> None:
        normalize = self._state[1]
        self._state = (value, normalize, self._variants.own(value, normalize))

    @property
    def config(self) -> dict[str, Any]:
        """The config the model was loaded with, with the current ``normalize`` and
        ``max_length`` (a new dict on every access)."""
        return {**self._rules.config, "normalize": self.normalize, "max_length": self.max_length}

    @property
    def tokens(self) -> tuple[str, ...]:
        """The tokens of the vocabulary, in the order of their ids."""
        return self._rules.tokens

    @property
    def embedding(self) -> ndarray:
        """The table of the rows (float32, read-only: model2vec's is writable)."""
        return self._rules.word_vectors.stored_vectors

    @property
    def weights(self) -> None:
        """Always None: models with vocabulary quantization are refused."""
        return None

    @property
    def token_mapping(self) -> None:
        """Always None: models with vocabulary quantization are refused."""
        return None

    @property
    def unk_token_id(self) -> int:
        """The id of the unknown token, which is dropped from every text."""
        return self._rules.parameters.unk_id

    @property
    def median_token_length(self) -> int:
        """The median length of the tokens, in characters (``max_length`` times it is the
        cut of a text in characters)."""
        return self._rules.median_token_length

    @property
    def base_model_name(self) -> str | list[str] | None:
        """The ``base_model`` of the model card."""
        return self._rules.base_model_name

    @property
    def language(self) -> str | list[str] | None:
        """The ``language`` of the model card."""
        return self._rules.language

    @property
    def embedding_dtype(self) -> str:
        """The dtype of the table: float32."""
        return self._rules.word_vectors.dtype

    @property
    def vocabulary_quantization(self) -> None:
        """Always None: models with vocabulary quantization are refused."""
        return None

    def __reduce__(self):
        raise TypeError("a StaticModel cannot be pickled; use StaticModel.from_pretrained")

    def __repr__(self) -> str:
        return (
            f"StaticModel({len(self.tokens)} tokens, {self.dim} dimensions, "
            f"normalize={self.normalize}, max_length={self.max_length})"
        )
