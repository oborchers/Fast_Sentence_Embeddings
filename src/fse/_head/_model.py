#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""What fse.Classifier and fse.Regressor share: an encoder and one linear head.

A model holds its encoder (or, standalone, the folded table as an encoder of k values
per row) and its head ``z = A x + b``. It does not change once made. On each backend it
prepares one head pipeline on first use: the folded table for a head on raw means (A
folds into the rows, b is added after pooling), the full mean for a head on normalized
means or when the vectors are asked for. It is saved complete (encoder and head) or
standalone (the folded table and b; only for heads on raw means), as one
``.safetensors`` file (docs/architecture/formats.md, "linear_head 1").
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy import ndarray

from fse._backends import PoolInput, get_backend
from fse._container import (
    CHECKSUM_PREFIX,
    CREATED_BY_KEY,
    FINGERPRINT_KEY,
    CorruptArtifactError,
    SafetensorsContainer,
    checksum,
    fingerprint,
    tensor_code,
)
from fse._contract import UnsupportedArtifactError
from fse._cpus import usable_cpus
from fse._encoder import Encoder, _folder, _pretrained
from fse._encoder._format import to_artifact as encoder_artifact
from fse._runtime import HeadPipeline, HeadSpec, make_head
from fse._runtime._head import fold
from fse._validation import PathLike, validate_arguments
from fse._wordvectors import _created_by

from . import _content, _format
from ._content import HeadContent
from ._outputs import output_identity

# The tensors of a standalone file made new by the fold, and the tensors of the encoder
# that the fold reads, by their fields in a PoolInput.
_FOLDED = frozenset({"vectors", "ngrams"})
_POOL_TENSORS = {
    "vectors": "vectors",
    "scales": "scales",
    "ngram_vectors": "ngrams",
    "ngram_scales": "ngram_scales",
}


class LinearModel:
    """An encoder and a linear head; ``fse.Classifier`` and ``fse.Regressor`` are its two
    tasks. Made by ``from_arrays``, ``from_sklearn`` or ``load``."""

    __slots__ = (
        "_checksums",
        "_content",
        "_encoder",
        "_fingerprint",
        "_head",
        "_lock",
        "_path",
        "_pipelines",
        "_source",
    )
    _checksums: dict[str, str] | None
    _content: HeadContent
    _encoder: Encoder
    _fingerprint: str | None
    _head: HeadSpec
    _lock: threading.Lock
    _path: Path | None
    _pipelines: dict[tuple[str, bool], HeadPipeline]
    _source: str | None

    TASK = ""

    def __init__(self, *_args: Any, **_kwargs: Any):
        name = type(self).__name__
        raise TypeError(
            f"make a {name} with {name}.from_sklearn, {name}.from_arrays or {name}.load"
        )

    @classmethod
    def _new(
        cls,
        encoder: Encoder,
        content: HeadContent,
        *,
        source: str | None = None,
        path: Path | None = None,
        checksums: dict[str, str] | None = None,
        fingerprint_: str | None = None,
    ):
        out = object.__new__(cls)
        out._encoder = encoder
        out._content = content
        out._head = HeadSpec(content.weight, content.bias, content.normalized, content.link)
        out._source = source
        out._path = path
        out._checksums = checksums
        out._fingerprint = fingerprint_
        out._pipelines = {}
        out._lock = threading.Lock()
        return out

    @classmethod
    def _build(
        cls,
        encoder: Any,
        weight: Any,
        bias: Any,
        *,
        normalized: bool | None,
        link: Any = None,
        labels: Any = None,
        source: str | None = None,
        flat: bool | None = None,
    ):
        """A model of ``encoder`` and the head of ``weight`` and ``bias``, checked."""
        if not isinstance(encoder, Encoder):
            raise TypeError(f"encoder must be an fse.Encoder, not {type(encoder).__name__}")
        w, b = _content.weights(weight, bias, encoder.output_dim)
        if normalized is None:
            normalized = "normalize@1" in encoder.steps
        k = len(b)
        if cls.TASK == "classification" and labels is None:
            raise TypeError("a classifier needs labels, one per output (two for one output)")
        chosen = None if cls.TASK == "regression" else _content.link(link, k)
        names = None if cls.TASK == "regression" else _content.labels(labels, k)
        if flat is None:
            flat = cls.TASK == "classification" or np.ndim(weight) == 1
        content = HeadContent(cls.TASK, w, b, normalized, chosen, names, flat=flat)
        return cls._new(encoder, content, source=source)

    # --- the head on a backend --------------------------------------------------------

    def _pipeline(self, backend: str, *, with_vectors: bool) -> HeadPipeline:
        key = (backend, with_vectors)
        pipeline = self._pipelines.get(key)
        if pipeline is None:
            with self._lock:  # made once, published whole; never held while predicting
                pipeline = self._pipelines.get(key)
                if pipeline is None:
                    own = self._encoder._pipeline(backend)
                    pipeline = make_head(
                        self._encoder._spec, self._head, backend, own=own, with_vectors=with_vectors
                    )
                    self._pipelines[key] = pipeline
        return pipeline

    @property
    def _tokenizer(self) -> str:
        return self._encoder.tokenizer

    @validate_arguments
    def quantize(self, dtype: Literal["float16", "int8"]):  # -> Self
        """A new model of the same class and head whose encoder holds a float16 or int8
        copy of its float32 table (``Encoder.quantize``); A and b are copied unchanged, a
        loaded model's compared with their checksums first. Its predictions are those of
        the quantized rows, on the cpu backend bit for bit as the reference computes them
        from the quantized table, not those of this model; how close they are is measured.
        A standalone model has no table to quantize and raises ``ValueError``: quantize the
        complete model, then ``save(standalone=True)``. A model of a quantized encoder
        raises as ``Encoder.quantize`` does."""
        if self._content.standalone:
            raise ValueError(
                "a standalone model holds a folded float32 table and cannot be quantized; "
                "quantize the complete model, then save it with standalone=True"
            )
        weight = self._verified("head.weight", self._content.weight)
        bias = self._verified("head.bias", self._content.bias)
        for array in (weight, bias):
            array.setflags(write=False)
        content = dataclasses.replace(self._content, weight=weight, bias=bias)
        return type(self)._new(self._encoder.quantize(dtype), content, source=self._source)

    # --- persistence ------------------------------------------------------------------

    @validate_arguments
    def save(self, path: PathLike, standalone: bool = False):
        """Write the model into the new ``.safetensors`` file ``path``: complete (the
        encoder's table and the head), or with ``standalone=True`` only the folded table
        (vocabulary x outputs) and the bias, which predicts the same bits but has no
        vectors; only for a head on raw means."""
        if standalone and self._content.normalized:
            raise ValueError(
                "a head trained on normalized vectors does not fold into the table; save it "
                "complete"
            )
        if standalone and not self._content.standalone:
            tensors, metadata, expected = self._standalone_artifact()
        else:
            tensors, metadata, expected = self._artifact()
        metadata[CREATED_BY_KEY] = _created_by()
        SafetensorsContainer().write(Path(path), tensors, metadata, expected=expected)

    def _artifact(self) -> tuple[dict[str, ndarray], dict[str, str], dict[str, str] | None]:
        tensors, metadata, expected = self._encoder._artifact()
        out, keys = _format.to_artifact(self._content, tensors, metadata, self._encoder.fingerprint)
        if expected is None and self._checksums is None:
            return out, keys, None  # made in memory: nothing to compare with
        # The writer compares every tensor once checksums are given: those the encoder
        # or the file recorded, and the head's own (copies the model owns) otherwise.
        known = {name: checksum(array) for name, array in out.items() if name in _format.TENSORS}
        known.update(expected or {})
        known.update(self._checksums or {})
        return out, keys, known

    def _standalone_artifact(self) -> tuple[dict[str, ndarray], dict[str, str], dict | None]:
        """The folded encoder part (the table times A, float32, no steps) and the head
        without its weight. The table is folded from a copy checked against the checksums
        it was loaded with, and every tensor kept as it is is compared while it is written,
        so data that changed since the load (a mapped file) never get new checksums."""
        weight = self._verified("head.weight", self._content.weight)
        recorded = self._encoder._checksums
        pool = fold(get_backend("cpu"), self._verified_pool(recorded), weight, usable_cpus())
        tensors, metadata = self._folded_encoder(pool)
        sums = {k: checksum(v) for k, v in tensors.items() if k in _FOLDED or recorded is None}
        sums.update({k: recorded[k] for k in tensors if recorded is not None and k not in _FOLDED})
        of_encoder = _format.encoder_fingerprint(tensors, metadata, sums)
        head = dataclasses.replace(self._content, weight=None, normalized=False, standalone=True)
        out, keys = _format.to_artifact(head, tensors, metadata, of_encoder)
        if recorded is None:
            return out, keys, None  # made in memory: the arrays are the encoder's own
        own = self._checksums or {}  # a loaded model: the head's checksums from its file
        heads = {n: own.get(n) or checksum(a) for n, a in out.items() if n in _format.TENSORS}
        return out, keys, {**sums, **heads}

    def _folded_encoder(self, pool: PoolInput) -> tuple[dict[str, ndarray], dict[str, str]]:
        """The encoder artifact of the folded table: its rows and the encoder's vocabulary,
        tokenizer and FastText keys, k float32 values per row, no steps."""
        table, keys, _ = self._encoder.word_vectors._parts()
        table = {k: v for k, v in table.items() if k in ("vocab.bytes", "vocab.offsets")}
        table["vectors"] = pool.vectors
        if pool.ngram_vectors is not None:
            table["ngrams"] = pool.ngram_vectors
        keys = {k: v for k, v in keys.items() if k in ("fse.fasttext", "fse.ngram_hash")}
        keys.update({"fse.dim": str(self._content.k), "fse.dtype": "float32"})
        folded = dataclasses.replace(self._encoder._content, steps=())  # tokenizer kept
        return encoder_artifact(folded, table, keys)

    def _verified(self, name: str, array: ndarray | None) -> ndarray:
        """A copy of the head tensor ``name`` that matches the checksum of the file the
        model was loaded from (the array itself for a model made in memory)."""
        assert array is not None
        if not self._checksums or name not in self._checksums:
            return array
        copy = np.array(array, copy=True)
        if checksum(copy) != self._checksums[name]:
            raise CorruptArtifactError(
                f"tensor {name} of {self._path} changed since it was loaded; nothing was written"
            )
        return copy

    def _verified_pool(self, recorded: dict[str, str] | None) -> PoolInput:
        """The pool of the encoder; for a loaded one, copies of its rows and scales that
        match the checksums they were loaded with."""
        pool = self._encoder._spec.pool_input()
        if recorded is None:
            return pool
        copies: dict[str, Any] = {}
        for field, name in _POOL_TENSORS.items():
            array = getattr(pool, field)
            if array is not None:
                copy = np.array(array, copy=True)
                if checksum(copy) != recorded.get(name):
                    raise CorruptArtifactError(
                        f"tensor {name} of {self._encoder._path} changed since it was loaded; "
                        "nothing was written"
                    )
                copies[field] = copy
        return dataclasses.replace(pool, **copies)

    @classmethod
    def _load(cls, path: Any, mmap: bool = False):
        path = Path(path)
        tensors, metadata = SafetensorsContainer().open(path, mmap=mmap)
        return cls._opened(tensors, metadata, path, path)

    @classmethod
    def _opened(cls, tensors: dict, metadata: dict, path: Path | None, where: Path):
        """The model of an opened artifact: from the file ``path``, or of checked bytes
        (``path`` None) that ``where`` names in errors."""
        if metadata.get("fse.kind") == _format.ENCODER_KIND:
            raise UnsupportedArtifactError(
                f"{where} is an encoder, not a linear head; make a {cls.__name__} of it with "
                "from_arrays or from_sklearn"
            )
        content, enc_tensors, enc_metadata, of_encoder = _format.from_artifact(tensors, metadata)
        if content.task != cls.TASK:
            other = "Regressor" if content.task == "regression" else "Classifier"
            raise ValueError(
                f"{where} holds a {content.task} head; open it with fse.{other}.load or "
                f"fse.{other}.from_pretrained"
            )
        encoder = Encoder._from_artifact(enc_tensors, enc_metadata, path, of_encoder)
        checksums = {
            k[len(CHECKSUM_PREFIX) :]: v
            for k, v in metadata.items()
            if k.startswith(CHECKSUM_PREFIX)
        }
        return cls._new(
            encoder,
            content,
            path=path,
            checksums=checksums,
            fingerprint_=metadata.get(FINGERPRINT_KEY),
        )

    @classmethod
    def _from_pretrained(
        cls, repo_or_path: Any, *, revision, token, cache_dir, local_files_only, mmap
    ):
        """The model of a local ``.safetensors`` file, or of an fse folder (local or of
        the Hub) whose config.json names a linear head; the config is compared with the
        metadata actually loaded."""
        files = _pretrained.files(
            repo_or_path,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        found = _pretrained.find_fse(files, _format.KIND, mmap)
        if found.data is not None:  # of the Hub, checked as these bytes
            tensors, metadata = SafetensorsContainer().open_bytes(*found.data)
            model = cls._opened(tensors, metadata, None, found.data[1])
        else:
            assert found.path is not None
            model = cls._load(found.path, mmap)
        _pretrained.check_loaded(found, model._fingerprint, _format.KIND)
        return model

    @validate_arguments
    def save_pretrained(
        self,
        path: PathLike,
        *,
        standalone: bool = False,
        license: str | list[str] | None = None,
        language: list[str] | None = None,
        model_name: str | None = None,
        extra_files: dict[str, PathLike] | None = None,
    ) -> Path:
        """Write the model as a Hugging Face folder at the new ``path`` (never half written
        or replaced): ``config.json``, ``model.safetensors`` (as ``save`` writes it,
        complete or ``standalone``), a ``README.md`` model card of that file with the
        license its encoder records (or ``license`` if it records none) and
        ``extra_files`` (name to local text file, such as LICENSE). ``from_pretrained``
        reads it. docs/architecture/hub.md, "Hub folders"."""

        def write(file: Path) -> None:
            self.save(file, standalone=standalone)

        card: dict[str, Any] = {"license": license, "language": language, "extra": extra_files}
        return _folder.save(path, write, model_name=model_name, **card)[0]

    @validate_arguments
    def push_to_hub(
        self,
        repo_id: str,
        *,
        standalone: bool = False,
        token: str | bool | None = None,
        private: bool | None = None,
        revision: str | None = None,
        create_pr: bool = False,
        commit_message: str | None = None,
        replace: bool = False,
        license: str | list[str] | None = None,
        language: list[str] | None = None,
        extra_files: dict[str, PathLike] | None = None,
    ) -> str:
        """Publish the folder of ``save_pretrained`` to the Hub repository ``repo_id`` in
        one commit and return its URL, as ``Encoder.push_to_hub`` does: a license is
        required, other content under the same names and other files are refused unless
        ``replace=True``, license files already there are kept, every file is checked at
        the new commit."""

        def write(file: Path) -> None:
            self.save(file, standalone=standalone)

        hub = {"token": token, "private": private, "revision": revision, "replace": replace}
        hub |= {"create_pr": create_pr, "commit_message": commit_message}
        card: dict[str, Any] = {"license": license, "language": language, "extra": extra_files}
        return _folder.push(
            repo_id, write, type(self).__name__.lower(), hub, card | {"model_name": repo_id}
        )

    # --- properties -------------------------------------------------------------------

    @property
    def encoder(self) -> Encoder:
        """The encoder whose rows the head reads; a standalone model has none (its table is
        folded) and raises."""
        if self._content.standalone:
            raise ValueError("a standalone model has no encoder: its table is folded")
        return self._encoder

    @property
    def normalized(self) -> bool:
        """Whether the head was trained on normalized rows (then it reads them unfolded)."""
        return self._content.normalized

    @property
    def link(self) -> str:
        """``sigmoid@1``, ``softmax@1``, ``ovr@1`` or ``none`` (no probabilities)."""
        return self._content.link_id

    @property
    def n_outputs(self) -> int:
        """k, the rows of A: the logits of a classifier, the targets of a regressor."""
        return self._content.k

    @property
    def standalone(self) -> bool:
        return self._content.standalone

    @property
    def fingerprint(self) -> str:
        """The fingerprint of the model as ``save`` writes it complete (standalone: as it
        was loaded), the same in memory and after save and load."""
        if self._fingerprint is None:
            tensors, metadata, expected = self._artifact()
            known = expected or {}
            entries = {
                k: (tensor_code(v), v.shape, known.get(k) or checksum(v))
                for k, v in tensors.items()
            }
            self._fingerprint = fingerprint(entries, metadata)
        return self._fingerprint

    @property
    def encoder_fingerprint(self) -> str:
        """The fingerprint of the encoder whose rows the head reads (of the folded table
        for a standalone model); a store must hold this encoder's rows."""
        return self._encoder.fingerprint

    @property
    def coef_(self) -> ndarray:
        """A copy of A, (k, d) float32; a standalone model has none and raises."""
        if self._content.weight is None:
            raise ValueError("a standalone model holds no weights: they are folded into its table")
        return self._content.weight.copy()

    @property
    def intercept_(self) -> ndarray:
        """A copy of b, (k,) float32."""
        return self._content.bias.copy()

    def output_identity(self, output: str) -> str:
        """The identity of the rows ``output`` (``decision``, ``proba`` or ``values``) of
        this model: a sink that names an ``encoder_fingerprint`` must name it."""
        return output_identity(self.fingerprint, output)

    def __reduce__(self):
        name = type(self).__name__
        raise TypeError(f"a {name} cannot be pickled; use save() and {name}.load()")

    def __repr__(self) -> str:
        kind = "standalone, " if self.standalone else ""
        flag = "normalized" if self.normalized else "raw means"
        return (
            f"{type(self).__name__}({kind}{self.n_outputs} outputs, {flag}, {self.link}, "
            f"{self._encoder!r})"
        )
