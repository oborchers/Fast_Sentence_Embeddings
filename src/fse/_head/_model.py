#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""What fse.Classifier and fse.Regressor share: an encoder and one linear head.

A model holds its encoder (or, standalone, the folded table as an encoder of k values
per row) and its head ``z = A x + b``. It does not change once made. On each backend it
prepares one head pipeline on first use: the folded table for a head on raw means (A
folds into the rows, b is added after pooling), the full mean for a head on normalized
means or when the vectors are asked for. It is saved complete (encoder and head) or
standalone (the folded table and b; only for heads on raw means), as one
``.safetensors`` file (ARCHITECTURE.md, "linear_head 1").
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from numpy import ndarray

from fse._backends import get_backend
from fse._container import (
    CHECKSUM_PREFIX,
    CREATED_BY_KEY,
    FINGERPRINT_KEY,
    SafetensorsContainer,
    checksum,
    fingerprint,
    tensor_code,
)
from fse._contract import UnsupportedArtifactError
from fse._cpus import usable_cpus
from fse._encoder import Encoder
from fse._encoder._format import EncoderContent
from fse._encoder._format import to_artifact as encoder_artifact
from fse._runtime import HeadPipeline, HeadSpec, make_head
from fse._runtime._head import fold
from fse._validation import PathLike, validate_arguments
from fse._wordvectors import _created_by

from . import _content, _format
from ._content import HeadContent
from ._outputs import output_identity


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
        content = HeadContent(cls.TASK, w, b, normalized, chosen, names)
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

    def _standalone_artifact(self) -> tuple[dict[str, ndarray], dict[str, str], None]:
        """The folded encoder part (the table times A, float32, no steps) and the head
        without its weight."""
        spec = self._encoder._spec
        weight = self._content.weight
        assert weight is not None
        pool = fold(get_backend("cpu"), spec.pool_input(), weight, usable_cpus())
        wv = self._encoder.word_vectors
        table, keys, _ = wv._parts()
        table = {k: v for k, v in table.items() if k in ("vocab.bytes", "vocab.offsets")}
        table["vectors"] = pool.vectors
        if pool.ngram_vectors is not None:
            table["ngrams"] = pool.ngram_vectors
        keys = {k: v for k, v in keys.items() if k in ("fse.fasttext", "fse.ngram_hash")}
        keys.update({"fse.dim": str(self._content.k), "fse.dtype": "float32"})
        content = self._encoder._content
        folded = EncoderContent(content.tokenizer, (), content.oov_weight, content.max_ngrams)
        tensors, metadata = encoder_artifact(folded, table, keys)
        sums = {k: checksum(v) for k, v in tensors.items()}
        of_encoder = _format.encoder_fingerprint(tensors, metadata, sums)
        head = HeadContent(
            self._content.task,
            None,
            self._content.bias,
            False,
            self._content.link,
            self._content.labels,
            True,
        )
        out, keys = _format.to_artifact(head, tensors, metadata, of_encoder)
        return out, keys, None

    @classmethod
    def _load(cls, path: Any, mmap: bool = False):
        path = Path(path)
        tensors, metadata = SafetensorsContainer().open(path, mmap=mmap)
        if metadata.get("fse.kind") == _format.ENCODER_KIND:
            raise UnsupportedArtifactError(
                f"{path} is an encoder, not a linear head; make a {cls.__name__} of it with "
                "from_arrays or from_sklearn"
            )
        content, enc_tensors, enc_metadata, of_encoder = _format.from_artifact(tensors, metadata)
        if content.task != cls.TASK:
            other = "Regressor" if content.task == "regression" else "Classifier"
            raise ValueError(f"{path} holds a {content.task} head; open it with fse.{other}.load")
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
