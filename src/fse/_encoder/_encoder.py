#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The frozen encoder: word vectors, weights and steps, turned into sentence vectors.

An Encoder does not change once made (by ``fit`` of a recipe or ``load``). It builds the
VocabTable of its vocabulary once and one Pipeline per backend, both on first use, and
several threads may encode with it at once. It is saved as one safetensors file
(ARCHITECTURE.md, "encoder 1").
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _codes
from fse._backends import BACKEND_NAMES
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
from fse._runtime import EncodeSpec, Pipeline, QuantizedRows, check_workers
from fse._text import Vocabulary, make_vocabulary
from fse._validation import PathLike, validate_arguments
from fse._wordvectors import WordVectors, _created_by
from fse._wordvectors_format import FORMAT_VERSION as WORD_VECTORS_VERSION

from . import _batches
from ._batches import EncodedBatch, check_sink, flush, run_into
from ._format import (
    WORD_VECTOR_KEYS,
    WORD_VECTOR_TENSORS,
    EncoderContent,
    from_artifact,
    to_artifact,
)
from ._sources import opened


class Encoder:
    """Turns sentences into vectors with frozen word vectors, weights and steps.

    Make one with ``fit`` of a recipe (``fse.Average``, ``fse.SIF``, ``fse.USIF``) or
    with ``Encoder.load``; it cannot be changed afterwards.
    """

    __slots__ = (
        "_checksums",
        "_content",
        "_fingerprint",
        "_lock",
        "_path",
        "_pipelines",
        "_spec",
        "_vocabulary",
        "_word_vectors",
    )
    _checksums: dict[str, str] | None
    _content: EncoderContent
    _fingerprint: str | None
    _lock: threading.Lock
    _path: Path | None
    _pipelines: dict[str, Pipeline]
    _spec: EncodeSpec
    _vocabulary: Vocabulary | None
    _word_vectors: WordVectors

    def __init__(self, *_args: Any, **_kwargs: Any):
        raise TypeError("make an Encoder with fit of a recipe or with Encoder.load")

    @classmethod
    def _new(
        cls,
        content: EncoderContent,
        word_vectors: WordVectors,
        *,
        path: Path | None = None,
        checksums: dict[str, str] | None = None,
        fingerprint_: str | None = None,
        vocabulary: Vocabulary | None = None,
    ) -> Encoder:
        _check_binding(content, word_vectors)
        out = object.__new__(cls)
        out._content = content
        out._word_vectors = word_vectors
        weights = content.weights
        if weights is None:
            weights = np.ones(len(word_vectors), dtype=np.float32)
        out._spec = EncodeSpec(
            word_vectors,
            weights,
            content.steps,
            content.tokenizer,
            oov_weight=None if content.oov_weight is None else content.oov_weight[()],
            max_ngrams=content.max_ngrams,
            aggregation=content.aggregation,
        )
        out._path = path
        out._checksums = checksums
        out._fingerprint = fingerprint_
        out._vocabulary = vocabulary
        out._pipelines = {}
        out._lock = threading.Lock()
        return out

    # --- encoding ---------------------------------------------------------------------

    def encode(
        self,
        texts: Any,
        *,
        dtype: str | None = None,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
    ) -> ndarray | QuantizedRows | None:
        """The vectors of ``texts``, one row per sentence, as ``dtype`` (float32 by
        default): float32 or float16 rows, or int8 ``QuantizedRows``. With a ``sink``
        (``fse.Sink``, such as ``fse.ShardSink``), the rows go there in its dtype and
        nothing is returned; its ``flush`` runs after the last row.

        ``texts`` is a list of str, a path of a text file (one sentence per line), token
        lists for the external tokenizer, or any iterable of str, read once.
        """
        if sink is not None:
            check_sink(sink, dtype, self.fingerprint, self.output_dim)
            pipeline = self._pipeline(backend)
            with opened(texts, self._content.tokenizer) as source:
                run_into(pipeline, source, sink, workers, 0)
            flush(sink)
            return None
        dtype = _codes.check_dtype("float32" if dtype is None else dtype)
        pipeline = self._pipeline(backend)
        with opened(texts, self._content.tokenizer) as source:
            return pipeline.infer(source, workers=workers, dtype=dtype)

    def encode_batches(
        self,
        batches: Any,
        *,
        dtype: str | None = None,
        workers: int | None = None,
        backend: str = "cpu",
        sink: Any = None,
    ) -> Iterator[EncodedBatch]:
        """Encode a stream batch by batch: ``batches`` yields finite lists of sentences
        (or token lists), as a queue consumer polls them, and every batch gives one
        ``EncodedBatch`` once its rows are finished. With a ``sink`` the rows go there,
        and ``durable_rows`` says how many rows of the run are durable, so a consumer
        confirms its input only up to there; without one the batch holds its rows.

        The arguments and the encoder are checked when this is called, before any batch
        is read; closing the iterator early reads no further batch.
        """
        if sink is not None:
            dtype = check_sink(sink, dtype, self.fingerprint, self.output_dim)
        else:
            dtype = _codes.check_dtype("float32" if dtype is None else dtype)
        check_workers(workers)
        pipeline = self._pipeline(backend)
        return _batches.batches(
            pipeline,
            iter(batches),
            self._content.tokenizer,
            dtype=dtype,
            workers=workers,
            sink=sink,
        )

    def _pipeline(self, backend: str) -> Pipeline:
        if not isinstance(backend, str) or backend not in BACKEND_NAMES:
            raise ValueError(f"backend must be one of {list(BACKEND_NAMES)}, not {backend!r}")
        pipeline = self._pipelines.get(backend)
        if pipeline is None:
            with self._lock:  # built once, published whole; never held while encoding
                pipeline = self._pipelines.get(backend)
                if pipeline is None:
                    if self._vocabulary is None:
                        self._vocabulary = make_vocabulary(self._word_vectors)
                    pipeline = Pipeline(self._spec, backend, vocabulary=self._vocabulary)
                    self._pipelines[backend] = pipeline
        return pipeline

    # --- persistence ------------------------------------------------------------------

    @validate_arguments
    def save(self, path: PathLike):
        """Write the encoder into the new ``.safetensors`` file ``path``; a folded encoder
        includes its word vectors. Data loaded from a file are compared with the checksums
        they were loaded with while they are written."""
        tensors, metadata, expected = self._artifact()
        metadata[CREATED_BY_KEY] = _created_by()
        SafetensorsContainer().write(Path(path), tensors, metadata, expected=expected)

    def _artifact(self) -> tuple[dict[str, ndarray], dict[str, str], dict[str, str] | None]:
        embedded = None
        expected = dict(self._checksums or {})
        if self.folded:
            words, keys, loaded = self._word_vectors._parts()
            words = {k: v for k, v in words.items() if k != "counts"}
            embedded = (words, {k: v for k, v in keys.items() if k in WORD_VECTOR_KEYS})
            expected.update({k: v for k, v in (loaded or {}).items() if k in words})
        tensors, metadata = to_artifact(self._content, embedded)
        tensors.update(self._used())
        return tensors, metadata, expected or None

    def _used(self) -> dict[str, ndarray]:
        """The copies the pipelines encode with, as tensors: the weights and oov_weight.
        Saving and verifying use these, so neither can pass data the encoder does not use."""
        used = {}
        if self._content.weights is not None:
            used["weights"] = self._spec.weights
        if self._spec.oov_weight is not None:
            used["oov_weight"] = np.array(self._spec.oov_weight, dtype=np.float32)
        return used

    @classmethod
    @validate_arguments
    def load(cls, path: PathLike, word_vectors: Any = None, mmap: bool = False):  # -> Encoder
        """Open an encoder saved with ``save``. An unfolded encoder needs the
        ``word_vectors`` it was fitted on (checked by fingerprint); a folded one holds
        them. With ``mmap`` the arrays are mapped read-only (see ``verify``)."""
        path = Path(path)
        tensors, metadata = SafetensorsContainer().open(path, mmap=mmap)
        content, embedded = from_artifact(tensors, metadata)
        checksums = {
            k[len(CHECKSUM_PREFIX) :]: v
            for k, v in metadata.items()
            if k.startswith(CHECKSUM_PREFIX)
        }
        if embedded is None:
            if not isinstance(word_vectors, WordVectors):
                raise TypeError(
                    f"{path} is an unfolded encoder: pass the word_vectors it was fitted on "
                    f"(fingerprint {content.word_vectors})"
                )
        else:
            if word_vectors is not None:
                raise ValueError(f"{path} is a folded encoder and holds its word vectors")
            word_vectors = _embedded(embedded, metadata, checksums, content.word_vectors, path)
        return cls._new(
            content,
            word_vectors,
            path=path,
            checksums=checksums,
            fingerprint_=metadata.get(FINGERPRINT_KEY),
        )

    def verify(self):
        """Check the data this encoder holds (as it encodes with them) against the
        checksums it was loaded with, its file (if loaded), and its word vectors."""
        if self._checksums is not None:
            tensors, _, _ = self._artifact()
            held = {k: v for k, v in tensors.items() if k in self._checksums}
            for name, array in held.items():
                if name not in WORD_VECTOR_TENSORS and checksum(array) != self._checksums[name]:
                    raise CorruptArtifactError(
                        f"tensor {name} of {self._path} does not match its checksum"
                    )
        if self._path is not None:
            SafetensorsContainer().verify(self._path)
        self._word_vectors.verify()

    # --- properties -------------------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        """The fingerprint of the encoder (ARCHITECTURE.md, "Identities"), the same in
        memory and after save and load."""
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
    def word_vectors(self) -> WordVectors:
        return self._word_vectors

    @property
    def tokenizer(self) -> str:
        return self._content.tokenizer

    @property
    def steps(self) -> tuple[str, ...]:
        """The identifiers of the steps applied per row, in order."""
        return tuple(s.op.id for s in self._content.steps)

    @property
    def folded(self) -> bool:
        """Whether learned linear steps were folded into the encoder's own word vectors."""
        return bool(self._content.folded)

    @property
    def input_dim(self) -> int:
        return self._content.input_dim

    @property
    def output_dim(self) -> int:
        return self._content.output_dim

    @property
    def recipe(self) -> dict[str, Any]:
        """The recipe and its parameters, as recorded (provenance only)."""
        return copy.deepcopy(self._content.recipe)

    def __reduce__(self):
        raise TypeError("an Encoder cannot be pickled; use save() and Encoder.load()")

    def __repr__(self) -> str:
        name = self._content.recipe.get("name", "encoder")
        kind = ", folded" if self.folded else ""
        return f"Encoder({name}, {self.input_dim} -> {self.output_dim} dimensions{kind})"


def _check_binding(content: EncoderContent, word_vectors: WordVectors):
    """The word vectors must be the ones the content names."""
    if word_vectors.dim != content.input_dim:
        raise ValueError(
            f"the encoder pools vectors of {content.input_dim} dimensions, not {word_vectors.dim}"
        )
    if content.weights is not None and len(content.weights) != len(word_vectors):
        raise ValueError(
            f"the encoder holds {len(content.weights)} weights for {len(word_vectors)} words"
        )
    for name, want, got in (
        ("vocabulary", content.vocabulary, word_vectors.vocabulary_fingerprint),
        ("word vectors", content.word_vectors, word_vectors.fingerprint),
    ):
        if want != got:
            raise ValueError(f"the encoder was fitted on the {name} {want}, not {got}")


def _embedded(
    embedded: tuple[dict[str, ndarray], dict[str, str]],
    metadata: dict[str, str],
    checksums: dict[str, str],
    recorded: str,
    path: Path,
) -> WordVectors:
    """The word vectors a folded encoder holds, as loaded word vectors of its file; their
    fingerprint from the recorded checksums must be the one the encoder names."""
    tensors, keys = embedded
    content = {"fse.kind": "word_vectors", "fse.format_version": WORD_VECTORS_VERSION, **keys}
    entries = {k: (tensor_code(v), v.shape, checksums[k]) for k, v in tensors.items()}
    if fingerprint(entries, content) != recorded:
        raise CorruptArtifactError(
            f"the word vectors held by {path} do not have the fingerprint it names ({recorded})"
        )
    loaded = {
        **content,
        CREATED_BY_KEY: metadata.get(CREATED_BY_KEY, "fse"),
        FINGERPRINT_KEY: recorded,
        **{CHECKSUM_PREFIX + k: checksums[k] for k in tensors},
    }
    return WordVectors(tensors, loaded, path)
