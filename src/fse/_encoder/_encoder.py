#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The encoder: a finished model that turns sentences into vectors.

An Encoder is a table of word rows (``WordVectors``), a tokenizer and whether every row
is normalized; it pools the rows of a sentence with ``weighted_mean@1``, every word with
the weight 1. It does not change once made (``from_word_vectors``, ``from_arrays`` or
``load``). It builds the VocabTable of its vocabulary once and one Pipeline per backend,
both on first use, and several threads may encode with it at once. It is saved as one
self-contained safetensors file (ARCHITECTURE.md, "encoder 1").
"""

from __future__ import annotations

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
from fse._contract import Normalize, check_tokenizer
from fse._runtime import EncodeSpec, Pipeline, QuantizedRows, check_workers
from fse._text import MAX_NGRAMS_LIMIT, Vocabulary, make_vocabulary
from fse._validation import PathLike, PositiveInt, validate_arguments
from fse._wordvectors import WordVectors, _created_by
from fse._wordvectors_format import FORMAT_VERSION as WORD_VECTORS_VERSION

from . import _batches
from ._batches import EncodedBatch, check_sink, flush, run_into
from ._format import TABLE_KEYS, TABLE_TENSORS, EncoderContent, from_artifact, to_artifact
from ._sources import opened

# The ngrams an unknown FastText word uses at most, unless an encoder is given another.
DEFAULT_MAX_NGRAMS = 40


class Encoder:
    """Turns sentences into vectors with a finished table of word rows.

    Make one with ``Encoder.from_word_vectors``, ``Encoder.from_arrays`` or
    ``Encoder.load``; it cannot be changed afterwards.
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
        raise TypeError(
            "make an Encoder with Encoder.from_word_vectors, Encoder.from_arrays or Encoder.load"
        )

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
        out = object.__new__(cls)
        out._content = content
        out._word_vectors = word_vectors
        out._spec = EncodeSpec(
            word_vectors,
            np.ones(len(word_vectors), dtype=np.float32),  # a finished table: every word weighs 1
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

    @classmethod
    @validate_arguments
    def from_word_vectors(
        cls,
        word_vectors: Any,
        *,
        tokenizer: str = "fse_words@1",
        normalize: bool = True,
        max_ngrams: PositiveInt | None = None,
        oov_weight: float | None = None,
    ):  # -> Encoder; pydantic cannot resolve it here
        """The encoder of a finished table ``word_vectors`` (``fse.WordVectors``): every
        sentence becomes the mean of the rows of its known words, with ``tokenizer``
        (``fse_words@1`` or ``external``), and with ``normalize`` scaled to unit length.

        For FastText vectors an unknown word adds its first ``max_ngrams`` ngram rows
        (default 40), together with the weight ``oov_weight`` (default 1); other vectors
        take neither. Weights, removed components or projections of a model are folded
        into its rows before (ARCHITECTURE.md, "2.0 design").
        """
        if not isinstance(word_vectors, WordVectors):
            raise TypeError(f"word_vectors must be WordVectors, not {type(word_vectors).__name__}")
        content = EncoderContent(
            tokenizer=check_tokenizer(tokenizer),
            steps=(Normalize(),) if normalize else (),
            **_fasttext(word_vectors, max_ngrams, oov_weight),
        )
        return cls._new(content, word_vectors)

    @classmethod
    @validate_arguments
    def from_arrays(
        cls,
        vectors: Any,
        *,
        words: Any = None,
        vocab_bytes: Any = None,
        vocab_offsets: Any = None,
        scales: Any = None,
        fasttext: Any = None,
        ngrams: Any = None,
        ngram_scales: Any = None,
        tokenizer: str = "fse_words@1",
        normalize: bool = True,
        max_ngrams: PositiveInt | None = None,
        oov_weight: float | None = None,
    ):  # -> Encoder
        """``from_word_vectors`` of ``fse.WordVectors.from_arrays`` of the same arrays:
        the rows (float32, float16 or int8 with ``scales``), the vocabulary (``words``, or
        ``vocab_bytes`` and ``vocab_offsets``) and for FastText ``fasttext`` and ``ngrams``
        (with ``ngram_scales`` for int8)."""
        word_vectors = WordVectors.from_arrays(
            vectors,
            words=words,
            vocab_bytes=vocab_bytes,
            vocab_offsets=vocab_offsets,
            scales=scales,
            fasttext=fasttext,
            ngrams=ngrams,
            ngram_scales=ngram_scales,
        )
        return cls.from_word_vectors(
            word_vectors,
            tokenizer=tokenizer,
            normalize=normalize,
            max_ngrams=max_ngrams,
            oov_weight=oov_weight,
        )

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
        default): float32 or float16 rows, or int8 ``QuantizedRows``. Without a sink the
        whole result is held in memory. With a ``sink`` (``fse.Sink``, such as
        ``fse.ShardSink``), the rows go there and nothing is returned; ``dtype`` is then
        the sink's (another one raises) and its ``flush`` runs after the last row.

        ``texts`` is a list, tuple or numpy array of str, or of token lists for the
        external tokenizer; a ``pathlib.Path`` (or other
        ``os.PathLike``) of a text file, one sentence per line (a str is one sentence,
        never a path: pass ``[text]`` for one sentence); or any other iterable of str, a
        stream read once.
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
        is read; closing the iterator early reads no further batch. After a successful
        full iteration the sink's ``flush`` (where it has one) has run, and its own
        ``durable_rows`` then says what is durable; for a ``fse.ShardSink`` that is every
        row of the run, while the last batch may report fewer. Stopping
        early (``break``, ``close()``) does not flush: the owner of the sink closes it
        (``ShardSink.close`` publishes the rows committed so far). Without a sink every
        batch holds its rows in memory.
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
        """Write the encoder, with its table, into the new ``.safetensors`` file ``path``.
        Data loaded from a file are compared with the checksums they were loaded with
        while they are written."""
        tensors, metadata, expected = self._artifact()
        metadata[CREATED_BY_KEY] = _created_by()
        SafetensorsContainer().write(Path(path), tensors, metadata, expected=expected)

    def _artifact(self) -> tuple[dict[str, ndarray], dict[str, str], dict[str, str] | None]:
        table, keys, loaded = self._word_vectors._parts()
        expected = dict(self._checksums or {})
        expected.update({k: v for k, v in (loaded or {}).items() if k in table})
        tensors, metadata = to_artifact(
            self._content, table, {k: v for k, v in keys.items() if k in TABLE_KEYS}
        )
        if self._spec.oov_weight is not None:
            # The copy the pipelines encode with, so saving and verifying cannot pass a
            # value the encoder does not use.
            tensors["oov_weight"] = np.array(self._spec.oov_weight, dtype=np.float32)
        return tensors, metadata, expected or None

    @classmethod
    @validate_arguments
    def load(cls, path: PathLike, mmap: bool = False):  # -> Encoder
        """Open an encoder saved with ``save``; it holds its table. With ``mmap`` the
        arrays are mapped read-only (see ``verify``). An encoder written by a pre-release
        of 2.0 raises ``fse.UnsupportedArtifactError``: rebuild it."""
        path = Path(path)
        tensors, metadata = SafetensorsContainer().open(path, mmap=mmap)
        content, table, keys = from_artifact(tensors, metadata)
        checksums = {
            k[len(CHECKSUM_PREFIX) :]: v
            for k, v in metadata.items()
            if k.startswith(CHECKSUM_PREFIX)
        }
        return cls._new(
            content,
            _table(table, keys, metadata, checksums, path),
            path=path,
            checksums=checksums,
            fingerprint_=metadata.get(FINGERPRINT_KEY),
        )

    def verify(self):
        """Check the data this encoder holds (as it encodes with them) against the
        checksums it was loaded with, its file (if loaded), and its table."""
        if self._checksums is not None:
            tensors, _, _ = self._artifact()
            held = {k: v for k, v in tensors.items() if k in self._checksums}
            for name, array in held.items():
                if name not in TABLE_TENSORS and checksum(array) != self._checksums[name]:
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
    def output_dim(self) -> int:
        """The dimension of the rows the encoder gives (that of its table)."""
        return self._spec.output_dim

    def __reduce__(self):
        raise TypeError("an Encoder cannot be pickled; use save() and Encoder.load()")

    def __repr__(self) -> str:
        wv = self._word_vectors
        kind = ", FastText" if wv.fasttext else ""
        steps = "".join(f", {step}" for step in self.steps)
        return (
            f"Encoder({len(wv)} words, {self.output_dim} dimensions, {wv.dtype}{kind}, "
            f"{self.tokenizer}{steps})"
        )


def _fasttext(word_vectors: WordVectors, max_ngrams: int | None, oov_weight: float | None):
    """``max_ngrams`` and ``oov_weight`` of an encoder of ``word_vectors``, with their
    defaults for FastText vectors; other vectors take neither."""
    if word_vectors.fasttext is None:
        if max_ngrams is not None or oov_weight is not None:
            raise ValueError("max_ngrams and oov_weight apply only to FastText vectors")
        return {"max_ngrams": None, "oov_weight": None}
    if max_ngrams is not None and max_ngrams > MAX_NGRAMS_LIMIT:
        raise ValueError(f"max_ngrams must be at most {MAX_NGRAMS_LIMIT}, not {max_ngrams}")
    with np.errstate(over="ignore"):
        weight = np.array(1.0 if oov_weight is None else oov_weight, dtype=np.float32)
    if not np.isfinite(weight):
        raise ValueError(f"oov_weight must be finite in float32, not {oov_weight!r}")
    return {"max_ngrams": max_ngrams or DEFAULT_MAX_NGRAMS, "oov_weight": weight}


def _table(
    tensors: dict[str, ndarray],
    keys: dict[str, str],
    metadata: dict[str, str],
    checksums: dict[str, str],
    path: Path,
) -> WordVectors:
    """The table an encoder holds, as word vectors loaded from its file, with the
    fingerprint of their recorded checksums."""
    content = {"fse.kind": "word_vectors", "fse.format_version": WORD_VECTORS_VERSION, **keys}
    entries = {k: (tensor_code(v), v.shape, checksums[k]) for k, v in tensors.items()}
    loaded = {
        **content,
        CREATED_BY_KEY: metadata.get(CREATED_BY_KEY, "fse"),
        FINGERPRINT_KEY: fingerprint(entries, content),
        **{CHECKSUM_PREFIX + k: checksums[k] for k in tensors},
    }
    return WordVectors._open(tensors, loaded, path)
