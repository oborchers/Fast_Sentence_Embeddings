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
from typing import Any, Literal

from numpy import ndarray

from fse import _codes
from fse._backends import BACKEND_NAMES
from fse._container import (
    CREATED_BY_KEY,
    FINGERPRINT_KEY,
    SafetensorsContainer,
)
from fse._contract import Normalize, UnsupportedArtifactError
from fse._runtime import EncodeSpec, Pipeline, QuantizedRows, check_workers
from fse._text import Vocabulary, make_vocabulary
from fse._tokenizer_spec import TokenizerSpec
from fse._validation import PathLike, PositiveInt, validate_arguments
from fse._wordvectors import WordVectors, _created_by

from . import _batches, _folder, _persist, _pretrained
from ._batches import EncodedBatch, check_sink, flush, run_into
from ._format import (
    EncoderContent,
    Source,
    fasttext_parts,
)
from ._sources import opened


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
        out._spec = _persist.spec(content, word_vectors)
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
            tokenizer=TokenizerSpec.of(tokenizer),
            steps=(Normalize(),) if normalize else (),
            **fasttext_parts(word_vectors, max_ngrams, oov_weight),
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
        truncate: bool = True,
        max_length: PositiveInt | Literal["model"] | None = "model",
        truncate_dim: PositiveInt | Literal["model"] | None = "model",
        normalize: bool | Literal["model"] = "model",
        mmap: bool = False,
    ):  # -> Encoder
        """The encoder of a pretrained model: a local ``.safetensors`` file or folder, or
        a Hugging Face Hub repository (``namespace/name``) at
        ``revision`` (default branch if None), checked against its commit. An fse encoder
        is loaded (``mmap`` as for ``load``); a Model2Vec model is converted, with its
        length rule unless ``truncate=False`` (``max_length`` sets another token limit,
        None none, as in model2vec); a sentence-transformers model of one StaticEmbedding
        is converted with the rules of its encode, its first ``truncate_dim`` dimensions
        (``"model"``: the model's, None: all) and, with ``normalize=True``, fse's own
        ``normalize@1`` after the cut. ARCHITECTURE.md, "from_pretrained"."""
        files = _pretrained.files(
            repo_or_path,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        found = _pretrained.find(
            files,
            truncate=truncate,
            max_length=max_length,
            truncate_dim=truncate_dim,
            normalize=normalize,
            mmap=mmap,
        )
        if found.converted is None:
            if found.data is not None:  # an fse encoder of the Hub, checked as these bytes
                tensors, metadata = SafetensorsContainer().open_bytes(*found.data)
                encoder = cls._from_artifact(tensors, metadata, None, metadata[FINGERPRINT_KEY])
            else:
                assert found.path is not None
                encoder = cls.load(found.path, mmap=mmap)
            _pretrained.check_loaded(found, encoder._fingerprint, "encoder")
            return encoder
        model = found.converted
        return cls._converted(
            model.word_vectors,
            model.tokenizer,
            normalize=model.normalize,
            source=found.source,
            license=found.license,
        )

    @classmethod
    def _converted(
        cls,
        word_vectors: WordVectors,
        tokenizer: TokenizerSpec,
        *,
        normalize: bool,
        source: Source | None = None,
        license: str | tuple[str, ...] | None = None,
    ) -> Encoder:
        """The encoder of a model converted from another library (the readers of PR 12a2):
        its table, its tokenizer with parameters, whether it normalizes, its provenance."""
        content = _persist.converted(word_vectors, tokenizer, normalize, source, license)
        return cls._new(content, word_vectors)

    @validate_arguments
    def quantize(self, dtype: Literal["float16", "int8"]):  # -> Encoder
        """A new encoder with a float16 or int8 copy of this encoder's float32 table
        (``quantize_rows@1``, int8 with one scale per row), in memory; ``save`` writes it.
        Its rows are those of the quantized table, bit for bit as its reference computes
        them. They are not the rows of this encoder; how close they are is measured.
        Quantizing a float16 or int8 encoder again raises ``ValueError``."""
        content, table = _persist.quantized(
            self._content,
            self._word_vectors,
            self._checksums,
            self._path,
            self._spec.oov_weight,
            dtype,
        )
        return Encoder._new(content, table, vocabulary=self._vocabulary)

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
            with opened(texts, self.tokenizer) as source:
                run_into(pipeline, source, sink, workers, 0)
            flush(sink)
            return None
        dtype = _codes.check_dtype("float32" if dtype is None else dtype)
        pipeline = self._pipeline(backend)
        with opened(texts, self.tokenizer) as source:
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
            self.tokenizer,
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
        return _persist.artifact(
            self._content, self._word_vectors, self._checksums, self._spec.oov_weight
        )

    @validate_arguments
    def save_pretrained(
        self,
        path: PathLike,
        *,
        license: str | list[str] | None = None,
        language: list[str] | None = None,
        model_name: str | None = None,
        extra_files: dict[str, PathLike] | None = None,
    ) -> Path:
        """Write the encoder as a Hugging Face folder at the new ``path`` (never half written
        or replaced): ``config.json``, ``model.safetensors`` (as ``save`` writes it), a
        ``README.md`` model card with the license the encoder records (or ``license`` if it
        records none) and ``extra_files`` (name to local text file, such as LICENSE).
        ``from_pretrained`` reads it. ARCHITECTURE.md, "Hub folders"."""
        card: dict[str, Any] = {"license": license, "language": language, "extra": extra_files}
        return _folder.save(path, self.save, model_name=model_name, **card)[0]

    @validate_arguments
    def push_to_hub(
        self,
        repo_id: str,
        *,
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
        one commit and return its URL. A license is required. Files of another content
        under the same names, and other files, are refused unless ``replace=True``; license
        files already there are kept. Every file is checked at the new commit."""
        hub = {"token": token, "private": private, "revision": revision, "replace": replace}
        hub |= {"create_pr": create_pr, "commit_message": commit_message}
        card: dict[str, Any] = {"license": license, "language": language, "extra": extra_files}
        return _folder.push(repo_id, self.save, "encoder", hub, card | {"model_name": repo_id})

    @classmethod
    @validate_arguments
    def load(cls, path: PathLike, mmap: bool = False):  # -> Encoder
        """Open an encoder saved with ``save``; it holds its table. With ``mmap`` the
        arrays are mapped read-only (see ``verify``). An encoder written by a pre-release
        of 2.0 raises ``fse.UnsupportedArtifactError``: rebuild it."""
        path = Path(path)
        tensors, metadata = SafetensorsContainer().open(path, mmap=mmap)
        if metadata.get("fse.kind") == "linear_head":
            raise UnsupportedArtifactError(
                f"{path} is a linear head (fse.kind linear_head), not an encoder; open it with "
                "fse.Classifier.load or fse.Regressor.load"
            )
        return cls._from_artifact(tensors, metadata, path, metadata.get(FINGERPRINT_KEY))

    @classmethod
    def _from_artifact(
        cls,
        tensors: dict[str, ndarray],
        metadata: dict[str, str],
        path: Path | None,
        fingerprint_: str | None,
    ) -> Encoder:
        """The encoder of an encoder artifact opened from ``path`` (also the encoder part of
        a head file), with its recorded checksums and ``fingerprint_``."""
        content, table, checksums = _persist.opened(tensors, metadata, path)
        return cls._new(
            content,
            table,
            path=path,
            checksums=checksums if path is not None else None,
            fingerprint_=fingerprint_,
        )

    def verify(self):
        """Check the data this encoder holds (as it encodes with them) against the
        checksums it was loaded with, its file (if loaded), and its table."""
        _persist.check_held(self._artifact()[0], self._checksums, self._path)
        if self._path is not None:
            SafetensorsContainer().verify(self._path)
        self._word_vectors.verify()

    # --- properties -------------------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        """The fingerprint of the encoder (ARCHITECTURE.md, "Identities"), the same in
        memory and after save and load."""
        if self._fingerprint is None:
            self._fingerprint = _persist.fingerprint_of(*self._artifact())
        return self._fingerprint

    @property
    def word_vectors(self) -> WordVectors:
        return self._word_vectors

    @property
    def tokenizer(self) -> str:
        """The identifier of the tokenizer (or ``external``)."""
        return self._content.tokenizer.name

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
            f"{self._content.tokenizer.describe()}{steps})"
        )
