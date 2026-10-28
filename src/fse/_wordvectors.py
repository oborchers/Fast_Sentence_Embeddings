#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Word vectors in the 2.0 format: immutable, memory-mappable, without pickle.

ARCHITECTURE.md, "word_vectors 1", specifies the artifact. The vocabulary is the UTF-8
bytes of the words with int64 offsets; a word is found by an exact match of its bytes.
Vectors are stored as float32, float16 or int8 codes with a float32 scale per row, and
read as float32. FastText vectors also hold their ngram rows and hashing parameters.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy import ndarray

from fse import _codes
from fse._container import (
    CHECKSUM_PREFIX,
    CREATED_BY_KEY,
    FINGERPRINT_KEY,
    CorruptArtifactError,
    SafetensorsContainer,
    array_sha256,
    fingerprint,
    tensor_code,
)
from fse._validation import PathLike, validate_arguments
from fse._wordvectors_format import (
    _BLOCK_ROWS,
    QUANTIZATION,
    FastTextParams,
    _check_structure,
    _check_values,
    _lookup,
    _new_metadata,
    _new_tensors,
    _parse_fasttext,
    _vocabulary,
)


def _created_by() -> str:
    try:
        return f"fse {version('fse')}"
    except PackageNotFoundError:  # pragma: no cover
        return "fse"


def _view(array: ndarray | None) -> ndarray | None:
    """A new read-only header on internal data: neither writable nor reshaping the data."""
    return None if array is None else array.view()


def _frozen(array: ndarray) -> ndarray:
    copy = np.array(array, copy=True, order="C")
    copy.setflags(write=False)
    return copy


def _read_only(array: ndarray) -> ndarray:
    """``array`` itself, made read-only, if it is C-contiguous; otherwise a copy."""
    if not array.flags.c_contiguous:
        return _frozen(array)
    array.setflags(write=False)
    return array


class WordVectors:
    """Word vectors with their vocabulary, read as float32 and stored in a smaller format.

    Create them with ``from_arrays``, ``fse.gensim.from_gensim`` or ``load``; they cannot
    be changed afterwards. ``quantize`` gives float16 or int8 copies, ``save`` writes them.
    """

    __slots__ = ("_fingerprint", "_index", "_metadata", "_path", "_tensors", "_verified")

    def __init__(
        self, tensors: dict[str, ndarray], metadata: dict[str, str], path: Path | None = None
    ):
        _check_structure(tensors, metadata)
        self._tensors = tensors
        self._metadata = metadata
        self._path = path
        self._index: dict[bytes, int] | None = None
        self._fingerprint: str | None = metadata.get(FINGERPRINT_KEY)
        # Whether the values, the words and (if loaded) the data on disk were checked.
        self._verified = False

    @classmethod
    def from_arrays(
        cls,
        vectors: Any,
        *,
        words: Sequence[str] | None = None,
        vocab_bytes: Any = None,
        vocab_offsets: Any = None,
        scales: Any = None,
        counts: Any = None,
        fasttext: FastTextParams | None = None,
        ngrams: Any = None,
        ngram_scales: Any = None,
        _copy: bool = True,
    ) -> WordVectors:
        """Word vectors from arrays, copied and checked.

        ``vectors`` holds one row per word as float32, float16 or int8 codes (with
        ``scales``). The vocabulary is either ``words`` or its UTF-8 ``vocab_bytes`` and
        int64 ``vocab_offsets``. FastText vectors also take ``fasttext`` and the ``ngrams``
        rows (with ``ngram_scales`` for int8).
        """
        copy = _copy  # False only for arrays the caller created and hands over
        if not isinstance(vectors, ndarray):
            raise TypeError(f"vectors must be a numpy array, not {type(vectors).__name__}")
        vocab_bytes, vocab_offsets = _vocabulary(words, vocab_bytes, vocab_offsets)
        tensors = _new_tensors(
            {
                "vectors": vectors,
                "vocab.bytes": vocab_bytes,
                "vocab.offsets": vocab_offsets,
                "scales": scales,
                "counts": counts,
                "ngrams": ngrams,
                "ngram_scales": ngram_scales,
            }
        )
        metadata = _new_metadata(vectors, fasttext)
        return cls._new(
            {k: _frozen(v) if copy else _read_only(v) for k, v in tensors.items()}, metadata
        )

    @classmethod
    def _new(
        cls,
        tensors: dict[str, ndarray],
        metadata: dict[str, str],
        index: dict[bytes, int] | None = None,
    ) -> WordVectors:
        """New word vectors from arrays this module owns; checks values and words.

        ``index`` is the checked lookup of the same vocabulary, if there is one.
        """
        out = cls(tensors, metadata)
        _check_values(tensors)
        out._index = index or _lookup(tensors["vocab.bytes"], tensors["vocab.offsets"])
        out._verified = True
        return out

    @classmethod
    @validate_arguments
    def load(
        cls, path: PathLike, mmap: bool = False
    ):  # -> WordVectors; pydantic cannot resolve it here
        """Open word vectors saved with ``save``.

        Without ``mmap`` the arrays are read into memory and every checksum is checked;
        with ``mmap`` they are mapped read-only and only the structure is checked (see
        ``verify``). Codes are never converted to float32.
        """
        path = Path(path)
        if path.is_dir():
            raise ValueError(
                f"{path} is a directory; word vectors are one {SafetensorsContainer.suffix} "
                "file (directories of .npy files hold stored sentence vectors)"
            )
        tensors, metadata = SafetensorsContainer().open(path, mmap=mmap)
        return cls(tensors, metadata, path)

    @validate_arguments
    def save(self, path: PathLike):
        """Write the word vectors into the new ``.safetensors`` file ``path``.

        Loaded vectors are checked first, and every tensor is compared with the checksum
        it was loaded with while it is written, so damaged or changed data never gets new
        checksums.
        """
        if not self._verified or self._mapped():
            self._check_contents()
        metadata = {
            k: v
            for k, v in self._metadata.items()
            if k != FINGERPRINT_KEY and k != CREATED_BY_KEY and not k.startswith(CHECKSUM_PREFIX)
        }
        metadata[CREATED_BY_KEY] = _created_by()
        SafetensorsContainer().write(
            Path(path), self._tensors, metadata, expected=self._checksums()
        )

    def _mapped(self) -> bool:
        return any(isinstance(t, np.memmap) for t in self._tensors.values())

    def _checksums(self) -> dict[str, str] | None:
        """The checksums the tensors were loaded with, or None for vectors made in memory."""
        if self._path is None:
            return None
        return {
            k[len(CHECKSUM_PREFIX) :]: v
            for k, v in self._metadata.items()
            if k.startswith(CHECKSUM_PREFIX)
        }

    def _check_contents(self):
        """Check the data this object holds: checksums (if loaded), values and words."""
        expected = self._checksums()
        if expected is not None:
            for name, array in self._tensors.items():
                if array_sha256(array) != expected[name]:
                    raise CorruptArtifactError(
                        f"tensor {name} of {self._path} does not match its checksum"
                    )
        _check_structure(self._tensors, self._metadata)
        _check_values(self._tensors)
        if self._index is None:
            self._index = _lookup(self._tensors["vocab.bytes"], self._tensors["vocab.offsets"])
        # Mapped data can change through the file, so its check is never remembered.
        self._verified = not self._mapped()

    def verify(self):
        """Check everything: the data this object holds against the checksums it was loaded
        with, the files on disk, the values and the words.
        """
        self._check_contents()
        if self._path is not None:
            SafetensorsContainer().verify(self._path)

    @validate_arguments
    def quantize(self, dtype: Literal["float16", "int8"]):  # -> WordVectors, as for load
        """float16 or int8 copies of float32 word vectors, quantized in blocks.

        The result owns all its arrays. For loaded vectors, every block is read once, and
        the same copy is hashed and quantized, so the result is exactly the content the
        checksums describe, even if the file changes meanwhile.
        """
        if self.dtype != "float32":
            raise ValueError(f"only float32 word vectors can be quantized, these are {self.dtype}")
        if not self._verified:
            self._check_contents()
        expected = self._checksums()
        tensors = {
            name: self._checked_copy(name, array, expected)
            for name, array in self._tensors.items()
            if name not in ("vectors", "ngrams")
        }
        for name, scales in (("vectors", "scales"), ("ngrams", "ngram_scales")):
            if name not in self._tensors:
                continue
            codes, codes_scales = self._quantized(name, dtype, expected)
            tensors[name] = codes
            if codes_scales is not None:
                tensors[scales] = codes_scales
        # The copied vocabulary matches the checksums, so the checked lookup still holds.
        return WordVectors._new(tensors, self._quantized_metadata(dtype), self._index)

    def _checked_copy(self, name: str, array: ndarray, expected) -> ndarray:
        """A frozen copy of tensor ``name``, which must still match its loaded checksum."""
        copy = _frozen(array)
        if expected is not None and array_sha256(copy) != expected[name]:
            raise CorruptArtifactError(f"tensor {name} of {self._path} changed since it was loaded")
        return copy

    def _quantized(self, name: str, dtype: str, expected) -> tuple[ndarray, ndarray | None]:
        """The codes (and int8 scales) of rows ``name``: every block is read once, and the
        same copy is hashed and quantized."""
        source = self._tensors[name]
        codes = np.empty(source.shape, dtype=_codes.STORAGE_TYPES[dtype])
        codes_scales = np.empty(len(source), dtype=np.float32) if dtype == "int8" else None
        digest = hashlib.sha256()
        for begin in range(0, len(source), _BLOCK_ROWS):
            block = np.array(source[begin : begin + _BLOCK_ROWS], dtype=np.float32, copy=True)
            digest.update(memoryview(block.reshape(-1).view(np.uint8)))
            _codes.quantize_into(
                block,
                codes[begin : begin + len(block)],
                None if codes_scales is None else codes_scales[begin : begin + len(block)],
                first_row=begin,
                kind=f"row of {name}",
            )
        if expected is not None and digest.hexdigest() != expected[name]:
            raise CorruptArtifactError(f"tensor {name} of {self._path} changed since it was loaded")
        codes.setflags(write=False)
        if codes_scales is not None:
            codes_scales.setflags(write=False)
        return codes, codes_scales

    def _quantized_metadata(self, dtype: str) -> dict[str, str]:
        metadata = {
            k: v
            for k, v in self._metadata.items()
            if k != FINGERPRINT_KEY and k != CREATED_BY_KEY and not k.startswith(CHECKSUM_PREFIX)
        }
        metadata["fse.dtype"] = dtype
        metadata["fse.quantization"] = QUANTIZATION
        metadata["fse.quantized_from"] = self.fingerprint
        return metadata

    def __reduce__(self):
        raise TypeError("WordVectors cannot be pickled; use save() and WordVectors.load()")

    def __len__(self) -> int:
        return len(self._tensors["vectors"])

    def __repr__(self) -> str:
        kind = ", FastText" if self.fasttext else ""
        return f"WordVectors({len(self)} words, {self.dim} dimensions, {self.dtype}{kind})"

    @property
    def dim(self) -> int:
        return int(self._tensors["vectors"].shape[1])

    @property
    def dtype(self) -> str:
        return self._metadata["fse.dtype"]

    @property
    def vectors(self) -> ndarray:
        """The stored rows: float32 values, or float16 or int8 codes."""
        return self._tensors["vectors"].view()

    @property
    def scales(self) -> ndarray | None:
        return _view(self._tensors.get("scales"))

    @property
    def counts(self) -> ndarray | None:
        return _view(self._tensors.get("counts"))

    @property
    def vocab_bytes(self) -> ndarray:
        return self._tensors["vocab.bytes"].view()

    @property
    def vocab_offsets(self) -> ndarray:
        return self._tensors["vocab.offsets"].view()

    @property
    def fasttext(self) -> FastTextParams | None:
        text = self._metadata.get("fse.fasttext")
        return None if text is None else _parse_fasttext(text)

    @property
    def ngrams(self) -> ndarray | None:
        return _view(self._tensors.get("ngrams"))

    @property
    def ngram_scales(self) -> ndarray | None:
        return _view(self._tensors.get("ngram_scales"))

    @property
    def path(self) -> Path | None:
        """The directory the vectors were loaded from, if any."""
        return self._path

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self._metadata)

    @property
    def fingerprint(self) -> str:
        """The fingerprint of the content (ARCHITECTURE.md, "Identities")."""
        if self._fingerprint is None:
            entries = {
                k: (tensor_code(v), v.shape, array_sha256(v)) for k, v in self._tensors.items()
            }
            self._fingerprint = fingerprint(entries, self._metadata)
        return self._fingerprint

    def _words(self) -> dict[bytes, int]:
        if self._index is None:
            self._index = _lookup(self._tensors["vocab.bytes"], self._tensors["vocab.offsets"])
        return self._index

    def index(self, word: str) -> int:
        """The id of ``word``, matched by its exact UTF-8 bytes; raises KeyError if unknown."""
        if not isinstance(word, str):
            raise TypeError(f"a word must be a str, not {type(word).__name__}")
        try:
            return self._words()[word.encode("utf-8")]
        except (KeyError, UnicodeEncodeError):
            raise KeyError(word) from None

    def __contains__(self, word: object) -> bool:
        if not isinstance(word, str):
            return False
        try:
            self.index(word)
        except KeyError:
            return False
        return True

    def word(self, index: int) -> str:
        """The word with id ``index``."""
        i = self._check_index(index)
        offsets = self._tensors["vocab.offsets"]
        return self._tensors["vocab.bytes"][offsets[i] : offsets[i + 1]].tobytes().decode("utf-8")

    def _check_index(self, index: object) -> int:
        if isinstance(index, bool) or not isinstance(index, int | np.integer):
            raise TypeError(f"an index must be an int, not {type(index).__name__}")
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} words")
        return int(index)

    def get_vector(self, key: str | int) -> ndarray:
        """The float32 row of a word or word id, as a new array.

        For FastText this is the row of the word itself, without its ngrams (gensim's
        ``kv[word]`` averages both); unknown words raise KeyError.
        """
        i = self.index(key) if isinstance(key, str) else self._check_index(key)
        return self.get_vectors(np.array([i]))[0]

    def get_vectors(self, indices: Any) -> ndarray:
        """The float32 rows of the word ids ``indices``, as a new (k, dim) array."""
        ids = np.asarray(indices)
        if ids.size == 0 and ids.ndim == 1:
            ids = ids.astype(np.int64)
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise TypeError("indices must be a one-dimensional array of integers")
        if ids.size and (ids.min() < 0 or ids.max() >= len(self)):
            raise IndexError(f"indices must be between 0 and {len(self) - 1}")
        scales = self.scales
        return _codes.dequantize(self.vectors[ids], None if scales is None else scales[ids])
