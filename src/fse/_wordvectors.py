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

import codecs
import json
import re
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy import ndarray

from fse import _codes
from fse._container import (
    CHECKSUM_PREFIX,
    CREATED_BY_KEY,
    FINGERPRINT_KEY,
    INTEGER,
    NpyContainer,
    array_sha256,
    canonical_json,
    fingerprint,
    tensor_code,
)
from fse._contract import OOV_TOKEN, UnsupportedArtifactError
from fse._validation import PathLike, validate_arguments

KIND = "word_vectors"
FORMAT_VERSION = "1"
QUANTIZATION = "quantize_rows@1"
NGRAM_HASH = "fasttext@1"

_KEYS = {
    "fse.kind",
    "fse.format_version",
    CREATED_BY_KEY,
    "fse.dim",
    "fse.dtype",
    "fse.quantization",
    "fse.quantized_from",
    "fse.fasttext",
    "fse.ngram_hash",
}
_TENSORS = {"vectors", "scales", "counts", "vocab.bytes", "vocab.offsets", "ngrams", "ngram_scales"}
_DTYPE_OF = {"F32": "float32", "F16": "float16", "I8": "int8"}
_BLOCK_ROWS = 16_384
_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_WORDS = OOV_TOKEN  # word ids must stay below the reserved token ids


class FastTextParams(NamedTuple):
    """How unknown words are hashed into ngram buckets (the FastText scheme)."""

    min_n: int
    max_n: int
    bucket: int


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


def _encode_words(words: Sequence[str]) -> tuple[ndarray, ndarray]:
    """The UTF-8 bytes and offsets of ``words``, without a list of bytes objects."""
    for i, word in enumerate(words):
        if not isinstance(word, str):
            raise TypeError(f"word {i} must be a str, not {type(word).__name__}")
    try:
        lengths = np.fromiter(
            (len(w.encode("utf-8")) for w in words), dtype=np.int64, count=len(words)
        )
    except UnicodeEncodeError:
        bad = next(i for i, w in enumerate(words) if not _has_utf8(w))
        raise ValueError(f"word {bad} ({words[bad]!r}) has no UTF-8 form") from None
    offsets = np.zeros(len(words) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    # Every word encodes on its own, so their concatenation encodes to the joined bytes.
    data = np.frombuffer("".join(words).encode("utf-8"), dtype=np.uint8)
    return data, offsets


def _has_utf8(word: str) -> bool:
    try:
        word.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _integer(metadata: dict[str, str], key: str, error=UnsupportedArtifactError) -> int:
    value = metadata.get(key)
    if value is None or not INTEGER.fullmatch(value):
        raise error(f"{key} must be a non-negative decimal integer, not {value!r}")
    return int(value)


def _check_structure(tensors: dict[str, ndarray], metadata: dict[str, str]):
    """Check metadata and tensors of word vectors against each other (every open)."""
    kind = metadata.get("fse.kind")
    if kind != KIND:
        raise UnsupportedArtifactError(f"the artifact is a {kind!r}, not {KIND!r}")
    if metadata.get("fse.format_version") != FORMAT_VERSION:
        raise UnsupportedArtifactError(
            f"word vectors of format version {metadata.get('fse.format_version')!r} are not "
            f"supported by this fse; supported: {FORMAT_VERSION}"
        )
    unknown = {
        k
        for k in metadata
        if k not in _KEYS and k != FINGERPRINT_KEY and not k.startswith(CHECKSUM_PREFIX)
    }
    if unknown:
        raise UnsupportedArtifactError(f"unknown metadata keys of word vectors: {sorted(unknown)}")
    if set(tensors) - _TENSORS:
        raise UnsupportedArtifactError(
            f"unknown tensors of word vectors: {sorted(set(tensors) - _TENSORS)}"
        )
    for name in ("vectors", "vocab.bytes", "vocab.offsets"):
        if name not in tensors:
            raise ValueError(f"word vectors need the tensor {name}")

    vectors = tensors["vectors"]
    dtype = metadata.get("fse.dtype")
    if dtype not in _codes.STORAGE_TYPES:
        raise UnsupportedArtifactError(
            f"fse.dtype must be one of {sorted(_codes.STORAGE_TYPES)}, not {dtype!r}"
        )
    code = tensor_code(vectors)
    if _DTYPE_OF.get(code) != dtype or vectors.ndim != 2:
        raise ValueError(
            f"vectors must be a matrix of {dtype}, not {code} with {vectors.ndim} dimensions"
        )
    n, dim = vectors.shape
    if _integer(metadata, "fse.dim", ValueError) != dim:
        raise ValueError(f"fse.dim is {metadata['fse.dim']}, the vectors have {dim} values per row")
    if dim < 1:
        raise ValueError("word vectors need at least one dimension")
    if not 1 <= n < _MAX_WORDS:
        raise ValueError(f"word vectors need between 1 and {_MAX_WORDS - 1} words, not {n}")
    quantized = dtype != "float32"
    if quantized != ("fse.quantization" in metadata):
        raise ValueError("fse.quantization is given exactly for float16 and int8 vectors")
    origin = metadata.get("fse.quantized_from")
    if origin is not None and (not quantized or not _FINGERPRINT.fullmatch(origin)):
        raise ValueError(
            f"fse.quantized_from must be the fingerprint of float32 vectors and is given "
            f"only for float16 and int8, not {origin!r}"
        )
    if quantized and metadata["fse.quantization"] != QUANTIZATION:
        raise UnsupportedArtifactError(
            f"quantization {metadata['fse.quantization']!r} is not supported; supported: {QUANTIZATION}"
        )
    _check_scales(tensors, "scales", n, dtype == "int8")
    if "counts" in tensors:
        _check_vector(tensors["counts"], "counts", "I64", n)

    data = tensors["vocab.bytes"]
    offsets = tensors["vocab.offsets"]
    _check_vector(data, "vocab.bytes", "U8", len(data))
    _check_vector(offsets, "vocab.offsets", "I64", n + 1)
    if offsets[0] != 0 or offsets[-1] != len(data) or _decreases(offsets):
        raise ValueError(
            f"vocab.offsets must rise from 0 to {len(data)}, the number of vocabulary bytes"
        )
    _check_utf8(data, offsets)

    has_ngrams = "ngrams" in tensors
    if has_ngrams != ("fse.fasttext" in metadata) or has_ngrams != ("fse.ngram_hash" in metadata):
        raise ValueError("fse.fasttext, fse.ngram_hash and the ngrams tensor are given together")
    if has_ngrams:
        params = _parse_fasttext(metadata["fse.fasttext"])
        if metadata["fse.ngram_hash"] != NGRAM_HASH:
            raise UnsupportedArtifactError(
                f"ngram hash {metadata['fse.ngram_hash']!r} is not supported; supported: {NGRAM_HASH}"
            )
        ngrams = tensors["ngrams"]
        if tensor_code(ngrams) != code or ngrams.shape != (params.bucket, dim):
            raise ValueError(f"ngrams must be a ({params.bucket}, {dim}) matrix of {dtype}")
        _check_scales(tensors, "ngram_scales", params.bucket, dtype == "int8")
    elif "ngram_scales" in tensors:
        raise ValueError("ngram_scales without ngrams")


def _check_scales(tensors: dict[str, ndarray], name: str, rows: int, needed: bool):
    if needed != (name in tensors):
        raise ValueError(f"{name} are given exactly for int8 rows")
    if needed:
        _check_vector(tensors[name], name, "F32", rows)


def _check_vector(array: ndarray, name: str, code: str, length: int):
    if tensor_code(array) != code or array.shape != (length,):
        raise ValueError(
            f"{name} must be a vector of {length} {code} values, not {array.dtype.str} {array.shape}"
        )


def _decreases(offsets: ndarray) -> bool:
    """Whether ``offsets`` ever decrease, compared in blocks to bound the temporaries."""
    step = 2**20
    for begin in range(0, len(offsets) - 1, step):
        block = np.asarray(offsets[begin : begin + step + 1])
        if (block[1:] < block[:-1]).any():
            return True
    return False


def _check_utf8(data: ndarray, offsets: ndarray, chunk_bytes: int = 2**24):
    """Check that the vocabulary is UTF-8 and every word starts at a character boundary.

    An incremental decoder takes fixed blocks of bytes, whatever the words, so memory
    stays bounded also for a single huge word.
    """
    step = 2**20
    for begin in range(1, len(offsets) - 1, step):
        inner = np.asarray(offsets[begin : min(begin + step, len(offsets) - 1)])
        inner = inner[inner < len(data)]
        if len(inner) and ((np.asarray(data[inner]) & 0xC0) == 0x80).any():
            raise ValueError("a word of the vocabulary starts inside a UTF-8 character")
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    try:
        for begin in range(0, len(data), chunk_bytes):
            decoder.decode(data[begin : begin + chunk_bytes].tobytes())
        decoder.decode(b"", final=True)
    except UnicodeDecodeError as err:
        raise ValueError(f"the vocabulary is not valid UTF-8: {err}") from None


def _parse_fasttext(text: str) -> FastTextParams:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"fse.fasttext is not JSON: {text!r}") from None
    if not isinstance(value, dict) or set(value) != {"bucket", "max_n", "min_n"}:
        raise ValueError("fse.fasttext must hold exactly bucket, max_n and min_n")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in value.values()):
        raise ValueError("fse.fasttext must hold integers")
    if canonical_json(value) != text:
        raise ValueError(f"fse.fasttext must be canonical JSON, not {text!r}")
    params = FastTextParams(value["min_n"], value["max_n"], value["bucket"])
    _check_fasttext(params)
    return params


def _check_fasttext(params: FastTextParams):
    if not 1 <= params.min_n <= params.max_n:
        raise ValueError(
            f"FastText needs 1 <= min_n <= max_n, not {params.min_n} and {params.max_n}"
        )
    if not 1 <= params.bucket <= 2**32 - 1:
        raise ValueError(f"FastText needs between 1 and 2**32 - 1 buckets, not {params.bucket}")


def _check_values(tensors: dict[str, ndarray]):
    """Check the values of word vectors in blocks (verify and every writer)."""
    for name, scales in (("vectors", "scales"), ("ngrams", "ngram_scales")):
        if name not in tensors:
            continue
        rows = tensors[name]
        for begin in range(0, len(rows), _BLOCK_ROWS):
            block = np.asarray(rows[begin : begin + _BLOCK_ROWS])
            if block.dtype == np.int8:
                bad = np.flatnonzero((block == -128).any(axis=1))
                problem = "has the code -128 (codes are -127 to 127)"
            else:
                bad = np.flatnonzero(~np.isfinite(block).all(axis=1))
                problem = "contains NaN or infinite values"
            if len(bad):
                raise ValueError(f"row {begin + int(bad[0])} of {name} {problem}")
        if scales in tensors:
            values = np.asarray(tensors[scales])
            bad = np.flatnonzero(~np.isfinite(values) | (values < 0))
            if len(bad):
                raise ValueError(f"{scales} {int(bad[0])} is negative or not finite")
            # code * scale must stay finite, or reading the row would give infinity.
            for begin in range(0, len(rows), _BLOCK_ROWS):
                block = np.asarray(rows[begin : begin + _BLOCK_ROWS])
                top = np.abs(block.astype(np.float32)).max(axis=1) if block.shape[1] else 0
                with np.errstate(over="ignore"):
                    product = top * values[begin : begin + _BLOCK_ROWS]
                bad = np.flatnonzero(~np.isfinite(product))
                if len(bad):
                    raise ValueError(
                        f"row {begin + int(bad[0])} of {name} overflows float32 with its scale"
                    )
    if "counts" in tensors:
        bad = np.flatnonzero(np.asarray(tensors["counts"]) < 0)
        if len(bad):
            raise ValueError(f"count {int(bad[0])} is negative")


def _lookup(data: ndarray, offsets: ndarray) -> dict[bytes, int]:
    raw = data.tobytes()
    bounds = offsets.tolist()
    index: dict[bytes, int] = {}
    for i in range(len(bounds) - 1):
        word = raw[bounds[i] : bounds[i + 1]]
        if index.setdefault(word, i) != i:
            raise ValueError(f"word {i} ({word.decode('utf-8')!r}) is also word {index[word]}")
    return index


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
        if (words is None) == (vocab_bytes is None or vocab_offsets is None) or (
            words is not None and (vocab_bytes is not None or vocab_offsets is not None)
        ):
            raise ValueError("give either words or both vocab_bytes and vocab_offsets")
        if words is not None:
            if isinstance(words, str) or not isinstance(words, Sequence):
                raise TypeError("words must be a sequence of str")
            vocab_bytes, vocab_offsets = _encode_words(words)
        tensors = {"vectors": vectors, "vocab.bytes": vocab_bytes, "vocab.offsets": vocab_offsets}
        for name, value in (
            ("scales", scales),
            ("counts", counts),
            ("ngrams", ngrams),
            ("ngram_scales", ngram_scales),
        ):
            if value is not None:
                tensors[name] = value
        for name, value in tensors.items():
            if not isinstance(value, ndarray):
                raise TypeError(f"{name} must be a numpy array, not {type(value).__name__}")
        try:
            dtype = _DTYPE_OF.get(tensor_code(vectors))
        except ValueError:
            dtype = None
        if dtype is None:
            raise ValueError(f"vectors must be float32, float16 or int8, not {vectors.dtype}")
        metadata = {
            "fse.kind": KIND,
            "fse.format_version": FORMAT_VERSION,
            "fse.dim": str(vectors.shape[1]) if vectors.ndim == 2 else "0",
            "fse.dtype": dtype,
        }
        if dtype != "float32":
            metadata["fse.quantization"] = QUANTIZATION
        if fasttext is not None:
            if not isinstance(fasttext, FastTextParams):
                raise TypeError("fasttext must be FastTextParams")
            _check_fasttext(fasttext)
            metadata["fse.fasttext"] = canonical_json(fasttext._asdict())
            metadata["fse.ngram_hash"] = NGRAM_HASH
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
        tensors, metadata = NpyContainer().open(path, mmap=mmap)
        return cls(tensors, metadata, path)

    @validate_arguments
    def save(self, path: PathLike):
        """Write the word vectors into the new directory ``path``.

        Loaded vectors are verified first, so damaged data never gets new checksums.
        """
        if not self._verified:
            self.verify()
        metadata = {
            k: v
            for k, v in self._metadata.items()
            if k != FINGERPRINT_KEY and k != CREATED_BY_KEY and not k.startswith(CHECKSUM_PREFIX)
        }
        metadata[CREATED_BY_KEY] = _created_by()
        NpyContainer().write(Path(path), self._tensors, metadata)

    def verify(self):
        """Check everything: the data on disk against its checksums, the values and words."""
        if self._path is not None:
            NpyContainer().verify(self._path)
        _check_structure(self._tensors, self._metadata)
        _check_values(self._tensors)
        if self._index is None:
            self._index = _lookup(self._tensors["vocab.bytes"], self._tensors["vocab.offsets"])
        self._verified = True

    @validate_arguments
    def quantize(self, dtype: Literal["float16", "int8"]):  # -> WordVectors, as for load
        """float16 or int8 copies of float32 word vectors, quantized in blocks."""
        if self.dtype != "float32":
            raise ValueError(f"only float32 word vectors can be quantized, these are {self.dtype}")
        if not self._verified:
            self.verify()
        tensors = {k: v for k, v in self._tensors.items() if k not in ("vectors", "ngrams")}
        for name, scales in (("vectors", "scales"), ("ngrams", "ngram_scales")):
            if name in self._tensors:
                codes, codes_scales = _codes.quantize(
                    self._tensors[name], dtype, kind=f"row of {name}"
                )
                codes.setflags(write=False)
                tensors[name] = codes
                if codes_scales is not None:
                    codes_scales.setflags(write=False)
                    tensors[scales] = codes_scales
        metadata = {
            k: v
            for k, v in self._metadata.items()
            if k != FINGERPRINT_KEY and k != CREATED_BY_KEY and not k.startswith(CHECKSUM_PREFIX)
        }
        metadata["fse.dtype"] = dtype
        metadata["fse.quantization"] = QUANTIZATION
        metadata["fse.quantized_from"] = self.fingerprint
        return WordVectors._new(tensors, metadata, self._index)

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
