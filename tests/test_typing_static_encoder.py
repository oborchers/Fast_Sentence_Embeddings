"""Caller-facing encoder types, checked by mypy and executed by pytest."""

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal, assert_type

import numpy as np
from numpy.typing import NDArray

import fse
from fse.typing import Encoding, LengthLevel, SinkRows, Tokenizer

TEXTS = ["a b", "c", ""]


class _SpecialEncoder(fse.Encoder):
    pass


class _SpecialWords(fse.WordVectors):
    pass


class _SpecialStatic(fse.StaticModel):
    pass


def _encoder() -> fse.Encoder:
    return fse.Encoder.from_arrays(np.eye(3, dtype=np.float32), words=["a", "b", "c"])


def test_encoder_factory_types(tmp_path: Path) -> None:
    vectors = fse.WordVectors.from_arrays(np.eye(3, dtype=np.float32), words=["a", "b", "c"])
    assert_type(vectors, fse.WordVectors)
    encoder = fse.Encoder.from_word_vectors(vectors)
    assert_type(encoder, fse.Encoder)
    assert isinstance(encoder, fse.Encoder)
    arrays = fse.Encoder.from_arrays(np.eye(3, dtype=np.float32), words=["a", "b", "c"])
    assert_type(arrays, fse.Encoder)
    assert isinstance(arrays, fse.Encoder)
    path = tmp_path / "encoder.safetensors"
    assert_type(encoder.save(path), None)
    assert_type(encoder.verify(), None)
    loaded = fse.Encoder.load(path)
    assert_type(loaded, fse.Encoder)
    assert isinstance(loaded, fse.Encoder)
    folder = encoder.save_pretrained(tmp_path / "folder", license="mit")
    pretrained = fse.Encoder.from_pretrained(folder)
    assert_type(pretrained, fse.Encoder)
    assert isinstance(pretrained, fse.Encoder)
    quantized = encoder.quantize("int8")
    assert_type(quantized, fse.Encoder)
    assert type(quantized) is fse.Encoder
    word_path = tmp_path / "words.safetensors"
    assert_type(vectors.save(word_path), None)
    assert_type(vectors.verify(), None)
    words = fse.WordVectors.load(word_path)
    assert_type(words, fse.WordVectors)
    assert isinstance(words, fse.WordVectors)
    qwords = vectors.quantize("int8")
    assert_type(qwords, fse.WordVectors)
    assert type(qwords) is fse.WordVectors


def test_subclasses_and_honest_base_quantization(tmp_path: Path) -> None:
    words = _SpecialWords.from_arrays(np.eye(3, dtype=np.float32), words=["a", "b", "c"])
    assert_type(words, _SpecialWords)
    assert type(words) is _SpecialWords
    words.save(tmp_path / "words.safetensors")
    reloaded_words = _SpecialWords.load(tmp_path / "words.safetensors")
    assert_type(reloaded_words, _SpecialWords)
    assert type(reloaded_words) is _SpecialWords
    base_words = words.quantize("int8")
    assert_type(base_words, fse.WordVectors)
    assert type(base_words) is fse.WordVectors
    encoder = _SpecialEncoder.from_word_vectors(words)
    assert_type(encoder, _SpecialEncoder)
    assert type(encoder) is _SpecialEncoder
    array_encoder = _SpecialEncoder.from_arrays(np.eye(3, dtype=np.float32), words=["a", "b", "c"])
    assert_type(array_encoder, _SpecialEncoder)
    assert type(array_encoder) is _SpecialEncoder
    encoder.save(tmp_path / "encoder.safetensors")
    loaded = _SpecialEncoder.load(tmp_path / "encoder.safetensors")
    assert_type(loaded, _SpecialEncoder)
    assert type(loaded) is _SpecialEncoder
    folder = encoder.save_pretrained(tmp_path / "folder", license="mit")
    options: tuple[Literal["model"] | bool, ...] = ("model", False, True)
    for normalize in options:
        pretrained = _SpecialEncoder.from_pretrained(folder, normalize=normalize)
        assert_type(pretrained, _SpecialEncoder)
        assert type(pretrained) is _SpecialEncoder
    quantized = encoder.quantize("int8")
    assert_type(quantized, fse.Encoder)
    assert type(quantized) is fse.Encoder


def test_encode_returns_and_reports(tmp_path: Path) -> None:
    encoder = _encoder()
    rows = encoder.encode(TEXTS)
    assert_type(rows, NDArray[Any])
    assert isinstance(rows, np.ndarray)
    assert rows.dtype == np.float32
    half = encoder.encode(TEXTS, dtype="float16")
    assert_type(half, NDArray[Any])
    assert isinstance(half, np.ndarray)
    assert half.dtype == np.float16
    quantized = encoder.encode(TEXTS, dtype="int8")
    assert_type(quantized, fse.QuantizedRows)
    assert_type(quantized, NDArray[Any])  # type: ignore[assert-type]
    assert isinstance(quantized, fse.QuantizedRows)
    assert_type(quantized.codes, NDArray[np.int8])
    assert_type(quantized.scales, NDArray[np.float32])
    reported = encoder.encode(TEXTS, lengths="summary")
    assert_type(reported, tuple[NDArray[Any], fse.LengthReport])
    assert isinstance(reported[0], np.ndarray)
    assert_type(reported[1].lengths, LengthLevel)
    half_report = encoder.encode(TEXTS, dtype="float16", lengths="per_row")
    assert_type(half_report, tuple[NDArray[Any], fse.LengthReport])
    assert half_report[0].dtype == np.float16
    q_report = encoder.encode(TEXTS, dtype="int8", lengths="per_row")
    assert_type(q_report, tuple[fse.QuantizedRows, fse.LengthReport])
    assert isinstance(q_report[0], fse.QuantizedRows)
    sink = fse.MemorySink(3)
    result = encoder.encode(TEXTS, sink=sink)
    assert_type(result, None)
    assert_type(result, NDArray[Any])  # type: ignore[assert-type]
    assert result is None
    reported_sink = encoder.encode(TEXTS, sink=fse.MemorySink(3), lengths="summary")
    assert_type(reported_sink, tuple[None, fse.LengthReport])
    assert reported_sink[0] is None
    with fse.ShardSink(tmp_path / "shards", encoder) as shards:
        stored = encoder.encode(TEXTS, sink=shards)
        assert_type(stored, None)
        assert stored is None
    _dynamic_encode(encoder, "float32", None, "summary")
    _dynamic_encode(encoder, "int8", fse.MemorySink(3, dtype="int8"), "per_row")


def _dynamic_encode(
    encoder: fse.Encoder, dtype: str, sink: fse.Sink | None, lengths: LengthLevel
) -> None:
    plain = encoder.encode(TEXTS, dtype=dtype, sink=sink)
    assert_type(plain, NDArray[Any] | fse.QuantizedRows | None)
    assert plain is None or isinstance(plain, np.ndarray | fse.QuantizedRows)
    reported = encoder.encode(TEXTS, dtype=dtype, lengths=lengths)
    assert_type(reported, tuple[NDArray[Any] | fse.QuantizedRows | None, fse.LengthReport])
    assert isinstance(reported[1], fse.LengthReport)


def test_encoded_batch_fields_and_generators() -> None:
    encoder = _encoder()
    gen = encoder.encode_batches([TEXTS])
    assert_type(gen, Generator[fse.EncodedBatch, None, None])
    batch = next(gen)
    assert_type(batch.rows, NDArray[Any] | fse.QuantizedRows | None)
    assert_type(batch.vectors, NDArray[Any] | fse.QuantizedRows | None)
    assert isinstance(batch.rows, np.ndarray)
    assert batch.rows.dtype == np.float32
    gen.close()
    for dtype in ("int8", "float16"):
        with_rows = encoder.encode_batches([TEXTS], dtype=dtype)
        item = next(with_rows)
        assert isinstance(item.rows, fse.QuantizedRows if dtype == "int8" else np.ndarray)
        with_rows.close()
    no_rows = encoder.encode_batches([TEXTS], sink=fse.MemorySink(3))
    assert next(no_rows).rows is None
    no_rows.close()
    reports = encoder.encode_batches([TEXTS], lengths="per_row")
    assert_type(reports, Generator[tuple[fse.EncodedBatch, fse.LengthReport], None, None])
    assert isinstance(next(reports)[1], fse.LengthReport)
    reports.close()


class _CustomSink(fse.Sink):
    dim: int

    def __init__(self, dim: int):
        self.dim = dim
        self.rows = np.empty((3, dim), np.float32)

    def target(self, first_row: int, n_rows: int) -> SinkRows:
        return self.rows[first_row : first_row + n_rows]

    def commit(self, first_row: int, rows: SinkRows) -> None:
        assert first_row >= 0
        assert isinstance(rows, np.ndarray)


def test_explicit_sink_subclass_can_assign_dim() -> None:
    sink = _CustomSink(3)
    assert sink.dim == 3
    protocol: fse.Sink = sink
    result = _encoder().encode(TEXTS, sink=protocol)
    assert_type(result, None)
    assert result is None
    assert sink.rows.dtype == np.float32


def _static_folder(path: Path) -> Path:
    from fse._container import SafetensorsContainer

    path.mkdir()
    tokens = ["[UNK]", "a", "b", "c"]
    table = np.arange(12, dtype=np.float32).reshape(4, 3)
    SafetensorsContainer().write(path / "model.safetensors", {"embeddings": table}, {})
    (path / "config.json").write_text(json.dumps({"normalize": False, "max_length": 8}))
    document = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": {
            "type": "BertNormalizer",
            "clean_text": True,
            "handle_chinese_chars": True,
            "strip_accents": None,
            "lowercase": True,
        },
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": dict(zip(tokens, range(4), strict=True)),
        },
    }
    (path / "tokenizer.json").write_text(json.dumps(document))
    return path


def test_static_model_and_tokenizer_public_types(tmp_path: Path) -> None:
    folder = _static_folder(tmp_path / "static")
    model = fse.StaticModel.from_pretrained(folder)
    assert_type(model, fse.StaticModel)
    assert isinstance(model, fse.StaticModel)
    tokenizer = model.tokenizer
    assert_type(tokenizer, Tokenizer)
    encoded = tokenizer.encode("a b", add_special_tokens=False)
    assert_type(encoded, Encoding)
    assert encoded.ids == [1, 2]
    batch = tokenizer.encode_batch(TEXTS, add_special_tokens=False)
    assert_type(batch, list[Encoding])
    assert len(batch) == len(TEXTS)
    subclass = _SpecialStatic.from_pretrained(folder)
    assert_type(subclass, _SpecialStatic)
    assert type(subclass) is _SpecialStatic
    encoder = _SpecialEncoder.from_pretrained(folder)
    assert_type(encoder, _SpecialEncoder)
    assert type(encoder) is _SpecialEncoder


def test_union_dtype_literal_and_omitted_sink() -> None:
    encoder = _encoder()
    dtype: Literal["float16", "int8"] = "int8"
    result = encoder.encode(TEXTS, dtype=dtype)
    assert_type(result, NDArray[Any] | fse.QuantizedRows)
    assert isinstance(result, np.ndarray | fse.QuantizedRows)
