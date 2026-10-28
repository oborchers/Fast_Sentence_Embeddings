import hashlib
import json
import pickle
import runpy
import struct
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from fse import _reference, _text
from fse._encoder import Encoder, QuantizedRows
from fse._runtime import Pipeline
from fse._wordvectors import WordVectors
from support.encoder import folded, unfolded
from support.runtime import expected, texts, word_vectors

SENTENCES = texts(700)


def _bytes(rows):
    return b"".join(a.tobytes() for a in rows) if isinstance(rows, tuple) else rows.tobytes()


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("make", [unfolded, folded], ids=["unfolded", "folded"])
def test_encode_gives_the_reference_rows(make, dtype):
    encoder = make()
    want = expected(encoder._spec, SENTENCES)
    got = encoder.encode(SENTENCES, dtype=dtype, workers=3)
    if dtype == "int8":
        assert isinstance(got, QuantizedRows)
    want_codes, want_scales = _reference.quantize_rows(want, dtype)
    assert _bytes(got) == want_codes.tobytes() + (
        b"" if want_scales is None else want_scales.tobytes()
    )
    assert _bytes(encoder.encode(SENTENCES, dtype=dtype, backend="numpy")) == _bytes(got)


@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("make", [unfolded, folded], ids=["unfolded", "folded"])
def test_save_and_load_keep_the_fingerprint_and_the_rows(tmp_path, make, mmap):
    encoder = make()
    path = tmp_path / "encoder.safetensors"
    encoder.save(path)
    if encoder.folded:
        loaded = Encoder.load(path, mmap=mmap)
    else:
        loaded = Encoder.load(path, word_vectors=encoder.word_vectors, mmap=mmap)
    assert loaded.fingerprint == encoder.fingerprint
    assert (loaded.steps, loaded.folded, loaded.recipe) == (
        encoder.steps,
        encoder.folded,
        encoder.recipe,
    )
    assert loaded.encode(SENTENCES).tobytes() == encoder.encode(SENTENCES).tobytes()
    loaded.verify()
    copy = tmp_path / "copy.safetensors"
    loaded.save(copy)
    assert (
        Encoder.load(
            copy, word_vectors=None if encoder.folded else encoder.word_vectors
        ).fingerprint
        == encoder.fingerprint
    )


def test_fasttext_and_the_external_tokenizer(tmp_path):
    encoder = unfolded("int8", fasttext=True)
    want = expected(encoder._spec, SENTENCES)
    assert encoder.encode(SENTENCES).tobytes() == want.tobytes()
    encoder.save(tmp_path / "ft.safetensors")
    loaded = Encoder.load(tmp_path / "ft.safetensors", word_vectors=encoder.word_vectors)
    assert loaded.encode(SENTENCES).tobytes() == want.tobytes()
    external = unfolded(tokenizer="external")
    tokens = [s.split() for s in SENTENCES]
    assert external.encode(tokens).tobytes() == external.encode(SENTENCES).tobytes()


def test_an_unfolded_encoder_loads_only_with_its_word_vectors(tmp_path):
    encoder = unfolded()
    path = tmp_path / "e.safetensors"
    encoder.save(path)
    with pytest.raises(TypeError, match="unfolded encoder: pass the word_vectors"):
        Encoder.load(path)
    with pytest.raises(ValueError, match="fitted on the word vectors"):
        Encoder.load(path, word_vectors=word_vectors(seed=1))
    reordered = WordVectors.from_arrays(
        encoder.word_vectors.vectors, words=[f"w{i}" for i in reversed(range(200))]
    )
    with pytest.raises(ValueError, match="fitted on the vocabulary"):
        Encoder.load(path, word_vectors=reordered)
    with pytest.raises(ValueError, match="200 weights for 150 words"):
        Encoder.load(path, word_vectors=word_vectors(n=150))
    folded_path = tmp_path / "f.safetensors"
    folded().save(folded_path)
    with pytest.raises(ValueError, match="holds its word vectors"):
        Encoder.load(folded_path, word_vectors=encoder.word_vectors)


def test_a_folded_encoder_checks_the_word_vectors_it_holds(tmp_path):
    from fse._container import SafetensorsContainer

    encoder = folded("int8")
    tensors, metadata, _ = encoder._artifact()
    metadata["fse.word_vectors"] = "sha256:" + "0" * 64
    SafetensorsContainer().write(
        tmp_path / "bad.safetensors", tensors, {**metadata, "fse.created_by": "test"}
    )
    with pytest.raises(ValueError, match="do not have the fingerprint"):
        Encoder.load(tmp_path / "bad.safetensors")


def test_encoders_are_frozen():
    encoder = unfolded()
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(encoder)
    with pytest.raises(TypeError, match="make an Encoder with fit"):
        Encoder()
    with pytest.raises(AttributeError):
        encoder.tokenizer = "external"
    assert repr(encoder) == "Encoder(test, 12 -> 6 dimensions)"
    assert (encoder.input_dim, encoder.output_dim) == (12, 6)
    assert encoder.steps == ("remove_components@1", "project@1", "normalize@1")


def test_arguments_are_checked_before_reading():
    encoder = unfolded()
    reads = []

    def counted():
        reads.append(1)
        yield "w1"

    with pytest.raises(ValueError, match="dtype must be one of"):
        encoder.encode(counted(), dtype="float64")
    with pytest.raises(ValueError, match="backend must be one of"):
        encoder.encode(counted(), backend="gpu")
    with pytest.raises(ValueError, match="workers must be a positive int"):
        encoder.encode(counted(), workers=0)
    with pytest.raises(TypeError, match="not one str"):
        encoder.encode("w1 w2")
    assert reads == []


def test_the_first_use_from_many_threads_builds_one_table_and_one_pipeline_per_backend(monkeypatch):
    built, pipelines = [], []
    real_table, real_pipeline = _text._native.VocabTable, Pipeline.__init__

    def table(*args):
        built.append(1)
        return real_table(*args)

    def pipeline(self, *args, **kwargs):
        pipelines.append(args[1] if len(args) > 1 else kwargs.get("backend", "cpu"))
        real_pipeline(self, *args, **kwargs)

    monkeypatch.setattr(_text._native, "VocabTable", table)
    monkeypatch.setattr(Pipeline, "__init__", pipeline)
    encoder = unfolded()
    want = encoder._spec
    barrier = threading.Barrier(8)
    results, errors = [], []

    def run(i):
        barrier.wait()
        try:
            results.append(
                encoder.encode(SENTENCES[:50], workers=1, backend=("cpu", "numpy")[i % 2])
            )
        except BaseException as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert built == [1]
    assert sorted(pipelines) == ["cpu", "numpy"]
    rows = expected(want, SENTENCES[:50]).tobytes()
    assert all(r.tobytes() == rows for r in results)


def test_a_failed_first_use_can_be_retried(monkeypatch):
    encoder = unfolded()
    real = Pipeline.__init__
    calls = []

    def failing(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError("no memory for the table")
        real(self, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "__init__", failing)
    with pytest.raises(MemoryError):
        encoder.encode(SENTENCES[:5])
    assert encoder.encode(SENTENCES[:5]).shape == (5, 6)


GOLDEN = Path(__file__).parent / "test_data" / "v2"


@pytest.mark.parametrize(
    ("name", "fingerprint", "rows"),
    [
        (
            "encoder_sif_unfolded",
            "sha256:c5bc6326324bd04c938ee2e73f9c5defbc0790f2a2a03c4710b0486c026febb7",
            "62b7fcaa61f113edc40f88e3125708faae27c3bf52300f7d6d2530f7f5bee472",
        ),
        (
            "encoder_usif_folded_int8",
            "sha256:ce1563cdd99c80480b72294c2a759b3a6b026fee74054bb74ce9c3fc655bdc8d",
            "2a567828aaccdc0494e97c399fe475321695602ab1e6e1eb79f2ca75e73df74a",
        ),
    ],
)
@pytest.mark.parametrize("mmap", [False, True])
def test_the_golden_encoders_load_and_give_their_pinned_rows(name, fingerprint, rows, mmap):
    """encoder 1 is fixed: these files load, keep their fingerprint and give the same bits
    on every platform (the encoding is contract arithmetic; only fit depends on BLAS)."""
    corpus = runpy.run_path(str(GOLDEN / "make_encoder_golden.py"))["CORPUS"]
    path = GOLDEN / f"{name}.safetensors"
    if "unfolded" in name:
        wv = WordVectors.load(GOLDEN / "word_vectors_float32.safetensors")
        encoder = Encoder.load(path, word_vectors=wv, mmap=mmap)
    else:
        encoder = Encoder.load(path, mmap=mmap)
    assert encoder.fingerprint == fingerprint
    assert hashlib.sha256(encoder.encode(corpus).tobytes()).hexdigest() == rows
    encoder.verify()


def _tensor_span(path, name):
    """The bytes of tensor ``name`` in a safetensors file."""
    data = path.read_bytes()
    (length,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + length])
    begin, end = header[name]["data_offsets"]
    return 8 + length + begin, 8 + length + end


def test_verify_checks_the_data_the_encoder_holds(tmp_path):
    """A mapped load copies the weights; if the file was damaged at load and repaired
    since, the file verifies, but the weights the encoder uses do not."""
    path = tmp_path / "e.safetensors"
    encoder = unfolded()
    encoder.save(path)
    original = path.read_bytes()
    begin, _ = _tensor_span(path, "weights")
    damaged = bytearray(original)
    damaged[begin] ^= 1
    path.write_bytes(bytes(damaged))
    loaded = Encoder.load(path, word_vectors=encoder.word_vectors, mmap=True)
    path.write_bytes(original)
    with pytest.raises(ValueError, match=r"weights of .* does not match its checksum"):
        loaded.verify()
    Encoder.load(path, word_vectors=encoder.word_vectors, mmap=True).verify()


def test_the_recipe_is_a_copy_all_the_way_down():
    encoder = unfolded()
    content = replace(encoder._content, recipe={"name": "x", "nested": {"a": [1, 2]}})
    nested = Encoder._new(content, encoder.word_vectors)
    nested.recipe["nested"]["a"].append(3)
    assert nested.recipe == {"name": "x", "nested": {"a": [1, 2]}}
