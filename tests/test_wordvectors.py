import gc
import json
import pickle
import sys
import tracemalloc

import numpy as np
import pytest

from fse._container import SafetensorsContainer
from fse._reference import dequantize, quantize_rows
from fse._wordvectors import FastTextParams, WordVectors
from support.container import fake_fingerprint
from support.wordvectors import (
    F32,
    WORDS,
    _flip_last_byte,
    _overwrite,
    _rewrite,
    _same,
    _saved,
    _wv,
)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("fasttext", [False, True])
@pytest.mark.parametrize("mmap", [False, True])
def test_round_trip(tmp_path, dtype, fasttext, mmap):
    wv = _wv(dtype, fasttext)
    wv.save(tmp_path / "wv.safetensors")
    loaded = WordVectors.load(tmp_path / "wv.safetensors", mmap=mmap)
    _same(wv, loaded)
    assert loaded.path == tmp_path / "wv.safetensors"
    assert loaded.dtype == dtype
    assert isinstance(loaded.vectors, np.memmap) == mmap
    loaded.verify()
    assert loaded.metadata["fse.created_by"].startswith("fse")
    # Saving again gives the same content and fingerprint.
    loaded.save(tmp_path / "again.safetensors")
    _same(loaded, WordVectors.load(tmp_path / "again.safetensors"))


def test_lookup_is_byte_exact():
    wv = _wv()
    assert len(wv) == len(WORDS)
    for i, word in enumerate(WORDS):
        assert wv.index(word) == i
        assert wv.word(i) == word
        assert word in wv
    # NFC and NFD spellings are different words.
    assert wv.index("é") != wv.index("é")
    for missing in ("THE", "x", "\ud800", 3, None):
        assert missing not in wv
    with pytest.raises(KeyError):
        wv.index("\ud800")
    with pytest.raises(TypeError, match="must be a str"):
        wv.index(b"the")
    with pytest.raises(IndexError, match="out of range"):
        wv.word(len(WORDS))
    with pytest.raises(TypeError, match="must be an int"):
        wv.word(True)
    assert wv.word(np.int64(1)) == "é"


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_get_vector_is_float32_and_new(dtype):
    source = _wv()
    wv = source.quantize(dtype) if dtype != "float32" else source
    expected = dequantize(wv.vectors, wv.scales)
    for i, word in enumerate(WORDS):
        got = wv.get_vector(word)
        assert got.dtype == F32
        assert got.view(np.uint32).tolist() == expected[i].view(np.uint32).tolist()
    rows = wv.get_vectors(np.array([2, 0]))
    np.testing.assert_array_equal(rows.view(np.uint32), expected[[2, 0]].view(np.uint32))
    rows[0, 0] = 99
    assert wv.get_vector(2)[0] != 99
    assert wv.get_vectors(np.array([], dtype=np.int64)).shape == (0, 5)
    with pytest.raises(IndexError, match="between 0 and 5"):
        wv.get_vectors(np.array([6]))
    with pytest.raises(TypeError, match="integers"):
        wv.get_vectors(np.array([0.5]))
    with pytest.raises(KeyError):
        wv.get_vector("missing")


@pytest.mark.parametrize("dtype", ["float16", "int8"])
def test_quantize(dtype):
    wv = _wv(fasttext=True)
    q = wv.quantize(dtype)
    codes, scales = quantize_rows(np.asarray(wv.vectors), dtype)
    assert q.vectors.tobytes() == codes.tobytes()
    ngram_codes, ngram_scales = quantize_rows(np.asarray(wv.ngrams), dtype)
    assert q.ngrams.tobytes() == ngram_codes.tobytes()
    if dtype == "int8":
        assert q.scales.tobytes() == scales.tobytes()
        assert q.ngram_scales.tobytes() == ngram_scales.tobytes()
    assert q.metadata["fse.quantized_from"] == wv.fingerprint
    assert q.metadata["fse.quantization"] == "quantize_rows@1"
    np.testing.assert_array_equal(q.counts, wv.counts)
    with pytest.raises(ValueError, match="only float32"):
        q.quantize("int8")
    with pytest.raises(Exception, match="float16"):
        wv.quantize("float32")


def test_immutable_and_not_picklable():
    wv = _wv()
    for array in (wv.vectors, wv.vocab_bytes, wv.vocab_offsets, wv.counts):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 1
    with pytest.raises(AttributeError):
        wv.extra = 1
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(wv)
    assert "6 words, 5 dimensions, float32" in repr(wv)
    assert "FastText" in repr(_wv(fasttext=True))


def test_from_arrays_copies():
    rows = np.ones((2, 3), dtype=F32)
    wv = WordVectors.from_arrays(rows, words=["a", "b"])
    rows[0, 0] = 5
    assert wv.vectors[0, 0] == 1
    assert wv.counts is None
    assert wv.fasttext is None
    assert wv.scales is None


def test_from_arrays_with_vocabulary_bytes():
    data = np.frombuffer(b"abc", dtype=np.uint8)
    wv = WordVectors.from_arrays(
        np.ones((2, 2), F32), vocab_bytes=data, vocab_offsets=np.array([0, 1, 3], np.int64)
    )
    assert wv.word(1) == "bc"


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"vectors": [[1.0]]}, TypeError, "numpy array"),
        ({"vectors": np.ones((2, 2))}, ValueError, "float32, float16 or int8"),
        ({"vectors": np.ones(2, F32)}, ValueError, "matrix"),
        ({"vectors": np.ones((2, 0), F32)}, ValueError, "at least one dimension"),
        ({"vectors": np.ones((0, 2), F32), "words": []}, ValueError, "between 1 and"),
        ({"words": None}, ValueError, "either words or"),
        ({"vocab_bytes": np.zeros(1, np.uint8)}, ValueError, "either words or"),
        ({"words": "ab"}, TypeError, "sequence of str"),
        ({"words": ["a", 1]}, TypeError, "word 1 must be a str"),
        ({"words": ["a", "\ud800"]}, ValueError, "word 1 .* has no UTF-8 form"),
        ({"words": ["a", "a"]}, ValueError, "word 1 .*'a'.* is also word 0"),
        ({"words": ["a"]}, ValueError, "vocab.offsets must be a vector of 3"),
        ({"counts": np.array([1, -1])}, ValueError, "count 1 is negative"),
        ({"counts": np.array([1, 1], np.int32)}, ValueError, "not one of the format"),
        ({"scales": np.ones(2, F32)}, ValueError, "scales are given exactly for int8"),
        (
            {"vectors": np.array([[1, -128], [0, 0]], np.int8), "scales": np.ones(2, F32)},
            ValueError,
            "code -128",
        ),
        (
            {"vectors": np.array([[1, 1], [0, 0]], np.int8), "scales": np.array([1, -1], F32)},
            ValueError,
            "scales 1 is negative",
        ),
        (
            {"vectors": np.array([[1, np.inf], [0, 0]], F32)},
            ValueError,
            "row 0 of vectors contains NaN",
        ),
        ({"ngrams": np.ones((3, 2), F32)}, ValueError, "given together"),
        ({"fasttext": (1, 2, 3)}, TypeError, "FastTextParams"),
        (
            {"fasttext": FastTextParams(3, 2, 5), "ngrams": np.ones((5, 2), F32)},
            ValueError,
            "min_n <= max_n",
        ),
        (
            {"fasttext": FastTextParams(1, 2, 0), "ngrams": np.ones((0, 2), F32)},
            ValueError,
            "buckets",
        ),
        (
            {"fasttext": FastTextParams(1, 2, 4), "ngrams": np.ones((3, 2), F32)},
            ValueError,
            r"\(4, 2\) matrix",
        ),
        (
            {"vectors": np.ones((2, 2), F32), "counts": [1, 2]},
            TypeError,
            "counts must be a numpy array",
        ),
    ],
)
def test_from_arrays_rejects(kwargs, error, match):
    args = {"vectors": np.ones((2, 2), F32), "words": ["a", "b"], **kwargs}
    with pytest.raises(error, match=match):
        WordVectors.from_arrays(**args)


def test_verify_checks_the_files(tmp_path):
    _wv().save(tmp_path / "wv.safetensors")
    _flip_last_byte(tmp_path / "wv.safetensors", "vectors")
    wv = WordVectors.load(tmp_path / "wv.safetensors", mmap=True)
    with pytest.raises(ValueError, match="does not match its checksum"):
        wv.verify()
    with pytest.raises(ValueError, match="does not match its checksum"):
        WordVectors.load(tmp_path / "wv.safetensors")


def test_save_never_overwrites(tmp_path):
    wv = _wv()
    wv.save(tmp_path / "wv.safetensors")
    with pytest.raises(FileExistsError):
        wv.save(tmp_path / "wv.safetensors")
    with pytest.raises(Exception, match="path"):
        wv.save(3)


@pytest.mark.usefixtures("windows_file_maps")
def test_released_maps_allow_removal(tmp_path):
    _wv("int8").save(tmp_path / "wv.safetensors")
    wv = WordVectors.load(tmp_path / "wv.safetensors", mmap=True)
    assert wv.index("é") == 1  # the lookup holds no map
    del wv
    gc.collect()
    (tmp_path / "wv.safetensors").unlink()


def _peak(fn):
    gc.collect()
    tracemalloc.start()
    try:
        result = fn()
        return result, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_loading_int8_makes_no_float32_copy(tmp_path):
    """The gate of the board entry: loading codes needs about their bytes, not float32."""
    wv = _wv("int8", n=20_000, dim=300, counts=False)
    wv.save(tmp_path / "wv.safetensors")
    codes = 20_000 * 300
    loaded, peak = _peak(lambda: WordVectors.load(tmp_path / "wv.safetensors"))
    assert loaded.vectors.dtype == np.int8
    float32_copy = 4 * codes
    assert peak < codes + 2 * 2**20, peak  # codes, scales and vocabulary, far below float32
    assert peak < float32_copy / 3
    _, mapped = _peak(lambda: WordVectors.load(tmp_path / "wv.safetensors", mmap=True))
    assert mapped < 2 * 2**20, mapped


def test_quantize_makes_no_float32_copy():
    # Many rows of few values: a float32 copy (4 bytes per value) would exceed the bound
    # of the codes (1 byte per value) plus a few blocks of float32 temporaries.
    rows, dim = 400_000, 16
    words = [str(i) for i in range(rows)]
    wv = WordVectors.from_arrays(
        np.random.default_rng(0).standard_normal((rows, dim)).astype(F32), words=words
    )
    codes = rows * dim
    q, peak = _peak(lambda: wv.quantize("int8"))
    float32_copy = 4 * codes
    block = 16_384 * dim * 4  # bytes of one block of float32 temporaries
    vocabulary = wv.vocab_bytes.nbytes + wv.vocab_offsets.nbytes  # copied into the result
    assert peak < codes + 4 * rows + vocabulary + 10 * block, peak
    assert peak < float32_copy, peak
    assert q.dtype == "int8"


def test_fingerprint_of_unsaved_vectors_equals_the_saved_one(tmp_path):
    wv = _wv("int8", fasttext=True)
    wv.save(tmp_path / "wv.safetensors")
    assert wv.fingerprint == WordVectors.load(tmp_path / "wv.safetensors", mmap=True).fingerprint
    raw = (tmp_path / "wv.safetensors").read_bytes()
    metadata = json.loads(raw[8 : 8 + int.from_bytes(raw[:8], "little")])["__metadata__"]
    assert metadata["fse.fingerprint"] == wv.fingerprint
    assert metadata["fse.ngram_hash"] == "fasttext@1"
    assert metadata["fse.fasttext"] == '{"bucket":7,"max_n":5,"min_n":3}'


def test_returned_arrays_cannot_change_the_vectors():
    wv = _wv("int8", fasttext=True)
    fingerprint = wv.fingerprint
    for name in (
        "vectors",
        "scales",
        "counts",
        "vocab_bytes",
        "vocab_offsets",
        "ngrams",
        "ngram_scales",
    ):
        array = getattr(wv, name)
        with pytest.raises(ValueError, match=r"WRITEABLE|read-only"):
            array.setflags(write=True)
        assert array is not getattr(wv, name)  # a new header each time
    assert wv.vectors.shape == (len(WORDS), 5)
    assert wv.fingerprint == fingerprint
    assert wv.index("é") == 1


def test_int8_scales_that_overflow_are_rejected():
    with pytest.raises(ValueError, match="row 0 of vectors overflows float32 with its scale"):
        WordVectors.from_arrays(
            np.array([[127, 0]], np.int8), words=["a"], scales=np.array([np.finfo(F32).max], F32)
        )


def test_get_vectors_of_an_empty_list():
    assert _wv().get_vectors([]).shape == (0, 5)


def test_quantized_from_must_be_a_fingerprint(tmp_path):
    _wv("int8").save(tmp_path / "wv.safetensors")
    path = _rewrite(
        tmp_path / "wv.safetensors", None, lambda m: m.update({"fse.quantized_from": "yesterday"})
    )
    with pytest.raises(ValueError, match=r"fse\.quantized_from must be the fingerprint"):
        WordVectors.load(path)
    _wv().save(tmp_path / "f32.safetensors")
    fake = fake_fingerprint("0")
    path = _rewrite(
        tmp_path / "f32.safetensors",
        None,
        lambda m: m.update({"fse.quantized_from": fake}),
        "edited32",
    )
    with pytest.raises(ValueError, match="only for float16 and int8"):
        WordVectors.load(path)


def test_save_of_damaged_mapped_vectors_raises(tmp_path):
    _wv().save(tmp_path / "wv.safetensors")
    _flip_last_byte(tmp_path / "wv.safetensors", "vectors")
    wv = WordVectors.load(tmp_path / "wv.safetensors", mmap=True)
    with pytest.raises(ValueError, match="does not match its checksum"):
        wv.save(tmp_path / "copy.safetensors")
    with pytest.raises(ValueError, match="does not match its checksum"):
        wv.quantize("int8")
    assert not (tmp_path / "copy.safetensors").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows does not let a mapped file change")
def test_mapped_data_changed_after_verify_is_never_saved(tmp_path):
    _wv().save(tmp_path / "wv.safetensors")
    wv = WordVectors.load(tmp_path / "wv.safetensors", mmap=True)
    wv.verify()
    _flip_last_byte(tmp_path / "wv.safetensors", "vectors")
    with pytest.raises(ValueError, match="does not match its checksum"):
        wv.save(tmp_path / "copy.safetensors")
    assert not (tmp_path / "copy.safetensors").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows does not let a mapped file change")
def test_quantized_vectors_own_their_arrays(tmp_path):
    _wv().save(tmp_path / "wv.safetensors")
    wv = WordVectors.load(tmp_path / "wv.safetensors", mmap=True)
    q = wv.quantize("int8")
    for name in ("vectors", "scales", "counts", "vocab_bytes", "vocab_offsets"):
        assert not isinstance(getattr(q, name), np.memmap), name
    # Changing the source file afterwards changes nothing in the quantized vectors.
    _overwrite(tmp_path / "wv.safetensors", "vocab.bytes", b"x")
    assert [q.word(i) for i in range(len(q))] == WORDS
    q.verify()
    q.save(tmp_path / "q.safetensors")


@pytest.mark.skipif(sys.platform == "win32", reason="Windows does not let a mapped file change")
def test_quantize_of_changed_mapped_vectors_raises(tmp_path):
    _wv().save(tmp_path / "wv.safetensors")
    wv = WordVectors.load(tmp_path / "wv.safetensors", mmap=True)
    wv.verify()
    _overwrite(tmp_path / "wv.safetensors", "vectors", np.float32(7).tobytes())
    with pytest.raises(
        ValueError, match=r"does not match its checksum|changed since it was loaded"
    ):
        wv.quantize("int8")


def test_word_vectors_are_never_a_directory(tmp_path):
    from fse._container import NpyContainer

    wv = _wv()
    with pytest.raises(ValueError, match=r"must end with \.safetensors"):
        wv.save(tmp_path / "wv")
    tensors, metadata = SafetensorsContainer().open(_saved(wv, tmp_path), mmap=False)
    plain = {
        k: v
        for k, v in metadata.items()
        if k != "fse.fingerprint" and not k.startswith("fse.checksum.")
    }
    NpyContainer().write(tmp_path / "as_npy", tensors, plain)
    with pytest.raises(
        ValueError, match=r"is a directory; word vectors are one \.safetensors file"
    ):
        WordVectors.load(tmp_path / "as_npy")


def test_the_vocabulary_fingerprint_depends_on_the_words_and_their_order_alone(tmp_path):
    words = ["a", "b", "c"]
    wv = WordVectors.from_arrays(np.arange(9, dtype=np.float32).reshape(3, 3), words=words)
    other = WordVectors.from_arrays(np.ones((3, 3), np.float32), words=words)
    assert wv.vocabulary_fingerprint == other.vocabulary_fingerprint
    assert wv.quantize("int8").vocabulary_fingerprint == wv.vocabulary_fingerprint
    reordered = WordVectors.from_arrays(np.ones((3, 3), np.float32), words=words[::-1])
    assert reordered.vocabulary_fingerprint != wv.vocabulary_fingerprint
    assert wv.vocabulary_fingerprint != wv.fingerprint
    wv.save(tmp_path / "wv.safetensors")
    for mmap in (False, True):
        loaded = WordVectors.load(tmp_path / "wv.safetensors", mmap=mmap)
        assert loaded.vocabulary_fingerprint == wv.vocabulary_fingerprint
