import gc
import json
import pickle
import tracemalloc

import numpy as np
import pytest

from fse._container import NpyContainer
from fse._contract import UnsupportedArtifactError
from fse._reference import dequantize, quantize_rows
from fse._wordvectors import FastTextParams, WordVectors

F32 = np.float32
WORDS = ["the", "é", "é", "", "naïve", "日本"]


def _wv(dtype="float32", fasttext=False, counts=True, n=None, dim=5, seed=0):
    rng = np.random.default_rng(seed)
    words = WORDS if n is None else [f"w{i}" for i in range(n)]
    rows = rng.standard_normal((len(words), dim)).astype(F32)
    kwargs = {"counts": np.arange(len(words), dtype=np.int64)} if counts else {}
    if fasttext:
        kwargs["fasttext"] = FastTextParams(3, 5, 7)
        kwargs["ngrams"] = rng.standard_normal((7, dim)).astype(F32)
    wv = WordVectors.from_arrays(rows, words=words, **kwargs)
    return wv if dtype == "float32" else wv.quantize(dtype)


def _same(a: WordVectors, b: WordVectors):
    for name in (
        "vectors",
        "scales",
        "counts",
        "vocab_bytes",
        "vocab_offsets",
        "ngrams",
        "ngram_scales",
    ):
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert x.dtype == y.dtype, name
            assert x.shape == y.shape, name
            assert np.asarray(x).tobytes() == np.asarray(y).tobytes(), name
    assert a.fasttext == b.fasttext
    assert a.fingerprint == b.fingerprint


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("fasttext", [False, True])
@pytest.mark.parametrize("mmap", [False, True])
def test_round_trip(tmp_path, dtype, fasttext, mmap):
    wv = _wv(dtype, fasttext)
    wv.save(tmp_path / "wv")
    loaded = WordVectors.load(tmp_path / "wv", mmap=mmap)
    _same(wv, loaded)
    assert loaded.path == tmp_path / "wv"
    assert loaded.dtype == dtype
    assert isinstance(loaded.vectors, np.memmap) == mmap
    loaded.verify()
    assert loaded.metadata["fse.created_by"].startswith("fse")
    # Saving again gives the same content and fingerprint.
    loaded.save(tmp_path / "again")
    _same(loaded, WordVectors.load(tmp_path / "again"))


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


def _rewrite(path, edit_tensors=None, edit_metadata=None):
    """Rewrite a saved artifact with edited tensors or metadata (fresh checksums)."""
    tensors, metadata = NpyContainer().open(path, mmap=False)
    tensors = {k: np.array(v) for k, v in tensors.items()}
    metadata = {
        k: v
        for k, v in metadata.items()
        if k != "fse.fingerprint" and not k.startswith("fse.checksum.")
    }
    if edit_tensors:
        edit_tensors(tensors)
    if edit_metadata:
        edit_metadata(metadata)
    target = path.parent / "edited"
    NpyContainer().write(target, tensors, metadata)
    return target


SCHEMA_ERRORS = {
    "kind": (
        None,
        lambda m: m.update({"fse.kind": "encoder"}),
        UnsupportedArtifactError,
        "not 'word_vectors'",
    ),
    "format version": (
        None,
        lambda m: m.update({"fse.format_version": "2"}),
        UnsupportedArtifactError,
        "format version",
    ),
    "unknown key": (
        None,
        lambda m: m.update({"x.note": "1"}),
        UnsupportedArtifactError,
        "unknown metadata keys",
    ),
    "unknown tensor": (
        lambda t: t.update(extra=np.zeros(1, F32)),
        None,
        UnsupportedArtifactError,
        "unknown tensors",
    ),
    "missing tensor": (
        lambda t: t.pop("vocab.offsets"),
        None,
        ValueError,
        "need the tensor vocab.offsets",
    ),
    "dim plus": (None, lambda m: m.update({"fse.dim": "+5"}), ValueError, "decimal integer"),
    "dim leading zero": (
        None,
        lambda m: m.update({"fse.dim": "05"}),
        ValueError,
        "decimal integer",
    ),
    "dim float": (None, lambda m: m.update({"fse.dim": "5.0"}), ValueError, "decimal integer"),
    "dim wrong": (None, lambda m: m.update({"fse.dim": "4"}), ValueError, "fse.dim is 4"),
    "dtype unknown": (
        None,
        lambda m: m.update({"fse.dtype": "int4"}),
        UnsupportedArtifactError,
        "fse.dtype",
    ),
    "dtype mismatch": (
        None,
        lambda m: m.update({"fse.dtype": "float16", "fse.quantization": "quantize_rows@1"}),
        ValueError,
        "matrix of float16",
    ),
    "quantization missing": (
        None,
        lambda m: m.pop("fse.quantization"),
        ValueError,
        "fse.quantization",
    ),
    "quantization unknown": (
        None,
        lambda m: m.update({"fse.quantization": "absmax@2"}),
        UnsupportedArtifactError,
        "quantization",
    ),
    "scales missing": (lambda t: t.pop("scales"), None, ValueError, "scales are given exactly"),
    "scales shape": (
        lambda t: t.update(scales=np.ones(2, F32)),
        None,
        ValueError,
        "scales must be a vector",
    ),
    "offsets start": (
        lambda t: t["vocab.offsets"].__setitem__(0, 1),
        None,
        ValueError,
        "rise from 0",
    ),
    "offsets decrease": (
        lambda t: t["vocab.offsets"].__setitem__(2, 0),
        None,
        ValueError,
        "rise from 0",
    ),
    "offsets end": (
        lambda t: t["vocab.offsets"].__setitem__(-1, 99),
        None,
        ValueError,
        "rise from 0",
    ),
    "invalid utf-8": (
        lambda t: t["vocab.bytes"].__setitem__(0, 0xFF),
        None,
        ValueError,
        "not valid UTF-8",
    ),
    "inside a character": (
        lambda t: t["vocab.offsets"].__setitem__(2, 4),
        None,
        ValueError,
        "inside a UTF-8 character",
    ),
    "fasttext not json": (None, lambda m: m.update({"fse.fasttext": "{"}), ValueError, "not JSON"),
    "fasttext keys": (
        None,
        lambda m: m.update({"fse.fasttext": '{"bucket":7}'}),
        ValueError,
        "exactly bucket",
    ),
    "fasttext floats": (
        None,
        lambda m: m.update({"fse.fasttext": '{"bucket":7,"max_n":5,"min_n":3.0}'}),
        ValueError,
        "integers",
    ),
    "fasttext not canonical": (
        None,
        lambda m: m.update({"fse.fasttext": '{"min_n":3,"max_n":5,"bucket":7}'}),
        ValueError,
        "canonical",
    ),
    "ngram hash": (
        None,
        lambda m: m.update({"fse.ngram_hash": "murmur@1"}),
        UnsupportedArtifactError,
        "ngram hash",
    ),
    "ngrams without params": (
        None,
        lambda m: [m.pop("fse.fasttext"), m.pop("fse.ngram_hash")],
        ValueError,
        "given together",
    ),
    "ngram scales missing": (
        lambda t: t.pop("ngram_scales"),
        None,
        ValueError,
        "ngram_scales are given exactly",
    ),
    "ngram scales alone": (
        lambda t: [t.pop("ngrams")],
        lambda m: [m.pop("fse.fasttext"), m.pop("fse.ngram_hash")],
        ValueError,
        "ngram_scales without ngrams",
    ),
}


@pytest.mark.parametrize("case", sorted(SCHEMA_ERRORS))
def test_load_rejects_broken_word_vectors(tmp_path, case):
    edit_tensors, edit_metadata, error, match = SCHEMA_ERRORS[case]
    _wv("int8", fasttext=True).save(tmp_path / "wv")
    path = _rewrite(tmp_path / "wv", edit_tensors, edit_metadata)
    for mmap in (False, True):
        with pytest.raises(error, match=match):
            WordVectors.load(path, mmap=mmap)


VALUE_ERRORS = {
    "code -128": (lambda t: t["vectors"].__setitem__((0, 0), -128), "code -128"),
    "nan scale": (
        lambda t: t["scales"].__setitem__(1, np.nan),
        "scales 1 is negative or not finite",
    ),
    "negative count": (lambda t: t["counts"].__setitem__(2, -1), "count 2 is negative"),
    "duplicate word": (
        lambda t: t["vocab.bytes"].__setitem__(slice(0, 3), np.frombuffer(b"tea", np.uint8)),
        None,
    ),
}


@pytest.mark.parametrize("case", sorted(VALUE_ERRORS))
def test_verify_checks_values(tmp_path, case):
    edit, match = VALUE_ERRORS[case]
    wv = WordVectors.from_arrays(
        np.ones((3, 2), np.int8),
        words=["the", "tea", "x"],
        scales=np.ones(3, F32),
        counts=np.ones(3, np.int64),
    )
    wv.save(tmp_path / "wv")
    path = _rewrite(tmp_path / "wv", edit)
    loaded = WordVectors.load(path)  # values are checked by verify, not by every open
    with pytest.raises(ValueError, match=match or "is also word"):
        loaded.verify()


def test_verify_checks_the_files(tmp_path):
    _wv().save(tmp_path / "wv")
    file = tmp_path / "wv" / "vectors.npy"
    raw = bytearray(file.read_bytes())
    raw[-1] ^= 1
    file.write_bytes(bytes(raw))
    wv = WordVectors.load(tmp_path / "wv", mmap=True)
    with pytest.raises(ValueError, match="does not match its checksum"):
        wv.verify()
    with pytest.raises(ValueError, match="does not match its checksum"):
        WordVectors.load(tmp_path / "wv")


def test_save_never_overwrites(tmp_path):
    wv = _wv()
    wv.save(tmp_path / "wv")
    with pytest.raises(FileExistsError):
        wv.save(tmp_path / "wv")
    with pytest.raises(Exception, match="path"):
        wv.save(3)


def test_released_maps_allow_removal(tmp_path, windows_file_maps):
    import shutil

    _wv("int8").save(tmp_path / "wv")
    wv = WordVectors.load(tmp_path / "wv", mmap=True)
    assert wv.index("é") == 1  # the lookup holds no map
    del wv
    gc.collect()
    shutil.rmtree(tmp_path / "wv")


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
    wv.save(tmp_path / "wv")
    codes = 20_000 * 300
    loaded, peak = _peak(lambda: WordVectors.load(tmp_path / "wv"))
    assert loaded.vectors.dtype == np.int8
    float32_copy = 4 * codes
    assert peak < codes + 2 * 2**20, peak  # codes, scales and vocabulary, far below float32
    assert peak < float32_copy / 3
    _, mapped = _peak(lambda: WordVectors.load(tmp_path / "wv", mmap=True))
    assert mapped < 2 * 2**20, mapped


def test_quantize_makes_no_float32_copy():
    wv = _wv(n=20_000, dim=300, counts=False)
    codes = 20_000 * 300
    q, peak = _peak(lambda: wv.quantize("int8"))
    # The codes plus one block of float32 temporaries.
    assert peak < codes + 4 * 16_384 * 300 * 3, peak
    assert q.dtype == "int8"


def test_fingerprint_of_unsaved_vectors_equals_the_saved_one(tmp_path):
    wv = _wv("int8", fasttext=True)
    wv.save(tmp_path / "wv")
    assert wv.fingerprint == WordVectors.load(tmp_path / "wv", mmap=True).fingerprint
    manifest = json.loads((tmp_path / "wv" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["fse.fingerprint"] == wv.fingerprint
    assert manifest["metadata"]["fse.ngram_hash"] == "fasttext@1"
    assert manifest["metadata"]["fse.fasttext"] == '{"bucket":7,"max_n":5,"min_n":3}'


GOLDEN = {
    "float32": "sha256:2700466c5f0e3b1c47624e9e52a4b5064e911e7fc1827e48070345d90cab89f1",
    "int8": "sha256:efce97495ce0ec596486aa46091b4afa5f27dc0ec2aa039da48dcd6616b68f37",
}


@pytest.mark.parametrize("dtype", sorted(GOLDEN))
@pytest.mark.parametrize("mmap", [False, True])
def test_golden_word_vectors(test_data_dir, dtype, mmap):
    """The committed artifacts of tests/test_data/v2: the format must keep reading them."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_golden", test_data_dir / "v2" / "make_golden.py"
    )
    make_golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_golden)
    a = make_golden.arrays()
    wv = WordVectors.load(test_data_dir / "v2" / f"word_vectors_{dtype}", mmap=mmap)
    wv.verify()
    assert wv.fingerprint == GOLDEN[dtype]
    assert [wv.word(i) for i in range(len(wv))] == a["words"]
    np.testing.assert_array_equal(wv.counts, a["counts"])
    assert wv.fasttext == a["fasttext"]
    if dtype == "float32":
        assert wv.vectors.tobytes() == a["vectors"].tobytes()
        assert wv.ngrams.tobytes() == a["ngrams"].tobytes()
    else:
        codes, scales = quantize_rows(a["vectors"], dtype)
        assert wv.vectors.tobytes() == codes.tobytes()
        assert wv.scales.tobytes() == scales.tobytes()
        assert wv.metadata["fse.quantized_from"] == GOLDEN["float32"]
    # Writing the same content again, through every container, gives the same fingerprint.
    from fse._container import CONTAINERS

    for container in CONTAINERS.values():
        assert container.name == "npy"
    rebuilt = WordVectors.from_arrays(
        a["vectors"],
        words=a["words"],
        counts=a["counts"],
        fasttext=a["fasttext"],
        ngrams=a["ngrams"],
    )
    rebuilt = rebuilt if dtype == "float32" else rebuilt.quantize(dtype)
    assert rebuilt.fingerprint == GOLDEN[dtype]


@pytest.mark.parametrize("chunk", [1, 2, 7, 64])
def test_utf8_check_in_chunks(chunk):
    from fse._wordvectors import _check_utf8, _encode_words

    words = ["é" * 5, "", "a", "日本" * 3, "", ""] * 50
    data, offsets = _encode_words(words)
    _check_utf8(data, offsets, chunk_bytes=chunk)
    bad = data.copy()
    bad[len(bad) // 2] = 0xFF
    with pytest.raises(ValueError, match="UTF-8"):
        _check_utf8(bad, offsets, chunk_bytes=chunk)
