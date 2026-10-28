"""The format rules of word_vectors 1: broken artifacts, values, golden files, UTF-8."""

import numpy as np
import pytest

from fse._container import SafetensorsContainer
from fse._contract import UnsupportedArtifactError
from fse._reference import quantize_rows
from fse._wordvectors import WordVectors
from support.wordvectors import F32, _rewrite, _wv

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
    _wv("int8", fasttext=True).save(tmp_path / "wv.safetensors")
    path = _rewrite(tmp_path / "wv.safetensors", edit_tensors, edit_metadata)
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
    wv.save(tmp_path / "wv.safetensors")
    path = _rewrite(tmp_path / "wv.safetensors", edit)
    loaded = WordVectors.load(path)  # values are checked by verify, not by every open
    with pytest.raises(ValueError, match=match or "is also word"):
        loaded.verify()


GOLDEN = {
    "float32": "sha256:2700466c5f0e3b1c47624e9e52a4b5064e911e7fc1827e48070345d90cab89f1",
    "int8": "sha256:efce97495ce0ec596486aa46091b4afa5f27dc0ec2aa039da48dcd6616b68f37",
}


@pytest.mark.parametrize("dtype", sorted(GOLDEN))
@pytest.mark.parametrize("mmap", [False, True])
def test_golden_word_vectors(test_data_dir, dtype, mmap):
    """The committed artifacts of tests/test_data/v2: the format must keep reading them."""
    import runpy

    # runpy writes no bytecode, so tests/test_data stays untouched.
    a = runpy.run_path(str(test_data_dir / "v2" / "make_golden.py"))["arrays"]()
    wv = WordVectors.load(test_data_dir / "v2" / f"word_vectors_{dtype}.safetensors", mmap=mmap)
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
    rebuilt = WordVectors.from_arrays(
        a["vectors"],
        words=a["words"],
        counts=a["counts"],
        fasttext=a["fasttext"],
        ngrams=a["ngrams"],
    )
    rebuilt = rebuilt if dtype == "float32" else rebuilt.quantize(dtype)
    assert rebuilt.fingerprint == GOLDEN[dtype]


@pytest.mark.parametrize("dtype", sorted(GOLDEN))
def test_every_container_gives_the_golden_fingerprint(tmp_path, test_data_dir, dtype):
    """The same tensors and metadata written through every container: same content, same
    fingerprint, as the fingerprint describes the content, not the container.
    """
    from fse._container import CONTAINERS

    golden = test_data_dir / "v2" / f"word_vectors_{dtype}.safetensors"
    tensors, metadata = SafetensorsContainer().open(golden, mmap=False)
    plain = {
        k: v
        for k, v in metadata.items()
        if k != "fse.fingerprint" and not k.startswith("fse.checksum.")
    }
    assert len(CONTAINERS) == 2
    for name, container in CONTAINERS.items():
        path = tmp_path / f"{name}{container.suffix}"
        container.write(path, tensors, plain)
        for mmap in (False, True):
            opened, written = container.open(path, mmap=mmap)
            assert written["fse.fingerprint"] == GOLDEN[dtype], name
            for key, array in tensors.items():
                assert opened[key].tobytes() == array.tobytes(), (name, key)


@pytest.mark.parametrize("chunk", [1, 2, 7, 64])
def test_utf8_check_in_chunks(chunk):
    from fse._wordvectors_format import _check_utf8, _encode_words

    words = ["é" * 5, "", "a", "日本" * 3, "", ""] * 50
    data, offsets = _encode_words(words)
    _check_utf8(data, offsets, chunk_bytes=chunk)
    bad = data.copy()
    bad[len(bad) // 2] = 0xFF
    with pytest.raises(ValueError, match="UTF-8"):
        _check_utf8(bad, offsets, chunk_bytes=chunk)
