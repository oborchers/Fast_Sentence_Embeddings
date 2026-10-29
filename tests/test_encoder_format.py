import runpy
from pathlib import Path

import numpy as np
import pytest

from fse._contract import Normalize, UnsupportedArtifactError
from fse._encoder import Encoder
from fse._encoder._format import EncoderContent, from_artifact, to_artifact
from fse._wordvectors import WordVectors
from support.wordvectors import _rewrite

F32 = np.float32
GOLDEN = Path(__file__).parent / "test_data" / "v2"


def _table(dim=3):
    tensors = {
        "vectors": np.ones((10, dim), F32),
        "vocab.bytes": np.frombuffer(b"abcdefghij", np.uint8),
        "vocab.offsets": np.arange(11, dtype=np.int64),
    }
    return tensors, {"fse.dim": str(dim), "fse.dtype": "float32"}


def _content(**changes):
    base = {
        "tokenizer": "fse_words@1",
        "steps": (Normalize(),),
        "oov_weight": None,
        "max_ngrams": None,
    }
    return EncoderContent(**{**base, **changes})


def _artifact(**changes):
    return to_artifact(_content(**changes), *_table())


def test_an_encoder_round_trips_with_its_table():
    tensors, metadata = _artifact()
    assert sorted(tensors) == ["vectors", "vocab.bytes", "vocab.offsets"]
    assert metadata == {
        "fse.kind": "encoder",
        "fse.format_version": "1",
        "fse.transform_version": "2",
        "fse.tokenizer": "fse_words@1",
        "fse.aggregation": "weighted_mean@1",
        "fse.steps": '["normalize@1"]',
        "fse.output_dim": "3",
        "fse.dim": "3",
        "fse.dtype": "float32",
    }
    content, table, keys = from_artifact(tensors, metadata)
    assert content.steps == (Normalize(),)
    assert (content.tokenizer, content.oov_weight, content.max_ngrams) == (
        "fse_words@1",
        None,
        None,
    )
    assert sorted(table) == sorted(tensors)
    assert keys == {"fse.dim": "3", "fse.dtype": "float32"}
    tensors, metadata = _artifact(steps=())
    assert metadata["fse.steps"] == "[]"
    assert from_artifact(tensors, metadata)[0].steps == ()


def test_fasttext_needs_oov_weight_and_max_ngrams():
    tensors, metadata = _artifact(oov_weight=np.array(0.25, F32), max_ngrams=40)
    content, _, _ = from_artifact(tensors, metadata)
    assert content.oov_weight.tobytes() == np.array(0.25, F32).tobytes()
    assert content.max_ngrams == 40
    del metadata["fse.max_ngrams"]
    with pytest.raises(ValueError, match="exactly for FastText"):
        from_artifact(tensors, metadata)
    with pytest.raises(ValueError, match="one finite float32 value"):
        from_artifact(
            {**tensors, "oov_weight": np.array(0.25)}, {**metadata, "fse.max_ngrams": "4"}
        )


def _changed_metadata(key, value):
    tensors, metadata = _artifact()
    if value is None:
        del metadata[key]
    else:
        metadata[key] = value
    return tensors, metadata


@pytest.mark.parametrize(
    ("key", "value", "error", "match"),
    [
        ("fse.kind", "word_vectors", UnsupportedArtifactError, "not 'encoder'"),
        ("fse.format_version", "2", UnsupportedArtifactError, "format version '2'"),
        ("fse.color", "blue", UnsupportedArtifactError, "unknown metadata keys"),
        ("fse.transform_version", "1", UnsupportedArtifactError, "transform_version 1"),
        ("fse.tokenizer", "bert@1", UnsupportedArtifactError, "tokenizer"),
        ("fse.aggregation", "max@1", UnsupportedArtifactError, "max"),
        ("fse.steps", '["normalize@2"]', UnsupportedArtifactError, "normalize"),
        ("fse.steps", '["project@1"]', UnsupportedArtifactError, "project@1 is not supported"),
        ("fse.steps", '["normalize@1","normalize@1"]', ValueError, "none or normalize once"),
        ("fse.steps", '[ "normalize@1"]', ValueError, "canonical"),
        ("fse.steps", "normalize@1", ValueError, "JSON list"),
        ("fse.output_dim", "6", ValueError, "aggregate to 3"),
        ("fse.dim", "03", ValueError, "decimal integer"),
        ("fse.dim", "0", ValueError, "at least 1"),
        ("fse.output_dim", None, ValueError, "needs the metadata key fse.output_dim"),
        ("fse.max_ngrams", "4", ValueError, "exactly for FastText"),
    ],
)
def test_metadata_is_checked(key, value, error, match):
    tensors, metadata = _changed_metadata(key, value)
    with pytest.raises(error, match=match):
        from_artifact(tensors, metadata)


@pytest.mark.parametrize(
    "key",
    [
        "fse.recipe",
        "fse.folded",
        "fse.folded_from",
        "fse.input_dim",
        "fse.word_vectors",
        "fse.vocabulary",
    ],
)
def test_the_keys_of_a_pre_release_are_refused_by_name(key):
    tensors, metadata = _changed_metadata(key, "x")
    with pytest.raises(
        UnsupportedArtifactError, match=rf"pre-release of 2.0 \(it holds {key}, .*rebuild"
    ):
        from_artifact(tensors, metadata)


@pytest.mark.parametrize(
    "name", ["weights", "step.0.components", "step.1.basis", "folded.0.weights"]
)
def test_the_tensors_of_a_pre_release_are_refused_by_name(name):
    tensors, metadata = _artifact()
    with pytest.raises(UnsupportedArtifactError, match=rf"pre-release of 2.0 \(it holds {name}, "):
        from_artifact({**tensors, name: np.ones(3, F32)}, metadata)


def test_an_encoder_without_its_table_is_a_pre_release():
    tensors, metadata = _artifact()
    with pytest.raises(UnsupportedArtifactError, match=r"pre-release of 2\.0 .*no table of its own"):
        from_artifact({}, metadata)
    with pytest.raises(ValueError, match="does not use"):
        from_artifact({**tensors, "row_ids": np.ones(3, np.int64)}, metadata)


def test_the_table_is_checked_as_word_vectors(tmp_path):
    Encoder.from_arrays(np.ones((2, 3), F32), words=["a", "b"]).save(tmp_path / "e.safetensors")
    path = _rewrite(
        tmp_path / "e.safetensors", edit_metadata=lambda m: m.update({"fse.dtype": "int8"})
    )
    with pytest.raises(ValueError, match="vectors must be a matrix of int8"):
        Encoder.load(path)


GOLDEN_ENCODERS = {
    "encoder_float32": (
        "xxh3-128:363c1e166b680245adb9fbb9df929d96",
        [
            "2ef9643f000000002ef9e43e",
            "2ef9e4be2ef9643f00000000",
            "0" * 24,
            "2ef9643f000000002ef9e43e",
        ],
    ),
    "encoder_int8": (
        "xxh3-128:900de12daf5acd9cd72ee4368740e1c4",
        [
            "0281c03f0000c0bf0e87c33e",
            "b0022cbfaeaa2a40b002acbe",
            "0" * 24,
            "bd022cbb0100803fabaaaa3e",
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(GOLDEN_ENCODERS))
@pytest.mark.parametrize("mmap", [False, True])
def test_the_golden_encoders_load_and_give_their_pinned_rows(name, mmap):
    """encoder 1 is fixed: these files load, keep their fingerprint and give the same bits
    on every platform (the encoding is contract arithmetic)."""
    golden = runpy.run_path(str(GOLDEN / "make_golden.py"))
    fingerprint, rows = GOLDEN_ENCODERS[name]
    encoder = Encoder.load(GOLDEN / "finished" / f"{name}.safetensors", mmap=mmap)
    assert encoder.fingerprint == fingerprint
    got = encoder.encode(golden["SENTENCES"], workers=1)
    assert [row.tobytes().hex() for row in got] == rows
    encoder.verify()
    wv = golden["word_vectors"]()
    made = golden["encoders"](wv, wv.quantize("int8"))[name]
    assert made.fingerprint == fingerprint  # the writer still gives the same file


@pytest.mark.parametrize("name", ["encoder_sif_unfolded", "encoder_usif_folded_int8"])
@pytest.mark.parametrize("mmap", [False, True])
def test_the_golden_encoders_of_a_pre_release_are_rejected(name, mmap):
    """The encoders of a pre-release (one referencing its word vectors, one folded), kept
    as they were written: refused by what they hold, or by their sha256 checksums."""
    with pytest.raises(UnsupportedArtifactError, match=r"pre-release of 2.0 .*fse.recipe.*rebuild"):
        Encoder.load(GOLDEN / f"{name}.safetensors", mmap=mmap)
    with pytest.raises(UnsupportedArtifactError, match=r"sha256 .* reads xxh3-128"):
        Encoder.load(GOLDEN / "sha256" / f"{name}.safetensors", mmap=mmap)


def test_word_vectors_are_no_encoder():
    with pytest.raises(UnsupportedArtifactError, match="is a 'word_vectors', not 'encoder'"):
        Encoder.load(GOLDEN / "finished" / "word_vectors_float32.safetensors")
    with pytest.raises(UnsupportedArtifactError, match="is a 'encoder', not 'word_vectors'"):
        WordVectors.load(GOLDEN / "finished" / "encoder_float32.safetensors")
