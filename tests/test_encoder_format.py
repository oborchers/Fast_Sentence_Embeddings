import numpy as np
import pytest

from fse._contract import Normalize, Project, RemoveComponents, UnsupportedArtifactError
from fse._encoder._format import EncoderContent, from_artifact, to_artifact

F32 = np.float32
FP_A = "sha256:" + "a" * 64
FP_B = "sha256:" + "b" * 64
FP_C = "sha256:" + "c" * 64


def _basis(k, d, seed=0):
    q = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, k)))[0]
    return np.ascontiguousarray(q.T, dtype=F32)


def _content(**changes):
    base = {
        "tokenizer": "fse_words@1",
        "steps": (
            RemoveComponents(_basis(2, 6), np.array([1.0, 0.5], F32)),
            Project(_basis(3, 6, 1)),
            Normalize(),
        ),
        "input_dim": 6,
        "weights": np.linspace(0.1, 1, 10).astype(F32),
        "oov_weight": None,
        "max_ngrams": None,
        "word_vectors": FP_A,
        "vocabulary": FP_B,
        "recipe": {"name": "sif", "version": 1, "alpha": 0.001},
    }
    return EncoderContent(**{**base, **changes})


def _embedded(dim=3):
    tensors = {
        "vectors": np.ones((10, dim), F32),
        "vocab.bytes": np.frombuffer(b"abcdefghij", np.uint8),
        "vocab.offsets": np.arange(11, dtype=np.int64),
    }
    return tensors, {"fse.dim": str(dim), "fse.dtype": "float32"}


def _folded():
    return _content(
        steps=(Normalize(),),
        input_dim=3,
        folded=(RemoveComponents(_basis(1, 6)), Project(_basis(3, 6, 2))),
        folded_from=FP_C,
    )


def _same_steps(got, want):
    assert [s.op for s in got] == [s.op for s in want]
    for a, b in zip(got, want, strict=True):
        for field in ("components", "weights", "basis"):
            x, y = getattr(a, field, None), getattr(b, field, None)
            assert (x is None) == (y is None)
            if x is not None:
                assert x.tobytes() == y.tobytes()


def test_an_unfolded_encoder_round_trips():
    content = _content()
    tensors, metadata = to_artifact(content, None)
    assert sorted(tensors) == ["step.0.components", "step.0.weights", "step.1.basis", "weights"]
    assert metadata["fse.steps"] == '["remove_components@1","project@1","normalize@1"]'
    assert metadata["fse.output_dim"] == "3"
    got, embedded = from_artifact(tensors, metadata)
    assert embedded is None
    _same_steps(got.steps, content.steps)
    assert got.weights.tobytes() == content.weights.tobytes()
    assert (got.input_dim, got.output_dim, got.recipe) == (6, 3, content.recipe)
    assert (got.word_vectors, got.vocabulary, got.folded, got.folded_from) == (FP_A, FP_B, (), None)


def test_a_folded_encoder_round_trips_with_its_word_vectors():
    content = _folded()
    tensors, metadata = to_artifact(content, _embedded())
    assert metadata["fse.folded"] == '["remove_components@1","project@1"]'
    got, (words, keys) = from_artifact(tensors, metadata)
    _same_steps(got.folded, content.folded)
    assert got.folded_from == FP_C
    assert sorted(words) == ["vectors", "vocab.bytes", "vocab.offsets"]
    assert keys == {"fse.dim": "3", "fse.dtype": "float32"}


def test_fasttext_needs_oov_weight_and_max_ngrams():
    content = _content(oov_weight=np.array(0.25, F32), max_ngrams=40)
    tensors, metadata = to_artifact(content, None)
    got, _ = from_artifact(tensors, metadata)
    assert got.oov_weight.tobytes() == np.array(0.25, F32).tobytes()
    assert got.max_ngrams == 40
    del metadata["fse.max_ngrams"]
    with pytest.raises(ValueError, match="exactly for FastText"):
        from_artifact(tensors, metadata)


def test_average_has_no_weights():
    tensors, metadata = to_artifact(_content(weights=None, steps=(Normalize(),)), None)
    assert tensors == {}
    assert from_artifact(tensors, metadata)[0].weights is None


def _artifact():
    return to_artifact(_content(), None)


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
        ("fse.steps", '[ "normalize@1"]', ValueError, "canonical"),
        ("fse.steps", "normalize@1", ValueError, "JSON list"),
        ("fse.output_dim", "6", ValueError, "the steps give 3"),
        ("fse.input_dim", "06", ValueError, "decimal integer"),
        ("fse.word_vectors", "sha256:x", ValueError, "fingerprint"),
        ("fse.recipe", "[]", ValueError, "JSON object"),
        ("fse.vocabulary", None, ValueError, "needs the metadata key fse.vocabulary"),
        ("fse.folded_from", FP_C, ValueError, "only a folded encoder"),
        ("fse.dim", "3", ValueError, "only a folded encoder"),
        ("fse.max_ngrams", "4", ValueError, "exactly for FastText"),
    ],
)
def test_metadata_is_checked(key, value, error, match):
    tensors, metadata = _changed_metadata(key, value)
    with pytest.raises(error, match=match):
        from_artifact(tensors, metadata)


def test_tensors_are_derived_from_the_content():
    tensors, metadata = _artifact()
    with pytest.raises(ValueError, match="does not use"):
        from_artifact({**tensors, "step.2.basis": _basis(3, 3)}, metadata)
    with pytest.raises(ValueError, match=r"needs the tensor step\.1\.basis"):
        from_artifact({k: v for k, v in tensors.items() if k != "step.1.basis"}, metadata)
    with pytest.raises(ValueError, match="only a folded encoder"):
        from_artifact({**tensors, "vectors": np.ones((10, 6), F32)}, metadata)
    with pytest.raises(ValueError, match="weights must be a float32 vector"):
        from_artifact({**tensors, "weights": np.ones(10)}, metadata)
    with pytest.raises(ValueError, match="orthonormal"):
        from_artifact({**tensors, "step.1.basis": np.ones((3, 6), F32)}, metadata)


def test_folded_encoders_are_checked():
    tensors, metadata = to_artifact(_folded(), _embedded())
    with pytest.raises(ValueError, match="keeps only normalize"):
        from_artifact(
            {**tensors, "step.0.components": _basis(1, 3)},
            {**metadata, "fse.steps": '["remove_components@1"]'},
        )
    with pytest.raises(ValueError, match="linear steps"):
        from_artifact(tensors, {**metadata, "fse.folded": '["normalize@1"]'})
    with pytest.raises(ValueError, match="names its original word vectors"):
        from_artifact(tensors, {k: v for k, v in metadata.items() if k != "fse.folded_from"})
    with pytest.raises(ValueError, match="holds its word vectors"):
        from_artifact({k: v for k, v in tensors.items() if k != "vectors"}, metadata)
    bad = {**tensors, "folded.1.basis": _basis(2, 6, 3)}
    with pytest.raises(ValueError, match="give rows of 2 values"):
        from_artifact(bad, metadata)
