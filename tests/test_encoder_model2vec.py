"""Model2Vec models: the vectors of model2vec within float32 rounding, and every refusal.

The models are tiny, written by model2vec 0.10.0 itself (tests/support/model2vec.py), so
nothing is downloaded.
"""

import json

import numpy as np
import pytest

from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._encoder import Encoder
from fse._encoder._model2vec import median_length
from support import model2vec as mv
from support import wordpiece as wp

model2vec = pytest.importorskip("model2vec")
TOKENS = wp.vocabulary()
TEXTS = [*wp.corpus(TOKENS, count=1200), " ".join(TOKENS[5:400]) * 3]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    return {
        (normalize, length): mv.save_model(
            root / f"{normalize}-{length}", normalize=normalize, max_length=length
        )
        for normalize in (True, False)
        for length in (512, 8, None)
    }


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("length", [512, 8, None])
def test_the_vectors_are_those_of_model2vec(models, normalize, length):
    folder = models[(normalize, length)]
    encoder = Encoder.from_pretrained(folder)
    want = model2vec.StaticModel.from_pretrained(folder).encode(TEXTS)
    got = encoder.encode(TEXTS)
    assert mv.assert_close_to_model2vec(got, want, encoder, TEXTS, normalize) <= 1
    assert encoder.steps == (("normalize@1",) if normalize else ())


@pytest.mark.parametrize(
    ("truncate", "max_length", "theirs"), [(True, 5, 5), (True, None, None), (False, "model", None)]
)
def test_another_length_rule(models, truncate, max_length, theirs):
    folder = models[(True, 512)]
    encoder = Encoder.from_pretrained(folder, truncate=truncate, max_length=max_length)
    want = model2vec.StaticModel.from_pretrained(folder).encode(TEXTS, max_length=theirs)
    mv.assert_close_to_model2vec(encoder.encode(TEXTS), want, encoder, TEXTS, True)
    spec = encoder._content.tokenizer
    assert spec.max_tokens == theirs
    median = median_length(TOKENS)
    assert spec.max_chars == (None if theirs is None else theirs * median)
    with pytest.raises(ValueError, match="needs truncate=True"):
        Encoder.from_pretrained(folder, truncate=False, max_length=5)


def test_the_median_is_that_of_numpy():
    for lengths in ([1], [1, 2], [1, 2, 3, 10], [5, 1, 4, 4], [3, 3, 9, 8, 1, 2]):
        tokens = ["x" * n for n in lengths]
        assert median_length(tokens) == int(np.median(lengths))
    with pytest.raises(CorruptArtifactError, match="median length"):
        median_length(["", "", "a"])


def _one_token_model(tmp_path, row):
    tokens = ["[PAD]", "[UNK]", "a", "b"]
    table = np.ones((4, 2), np.float32)
    table[2] = row
    return mv.save_model(tmp_path / "edge", table, tokens=tokens)


def test_a_tiny_row_raises_where_model2vec_scales_it_up(tmp_path):
    """The documented boundary (plan, finding 2): model2vec divides by norm + 1e-32 in
    float32, where the squares of 1e-30 underflow, and gives [100, 0]; normalize@1 raises
    for a nonzero row whose sum of squares underflows."""
    folder = _one_token_model(tmp_path, [1e-30, 0.0])
    want = model2vec.StaticModel.from_pretrained(folder).encode(["a"])
    np.testing.assert_allclose(want, [[100.0, 0.0]], rtol=1e-6)
    with pytest.raises(ValueError, match="underflow"):
        Encoder.from_pretrained(folder).encode(["a"])
    np.testing.assert_allclose(
        Encoder.from_pretrained(folder).encode(["b"]), [[2**-0.5] * 2], rtol=1e-6
    )


def test_a_huge_row_raises_where_model2vec_gives_no_unit_row(tmp_path):
    folder = _one_token_model(tmp_path, [3e19, 0.0])
    with np.errstate(over="ignore", invalid="ignore"):  # model2vec's norm overflows
        want = model2vec.StaticModel.from_pretrained(folder).encode(["a"])
    assert not np.isclose(np.linalg.norm(want.astype(np.float64)), 1.0)
    with pytest.raises(ValueError, match="overflow"):
        Encoder.from_pretrained(folder).encode(["a"])


def test_a_float16_model_is_refused(tmp_path):
    folder = mv.save_model(tmp_path / "f32")
    half = model2vec.StaticModel.from_pretrained(folder, quantize_to="float16")
    half.save_pretrained(tmp_path / "f16")
    with pytest.raises(UnsupportedArtifactError, match="float16 table"):
        Encoder.from_pretrained(tmp_path / "f16")
    int8 = model2vec.StaticModel.from_pretrained(folder, quantize_to="int8")
    int8.save_pretrained(tmp_path / "i8")
    with pytest.raises(UnsupportedArtifactError, match="int8 table"):
        Encoder.from_pretrained(tmp_path / "i8")


def test_a_vocabulary_quantized_model_is_refused(tmp_path):
    """As model2vec writes one: fewer rows, a mapping and weights, and the config key."""
    from safetensors.numpy import save_file

    folder = mv.save_model(tmp_path / "vq")
    tensors = {
        "embeddings": mv.rows(8),
        "mapping": np.arange(len(TOKENS), dtype=np.int64) % 8,
        "weights": np.ones(len(TOKENS), np.float32),
    }
    save_file(tensors, folder / "model.safetensors")
    with pytest.raises(UnsupportedArtifactError, match="vocabulary quantization"):
        Encoder.from_pretrained(folder)
    _edit_config(folder, vocabulary_quantization=8)
    with pytest.raises(UnsupportedArtifactError, match="vocabulary quantization"):
        Encoder.from_pretrained(folder)


def _edit_config(folder, **changes):
    path = folder / "config.json"
    config = json.loads(path.read_text())
    for key, value in changes.items():
        if value is KeyError:
            config.pop(key, None)
        else:
            config[key] = value
    path.write_text(json.dumps(config))
    return folder


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    [
        ({"head_config": {}}, UnsupportedArtifactError, "a classification head"),
        (
            {"pooling_mode": "max"},
            UnsupportedArtifactError,
            "keys this fse does not know: \\['pooling_mode'\\]",
        ),
        ({"normalize": 1}, CorruptArtifactError, "normalize must be true or false"),
        ({"max_length": 0}, CorruptArtifactError, "max_length must be a positive int"),
        ({"max_length": True}, CorruptArtifactError, "max_length must be a positive int"),
        ({"hidden_dim": 3}, CorruptArtifactError, "hidden_dim is 3"),
        ({"embedding_dtype": "float16"}, CorruptArtifactError, "embedding_dtype is 'float16'"),
    ],
)
def test_the_config_is_checked(tmp_path, changes, error, match):
    folder = _edit_config(mv.save_model(tmp_path / "m"), **changes)
    with pytest.raises(error, match=match):
        Encoder.from_pretrained(folder)


def test_provenance_keys_are_accepted_and_the_defaults_hold(tmp_path):
    folder = mv.save_model(tmp_path / "m", normalize=True)
    provenance = {"model_type": "model2vec", "architectures": ["StaticModel"], "apply_pca": 16}
    provenance |= {"apply_zipf": True, "tokenizer_name": "x", "seq_length": 10**6}
    _edit_config(folder, normalize=KeyError, max_length=KeyError, **provenance)
    encoder = Encoder.from_pretrained(folder)
    assert encoder.steps == ()  # normalize defaults to false
    assert encoder._content.tokenizer.max_tokens == 512  # max_length defaults to 512


def test_the_table_is_checked(tmp_path):
    from safetensors.numpy import save_file

    folder = mv.save_model(tmp_path / "m")
    files = {name: folder / name for name in ("config.json", "model.safetensors", "tokenizer.json")}
    table = mv.rows(len(TOKENS))
    cases = [
        ({"embeddings": table[:-1]}, CorruptArtifactError, "rows, but the tokenizer has"),
        (
            {"embeddings": table, "extra": table},
            CorruptArtifactError,
            "exactly the tensor embeddings",
        ),
        (
            {"embeddings": table, "weights": table[:, 0]},
            UnsupportedArtifactError,
            "vocabulary quantization",
        ),
        ({"embeddings": table[0]}, CorruptArtifactError, "must be a matrix"),
    ]
    for tensors, error, match in cases:
        save_file(tensors, files["model.safetensors"])
        with pytest.raises(error, match=match):
            Encoder.from_pretrained(folder)
