"""The API surface of fse.StaticModel against model2vec 0.10.0: names, order, kinds and
defaults of the signatures, every public name provided or listed with its reason (so an
upgrade of the pin fails loudly), and every refusal."""

import importlib.util
import inspect
import pickle

import numpy as np
import pytest
from pydantic import ValidationError

import fse
from fse._contract import UnsupportedArtifactError
from fse._encoder import Encoder
from fse._encoder._model2vec_rules import UNSET
from support import static_model as sm

model2vec = pytest.importorskip("model2vec")
pytest.importorskip("safetensors")
# What model2vec has and this PR does not provide, with the reason.
NOT_PROVIDED = {
    "quantize_model": "out of scope: fse.StaticModel has float32 tables and no vocabulary clusters",
}
# The submodules of model2vec for other jobs than encoding with a finished model.
OTHER_MODULES = {
    "distill": "out of scope: fse trains and distills nothing",
    "train": "out of scope: fse trains nothing",
    "inference": "out of scope: fse.Classifier and fse.Regressor are its heads",
    "onnx": "out of scope: fse has no ONNX export",
}
# model2vec's **kwargs of encode, ignored there: only these names are accepted here.
NEUTRAL = ["convert_to_numpy", "convert_to_tensor", "device"]
# What fse adds to from_pretrained, after the parameters of model2vec, keyword-only.
EXTRAS = ["revision", "cache_dir", "local_files_only"]


def _parameters(function):
    return [
        (p.name, p.kind, p.default)
        for p in inspect.signature(function).parameters.values()
        if p.name != "self"
    ]


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return sm.model_folder(tmp_path_factory.mktemp("m") / "m")


@pytest.fixture(scope="module")
def model(folder):
    return fse.StaticModel.from_pretrained(folder)


def test_from_pretrained_has_the_parameters_of_model2vec_then_the_extras():
    ours = _parameters(fse.StaticModel.from_pretrained)
    theirs = _parameters(model2vec.StaticModel.from_pretrained)
    assert ours[: len(theirs)] == [
        (name, kind, UNSET if name == "max_length" else default) for name, kind, default in theirs
    ]
    extras = ours[len(theirs) :]
    assert [name for name, _, _ in extras] == EXTRAS
    assert {kind for _, kind, _ in extras} == {inspect.Parameter.KEYWORD_ONLY}
    assert [default for _, _, default in extras] == [None, None, False]


def test_encode_has_the_parameters_of_model2vec_and_a_closed_list_for_its_kwargs():
    ours = _parameters(fse.StaticModel.encode)
    theirs = _parameters(model2vec.StaticModel.encode)
    assert theirs[-1][1] is inspect.Parameter.VAR_KEYWORD
    named = theirs[:-1]
    assert [(n, k) for n, k, _ in ours[: len(named)]] == [(n, k) for n, k, _ in named]
    defaults = {n: (UNSET if n == "max_length" else d) for n, _, d in named if n != "sentences"}
    assert {n: d for n, _, d in ours if n in defaults} == defaults
    assert [n for n, _, _ in ours[len(named) :]] == NEUTRAL
    assert inspect.Parameter.VAR_KEYWORD not in {kind for _, kind, _ in ours}


def test_save_pretrained_and_push_to_hub_have_the_parameters_of_model2vec():
    for name in ("save_pretrained", "push_to_hub"):
        ours = _parameters(getattr(fse.StaticModel, name))
        theirs = _parameters(getattr(model2vec.StaticModel, name))
        assert ours == theirs, name
    assert [n for n, _, _ in _parameters(fse.StaticModel.push_to_hub)] == [
        "repo_id",
        "private",
        "token",
        "subfolder",
    ]


def test_encode_as_sequence_and_tokenize_have_the_parameters_of_model2vec():
    for name in ("encode_as_sequence", "tokenize"):
        ours = _parameters(getattr(fse.StaticModel, name))
        theirs = _parameters(getattr(model2vec.StaticModel, name))
        assert ours == theirs, name
    assert [n for n, _, _ in _parameters(fse.StaticModel.encode_as_sequence)][1:] == [
        "max_length",
        "batch_size",
        "show_progress_bar",
        "use_multiprocessing",
        "multiprocessing_threshold",
    ]


def test_every_public_name_of_model2vec_is_provided_or_listed_with_its_reason(folder):
    theirs = model2vec.StaticModel.from_pretrained(folder)
    names = {n for n in dir(theirs) if not n.startswith("_")}
    assert names >= {"encode", "from_pretrained", "dim", "tokens", "embedding", "config"}
    for name in names:
        if name in NOT_PROVIDED:
            assert not hasattr(fse.StaticModel, name), name
        else:
            assert hasattr(fse.StaticModel, name), f"{name} is neither provided nor listed"
    assert set(model2vec.__all__) - {"__version__"} == {"StaticModel", "quantize_model"}
    assert fse.StaticModel.__module__.startswith("fse.")
    assert not hasattr(fse, "quantize_model")
    assert "quantize_model" in NOT_PROVIDED
    for name in OTHER_MODULES:
        assert importlib.util.find_spec(f"model2vec.{name}") is not None, name
    assert all(reason for reason in [*NOT_PROVIDED.values(), *OTHER_MODULES.values()])
    # what fse adds
    assert hasattr(fse.StaticModel, "encoder")


def test_a_static_model_is_made_by_from_pretrained_only(model):
    with pytest.raises(TypeError, match=r"StaticModel\.from_pretrained"):
        fse.StaticModel()
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(model)
    assert isinstance(model.encoder, Encoder)
    assert fse.StaticModel is not model2vec.StaticModel


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"normalize_embeddings": True}, TypeError, "unexpected keyword argument"),
        ({"prompt": "x"}, TypeError, "unexpected keyword argument"),
        ({"convert_to_numpy": False}, ValueError, "convert_to_numpy=False"),
        ({"convert_to_tensor": True}, ValueError, "convert_to_tensor=True"),
        ({"device": "cuda"}, ValueError, "device='cuda'"),
        ({"device": 0}, ValueError, "device=0"),
        ({"show_progress_bar": 1}, TypeError, "show_progress_bar must be a bool"),
        ({"use_multiprocessing": None}, TypeError, "use_multiprocessing must be a bool"),
        ({"batch_size": 0}, ValueError, "batch_size must be at least 1"),
        ({"batch_size": True}, TypeError, "batch_size must be an int"),
        ({"batch_size": 1.5}, TypeError, "batch_size must be an int"),
        ({"multiprocessing_threshold": -1}, ValueError, "at least 0"),
        ({"max_length": 0}, ValueError, "max_length must be at least 1"),
        ({"max_length": "5"}, TypeError, "max_length must be an int"),
        ({"max_length": False}, TypeError, "max_length must be an int"),
        ({"normalize": 1}, TypeError, "normalize must be a bool"),
    ],
)
def test_encode_refuses_what_it_would_not_do(model, kwargs, error, match):
    with pytest.raises(error, match=match):
        model.encode(["hello"], **kwargs)


def test_encode_takes_the_neutral_values_and_numpy_integers(model):
    want = model.encode(["hello world"])
    got = model.encode(
        ["hello world"],
        batch_size=np.int64(3),
        max_length=np.int32(512),
        multiprocessing_threshold=np.int8(0),
        device=None,
        convert_to_numpy=True,
        convert_to_tensor=False,
    )
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize(
    "sentences", [iter(["a"]), (s for s in ["a"]), {"a"}, b"a", 5, None, [1, 2], ["a", None]]
)
def test_encode_takes_a_text_or_a_sequence_of_texts_only(model, sentences):
    with pytest.raises((TypeError, ValueError)):
        model.encode(sentences)


def test_a_stream_or_a_file_goes_to_the_encoder(model, tmp_path):
    path = tmp_path / "texts.txt"
    path.write_text("hello world\nthe cat\n", encoding="utf-8")
    with pytest.raises(TypeError, match=r"StaticModel\.encoder\.encode"):
        model.encode(path)
    assert (
        model.encoder.encode(path).tobytes() == model.encode(["hello world", "the cat"]).tobytes()
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"quantize_to": "float16"}, ValueError, "float32 tables only"),
        ({"quantize_to": "int8"}, ValueError, "float32 tables only"),
        ({"quantize_to": "float64"}, ValueError, "float32 tables only"),
        ({"quantize_to": "bfloat16"}, ValueError, "dtypes of model2vec"),
        ({"quantize_to": 32}, TypeError, "quantize_to must be a str"),
        ({"vocabulary_quantization": 8}, ValueError, "does not cluster the vocabulary"),
        ({"vocabulary_quantization": 0}, ValidationError, "greater than 0"),
        ({"max_length": 0}, ValidationError, "greater than 0"),
        ({"max_length": "5"}, ValidationError, "max_length"),
        ({"dimensionality": 0}, ValidationError, "greater than 0"),
        ({"dimensionality": 17}, ValueError, "greater than the 16 dimensions"),
        ({"normalize": "yes"}, ValidationError, "normalize"),
        ({"force_download": "no"}, ValidationError, "force_download"),
        ({"token": 5}, ValidationError, "token"),
        ({"subfolder": "../x"}, ValueError, "plain names"),
        ({"unknown": 1}, ValidationError, "unknown"),
        ({"revision": "main"}, ValueError, "are for the Hub"),
    ],
)
def test_from_pretrained_refuses_what_it_would_not_do(folder, kwargs, error, match):
    with pytest.raises(error, match=match):
        fse.StaticModel.from_pretrained(folder, **kwargs)


def test_the_setters_are_strict(model):
    for value in (1, "yes", None, np.bool_(True)):
        with pytest.raises(ValidationError):
            model.normalize = value
    for value in (0, -1, "5", 1.5, True):
        with pytest.raises(ValidationError):
            model.max_length = value
    assert (model.normalize, model.max_length) == (True, 512)
    model.max_length = np.int64(9)
    assert model.max_length == 9
    model.max_length = 512


def test_a_model_that_is_no_model2vec_model_is_refused_with_the_reason(tmp_path):
    from safetensors.numpy import load_file, save_file

    original = sm.model_folder(tmp_path / "original")
    half = model2vec.StaticModel.from_pretrained(original, quantize_to="float16")
    half.save_pretrained(tmp_path / "half")  # the config states the dtype of the table
    with pytest.raises(UnsupportedArtifactError, match="float16 table"):
        fse.StaticModel.from_pretrained(tmp_path / "half")
    for layout, tensor in (("model2vec", "embeddings"), ("st-root", "embedding.weight")):
        folder = sm.model_folder(tmp_path / layout, layout)
        path = folder / "model.safetensors"
        table = load_file(path)[tensor]
        for extra in ("weights", "mapping"):
            save_file({tensor: table, extra: np.zeros(len(table), np.float32)}, path)
            with pytest.raises(UnsupportedArtifactError, match="vocabulary quantization"):
                fse.StaticModel.from_pretrained(folder)
        if layout == "st-root":
            save_file({tensor: table.astype(np.float16)}, path)
            with pytest.raises(UnsupportedArtifactError, match="float16 table"):
                fse.StaticModel.from_pretrained(folder)
    with pytest.raises(FileNotFoundError, match="no local file or folder"):
        fse.StaticModel.from_pretrained(tmp_path / "missing")
