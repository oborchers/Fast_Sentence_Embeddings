"""Classifier heads from Model2Vec objects (fse._head._model2vec_objects), without torch: a
StaticModelPipeline written by model2vec 0.10.0, compared with its own predictions within
the bound of tests/support/model2vec_head.py, and every refusal."""

import importlib.metadata as metadata

import numpy as np
import pytest

import fse
from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from support import head as support
from support import model2vec_head as mh

pytest.importorskip("model2vec")
TEXTS = {t: mh.texts(t) for t in ("wordpiece", "unigram")}
F32 = np.float32


@pytest.mark.parametrize("tokenizer", ["wordpiece", "unigram"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    ("activation", "k"), [("softmax", 2), ("softmax", 4), ("sigmoid", 1), ("sigmoid", 3)]
)
def test_a_pipeline_predicts_what_model2vec_predicts(activation, k, normalize, tokenizer):
    pipe = mh.pipeline(activation, k, normalize=normalize, tokenizer=tokenizer)
    clf = fse.Classifier.from_model2vec(pipe)
    assert clf.normalized is normalize
    assert clf.link == ("softmax@1" if activation == "softmax" else "multilabel@1")
    assert clf.n_outputs == k
    assert list(clf.classes_) == list(pipe.classes_)
    mh.assert_matches(clf, pipe, TEXTS[tokenizer])


def test_the_head_and_the_table_are_those_of_the_pipeline():
    pipe = mh.pipeline("softmax", 3, normalize=False)
    clf = fse.Classifier.from_model2vec(pipe)
    layer = pipe.head.layers[0]
    assert clf.coef_.tobytes() == layer.weight.tobytes()
    assert clf.intercept_.tobytes() == layer.bias.tobytes()
    table = clf.encoder.word_vectors.stored_vectors
    assert np.asarray(table).tobytes() == pipe.model.embedding.tobytes()
    assert not clf.encoder.word_vectors.stored_vectors.flags.writeable
    assert "head imported from a Model2Vec StaticModelPipeline (softmax, 3 classes)" in str(
        clf._note
    )


def test_the_arrays_are_copied():
    pipe = mh.pipeline("softmax", 3)
    clf = fse.Classifier.from_model2vec(pipe)
    want = clf.predict_proba(TEXTS["wordpiece"])
    pipe.head.layers[0].weight[:] = 0
    pipe.model.embedding[:] = 0
    np.testing.assert_array_equal(clf.predict_proba(TEXTS["wordpiece"]), want)


def test_integer_classes_stay_integers():
    pipe = mh.pipeline("softmax", 3, classes=[3, 5, 9])
    clf = fse.Classifier.from_model2vec(pipe)
    assert clf.classes_.dtype == np.int64
    assert list(clf.predict(TEXTS["wordpiece"][:5])) == list(pipe.predict(TEXTS["wordpiece"][:5]))


def test_multilabel_thresholds_are_named_cutoffs():
    pipe = mh.pipeline("sigmoid", 3)
    texts = TEXTS["wordpiece"]
    default = fse.Classifier.from_model2vec(pipe)
    clf = fse.Classifier.from_model2vec(pipe, thresholds={"b": 0.9})
    assert dict(clf.thresholds) == pytest.approx({"a": 0.5, "b": 0.9, "c": 0.5})
    assert dict(default.thresholds) == {"a": 0.5, "b": 0.5, "c": 0.5}
    with np.errstate(over="ignore", invalid="ignore"):
        theirs = pipe.predict_proba(texts)
    np.testing.assert_array_equal(clf.predict(texts)[:, 1], theirs[:, 1] > np.float32(0.9))
    with pytest.raises(ValueError, match="only for multilabel heads"):
        fse.Classifier.from_model2vec(mh.pipeline("softmax", 3), thresholds=0.4)


@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
@pytest.mark.parametrize("backend", ["cpu", "numpy"])
@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("normalize", [False, True])
def test_every_backend_gives_the_bits_of_the_reference(backend, workers, normalize, activation):
    pipe = mh.pipeline(activation, 3, normalize=normalize)
    clf = fse.Classifier.from_model2vec(pipe)
    texts = TEXTS["wordpiece"]
    link = clf._content.link
    want = mh.reference_logits(clf, texts)
    got = clf.decision_function(texts, backend=backend, workers=workers)
    assert got.view(np.uint32).tobytes() == np.asarray(want, F32).view(np.uint32).tobytes()
    proba = clf.predict_proba(texts, backend=backend, workers=workers)
    assert proba.tobytes() == support.outputs(want, "proba", link).tobytes()
    labels = clf.predict(texts, backend=backend, workers=workers)
    expected = support.outputs(want, "labels", link)
    assert (labels == (expected if activation == "sigmoid" else clf.classes_[expected])).all()


@pytest.mark.parametrize("normalize", [False, True])
def test_save_and_load_keep_the_bits(tmp_path, normalize):
    pipe = mh.pipeline("sigmoid", 3, normalize=normalize)
    clf = fse.Classifier.from_model2vec(pipe)
    clf.save(tmp_path / "a.safetensors")
    loaded = fse.Classifier.load(tmp_path / "a.safetensors")
    texts = TEXTS["wordpiece"]
    assert loaded.fingerprint == clf.fingerprint
    assert loaded.predict_proba(texts).tobytes() == clf.predict_proba(texts).tobytes()
    folder = clf.save_pretrained(tmp_path / "folder", license="mit")
    again = fse.Classifier.from_pretrained(folder)
    assert again.predict_proba(texts).tobytes() == clf.predict_proba(texts).tobytes()
    assert (again.predict(texts) == clf.predict(texts)).all()
    if not normalize:
        clf.save(tmp_path / "s.safetensors", standalone=True)
        alone = fse.Classifier.load(tmp_path / "s.safetensors")
        assert alone.predict_proba(texts).tobytes() == clf.predict_proba(texts).tobytes()


def test_the_object_and_the_folder_of_a_pipeline_agree(tmp_path):
    folder, pipe = mh.saved(tmp_path / "export", activation="softmax", k=3, normalize=False)
    texts = TEXTS["wordpiece"]
    from_object = fse.Classifier.from_model2vec(pipe)
    from_folder = fse.Classifier.from_pretrained(folder)
    assert from_object.coef_.tobytes() == from_folder.coef_.tobytes()
    assert from_object.predict_proba(texts).tobytes() == from_folder.predict_proba(texts).tobytes()
    assert from_object.encoder._content.source.repo is None


@pytest.mark.parametrize("n", [0, 2, 3])
def test_hidden_layers_and_no_layer_are_refused(n):
    layers = [mh.layer(4, seed=i) for i in range(n)]
    layers = [(w[:, :16], b) for w, b in layers]
    if n >= 2:  # a hidden layer maps 16 to 4, the next 4 to k
        layers = [mh.layer(4), (np.ones((3, 4), F32), np.ones(3, F32))][:n]
    pipe = mh.pipeline("softmax", 3, layers=layers)
    with pytest.raises(UnsupportedArtifactError, match=r"hidden layers|no layer"):
        fse.Classifier.from_model2vec(pipe)


def test_regression_heads_are_refused():
    pipe = mh.pipeline("identity", 3)
    with pytest.raises(UnsupportedArtifactError, match="regression or projection"):
        fse.Classifier.from_model2vec(pipe)


@pytest.mark.parametrize("dtype", [np.float64, np.float16, np.int64])
def test_a_head_of_another_dtype_is_refused_like_its_export(dtype):
    pipe = mh.pipeline("softmax", 3, layers=[mh.layer(3, dtype=dtype)])
    with pytest.raises(UnsupportedArtifactError, match=f"is {np.dtype(dtype).name}"):
        fse.Classifier.from_model2vec(pipe)


@pytest.mark.parametrize("bias", [np.float32(0.5), np.zeros((3, 1), F32), np.zeros(2, F32)])
def test_a_bias_of_another_shape_is_refused(bias):
    weight, _ = mh.layer(3)
    pipe = mh.pipeline("softmax", 3, layers=[(weight, bias)])
    with pytest.raises((ValueError, TypeError)):
        fse.Classifier.from_model2vec(pipe)


def test_a_head_that_is_not_an_array_and_a_wrong_width():
    weight, bias = mh.layer(3)
    with pytest.raises(ValueError, match="must be a numpy array"):
        fse.Classifier.from_model2vec(mh.pipeline("softmax", 3, layers=[(weight.tolist(), bias)]))
    with pytest.raises(ValueError, match="the head reads 8 values"):
        fse.Classifier.from_model2vec(mh.pipeline("softmax", 3, layers=[(weight[:, :8], bias)]))


def test_nonfinite_weights_are_refused():
    weight, bias = mh.layer(3)
    weight[1, 2] = np.inf
    with pytest.raises(ValueError, match="not finite"):
        fse.Classifier.from_model2vec(mh.pipeline("softmax", 3, layers=[(weight, bias)]))


def test_classes_must_agree_and_exist():
    pipe = mh.pipeline("softmax", 3)
    pipe.classes_ = np.array(["x", "y", "z"])
    with pytest.raises(ValueError, match=r"differs from pipeline.head.classes_"):
        fse.Classifier.from_model2vec(pipe)
    pipe = mh.pipeline("softmax", 3)
    pipe.classes_ = pipe.head.classes_ = None
    with pytest.raises(ValueError, match="classes must be a non-empty list"):
        fse.Classifier.from_model2vec(pipe)


def test_the_config_must_describe_the_head():
    pipe = mh.pipeline("softmax", 3)
    pipe.model.config["head_config"] = {"n_layers": 1, "activation": "sigmoid", "classes": ["a"]}
    with pytest.raises(ValueError, match="does not describe the head"):
        fse.Classifier.from_model2vec(pipe)
    pipe = mh.pipeline("softmax", 3)
    pipe.model.config["head_config"] = {
        "n_layers": 1,
        "activation": "softmax",
        "classes": ["a", "b", "c"],
    }
    assert fse.Classifier.from_model2vec(pipe).n_outputs == 3


def test_separate_token_weights_and_a_token_mapping_are_refused():
    pipe = mh.pipeline("softmax", 3)
    pipe.model.weights = np.ones(len(pipe.model.embedding), F32)
    with pytest.raises(UnsupportedArtifactError, match="separate token weights"):
        fse.Classifier.from_model2vec(pipe)
    pipe = mh.pipeline("softmax", 3)
    pipe.model.token_mapping = np.arange(len(pipe.model.embedding))
    with pytest.raises(UnsupportedArtifactError, match="vocabulary quantization"):
        fse.Classifier.from_model2vec(pipe)


@pytest.mark.parametrize("dtype", [np.float16, np.float64])
def test_a_table_of_another_dtype_is_refused(dtype):
    pipe = mh.pipeline("softmax", 3)
    pipe.model.embedding = pipe.model.embedding.astype(dtype)
    with pytest.raises(UnsupportedArtifactError, match=f"{np.dtype(dtype).name} table"):
        fse.Classifier.from_model2vec(pipe)


def test_the_table_must_agree_with_its_config():
    pipe = mh.pipeline("softmax", 3)
    pipe.model.config["embedding_dtype"] = "float16"
    with pytest.raises(CorruptArtifactError, match="embedding_dtype is 'float16'"):
        fse.Classifier.from_model2vec(pipe)
    pipe = mh.pipeline("softmax", 3)
    pipe.model.config["hidden_dim"] = 7
    with pytest.raises(CorruptArtifactError, match="hidden_dim is 7"):
        fse.Classifier.from_model2vec(pipe)
    pipe = mh.pipeline("softmax", 3)
    pipe.model.config["vocabulary_quantization"] = 4
    with pytest.raises(UnsupportedArtifactError, match="vocabulary quantization"):
        fse.Classifier.from_model2vec(pipe)
    pipe = mh.pipeline("softmax", 3)
    pipe.model.embedding = pipe.model.embedding.tolist()
    with pytest.raises(TypeError, match="numpy array"):
        fse.Classifier.from_model2vec(pipe)


@pytest.mark.parametrize("thing", ["some/where", "", 3, None, object()])
def test_only_pipelines_and_trained_classifiers_are_objects(thing):
    with pytest.raises(TypeError, match="StaticModelPipeline or a trained"):
        fse.Classifier.from_model2vec(thing)


def test_a_model_without_a_head_and_other_model2vec_types_are_refused():
    from model2vec.inference import StaticModelPipeline

    for thing in (mh.static_model(), StaticModelPipeline):
        with pytest.raises(TypeError, match="StaticModelPipeline or a trained"):
            fse.Classifier.from_model2vec(thing)


@pytest.mark.parametrize("installed", ["0.9.2", "0.11.0", "1.0.0", "", "0.1"])
def test_only_the_version_window_is_read(monkeypatch, installed):
    pipe = mh.pipeline("softmax", 3)
    real = metadata.version

    def version(name):
        if name == "model2vec":
            if not installed:
                raise metadata.PackageNotFoundError(name)
            return installed
        return real(name)

    monkeypatch.setattr("fse._head._model2vec_objects.version", version)
    with pytest.raises(UnsupportedArtifactError, match=r"model2vec 0\.10 only"):
        fse.Classifier.from_model2vec(pipe)


@pytest.mark.parametrize("installed", ["0.10.0", "0.10.4", "0.10"])
def test_the_window_accepts_0_10(monkeypatch, installed):
    real = metadata.version
    monkeypatch.setattr(
        "fse._head._model2vec_objects.version",
        lambda name: installed if name == "model2vec" else real(name),
    )
    assert fse.Classifier.from_model2vec(mh.pipeline("softmax", 3)).n_outputs == 3


def test_a_path_is_not_an_object():
    with pytest.raises(TypeError, match=r"Classifier.from_pretrained"):
        fse.Classifier.from_model2vec("some/where")


def test_saturated_logits_compare_to_model2vec():
    for activation in ("sigmoid", "softmax"):
        pipe = mh.pipeline(activation, 3, scale=40.0, normalize=False)
        clf = fse.Classifier.from_model2vec(pipe)
        z, _ = mh.assert_matches(clf, pipe, TEXTS["wordpiece"])
        assert np.abs(z).max() > 90  # beyond the exponent range of float32 for exp(-z)


def test_the_applied_settings_must_equal_the_config():
    for name, value in (("normalize", False), ("max_length", 7)):
        pipe = mh.pipeline("softmax", 3, normalize=True)
        pipe.model.config[name] = value  # model2vec still applies model.<name>
        with pytest.raises(ValueError, match=rf"model\.{name} differs"):
            fse.Classifier.from_model2vec(pipe)


@pytest.mark.parametrize("stated", [None, [1], {"n_layers": True, "activation": "softmax"}])
def test_a_stated_head_config_is_checked_as_strictly_as_in_a_file(stated):
    pipe = mh.pipeline("softmax", 3)
    pipe.model.config["head_config"] = stated
    with pytest.raises(ValueError, match="does not describe the head"):
        fse.Classifier.from_model2vec(pipe)


def test_the_stated_head_config_with_floats_for_ints_is_refused():
    pipe = mh.pipeline("softmax", 3)
    pipe.model.config["head_config"] = {
        "n_layers": 1.0,
        "activation": "softmax",
        "classes": ["a", "b", "c"],
    }
    with pytest.raises(ValueError, match="does not describe the head"):
        fse.Classifier.from_model2vec(pipe)


@pytest.mark.parametrize("classes", [["a", 1, "c"], np.array(["a", "b", "c"], dtype=object)])
def test_classes_are_checked_before_they_are_converted(classes):
    pipe = mh.pipeline("softmax", 3)
    pipe.classes_ = pipe.head.classes_ = classes
    with pytest.raises(ValueError, match="one-dimensional numpy array of int or str"):
        fse.Classifier.from_model2vec(pipe)


def test_a_class_that_only_carries_the_names_is_refused():
    fake = type("StaticModelPipeline", (), {"__module__": "model2vec.inference.model"})
    with pytest.raises(TypeError, match="StaticModelPipeline or a trained"):
        fse.Classifier.from_model2vec(fake())


def test_fse_imports_neither_model2vec_nor_torch_to_read_an_export(tmp_path):
    import subprocess
    import sys

    folder, _ = mh.saved(tmp_path / "export", activation="softmax", k=3)
    code = (
        "import sys\n"
        "sys.modules['model2vec'] = None\nsys.modules['torch'] = None\n"
        "import fse\n"
        f"clf = fse.Classifier.from_pretrained({str(folder)!r})\n"
        "print(clf.n_outputs)\n"
    )
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "3"
