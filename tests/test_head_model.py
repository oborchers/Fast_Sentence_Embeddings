"""What Classifier and Regressor share (fse._head._model): construction, the flag,
properties, save and load, complete and standalone."""

import pickle

import numpy as np
import pytest

import fse
from fse._contract import UnsupportedArtifactError
from fse._head._model import LinearModel
from support.encoder import finished
from support.runtime import texts

F32 = np.float32


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def _classifier(encoder=None, k=3, **kwargs):
    encoder = encoder or finished(normalize=False)
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((k, encoder.output_dim)).astype(F32)
    names = ["neg", "pos"] if k == 1 else [f"c{i}" for i in range(k)]
    return fse.Classifier.from_arrays(
        encoder, weight, rng.standard_normal(k), labels=names, **kwargs
    )


def test_the_constructors_refuse_direct_calls():
    with pytest.raises(TypeError, match=r"Classifier\.from_sklearn, Classifier\.from_arrays"):
        fse.Classifier()
    with pytest.raises(TypeError, match="make a Regressor"):
        fse.Regressor(1, x=2)
    assert issubclass(fse.Classifier, LinearModel)


def test_the_flag_follows_the_encoder_unless_given():
    assert not _classifier(finished(normalize=False)).normalized
    assert _classifier(finished(normalize=True)).normalized
    assert not _classifier(finished(normalize=True), normalized=False).normalized
    assert _classifier(finished(normalize=False), normalized=True).normalized


def test_a_wrong_flag_gives_other_predictions():
    """The failure mode the default prevents: a head trained on normalized vectors read on
    raw means predicts something else."""
    encoder = finished(normalize=True)
    right = _classifier(encoder)
    wrong = _classifier(encoder, normalized=False)
    sentences = texts(300)
    assert (
        bits(right.decision_function(sentences)).tolist()
        != bits(wrong.decision_function(sentences)).tolist()
    )


def test_properties():
    model = _classifier(k=3, link="ovr")
    assert model.n_outputs == 3
    assert model.link == "ovr@1"
    assert not model.standalone
    assert model.coef_.shape == (3, 6)
    assert model.intercept_.dtype == F32
    model.coef_[0, 0] = 99  # a copy
    assert model.coef_[0, 0] != 99
    assert model.encoder_fingerprint == model.encoder.fingerprint
    assert model.fingerprint.startswith("xxh3-128:")
    assert "3 outputs, raw means, ovr@1" in repr(model)
    assert model.output_identity("decision") != model.output_identity("proba")


def test_the_constructors_check_their_arguments():
    encoder = finished(normalize=False)
    with pytest.raises(TypeError, match=r"encoder must be an fse\.Encoder"):
        fse.Classifier.from_arrays("enc", np.ones(6), labels=[0, 1])
    with pytest.raises(ValueError, match=r"shape \(k, 6\)"):
        fse.Classifier.from_arrays(encoder, np.ones(5), labels=[0, 1])
    with pytest.raises(TypeError, match="a classifier needs labels"):
        fse.Classifier.from_arrays(encoder, np.ones(6))
    with pytest.raises(ValueError, match="needs 2 labels"):
        fse.Classifier.from_arrays(encoder, np.ones(6), labels=[0, 1, 2])
    with pytest.raises(Exception, match="link"):
        fse.Classifier.from_arrays(encoder, np.ones(6), labels=[0, 1], link="tanh")
    with pytest.raises(Exception, match="normalized"):
        fse.Classifier.from_arrays(encoder, np.ones(6), labels=[0, 1], normalized="yes")


def test_models_cannot_be_pickled():
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(_classifier())


@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_save_and_load_give_the_same_bits(tmp_path, dtype, mmap):
    model = _classifier(finished(dtype, fasttext=True, normalize=False), k=2)
    model.save(tmp_path / "m.safetensors")
    loaded = fse.Classifier.load(tmp_path / "m.safetensors", mmap=mmap)
    assert loaded.fingerprint == model.fingerprint
    assert loaded.encoder_fingerprint == model.encoder.fingerprint
    sentences = texts(400)
    for output in ("predict_proba", "decision_function"):
        assert (
            bits(getattr(loaded, output)(sentences)).tolist()
            == bits(getattr(model, output)(sentences)).tolist()
        )
    assert loaded.predict(sentences).tolist() == model.predict(sentences).tolist()
    loaded.save(tmp_path / "again.safetensors")  # loaded data are compared while written
    assert fse.Classifier.load(tmp_path / "again.safetensors").fingerprint == model.fingerprint


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_a_standalone_file_predicts_the_bits_of_the_complete_model(tmp_path, dtype):
    model = _classifier(finished(dtype, fasttext=True, normalize=False), k=1)
    model.save(tmp_path / "s.safetensors", standalone=True)
    standalone = fse.Classifier.load(tmp_path / "s.safetensors")
    assert standalone.standalone
    assert standalone.classes_.tolist() == ["neg", "pos"]
    sentences = texts(400)
    assert (
        bits(standalone.predict_proba(sentences)).tolist()
        == bits(model.predict_proba(sentences)).tolist()
    )
    with pytest.raises(ValueError, match="standalone model has no encoder"):
        _ = standalone.encoder
    with pytest.raises(ValueError, match="holds no weights"):
        _ = standalone.coef_
    with pytest.raises(ValueError, match="standalone model has no vectors"):
        standalone.predict_batches([["w1"]], return_vectors=True)
    standalone.save(tmp_path / "s2.safetensors")
    assert fse.Classifier.load(tmp_path / "s2.safetensors").fingerprint == standalone.fingerprint
    assert standalone.encoder_fingerprint != model.encoder_fingerprint  # the folded table's


def test_a_normalized_head_is_saved_complete_only(tmp_path):
    model = _classifier(finished(normalize=True))
    with pytest.raises(ValueError, match="does not fold into the table; save it complete"):
        model.save(tmp_path / "s.safetensors", standalone=True)
    assert not (tmp_path / "s.safetensors").exists()


def test_load_names_the_class_of_the_file(tmp_path):
    encoder = finished(normalize=False)
    _classifier(encoder).save(tmp_path / "c.safetensors")
    fse.Regressor.from_arrays(encoder, np.ones(6)).save(tmp_path / "r.safetensors")
    encoder.save(tmp_path / "e.safetensors")
    with pytest.raises(ValueError, match=r"open it with fse\.Classifier\.load"):
        fse.Regressor.load(tmp_path / "c.safetensors")
    with pytest.raises(ValueError, match=r"open it with fse\.Regressor\.load"):
        fse.Classifier.load(tmp_path / "r.safetensors")
    with pytest.raises(UnsupportedArtifactError, match="is an encoder"):
        fse.Classifier.load(tmp_path / "e.safetensors")
    with pytest.raises(UnsupportedArtifactError, match=r"open it with fse\.Classifier\.load"):
        fse.Encoder.load(tmp_path / "c.safetensors")


def test_the_fingerprint_is_the_same_in_memory_and_after_load(tmp_path):
    model = fse.Regressor.from_arrays(finished("float16", normalize=False), np.ones((2, 6)), [1, 2])
    fingerprint = model.fingerprint
    model.save(tmp_path / "r.safetensors")
    assert fse.Regressor.load(tmp_path / "r.safetensors").fingerprint == fingerprint
    other = fse.Regressor.from_arrays(finished("float16", normalize=False), np.ones((2, 6)), [1, 3])
    assert other.fingerprint != fingerprint


def test_save_refuses_a_bad_flag(tmp_path):
    with pytest.raises(Exception, match="standalone"):
        _classifier().save(tmp_path / "x.safetensors", standalone=1)


def test_a_model_on_a_loaded_encoder_saves(tmp_path):
    finished("int8", fasttext=True, normalize=False).save(tmp_path / "e.safetensors")
    encoder = fse.Encoder.load(tmp_path / "e.safetensors")
    model = _classifier(encoder, k=2)
    model.save(tmp_path / "m.safetensors")
    loaded = fse.Classifier.load(tmp_path / "m.safetensors")
    assert loaded.fingerprint == model.fingerprint
    assert loaded.encoder_fingerprint == encoder.fingerprint


@pytest.mark.parametrize("source", ["encoder", "model"])
def test_a_standalone_export_refuses_data_that_changed_since_the_load(tmp_path, source):
    """A mapped file is read without hashing; when its rows differ from their checksums the
    export, which folds a checked copy, raises instead of writing them with new ones."""
    from fse import CorruptArtifactError

    encoder = finished(fasttext=True, normalize=False)
    path = tmp_path / "file.safetensors"
    (encoder if source == "encoder" else _classifier(encoder, k=2)).save(path)
    rows = encoder.word_vectors.stored_vectors.tobytes()
    data = bytearray(path.read_bytes())
    at = data.find(rows[:64])
    assert at > 0
    data[at : at + 8] = bytes(b ^ 0xFF for b in rows[:8])
    path.write_bytes(bytes(data))
    if source == "encoder":
        model = _classifier(fse.Encoder.load(path, mmap=True), k=2)
    else:
        model = fse.Classifier.load(path, mmap=True)
    with pytest.raises(CorruptArtifactError, match="changed since it was loaded"):
        model.save(tmp_path / "after.safetensors", standalone=True)
    assert not (tmp_path / "after.safetensors").exists()


def test_a_standalone_export_of_a_loaded_model_keeps_its_checksums(tmp_path):
    model = _classifier(finished("int8", fasttext=True, normalize=False), k=2)
    model.save(tmp_path / "m.safetensors")
    loaded = fse.Classifier.load(tmp_path / "m.safetensors", mmap=True)
    loaded.save(tmp_path / "s.safetensors", standalone=True)
    model.save(tmp_path / "s0.safetensors", standalone=True)
    assert (
        fse.Classifier.load(tmp_path / "s.safetensors").fingerprint
        == fse.Classifier.load(tmp_path / "s0.safetensors").fingerprint
    )
