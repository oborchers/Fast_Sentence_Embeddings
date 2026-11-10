"""One text-path protocol across model persistence, stores, and text inputs."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ConfigDict, TypeAdapter, ValidationError

import fse
from fse._encoder import _pretrained
from fse._validation import ModelRef, PathLike, file_path
from support import score_heads as sh
from support.encoder import finished
from support.paths import BytesPath, CountingPath, OwnPath
from support.runtime import word_vectors


@pytest.mark.parametrize("alias", [PathLike, ModelRef])
def test_path_validation_keeps_strings_and_normalizes_path_objects(alias, tmp_path):
    adapter = TypeAdapter(alias, config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    text = "namespace/model"
    assert adapter.validate_python(text) is text
    for value in (tmp_path, OwnPath(tmp_path)):
        assert adapter.validate_python(value) == tmp_path
        assert isinstance(adapter.validate_python(value), Path)


@pytest.mark.parametrize("value", [BytesPath(), b"path", 3, None, [], {"path": "x"}])
def test_path_validation_checks_the_original_type(value):
    adapter = TypeAdapter(PathLike, config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    with pytest.raises(ValidationError):
        adapter.validate_python(value)
    with pytest.raises(TypeError):
        file_path(value, "path")


@pytest.mark.parametrize("kind", ["encoder", "word_vectors", "classifier", "regressor"])
def test_model_save_and_load_accept_foreign_path_protocols(tmp_path, kind):
    encoder = finished()
    models = {
        "encoder": encoder,
        "word_vectors": word_vectors(),
        "classifier": sh.classifier(encoder),
        "regressor": sh.regressor(encoder),
    }
    model = models[kind]
    target = tmp_path / "model.safetensors"
    model.save(OwnPath(target))
    loaded = type(model).load(OwnPath(target))
    assert loaded.fingerprint == model.fingerprint
    with pytest.raises(ValidationError):
        model.save(BytesPath())
    with pytest.raises(ValidationError):
        type(model).load(BytesPath())


@pytest.mark.parametrize("kind", ["encoder", "classifier", "regressor"])
def test_pretrained_save_load_and_extra_files_accept_foreign_paths(tmp_path, kind):
    encoder = finished()
    model = {
        "encoder": encoder,
        "classifier": sh.classifier(encoder),
        "regressor": sh.regressor(encoder),
    }[kind]
    license_file = tmp_path / "license.txt"
    license_file.write_text("Test license", encoding="utf-8")
    target = tmp_path / "model"
    model.save_pretrained(
        OwnPath(target), license="mit", extra_files={"LICENSE": OwnPath(license_file)}
    )
    assert (target / "LICENSE").read_text(encoding="utf-8") == "Test license"
    loaded = type(model).from_pretrained(OwnPath(target))
    assert loaded.fingerprint == model.fingerprint


def test_a_path_object_that_looks_like_a_hub_id_stays_local(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def unexpected_hub(*_args, **_kwargs):
        pytest.fail("a path object reached the Hub")

    monkeypatch.setattr(_pretrained, "HubFiles", unexpected_hub)
    for model in (fse.Encoder, fse.Classifier, fse.Regressor, fse.StaticModel):
        with pytest.raises(FileNotFoundError):
            model.from_pretrained(OwnPath("namespace/model"))
        with pytest.raises(ValidationError):
            model.from_pretrained(BytesPath())


def test_a_hub_id_accepts_a_foreign_cache_path_offline(tmp_path):
    with pytest.raises(FileNotFoundError):
        fse.Encoder.from_pretrained(
            "namespace/missing", cache_dir=OwnPath(tmp_path), local_files_only=True
        )


@pytest.mark.parametrize("score_store", [False, True])
def test_sinks_stores_and_views_accept_foreign_paths(tmp_path, score_store):
    encoder = finished()
    group = sh.group(encoder)
    directory = tmp_path / "store"
    if score_store:
        with fse.ScoreSink(OwnPath(directory), group) as sink:
            list(group.predict_batches([["w1", "w2"]], score_sink=sink))
        store_type = fse.ScoreStore
    else:
        with fse.ShardSink(OwnPath(directory), encoder) as sink:
            list(encoder.encode_batches([["w1", "w2"]], sink=sink))
        store_type = fse.VectorStore
    with store_type.open(OwnPath(directory)) as opened:
        assert len(opened) == 2
    path = CountingPath(directory)
    with store_type.open_view([path]) as opened:
        assert len(opened) == 2
    assert path.calls == 1
    with pytest.raises(TypeError):
        store_type.open(BytesPath())
    with pytest.raises(TypeError):
        store_type.open_view([BytesPath()])
    sink_type = fse.ScoreSink if score_store else fse.ShardSink
    with pytest.raises(TypeError):
        sink_type(BytesPath(), group if score_store else encoder)
    if not score_store:
        assert fse.add_sign_bits(OwnPath(directory)) == 1
    assert fse.repair_store(OwnPath(directory)) == 2


@pytest.mark.parametrize(
    "method", ["encode", "predict", "predict_proba", "decision_function", "regressor", "heads"]
)
def test_text_paths_give_the_same_bits_as_the_lines(tmp_path, method):
    encoder = finished()
    classifier = sh.classifier(encoder)
    regressor = sh.regressor(encoder)
    group = fse.Heads(c=classifier, r=regressor)
    call = {
        "encode": encoder.encode,
        "predict": classifier.predict,
        "predict_proba": classifier.predict_proba,
        "decision_function": classifier.decision_function,
        "regressor": regressor.predict,
        "heads": group.predict,
    }[method]
    texts = ["w1 w2", "", "w3"]
    path = tmp_path / "texts.txt"
    path.write_text("\n".join(texts) + "\n", encoding="utf-8")
    expected = call(texts)
    for value in (path, OwnPath(path)):
        actual = call(value)
        if isinstance(expected, Mapping):
            assert actual.keys() == expected.keys()
            for name, rows in expected.items():
                assert actual[name].tobytes() == rows.tobytes()
        else:
            assert actual.tobytes() == expected.tobytes()
    with pytest.raises(TypeError):
        call(BytesPath())


def test_static_model_paths_accept_the_protocol(tmp_path):
    pytest.importorskip("model2vec")
    from support.model2vec import save_model

    folder = save_model(tmp_path / "original")
    model = fse.StaticModel.from_pretrained(OwnPath(folder))
    target = tmp_path / "saved"
    model.save_pretrained(OwnPath(target))
    loaded = fse.StaticModel.from_pretrained(OwnPath(target))
    assert loaded.encode(["hello"]).tobytes() == model.encode(["hello"]).tobytes()
