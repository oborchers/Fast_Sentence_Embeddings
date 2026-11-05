"""Classifier heads from the files of a Model2Vec export (fse._head._model2vec_files): a
local folder and a repository of the fake Hub, compared with model2vec's own pipeline, the
order in which formats are recognized, and every refusal."""

import json
import shutil

import huggingface_hub
import numpy as np
import pydantic
import pytest
from safetensors.numpy import load_file, save_file

import fse
import fse._hub as hub
from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._encoder import Encoder
from fse._encoder._format import Source
from support import model2vec as mv
from support import model2vec_head as mh
from support.hubcache import COMMIT, FakeHub

pytest.importorskip("model2vec")
TEXTS = {t: mh.texts(t) for t in ("wordpiece", "unigram")}
F32 = np.float32


def _export(tmp_path, name="export", **kwargs):
    kwargs = {"activation": "softmax", "k": 3, "normalize": False} | kwargs
    return mh.saved(tmp_path / name, **kwargs)


def _config(folder, **changes):
    path = folder / "config.json"
    config = json.loads(path.read_text())
    config.update(changes)
    path.write_text(json.dumps(config))
    return config


def _head_config(folder, **changes):
    path = folder / "config.json"
    config = json.loads(path.read_text())
    config["head_config"].update(changes)
    path.write_text(json.dumps(config))


def _head(folder, tensors):
    save_file(tensors, folder / "head.safetensors")


@pytest.mark.parametrize("tokenizer", ["wordpiece", "unigram"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    ("activation", "k"), [("softmax", 2), ("softmax", 4), ("sigmoid", 1), ("sigmoid", 3)]
)
def test_an_export_predicts_what_model2vec_predicts(tmp_path, activation, k, normalize, tokenizer):
    folder, pipe = _export(
        tmp_path, activation=activation, k=k, normalize=normalize, tokenizer=tokenizer
    )
    clf = fse.Classifier.from_pretrained(folder)
    assert clf.normalized is normalize
    assert list(clf.classes_) == list(pipe.classes_)
    mh.assert_matches(clf, pipe, TEXTS[tokenizer])
    assert clf.encoder._content.source == Source("model2vec")
    assert clf.encoder._content.license == "mit"  # of the model card


@pytest.mark.parametrize("normalize", [False, True])
def test_the_export_and_a_saved_export_keep_the_bits(tmp_path, normalize):
    folder, pipe = _export(tmp_path, activation="sigmoid", normalize=normalize)
    clf = fse.Classifier.from_pretrained(folder)
    texts = TEXTS["wordpiece"]
    assert clf.coef_.tobytes() == pipe.head.layers[0].weight.tobytes()
    clf.save(tmp_path / "a.safetensors")
    loaded = fse.Classifier.load(tmp_path / "a.safetensors")
    assert loaded.predict_proba(texts).tobytes() == clf.predict_proba(texts).tobytes()
    again = fse.Classifier.from_pretrained(clf.save_pretrained(tmp_path / "fse", license="mit"))
    assert again.fingerprint == clf.fingerprint
    assert again.predict_proba(texts).tobytes() == clf.predict_proba(texts).tobytes()
    if not normalize:
        clf.save(tmp_path / "s.safetensors", standalone=True)
        alone = fse.Classifier.load(tmp_path / "s.safetensors")
        assert alone.predict_proba(texts).tobytes() == clf.predict_proba(texts).tobytes()
    object_ = fse.Classifier.from_model2vec(pipe)
    assert object_.predict_proba(texts).tobytes() == clf.predict_proba(texts).tobytes()


def test_cutoffs_of_an_export_are_set_afterwards(tmp_path):
    folder, pipe = _export(tmp_path, activation="sigmoid")
    clf = fse.Classifier.from_pretrained(folder).with_thresholds({"a": 0.2})
    assert clf.thresholds["a"] == pytest.approx(0.2)
    with np.errstate(over="ignore", invalid="ignore"):
        proba = pipe.predict_proba(TEXTS["wordpiece"])
    np.testing.assert_array_equal(clf.predict(TEXTS["wordpiece"])[:, 0], proba[:, 0] > F32(0.2))
    with pytest.raises(pydantic.ValidationError, match="Unexpected keyword argument"):
        fse.Classifier.from_pretrained(folder, thresholds=0.2)


def test_no_base_model_is_loaded(tmp_path, monkeypatch):
    folder, _ = _export(tmp_path)

    def refuse(*_args, **_kwargs):
        raise AssertionError("the Hub was asked")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)
    monkeypatch.setattr(hub, "_library", refuse)
    assert fse.Classifier.from_pretrained(folder).n_outputs == 3


def test_an_fse_folder_next_to_foreign_files_is_still_an_fse_folder(tmp_path):
    folder, pipe = _export(tmp_path)
    clf = fse.Classifier.from_model2vec(pipe)
    saved = clf.save_pretrained(tmp_path / "fse", license="mit")
    shutil.copy(folder / "head.safetensors", saved / "head.safetensors")
    (saved / "pipeline.skops").write_bytes(b"not read")
    assert fse.Classifier.from_pretrained(saved).fingerprint == clf.fingerprint


# --- the Hub --------------------------------------------------------------------------


@pytest.fixture
def fake(tmp_path, monkeypatch):
    folder, pipe = _export(tmp_path, "hubbed", normalize=True)
    files = {p.name: p.read_bytes() for p in folder.iterdir()}
    lfs = {"model.safetensors", "head.safetensors"}
    hubbed = FakeHub(tmp_path / "cache", "acme/clf", files, lfs=lfs)
    monkeypatch.setattr(hub, "_library", lambda: hubbed)
    hubbed.folder_, hubbed.pipe = folder, pipe
    return hubbed


def test_an_export_of_the_hub_is_converted_at_its_commit(fake):
    clf = fse.Classifier.from_pretrained("acme/clf", revision="main", cache_dir=fake.cache)
    assert fake.asked == [("acme/clf", "main", None)]
    assert clf.encoder._content.source == Source("model2vec", "acme/clf", COMMIT)
    local = fse.Classifier.from_pretrained(fake.folder_)
    texts = TEXTS["wordpiece"]
    assert clf.predict_proba(texts).tobytes() == local.predict_proba(texts).tobytes()
    assert clf.fingerprint != local.fingerprint  # another provenance


def test_a_head_file_of_the_hub_must_be_an_lfs_file(fake):
    fake.lfs = {"model.safetensors"}
    with pytest.raises(CorruptArtifactError, match=r"head.safetensors"):
        fse.Classifier.from_pretrained("acme/clf", cache_dir=fake.cache)


def test_a_changed_head_blob_is_caught(fake):
    fse.Classifier.from_pretrained("acme/clf", cache_dir=fake.cache)
    import os

    blob = os.path.realpath(fake.folder() / "snapshots" / COMMIT / "head.safetensors")
    with open(blob, "r+b") as f:
        f.seek(-4, 2)
        f.write(b"\1\2\3\4")
    with pytest.raises(CorruptArtifactError, match=r"head\.safetensors has sha256"):
        fse.Classifier.from_pretrained("acme/clf", cache_dir=fake.cache)


def test_an_export_on_the_hub_needs_its_files(fake):
    del fake.files["head.safetensors"]
    with pytest.raises(CorruptArtifactError, match=r"head_config and head.safetensors"):
        fse.Classifier.from_pretrained("acme/clf", cache_dir=fake.cache)


# --- what is not read -----------------------------------------------------------------


def test_mmap_is_for_fse_models(tmp_path):
    folder, _ = _export(tmp_path)
    with pytest.raises(ValueError, match="converted in memory"):
        fse.Classifier.from_pretrained(folder, mmap=True)


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    [
        ({"n_layers": 2}, UnsupportedArtifactError, "hidden layers with ReLU"),
        ({"n_layers": 0}, UnsupportedArtifactError, "no layer"),
        ({"n_layers": "1"}, CorruptArtifactError, "n_layers must be"),
        ({"n_layers": True}, CorruptArtifactError, "n_layers must be"),
        ({"activation": "identity"}, UnsupportedArtifactError, "regression or projection"),
        ({"activation": "tanh"}, UnsupportedArtifactError, "activation 'tanh' is unknown"),
        ({"activation": 4}, CorruptArtifactError, "must be a str"),
        ({"classes": None}, CorruptArtifactError, "classes must be a non-empty list"),
        ({"classes": ["a", "b"]}, CorruptArtifactError, "do not describe one head"),
        ({"classes": ["a", "b", "b"]}, ValueError, "unique"),
        ({"classes": ["a", 1, "c"]}, TypeError, "all int or all str"),
        ({"classes": ["a", True, "c"]}, TypeError, "all int or all str|not a bool"),
        ({"extra": 1}, UnsupportedArtifactError, r"keys this fse does not know: \['extra'\]"),
    ],
)
def test_the_head_config_decides(tmp_path, changes, error, match):
    folder, _ = _export(tmp_path)
    _head_config(folder, **changes)
    with pytest.raises(error, match=match):
        fse.Classifier.from_pretrained(folder)


def test_a_head_config_that_is_no_object(tmp_path):
    folder, _ = _export(tmp_path)
    _config(folder, head_config=[1])
    with pytest.raises(CorruptArtifactError, match="head_config must be a JSON object"):
        fse.Classifier.from_pretrained(folder)


def test_a_head_config_without_keys_is_no_linear_head(tmp_path):
    folder, _ = _export(tmp_path)
    _config(folder, head_config={})
    with pytest.raises(UnsupportedArtifactError, match="no layer"):
        fse.Classifier.from_pretrained(folder)


def _tensors(folder):
    return dict(load_file(folder / "head.safetensors"))


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        (lambda t: {**t, "head.1.weight": t["head.0.weight"]}, CorruptArtifactError, "exactly"),
        (lambda t: {"head.0.weight": t["head.0.weight"]}, CorruptArtifactError, "exactly"),
        (
            lambda t: {k: v.astype(np.float64) for k, v in t.items()},
            UnsupportedArtifactError,
            "weight is float64",
        ),
        (
            lambda t: {**t, "head.0.bias": t["head.0.bias"].astype(np.float16)},
            UnsupportedArtifactError,
            "bias is float16",
        ),
        (
            lambda t: {**t, "head.0.bias": np.zeros((3, 1), F32)},
            CorruptArtifactError,
            "weight .k, d. and bias .k,.",
        ),
        (
            lambda t: {**t, "head.0.bias": np.zeros((), F32)},
            CorruptArtifactError,
            "weight .k, d. and bias .k,.",
        ),
        (
            lambda t: {**t, "head.0.bias": np.zeros(2, F32)},
            CorruptArtifactError,
            "do not describe one head",
        ),
        (
            lambda t: {**t, "head.0.weight": t["head.0.weight"][:, :8]},
            CorruptArtifactError,
            "the head reads 8 values",
        ),
        (
            lambda t: {**t, "head.0.weight": t["head.0.weight"].reshape(-1)},
            CorruptArtifactError,
            "weight .k, d. and bias",
        ),
    ],
)
def test_the_tensors_of_the_head(tmp_path, change, error, match):
    folder, _ = _export(tmp_path)
    _head(folder, change(_tensors(folder)))
    with pytest.raises(error, match=match):
        fse.Classifier.from_pretrained(folder)


def test_a_head_of_nonfinite_values(tmp_path):
    folder, _ = _export(tmp_path)
    tensors = _tensors(folder)
    tensors["head.0.weight"][0, 0] = np.nan
    _head(folder, tensors)
    with pytest.raises(ValueError, match="not finite"):
        fse.Classifier.from_pretrained(folder)


def test_a_softmax_head_of_one_class(tmp_path):
    folder, _ = _export(tmp_path)
    _head_config(folder, classes=["a"])
    _head(folder, {"head.0.weight": np.ones((1, mv.DIM), F32), "head.0.bias": np.ones(1, F32)})
    with pytest.raises(UnsupportedArtifactError, match="constant"):
        fse.Classifier.from_pretrained(folder)


def test_a_legacy_pipeline_is_never_opened(tmp_path):
    folder, _ = _export(tmp_path)
    (folder / "head.safetensors").unlink()
    (folder / "pipeline.skops").write_bytes(b"pickle")
    with pytest.raises(UnsupportedArtifactError, match="loads no pickle or skops"):
        fse.Classifier.from_pretrained(folder)
    (folder / "pipeline.skops").unlink()
    with pytest.raises(CorruptArtifactError, match=r"head_config and head.safetensors"):
        fse.Classifier.from_pretrained(folder)


def test_a_head_file_without_a_head_config(tmp_path):
    folder, _ = _export(tmp_path)
    config = json.loads((folder / "config.json").read_text())
    del config["head_config"]
    (folder / "config.json").write_text(json.dumps(config))
    with pytest.raises(CorruptArtifactError, match=r"head_config and head.safetensors"):
        fse.Classifier.from_pretrained(folder)


@pytest.mark.parametrize("missing", ["model.safetensors", "tokenizer.json"])
def test_a_missing_file_of_an_export(tmp_path, missing):
    folder, _ = _export(tmp_path)
    (folder / missing).unlink()
    with pytest.raises(CorruptArtifactError, match=f"lacks .*{missing}"):
        fse.Classifier.from_pretrained(folder)


def test_a_dense_module_is_refused(tmp_path):
    folder, _ = _export(tmp_path)
    modules = json.loads((folder / "modules.json").read_text())
    modules.append(
        {"idx": 1, "name": "1", "path": "1_Dense", "type": "sentence_transformers.models.Dense"}
    )
    (folder / "modules.json").write_text(json.dumps(modules))
    with pytest.raises(UnsupportedArtifactError, match="no static embedding model"):
        fse.Classifier.from_pretrained(folder)


def test_a_model_without_a_head_is_no_classifier(tmp_path):
    model = mv.save_model(tmp_path / "plain", normalize=False)
    with pytest.raises(UnsupportedArtifactError, match=r"has no head; read it with fse.Encoder"):
        fse.Classifier.from_pretrained(model)
    with pytest.raises(UnsupportedArtifactError, match="Model2Vec model"):
        fse.Regressor.from_pretrained(model)


def test_an_export_is_no_regressor_encoder_or_static_model(tmp_path):
    folder, _ = _export(tmp_path)
    with pytest.raises(
        UnsupportedArtifactError, match=r"classifier head.*Classifier.from_pretrained"
    ):
        fse.Regressor.from_pretrained(folder)
    with pytest.raises(
        UnsupportedArtifactError, match=r"Classifier.from_pretrained\(...\).encoder"
    ):
        Encoder.from_pretrained(folder)
    assert fse.Classifier.from_pretrained(folder).encoder.output_dim == mv.DIM


# --- tables the encoder reader refuses ------------------------------------------------


def _table(folder, **changes):
    path = folder / "model.safetensors"
    tensors = dict(load_file(path))
    tensors.update(changes)
    save_file(tensors, path)


def test_a_float64_table_is_refused_not_a_key_error(tmp_path):
    folder, _ = _export(tmp_path)
    _table(
        folder, embeddings=load_file(folder / "model.safetensors")["embeddings"].astype(np.float64)
    )
    _config(folder, embedding_dtype="float64")
    with pytest.raises(UnsupportedArtifactError, match="float64 table"):
        fse.Classifier.from_pretrained(folder)


def test_a_float64_model_without_a_head_is_refused_by_the_encoder(tmp_path):
    folder = mv.save_model(tmp_path / "plain")
    table = load_file(folder / "model.safetensors")["embeddings"]
    save_file({"embeddings": table.astype(np.float64)}, folder / "model.safetensors")
    _config(folder, embedding_dtype="float64")
    with pytest.raises(UnsupportedArtifactError, match="float64 table"):
        Encoder.from_pretrained(folder)


def test_separate_token_weights_are_refused(tmp_path):
    folder, pipe = _export(tmp_path)
    weights = np.ones(len(pipe.model.embedding), F32)
    _table(folder, weights=weights)
    with pytest.raises(UnsupportedArtifactError, match="separate token weights"):
        fse.Classifier.from_pretrained(folder)
    plain = mv.save_model(tmp_path / "plain")
    _table(plain, weights=weights)
    with pytest.raises(UnsupportedArtifactError, match="separate token weights"):
        Encoder.from_pretrained(plain)


def test_vocabulary_quantization_keeps_its_own_message(tmp_path):
    folder, pipe = _export(tmp_path)
    n = len(pipe.model.embedding)
    _table(folder, weights=np.ones(n, F32), mapping=np.arange(n, dtype=np.int64))
    with pytest.raises(UnsupportedArtifactError, match="vocabulary quantization"):
        fse.Classifier.from_pretrained(folder)
    folder, _ = _export(tmp_path, "second")
    _config(folder, vocabulary_quantization=4)
    with pytest.raises(UnsupportedArtifactError, match="vocabulary quantization"):
        fse.Classifier.from_pretrained(folder)


def test_a_table_of_another_width_than_the_head(tmp_path):
    folder, _ = _export(tmp_path)
    table = load_file(folder / "model.safetensors")["embeddings"][:, :8]
    _table(folder, embeddings=np.ascontiguousarray(table))
    _config(folder, hidden_dim=8)
    with pytest.raises(CorruptArtifactError, match="the head reads 16 values, the table has 8"):
        fse.Classifier.from_pretrained(folder)


def test_a_nonfinite_table(tmp_path):
    folder, _ = _export(tmp_path)
    table = load_file(folder / "model.safetensors")["embeddings"].copy()
    table[3, 3] = np.inf
    _table(folder, embeddings=table)
    with pytest.raises(ValueError, match="NaN or infinite"):
        fse.Classifier.from_pretrained(folder)


def test_an_unknown_key_in_the_config_still_raises(tmp_path):
    folder, _ = _export(tmp_path)
    _config(folder, something_new=1)
    with pytest.raises(UnsupportedArtifactError, match="something_new"):
        fse.Classifier.from_pretrained(folder)


def test_a_legacy_file_next_to_a_modern_head_is_ignored_unopened(tmp_path):
    folder, _ = _export(tmp_path)
    (folder / "pipeline.skops").write_bytes(b"pickle, never opened")
    clf = fse.Classifier.from_pretrained(folder)
    assert clf.predict_proba(TEXTS["wordpiece"]).shape == (len(TEXTS["wordpiece"]), 3)
