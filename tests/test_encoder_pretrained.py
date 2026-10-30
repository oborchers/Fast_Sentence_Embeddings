"""Encoder.from_pretrained for local files and folders: what a model is, and refusals."""

import json
import os
import shutil

import numpy as np
import pytest

from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._encoder import Encoder
from fse._encoder._format import Source
from fse._encoder._pretrained import card_license
from support import model2vec as mv

pytest.importorskip("model2vec")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return mv.save_model(tmp_path_factory.mktemp("m") / "model")


def _copy(model, tmp_path, name="copy"):
    return shutil.copytree(model, tmp_path / name)


def test_a_model2vec_folder_is_converted_with_its_provenance(model, tmp_path):
    encoder = Encoder.from_pretrained(model)
    assert encoder.tokenizer == "wordpiece@1"
    assert encoder._content.source == Source("model2vec", None, None)
    assert encoder._content.license == "mit"
    assert Encoder.from_pretrained(str(model)).fingerprint == encoder.fingerprint
    encoder.save(tmp_path / "e.safetensors")
    loaded = Encoder.from_pretrained(tmp_path / "e.safetensors")
    assert loaded.fingerprint == encoder.fingerprint
    texts = ["hello world", "ab [MASK]"]
    assert loaded.encode(texts).tobytes() == encoder.encode(texts).tobytes()


def test_an_fse_model_names_its_encoder_file(model, tmp_path):
    folder = tmp_path / "fse"
    folder.mkdir()
    encoder = Encoder.from_pretrained(model)
    encoder.save(folder / "encoder.safetensors")
    config = {
        "model_type": "fse",
        "fse_kind": "encoder",
        "fse_format_version": 1,
        "encoder": "encoder.safetensors",
    }
    (folder / "config.json").write_text(json.dumps(config))
    for mmap in (False, True):
        assert Encoder.from_pretrained(folder, mmap=mmap).fingerprint == encoder.fingerprint
    with pytest.raises(ValueError, match="holds its length rule"):
        Encoder.from_pretrained(folder, truncate=False)
    with pytest.raises(ValueError, match="holds its length rule"):
        Encoder.from_pretrained(folder / "encoder.safetensors", max_length=3)
    cases = [
        ({**config, "extra": 1}, UnsupportedArtifactError, "holds exactly"),
        ({**config, "fse_format_version": 2}, UnsupportedArtifactError, "format version 2"),
        ({**config, "fse_kind": "word_vectors"}, UnsupportedArtifactError, "'word_vectors'"),
        (
            {**config, "encoder": "../encoder.safetensors"},
            CorruptArtifactError,
            "must name a .safetensors file",
        ),
        (
            {**config, "encoder": "other.safetensors"},
            CorruptArtifactError,
            "no file other.safetensors",
        ),
    ]
    for value, error, match in cases:
        (folder / "config.json").write_text(json.dumps(value))
        with pytest.raises(error, match=match):
            Encoder.from_pretrained(folder)


def test_models_that_are_no_static_embeddings_are_refused(model, tmp_path):
    bert = tmp_path / "bert"
    bert.mkdir()
    (bert / "config.json").write_text('{"model_type": "bert"}')
    with pytest.raises(
        UnsupportedArtifactError, match=r"model_type 'bert'.* no static embedding model"
    ):
        Encoder.from_pretrained(bert)
    modules = [
        {"type": "sentence_transformers.models.Transformer"},
        {"type": "sentence_transformers.models.Pooling"},
    ]
    (bert / "modules.json").write_text(json.dumps(modules))
    with pytest.raises(UnsupportedArtifactError, match=r"its modules are .*Transformer.*Pooling"):
        Encoder.from_pretrained(bert)
    st = tmp_path / "st"
    (st / "0_StaticEmbedding").mkdir(parents=True)
    (st / "config_sentence_transformers.json").write_text("{}")
    with pytest.raises(UnsupportedArtifactError, match="StaticEmbedding is not supported yet"):
        Encoder.from_pretrained(st)
    (st / "config.json").write_text('{"model_type": "x"}')
    with pytest.raises(UnsupportedArtifactError, match="StaticEmbedding is not supported yet"):
        Encoder.from_pretrained(st)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(UnsupportedArtifactError, match=r"no config\.json"):
        Encoder.from_pretrained(empty)
    lacking = _copy(model, tmp_path, "lacking")
    (lacking / "tokenizer.json").unlink()
    with pytest.raises(UnsupportedArtifactError, match=r"a Model2Vec model has config\.json"):
        Encoder.from_pretrained(lacking)
    (lacking / "modules.json").write_text("{}")
    with pytest.raises(CorruptArtifactError, match="must be a list of modules"):
        Encoder.from_pretrained(lacking)


def test_the_arguments_are_checked(model, tmp_path):
    with pytest.raises(ValueError, match="mmap is for fse encoders"):
        Encoder.from_pretrained(model, mmap=True)
    with pytest.raises(ValueError, match="are for the Hub"):
        Encoder.from_pretrained(model, revision="main")
    with pytest.raises(ValueError, match="are for the Hub"):
        Encoder.from_pretrained(model, local_files_only=True)
    with pytest.raises(FileNotFoundError, match="no local file or folder"):
        Encoder.from_pretrained(tmp_path / "none")
    (tmp_path / "x.txt").write_text("x")
    with pytest.raises(ValueError, match=r"no \.safetensors file"):
        Encoder.from_pretrained(tmp_path / "x.txt")
    with pytest.raises(ValueError, match="validation error"):
        Encoder.from_pretrained(model, max_length=0)
    (model / "config.json").stat()  # untouched


def _cache_layout(model, tmp_path):
    """The layout of a Hugging Face cache: snapshot files are links into blobs."""
    repo = tmp_path / "models--x--y"
    (repo / "blobs").mkdir(parents=True)
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    for k, file in enumerate(sorted(model.iterdir())):
        blob = repo / "blobs" / f"blob{k}"
        shutil.copy(file, blob)
        (snapshot / file.name).symlink_to(os.path.relpath(blob, snapshot))
    return repo, snapshot


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symbolic links")
def test_a_cache_snapshot_follows_its_links_into_the_blobs(model, tmp_path):
    try:
        repo, snapshot = _cache_layout(model, tmp_path)
    except OSError:
        pytest.skip("symbolic links are not allowed here")
    assert (
        Encoder.from_pretrained(snapshot).fingerprint == Encoder.from_pretrained(model).fingerprint
    )
    outside = tmp_path / "outside.json"
    shutil.copy(model / "config.json", outside)
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").symlink_to(outside)
    with pytest.raises(CorruptArtifactError, match="outside the blobs"):
        Encoder.from_pretrained(snapshot)
    (snapshot / "config.json").unlink()
    (repo / "blobs" / "dir").mkdir()
    (snapshot / "config.json").symlink_to(os.path.relpath(repo / "blobs" / "dir", snapshot))
    with pytest.raises(CorruptArtifactError, match="no regular file"):
        Encoder.from_pretrained(snapshot)


@pytest.mark.parametrize(
    ("card", "want"),
    [
        ("---\nlicense: mit\nlibrary_name: model2vec\n---\n# x", "mit"),
        ("---\nlibrary_name: model2vec\nlicense: 'apache-2.0'\n---\n", "apache-2.0"),
        ("---\nlibrary_name: x\n---\nlicense: mit\n", None),
        ("# no front matter\nlicense: mit\n", None),
        ("", None),
    ],
)
def test_the_license_of_a_model_card(tmp_path, card, want):
    assert card_license(card.encode(), tmp_path / "README.md") == want
    assert card_license(None, tmp_path / "README.md") is None


def test_a_license_of_another_form_raises(tmp_path):
    with pytest.raises(CorruptArtifactError, match="states a license fse cannot read"):
        card_license(b"---\nlicense: {a: b}\n---\n", tmp_path / "README.md")


def test_the_rows_are_finite_numbers(model):
    rows = Encoder.from_pretrained(model).encode(["hello", ""])
    assert np.isfinite(rows).all()
    assert not rows[1].any()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_a_fifo_is_no_file_of_a_model(model, tmp_path):
    """A FIFO in place of a file raises at once instead of blocking a read."""
    folder = _copy(model, tmp_path, "fifo")
    for name in ("config.json", "tokenizer.json", "README.md"):
        original = folder / name
        saved = original.read_bytes()
        original.unlink()
        os.mkfifo(original)
        with pytest.raises(CorruptArtifactError, match="not a regular file"):
            Encoder.from_pretrained(folder)
        original.unlink()
        original.write_bytes(saved)


def test_the_size_limits_hold(model, tmp_path):
    import fse._encoder._pretrained as pretrained

    folder = _copy(model, tmp_path, "big")
    with open(folder / "README.md", "ab") as f:
        f.truncate(pretrained.MAX_CARD_BYTES + 1)
    with pytest.raises(CorruptArtifactError, match="more than the 16777216 it may have"):
        Encoder.from_pretrained(folder)


def _card(front):
    return ("---\nlibrary_name: model2vec\n" + front + "\ntags:\n- x\n---\n# card\n").encode()


@pytest.mark.parametrize(
    ("front", "want"),
    [
        ("license: mit", "mit"),
        ('license: "apache-2.0"  # the code', "apache-2.0"),
        ("license: mit # comment", "mit"),
        ("license: null", None),
        ("license: ~", None),
        ("license:", None),
        ("license:\n- mit\n- apache-2.0", ("mit", "apache-2.0")),
        ("license:\n  - 'cc-by-4.0'", ("cc-by-4.0",)),
        ("license: [mit, 'apache-2.0']", ("mit", "apache-2.0")),
        ("license:\n- mit\n# Kommentar\n- apache-2.0", ("mit", "apache-2.0")),
        ("license:\n\n  # the licenses\n  - mit\n\n  - other\n", ("mit", "other")),
        ("license:\n# none\n", None),
    ],
)
def test_the_yaml_rules_of_a_license(tmp_path, front, want):
    assert card_license(_card(front), tmp_path / "README.md") == want


@pytest.mark.parametrize(
    ("front", "match"),
    [
        ("license: true", "no string"),
        ("license: 123", "no string"),
        ("license: no", "no string"),
        ("license: {a: b}", "cannot read"),
        ("license: []", "list fse cannot read"),
        ("license: [mit,]", "no string"),
        ("license:\n- mit\n- null", "no string"),
        ("license: mit\nlicense: other", "license 2 times"),
        ("license: mit\n- other", "a value and a list"),
        ("license: 'mit", "cannot read"),
    ],
)
def test_a_license_of_another_form_raises_naming_it(tmp_path, front, match):
    with pytest.raises(CorruptArtifactError, match=match):
        card_license(_card(front), tmp_path / "README.md")


def test_a_card_without_front_matter_has_no_license(tmp_path):
    assert card_license(b"---\nlicense: mit\n", tmp_path / "README.md") is None  # unclosed
    assert card_license(b"license: mit\n", tmp_path / "README.md") is None


def test_several_licenses_reach_the_encoder(model, tmp_path):
    folder = _copy(model, tmp_path, "licenses")
    (folder / "README.md").write_bytes(_card("license:\n- mit\n- apache-2.0"))
    encoder = Encoder.from_pretrained(folder)
    assert encoder._content.license == ("mit", "apache-2.0")
    encoder.save(tmp_path / "e.safetensors")
    assert Encoder.load(tmp_path / "e.safetensors")._content.license == ("mit", "apache-2.0")


def test_a_classifier_on_a_converted_encoder(model, tmp_path):
    """A linear head on an encoder of from_pretrained: its labels are those of the rows,
    and its file keeps the tokenizer keys of wordpiece@1 through save and load."""
    import fse

    encoder = Encoder.from_pretrained(model)
    rng = np.random.default_rng(5)
    weight = rng.standard_normal((3, encoder.output_dim)).astype(np.float32)
    bias = rng.standard_normal(3).astype(np.float32)
    head = fse.Classifier.from_arrays(encoder, weight, bias, labels=["a", "b", "c"])
    texts = ["hello world", "the cat sat", "ab [MASK] x", "internationalization"]
    logits = encoder.encode(texts).astype(np.float64) @ weight.T + bias
    assert head.predict(texts).tolist() == [["a", "b", "c"][i] for i in logits.argmax(axis=1)]
    head.save(tmp_path / "head.safetensors")
    loaded = fse.Classifier.load(tmp_path / "head.safetensors")
    assert loaded.encoder.tokenizer == "wordpiece@1"
    assert loaded.encoder._content.tokenizer == encoder._content.tokenizer
    assert loaded.predict_proba(texts).tobytes() == head.predict_proba(texts).tobytes()
