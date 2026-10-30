"""The model card of a Hugging Face folder: its front matter as YAML readers parse it, the
license it states, and the facts of the model file it describes."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import fse
from fse._container import safetensors_header
from fse._container._common import _open_regular
from fse._encoder import _card
from fse._encoder._format import Source
from fse._encoder._pretrained import card_license
from support.encoder import finished

FINISHED = Path(__file__).parent / "test_data" / "v2" / "finished"
huggingface_hub = pytest.importorskip("huggingface_hub")


def _header(path):
    with _open_regular(path) as handle:
        _, entries, metadata = safetensors_header(handle, path)
    return dict(metadata), {name: tuple(spec[1]) for name, spec in entries.items()}


def _card_of(path, tmp_path, **options):
    metadata, shapes = _header(path)
    options = {"license": "mit", "language": None, "model_name": "m", "location": "a/b"} | options
    data = _card.card(metadata, shapes, **options)
    (tmp_path / "README.md").write_bytes(data)
    return data, huggingface_hub.metadata_load(tmp_path / "README.md")


@pytest.mark.parametrize(
    ("name", "task", "tag"),
    [
        ("encoder_float32", "feature-extraction", "sentence-embeddings"),
        ("encoder_int8", "feature-extraction", "sentence-embeddings"),
        ("classifier_float32", "text-classification", "text-classification"),
        ("classifier_standalone", "text-classification", "text-classification"),
        ("regressor_int8", None, "regression"),
    ],
)
def test_the_front_matter_parses_as_yaml(name, task, tag, tmp_path):
    data, meta = _card_of(FINISHED / f"{name}.safetensors", tmp_path, language=["en", "de"])
    want = {"library_name": "fse", "license": "mit", "language": ["en", "de"]}
    want["tags"] = ["fse", "static-embeddings", tag]
    if task is not None:
        want["pipeline_tag"] = task
    assert meta == want
    assert card_license(data, tmp_path / "README.md") == "mit"
    assert _card_of(FINISHED / f"{name}.safetensors", tmp_path, language=["en", "de"])[0] == data


def test_values_that_yaml_would_read_as_bools_or_numbers_stay_strings(tmp_path):
    path = FINISHED / "encoder_float32.safetensors"
    for license in ("no", "1.0", "on", "null"):
        data, meta = _card_of(path, tmp_path, license=license)
        assert meta["license"] == license
        assert card_license(data, tmp_path / "README.md") == license
    data, meta = _card_of(path, tmp_path, license=("mit", "apache-2.0"))
    assert meta["license"] == ["mit", "apache-2.0"]
    assert card_license(data, tmp_path / "README.md") == ("mit", "apache-2.0")
    data, meta = _card_of(path, tmp_path, license=None)
    assert "license" not in meta
    assert card_license(data, tmp_path / "README.md") is None
    assert "The model records no license." in data.decode()


def test_the_license_of_a_source_is_kept_and_never_replaced():
    recorded = {"fse.license": "mit"}
    assert _card.resolve_license(recorded, None) == "mit"
    assert _card.resolve_license(recorded, "mit") == "mit"
    assert _card.resolve_license(recorded, ["mit"]) == "mit"
    with pytest.raises(ValueError, match="records the license 'mit' of its source"):
        _card.resolve_license(recorded, "apache-2.0")
    listed = {"fse.license": json.dumps(["mit", "cc-by-4.0"])}
    assert _card.resolve_license(listed, None) == ("mit", "cc-by-4.0")
    assert _card.resolve_license(listed, ["mit", "cc-by-4.0"]) == ("mit", "cc-by-4.0")
    assert _card.resolve_license({}, None) is None
    assert _card.resolve_license({}, "apache-2.0") == "apache-2.0"
    for bad in ("MIT License", "", "a:b", []):
        with pytest.raises(ValueError, match="license id"):
            _card.resolve_license({}, bad)


def test_names_and_languages_are_checked(tmp_path):
    path = FINISHED / "encoder_float32.safetensors"
    for name in ("", "  ", "two\nlines"):
        with pytest.raises(ValueError, match="model_name must be"):
            _card_of(path, tmp_path, model_name=name)
    for code in ("english", "EN", "e"):
        with pytest.raises(ValueError, match="no language code"):
            _card_of(path, tmp_path, language=[code])


def test_the_text_states_the_facts_of_the_file(tmp_path):
    data, _ = _card_of(FINISHED / "classifier_float32.safetensors", tmp_path, model_name="clf")
    text = data.decode()
    metadata, _ = _header(FINISHED / "classifier_float32.safetensors")
    assert text.count("\n# clf\n") == 1
    assert f"fingerprint: `{metadata['fse.fingerprint']}`" in text
    assert 'model = fse.Classifier.from_pretrained("a/b")' in text
    assert "- link: softmax@1" in text
    assert "normalized to unit length" not in text
    data, _ = _card_of(FINISHED / "regressor_int8.safetensors", tmp_path)
    assert "fse.Regressor.from_pretrained" in data.decode()
    assert "(int8)" in data.decode()


def test_a_converted_model_names_its_source_and_its_differences(tmp_path):
    encoder = finished()
    content = dataclasses.replace(
        encoder._content, source=Source("model2vec", "minishlab/potion-base-8M", "abc")
    )
    converted = type(encoder)._new(content, encoder.word_vectors)
    converted.save(tmp_path / "m.safetensors")
    data, meta = _card_of(tmp_path / "m.safetensors", tmp_path)
    assert meta["base_model"] == "minishlab/potion-base-8M"
    text = data.decode()
    assert "Converted from the model2vec model minishlab/potion-base-8M at revision `abc`." in text
    assert "drops unknown token ids" in text
    assert "The license of the source model applies" in text
    local = type(encoder)._new(
        dataclasses.replace(content, source=Source("sentence-transformers")), encoder.word_vectors
    )
    local.save(tmp_path / "l.safetensors")
    data, meta = _card_of(tmp_path / "l.safetensors", tmp_path)
    assert "base_model" not in meta
    assert "from the sentence-transformers model a local folder" in data.decode()
    assert "`[UNK]` included" in data.decode()


def test_many_labels_are_cut_in_the_text(tmp_path):
    encoder = finished(normalize=False)
    labels = [f"l{i}" for i in range(25)]
    clf = fse.Classifier.from_arrays(encoder, np.ones((25, 6)), labels=labels)
    clf.save(tmp_path / "c.safetensors")
    data, _ = _card_of(tmp_path / "c.safetensors", tmp_path)
    assert "- labels: l0, l1," in data.decode()
    assert "l19 and 5 more" in data.decode()
