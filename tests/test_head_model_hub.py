"""The extracted Hub methods retain each model's normal artifact schema."""

import json

import numpy as np
import pytest

import fse
from fse._container import CorruptArtifactError
from fse._head._model_hub import HubModel
from support.encoder import finished


def test_hub_mixin_requires_a_concrete_artifact_writer():
    with pytest.raises(TypeError, match="abstract"):
        HubModel()


def test_multilabel_hub_folder_round_trip(tmp_path):
    encoder = finished(normalize=False)
    model = fse.Classifier.from_arrays(
        encoder,
        np.ones((2, encoder.output_dim), np.float32),
        labels=["sport", "news"],
        link="multilabel",
        thresholds={"sport": 0.4},
    )
    folder = model.save_pretrained(tmp_path / "classifier", license="mit")
    config = json.loads((folder / "config.json").read_text())
    assert config["fse_format_version"] == 2
    loaded = fse.Classifier.from_pretrained(folder)
    assert loaded.fingerprint == model.fingerprint
    np.testing.assert_array_equal(loaded.predict(["w1", ""]), model.predict(["w1", ""]))


@pytest.mark.parametrize("multilabel", [False, True])
def test_hub_config_rejects_both_format_version_mismatches(tmp_path, multilabel):
    encoder = finished(normalize=False)
    model = fse.Classifier.from_arrays(
        encoder,
        np.zeros((2, encoder.output_dim), np.float32),
        labels=["sport", "news"],
        link="multilabel" if multilabel else "softmax",
    )
    folder = model.save_pretrained(tmp_path / "classifier")
    path = folder / "config.json"
    config = json.loads(path.read_text())
    config["fse_format_version"] = 1 if multilabel else 2
    path.write_text(json.dumps(config))
    with pytest.raises(CorruptArtifactError, match="names format version.*file holds version"):
        fse.Classifier.from_pretrained(folder)
