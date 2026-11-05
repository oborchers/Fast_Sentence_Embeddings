"""Classifier heads of trained Model2Vec classifiers (fse._head._model2vec_objects): a
StaticModelForClassification trained for a few epochs on a tiny data set, converted by its
own to_pipeline, with a fine-tuned table and learned token weights.

These tests need torch (group ``st``) and skip without it, unless FSE_REQUIRE_ST is set
(``make test-st`` and the CI job test-st), when a missing package fails them.
"""

import os

import numpy as np
import pytest

import fse
from fse._contract import UnsupportedArtifactError
from support import model2vec_head as mh

if os.environ.get("FSE_REQUIRE_ST"):
    import torch  # noqa: F401
else:
    pytest.importorskip("torch")
pytest.importorskip("model2vec")

SENTENCES = mh.texts("wordpiece", count=120)
TRAIN = SENTENCES[:100]


def _trained(labels, *, n_layers=0, normalize=False, freeze_weights=False):
    from model2vec.train import StaticModelForClassification

    model = StaticModelForClassification.from_static_model(
        model=mh.static_model(normalize=normalize),
        n_layers=n_layers,
        hidden_dim=8,
        normalize=normalize,
        freeze_weights=freeze_weights,
    )
    model.fit(
        TRAIN,
        labels,
        max_epochs=3,
        min_epochs=1,
        batch_size=16,
        device="cpu",
        test_size=0.2,
        early_stopping_patience=None,
    )
    return model


def _single():
    return [("x", "y", "z")[len(t) % 3] for t in TRAIN]


def _multi():
    return [[("p", "q", "r")[len(t) % 3], "s"] if len(t) % 2 else ["p"] for t in TRAIN]


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("labels", [_single, _multi], ids=["single", "multi"])
def test_a_trained_classifier_is_imported(labels, normalize):
    model = _trained(labels(), normalize=normalize)
    pipe = model.to_pipeline()
    clf = fse.Classifier.from_model2vec(model)
    assert clf.link == ("multilabel@1" if model.multilabel else "softmax@1")
    assert list(clf.classes_) == list(model.classes_)
    assert clf.normalized is normalize
    # The fine-tuned table, with the learned token weights multiplied in, bit for bit.
    table = np.asarray(clf.encoder.word_vectors.stored_vectors)
    assert table.tobytes() == pipe.model.embedding.tobytes()
    assert not np.array_equal(table, mh.static_model().embedding)
    assert pipe.model.weights is None
    mh.assert_matches(clf, pipe, SENTENCES)


def test_the_object_pipeline_and_export_of_a_trained_model_agree(tmp_path):
    model = _trained(_single())
    pipe = model.to_pipeline()
    pipe.save_pretrained(str(tmp_path / "export"))
    texts = SENTENCES
    from_model = fse.Classifier.from_model2vec(model)
    from_pipe = fse.Classifier.from_model2vec(pipe)
    from_folder = fse.Classifier.from_pretrained(tmp_path / "export")
    for other in (from_pipe, from_folder):
        assert other.coef_.tobytes() == from_model.coef_.tobytes()
        assert other.predict_proba(texts).tobytes() == from_model.predict_proba(texts).tobytes()


def test_a_trained_head_with_hidden_layers_is_refused_as_object_and_export(tmp_path):
    model = _trained(_single(), n_layers=1)
    with pytest.raises(UnsupportedArtifactError, match=r"n_layers=1.*n_layers=0"):
        fse.Classifier.from_model2vec(model)
    pipe = model.to_pipeline()
    with pytest.raises(UnsupportedArtifactError, match="hidden layers with ReLU"):
        fse.Classifier.from_model2vec(pipe)
    pipe.save_pretrained(str(tmp_path / "mlp"))
    with pytest.raises(UnsupportedArtifactError, match="hidden layers with ReLU"):
        fse.Classifier.from_pretrained(tmp_path / "mlp")
