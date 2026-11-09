"""The extracted prediction mixins preserve validation titles and instance layout."""

from collections.abc import Mapping

import numpy as np
import pytest
from pydantic import ValidationError

import fse
from fse._head._model import LinearModel
from support.encoder import finished


def test_moved_methods_preserve_validation_titles_and_slots():
    encoder = finished(normalize=False)
    classifier = fse.Classifier.from_arrays(
        encoder, np.ones((2, encoder.output_dim)), labels=[0, 1]
    )
    regressor = fse.Regressor.from_arrays(encoder, np.ones(encoder.output_dim))
    heads = fse.Heads(c=classifier, r=regressor)
    cases = [
        (classifier.predict_proba, "_Predicting.predict_proba"),
        (classifier.decision_function, "_Predicting.decision_function"),
        (regressor.predict, "Regressor.predict"),
        (regressor.predict_batches, "Regressor.predict_batches"),
        (heads.predict, "Heads.predict"),
        (heads.predict_batches, "Heads.predict_batches"),
    ]
    for method, title in cases:
        assert method.__qualname__ == title
        with pytest.raises(ValidationError) as caught:
            method([], workers="1")
        assert caught.value.title == title
    for model in (classifier, regressor):
        assert type(model).__dictoffset__ == LinearModel.__dictoffset__
        assert type(model).__weakrefoffset__ == LinearModel.__weakrefoffset__
    assert type(heads).__dictoffset__ == Mapping.__dictoffset__
    assert type(heads).__weakrefoffset__ == Mapping.__weakrefoffset__
