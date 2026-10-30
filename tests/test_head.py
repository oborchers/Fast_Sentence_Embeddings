"""The package of linear heads and its public names."""

import fse
import fse._head


def test_the_public_names():
    for name in ("Classifier", "Regressor", "PredictedBatch"):
        assert getattr(fse, name) is getattr(fse._head, name)
        assert name in fse.__all__
    assert fse._head.__all__ == ["Classifier", "PredictedBatch", "Regressor"]
