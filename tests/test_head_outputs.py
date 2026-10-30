"""The outputs of a head: PredictedBatch and the identity of outputs."""

import pytest

from fse._container import is_fingerprint
from fse._head._outputs import SINK_OUTPUTS, PredictedBatch, output_identity

MODEL = "xxh3-128:" + "0" * 32


def test_output_identities_differ_by_output_and_model():
    found = {output_identity(MODEL, output) for output in SINK_OUTPUTS}
    assert len(found) == 3
    assert all(is_fingerprint(f) for f in found)
    assert output_identity("xxh3-128:" + "1" * 32, "proba") not in found
    assert output_identity(MODEL, "proba") == output_identity(MODEL, "proba")


def test_labels_have_no_identity():
    with pytest.raises(ValueError, match="output must be one of decision, proba, values"):
        output_identity(MODEL, "labels")


def test_predicted_batch():
    batch = PredictedBatch(10, 2, [1, 2], None, 12)
    assert (batch.first_row, batch.n_rows, batch.outputs, batch.vectors, batch.durable_rows) == (
        10,
        2,
        [1, 2],
        None,
        12,
    )
