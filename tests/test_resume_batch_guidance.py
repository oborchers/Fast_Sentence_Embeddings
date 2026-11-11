"""Resume-sink errors point to the working batch API before reading the source."""

import numpy as np
import pytest

import fse
from support.resume_heads import Poisoned


@pytest.mark.parametrize("kind", ["encoder", "classifier", "regressor", "heads"])
@pytest.mark.parametrize("sink_kind", ["vector", "score"])
def test_resume_batch_guidance(tmp_path, kind, sink_kind):
    if kind == "encoder" and sink_kind == "score":
        pytest.skip("encoders produce vectors only")
    encoder = fse.Encoder.from_arrays(np.eye(2, dtype=np.float32), words=["a", "b"])
    classifier = fse.Classifier.from_arrays(encoder, np.eye(2, dtype=np.float32), labels=["a", "b"])
    regressor = fse.Regressor.from_arrays(encoder, np.ones((1, 2), dtype=np.float32))
    model = {
        "encoder": encoder,
        "classifier": classifier,
        "regressor": regressor,
        "heads": fse.Heads(c=classifier, r=regressor),
    }[kind]
    method = model.encode_batches if kind == "encoder" else model.predict_batches
    if sink_kind == "vector":
        sink = fse.ShardSink(tmp_path / "store", encoder, resume=True)
        options = (
            {"sink": sink} if kind == "encoder" else {"vector_sink": sink, "return_vectors": True}
        )
    else:
        sink = fse.ScoreSink(
            tmp_path / "store", model, name=None if kind == "heads" else "head", resume=True
        )
        options = {"score_sink": sink}

    with sink:
        with pytest.raises(ValueError, match="positions=True"):
            list(method(Poisoned(), **options))
        assert not (tmp_path / "store").exists()
        results = list(method([(["a", "b"], "offset0")], positions=True, **options))
        assert len(results) == 1
        assert results[0].n_rows == 2
