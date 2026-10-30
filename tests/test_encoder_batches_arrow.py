"""Batches of Arrow columns in encode_batches and predict_batches: their rows come from the
source, never from len() of the batch (exporters without __len__ or with a wrong one), also
when a producer cannot export and is read as a stream. No pyarrow is needed."""

import numpy as np
import pytest

import fse
from support.encoder import finished
from support.runtime import ArrowChunks, texts

ENCODER = finished()
SENTENCES = texts(300)
WANT = ENCODER.encode(SENTENCES)


class Unsized(ArrowChunks):
    """An Arrow column without __len__."""

    def __init__(self, sentences):
        super().__init__(sentences, b"vu", 40)


class Missized(Unsized):
    """An Arrow column whose __len__ is wrong."""

    def __len__(self):
        return 1


class CannotExport:
    """A producer that cannot export (pandas without pyarrow), iterable, without __len__."""

    def __init__(self, sentences):
        self.sentences = sentences

    def __arrow_c_stream__(self, requested_schema=None):
        raise ImportError("pyarrow is not installed")

    def __iter__(self):
        return iter(self.sentences)


class CannotExportMissized(CannotExport):
    def __len__(self):
        return 7


def batches_of(kind):
    return [kind(SENTENCES[:120]), kind(SENTENCES[120:121]), kind(SENTENCES[121:])]


KINDS = [Unsized, Missized, CannotExport, CannotExportMissized]


@pytest.mark.parametrize("kind", KINDS)
def test_batches_count_the_rows_of_their_source(kind):
    got = list(ENCODER.encode_batches(batches_of(kind), workers=2))
    assert [(b.first_row, b.n_rows) for b in got] == [(0, 120), (120, 1), (121, 179)]
    rows = np.concatenate([b.rows for b in got])
    assert rows.view(np.uint32).tobytes() == WANT.view(np.uint32).tobytes()


@pytest.mark.parametrize("kind", KINDS)
def test_a_sink_confirms_the_rows_of_their_source(tmp_path, kind):
    with fse.ShardSink(tmp_path / "store", ENCODER, shard_rows=50) as sink:
        got = list(ENCODER.encode_batches(batches_of(kind), sink=sink))
    assert [(b.first_row, b.n_rows) for b in got] == [(0, 120), (120, 1), (121, 179)]
    assert all(b.durable_rows <= b.first_row + b.n_rows for b in got)
    stored = fse.VectorStore.open(tmp_path / "store").get_vectors()
    assert stored.view(np.uint32).tobytes() == WANT.view(np.uint32).tobytes()


@pytest.mark.parametrize("kind", KINDS)
def test_heads_count_the_rows_of_their_source(kind):
    rng = np.random.default_rng(0)
    model = fse.Classifier.from_arrays(
        ENCODER,
        rng.standard_normal((3, ENCODER.output_dim)).astype(np.float32),
        np.zeros(3, np.float32),
        labels=["a", "b", "c"],
    )
    got = list(model.predict_batches(batches_of(kind), output="proba"))
    assert [(b.first_row, b.n_rows) for b in got] == [(0, 120), (120, 1), (121, 179)]
    want = model.predict_proba(SENTENCES)
    assert np.concatenate([b.outputs for b in got]).tobytes() == want.tobytes()
