"""The durable rows a batch of Heads.predict_batches reports with a score sink."""

import numpy as np
import pytest

import fse
from support import score_heads as sh
from support import scores as sc

STORED = {"quality": "values", "toxic": "decision", "topic": "decision"}


class Spy:
    """A sink that records the order of publish, durable_rows and flush."""

    def __init__(self, sink):
        self.sink, self.calls = sink, []

    def __getattr__(self, name):
        return getattr(self.sink, name)

    def publish(self):
        self.calls.append("publish")
        return self.sink.publish()

    def flush(self):
        self.calls.append("flush")
        return self.sink.flush()

    @property
    def durable_rows(self):
        self.calls.append("durable_rows")
        return self.sink.durable_rows

    def target(self, first_row, n_rows):
        return self.sink.target(first_row, n_rows)

    def commit(self, *args, **kwargs):
        return self.sink.commit(*args, **kwargs)


def _batches(texts, size):
    return [texts[i : i + size] for i in range(0, len(texts), size)]


def test_the_counter_is_run_relative_monotonic_and_ends_at_the_sink_total(tmp_path):
    heads, texts = sh.group(), sh.texts(50)
    with fse.ScoreSink(tmp_path / "s", heads, shard_rows=16) as sink:
        got = list(heads.predict_batches(_batches(texts, 10), score_sink=sink, workers=2))
        durable = [b.durable_rows for b in got]
        assert durable == sorted(durable)
        assert all(0 <= d <= b.first_row + b.n_rows for d, b in zip(durable, got, strict=True))
        assert durable[0] == 0  # the first shard of 16 rows is not complete after 10 rows
        assert durable[-1] <= 48
        assert sink.durable_rows == 50  # the flush after the last batch


def test_a_second_run_counts_its_own_rows_not_those_of_the_store(tmp_path):
    heads, texts = sh.group(), sh.texts(20)
    with fse.ScoreSink(tmp_path / "s", heads, shard_rows=8) as sink:
        list(heads.predict_batches(_batches(texts, 10), score_sink=sink))
    with fse.ScoreSink(tmp_path / "s", heads, shard_rows=8) as sink:
        assert sink.first_row == 20
        got = list(heads.predict_batches(_batches(texts, 10), score_sink=sink))
        assert max(b.durable_rows for b in got) <= 20
        assert sink.durable_rows == 20
    assert sc.read_back(tmp_path / "s")[1].shape == (40, 7)


def test_publish_runs_before_each_value_and_flush_after_the_last_batch(tmp_path):
    heads, texts = sh.group(), sh.texts(20)
    with fse.ScoreSink(tmp_path / "s", heads, shard_rows=8) as inner:
        sink = Spy(inner)
        list(heads.predict_batches(_batches(texts, 10), score_sink=sink))
    assert sink.calls == ["publish", "durable_rows", "publish", "durable_rows", "flush"]


def test_stopping_early_does_not_flush_and_close_publishes_what_was_committed(tmp_path):
    heads, texts = sh.group(), sh.texts(30)
    with fse.ScoreSink(tmp_path / "s", heads, shard_rows=16) as inner:
        sink = Spy(inner)
        first = next(heads.predict_batches(_batches(texts, 10), score_sink=sink))
        assert first.n_rows == 10
    assert "flush" not in sink.calls
    assert sc.read_back(tmp_path / "s")[1].shape[0] >= 10


def test_none_reports_zero_durable_rows(tmp_path):
    heads, texts = sh.group(), sh.texts(20)
    with fse.ScoreSink(tmp_path / "s", heads, shard_rows=8, durability="none") as sink:
        got = list(heads.predict_batches(_batches(texts, 10), score_sink=sink))
    assert [b.durable_rows for b in got] == [0, 0]
    assert sink.published_rows == 20


def test_without_a_sink_the_field_is_none():
    heads = sh.group()
    (batch,) = heads.predict_batches([sh.texts(3)])
    assert batch.durable_rows is None


def test_empty_batches_report_the_rows_so_far(tmp_path):
    heads = sh.group()
    with fse.ScoreSink(tmp_path / "s", heads, shard_rows=4) as sink:
        got = list(heads.predict_batches([[], sh.texts(8), []], score_sink=sink))
    assert [b.n_rows for b in got] == [0, 8, 0]
    assert got[-1].durable_rows <= 8


def test_a_sink_with_a_wrong_durable_rows_is_refused(tmp_path):
    heads = sh.group()

    class Liar(Spy):
        @property
        def durable_rows(self):
            return 10**6

    with fse.ScoreSink(tmp_path / "s", heads, shard_rows=4) as inner:
        sink = Liar(inner)
        with pytest.raises(ValueError, match="durable_rows of the sink is"):
            list(heads.predict_batches([sh.texts(8)], score_sink=sink))


def test_the_ids_and_the_counter_work_together_with_the_arrow_free_lists(tmp_path):
    heads, texts = sh.group(), sh.texts(24)
    ids = np.arange(24)
    batches = [(texts[i : i + 8], ids[i : i + 8]) for i in range(0, 24, 8)]
    with fse.ScoreSink(tmp_path / "s", heads, doc_ids=True, shard_rows=8) as sink:
        got = list(heads.predict_batches(batches, score_sink=sink, doc_ids=True))
    assert got[-1].durable_rows <= 24
    assert sink.durable_rows == 24
