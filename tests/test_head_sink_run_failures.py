"""A failure between or inside the two sinks of a group run: no failed batch is delivered,
each folder stays a valid prefix of the clean run, and no folder is shorter than the last
confirmation."""

import errno

import pytest

import fse
from fse._store import _score_shard, _shard
from support import score_heads as sh
from support import sink_run as sr

TEXTS = sh.texts(90)


class BoomError(RuntimeError):
    pass


def _fail_on(obj, name, call, error=None):
    """Make the ``call``-th call of ``obj.name`` raise (counting from 1)."""
    real, count = getattr(obj, name), [0]

    def wrapper(*args, **kwargs):
        count[0] += 1
        if count[0] == call:
            raise error or BoomError("injected")
        return real(*args, **kwargs)

    setattr(obj, name, wrapper)


def _batches(heads, vectors, scores, batch, workers):
    """The batches of a run into both sinks, one by one, so that the delivered ones are
    kept when the run fails."""
    yield from heads.predict_batches(
        sr.batches_of(TEXTS, batch),
        return_vectors=True,
        vector_sink=vectors,
        score_sink=scores,
        workers=workers,
    )


def _run_failing(tmp_path, inject, *, batch=15, shard_rows=16, workers=1, expect=BoomError):
    heads = sh.group()
    reference = sr.reference(tmp_path, heads, TEXTS, batch=batch, shard_rows=shard_rows)
    vectors, scores = sr.sinks(tmp_path, heads, shard_rows=shard_rows)
    delivered: list = []

    def go():
        with vectors, scores:
            inject({"v": vectors, "s": scores})
            delivered.extend(_batches(heads, vectors, scores, batch, workers))

    with pytest.raises(expect):
        go()
    confirmed = delivered[-1].durable_rows if delivered else 0
    lengths = sr.prefix_report(tmp_path, reference)
    assert min(lengths) >= (confirmed or 0)
    for name in ("v", "s"):  # a repair keeps the prefix of each folder
        assert fse.repair_store(tmp_path / name) == lengths[0 if name == "v" else 1]
    return delivered, lengths


def test_a_vector_reservation_that_fails_after_the_score_reservation(tmp_path):
    delivered, lengths = _run_failing(tmp_path, lambda x: _fail_on(x["v"], "target", 3))
    assert len(delivered) <= 2
    assert lengths[1] <= lengths[0] + 15  # the score store is at most one task ahead


def test_a_vector_commit_that_fails_after_the_score_commit(tmp_path):
    delivered, lengths = _run_failing(tmp_path, lambda x: _fail_on(x["v"], "commit", 3))
    assert len(delivered) <= 2
    assert lengths[1] <= lengths[0] + 15


def test_a_score_commit_that_fails_leaves_the_vector_store_one_task_behind(tmp_path):
    delivered, lengths = _run_failing(tmp_path, lambda x: _fail_on(x["s"], "commit", 3))
    assert len(delivered) <= 2
    assert lengths[0] <= lengths[1] + 15


@pytest.mark.parametrize("which", ["v", "s"])
@pytest.mark.parametrize("call", [1, 2])
def test_a_publish_that_fails_in_the_first_or_second_sink(tmp_path, which, call):
    def inject(x):
        _fail_on(x[which], "publish", call + 1)

    delivered, _ = _run_failing(tmp_path, inject)
    assert len(delivered) == call


def test_a_flush_that_fails_in_the_first_sink_still_publishes_the_second_at_close(tmp_path):
    delivered, lengths = _run_failing(tmp_path, lambda x: _fail_on(x["v"], "flush", 1))
    assert len(delivered) == 6  # the batches were all delivered, then the flush failed
    assert lengths[1] == 90


def test_a_flush_that_fails_in_the_second_sink(tmp_path):
    delivered, lengths = _run_failing(tmp_path, lambda x: _fail_on(x["s"], "flush", 1))
    assert len(delivered) == 6
    assert lengths[0] == 90


def _no_space(monkeypatch, writer, after):
    real, count = writer.write, [0]

    def write(self, *args, **kwargs):
        count[0] += 1
        if count[0] > after:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(writer, "write", write)


@pytest.mark.parametrize(
    ("writer", "folder"),
    [(_shard.ShardWriter, "v"), (_score_shard.ScoreShardWriter, "s")],
)
def test_a_full_disk_in_one_folder_only(tmp_path, monkeypatch, writer, folder):
    _, lengths = _run_failing(
        tmp_path,
        lambda _: _no_space(monkeypatch, writer, 3),
        expect=OSError,
    )
    assert lengths[0 if folder == "v" else 1] < 90


@pytest.mark.parametrize("which", ["v", "s"])
def test_a_published_shard_that_is_not_durable_stops_the_run(tmp_path, which):
    error = fse.PublishedNotDurableError("injected")

    def inject(x):
        _fail_on(x[which], "publish", 2, error)

    delivered, _ = _run_failing(tmp_path, inject, expect=fse.PublishedNotDurableError)
    assert len(delivered) == 1


def test_a_keyboard_interrupt_during_the_inference_publishes_what_was_committed(tmp_path):
    heads = sh.group()
    reference = sr.reference(tmp_path, heads, TEXTS)
    vectors, scores = sr.sinks(tmp_path, heads)

    def go():
        with vectors, scores:
            for taken, _ in enumerate(_batches(heads, vectors, scores, 15, 1), start=1):
                if taken == 4:
                    raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        go()
    assert sr.prefix_report(tmp_path, reference) == (60, 60)
    assert vectors.durable_rows == scores.durable_rows == 60
