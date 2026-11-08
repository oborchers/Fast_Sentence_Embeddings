"""A group run into a vector sink and a score sink: the checks before a text is read, the
sharing of the document IDs and the rows that are durable in both."""

import numpy as np
import pytest

import fse
from fse._head._sink_run import SinkRun
from support import score_heads as sh
from support import sink_run as sr
from support.encoder import finished


def _call(heads, sink_kwargs, **kwargs):
    kwargs.setdefault("return_vectors", True)
    return heads.predict_batches(iter(()), **sink_kwargs, **kwargs)


# --- the checks ---------------------------------------------------------------------


def test_the_vectors_need_return_vectors_and_nothing_is_reserved(tmp_path):
    heads = sh.group()
    vectors, scores = sr.sinks(tmp_path, heads)
    with vectors, scores:
        with pytest.raises(ValueError, match=r"pass return_vectors=True.*unfolded"):
            _call(heads, {"vector_sink": vectors, "score_sink": scores}, return_vectors=False)
        assert vectors._reserved == 0
        assert not (tmp_path / "s").exists()


def test_a_score_sink_is_no_vector_sink(tmp_path):
    heads = sh.group()
    with (
        fse.ScoreSink(tmp_path / "s", heads) as scores,
        pytest.raises(ValueError, match=r"stores the scores of heads.*pass it as score_sink="),
    ):
        _call(heads, {"vector_sink": scores})


def test_a_vector_sink_follows_the_sink_protocol():
    heads = sh.group()
    with pytest.raises(TypeError, match="target and commit"):
        _call(heads, {"vector_sink": object()})
    with pytest.raises(ValueError, match="must take rows of 6 values"):
        _call(heads, {"vector_sink": fse.MemorySink(3)})


def test_a_vector_sink_of_another_encoder_is_refused(tmp_path):
    heads = sh.group()
    other = finished("float16", fasttext=True, normalize=False, seed=5)
    with (
        fse.ShardSink(tmp_path / "v", other) as sink,
        pytest.raises(ValueError, match="takes the rows of encoder"),
    ):
        _call(heads, {"vector_sink": sink})


def test_a_used_or_closed_shard_sink_is_refused_before_a_text_is_read(tmp_path):
    heads = sh.group()
    sink = fse.ShardSink(tmp_path / "v", heads["quality"].encoder, shard_rows=16)
    sink.target(0, 2)
    with pytest.raises(ValueError, match="took 2 rows of a run already"):
        _call(heads, {"vector_sink": sink})
    sink.close()
    with pytest.raises(RuntimeError, match="is closed"):
        _call(heads, {"vector_sink": sink})


def test_the_id_mode_must_match_each_sink(tmp_path):
    heads = sh.group()
    vectors, scores = sr.sinks(tmp_path, heads, doc_ids=True)
    with vectors, scores:
        for given in ({"vector_sink": vectors}, {"score_sink": scores}):
            with pytest.raises(ValueError, match="stores doc_ids; pass doc_ids=True"):
                _call(heads, given)
        name = "score_sink"
        with (
            _plain(tmp_path, heads) as plain,
            pytest.raises(ValueError, match=f"needs the {name} to have doc_ids"),
        ):
            _call(heads, {"vector_sink": vectors, "score_sink": plain}, doc_ids=True)
    with pytest.raises(ValueError, match="needs a vector_sink or a score_sink with doc_ids"):
        _call(heads, {}, doc_ids=True)


def _plain(tmp_path, heads):
    return fse.ScoreSink(tmp_path / "plain", heads)


def test_a_has_doc_ids_that_is_no_bool_is_a_type_error_with_ids_and_ignored_without():
    heads = sh.group()
    for value in (1, "yes"):
        sink = fse.MemorySink(heads["quality"].encoder.output_dim)
        sink.has_doc_ids = value
        with pytest.raises(TypeError, match="has_doc_ids of the vector_sink must be True"):
            _call(heads, {"vector_sink": sink}, doc_ids=True)
        run = _call(heads, {"vector_sink": sink})  # 2.0 behaviour: ignored without IDs
        assert list(run) == []


def test_nested_folders_are_refused(tmp_path):
    heads = sh.group()
    vectors = fse.ShardSink(tmp_path / "v", heads["quality"].encoder)
    scores = fse.ScoreSink(tmp_path / "v" / "s", heads)
    with vectors, scores:
        with pytest.raises(ValueError, match="two folders apart"):
            _call(heads, {"vector_sink": vectors, "score_sink": scores})
        assert vectors._reserved == 0


def test_a_rejected_call_leaves_no_score_store_and_reserves_no_row(tmp_path):
    heads = sh.group()
    vectors, scores = sr.sinks(tmp_path, heads, doc_ids=True)
    with vectors, scores:
        with pytest.raises(ValueError, match="stores doc_ids; pass doc_ids=True"):
            _call(heads, {"vector_sink": vectors, "score_sink": scores})
        assert not (tmp_path / "s").exists()
        assert vectors._reserved == 0


# --- stores that hold rows already ------------------------------------------------------


def _fill(tmp_path, heads, name, n, *, doc_ids=False):
    """A vector store and a score store of ``n`` rows each, written by a run."""
    texts = sh.texts(n)
    ids = np.arange(n, dtype=np.int64) if doc_ids else None
    sr.run(tmp_path, heads, texts, doc_ids=doc_ids, ids=ids, names=(f"v{name}", f"s{name}"))


def test_unequal_lengths_are_refused_without_ids_and_named_in_the_message(tmp_path):
    heads = sh.group()
    sr.run(tmp_path, heads, sh.texts(20), names=("v", "other"))
    _fill_scores_only(tmp_path, heads, "s", 12)
    vectors = fse.ShardSink(tmp_path / "v", heads["quality"].encoder, shard_rows=16)
    scores = fse.ScoreSink(tmp_path / "s", heads, shard_rows=16)
    with vectors, scores:
        with pytest.raises(ValueError, match=r"holds 20 rows.*score_sink 12"):
            _call(heads, {"vector_sink": vectors, "score_sink": scores})
        assert vectors._reserved == 0


def _fill_scores_only(tmp_path, heads, name, n, *, doc_ids=False):
    texts = sh.texts(n)
    with fse.ScoreSink(tmp_path / name, heads, doc_ids=doc_ids, shard_rows=16) as sink:
        batches = [(texts, np.arange(n, dtype=np.int64))] if doc_ids else [texts]
        list(heads.predict_batches(batches, score_sink=sink, doc_ids=doc_ids, return_vectors=True))


def test_unequal_lengths_are_taken_with_ids(tmp_path):
    heads = sh.group()
    sr.run(tmp_path, heads, sh.texts(20), doc_ids=True, ids=np.arange(20), names=("v", "other"))
    _fill_scores_only(tmp_path, heads, "s", 12, doc_ids=True)
    texts = sh.texts(10)
    ids = np.arange(100, 110, dtype=np.int64)
    vectors = fse.ShardSink(tmp_path / "v", heads["quality"].encoder, doc_ids=True, shard_rows=16)
    scores = fse.ScoreSink(tmp_path / "s", heads, doc_ids=True, shard_rows=16)
    with vectors, scores:
        list(
            heads.predict_batches(
                [(texts, ids)],
                return_vectors=True,
                vector_sink=vectors,
                score_sink=scores,
                doc_ids=True,
            )
        )
    assert (vectors.first_row, scores.first_row) == (20, 12)
    assert sr.read_vectors(tmp_path / "v")[1][20:].tolist() == ids.tolist()
    assert sr.read_scores(tmp_path / "s")[1][12:].tolist() == ids.tolist()


def test_a_writer_that_beats_the_attach_is_caught_after_the_attach(tmp_path):
    heads = sh.group()
    sr.run(tmp_path, heads, sh.texts(20), names=("v", "other"))
    scores = fse.ScoreSink(tmp_path / "s", heads, shard_rows=16)  # no store yet: first_row 0
    _fill_scores_only(tmp_path, heads, "s", 12)  # another writer is faster
    vectors = fse.ShardSink(tmp_path / "v", heads["quality"].encoder, shard_rows=16)
    with vectors, scores:
        with pytest.raises(ValueError, match=r"holds 20 rows.*score_sink 12"):
            _call(heads, {"vector_sink": vectors, "score_sink": scores})
        assert scores.first_row == 12
        assert vectors._reserved == 0


# --- other sinks ------------------------------------------------------------------------


def test_any_sink_after_the_protocol_takes_the_vectors(tmp_path):
    heads, texts = sh.group(), sh.texts(30)
    memory = fse.MemorySink(heads["quality"].encoder.output_dim)
    with fse.ScoreSink(tmp_path / "s", heads, shard_rows=16) as scores:
        got = list(
            heads.predict_batches(
                sr.batches_of(texts, 12),
                return_vectors=True,
                vector_sink=memory,
                score_sink=scores,
            )
        )
    assert [b.durable_rows for b in got] == [None] * 3  # a MemorySink promises nothing
    assert memory.result().shape == (30, 6)
    assert all(b.vectors is None for b in got)


def test_a_vector_sink_without_a_score_sink_gives_outputs_and_durable_rows(tmp_path):
    heads, texts = sh.group(), sh.texts(30)
    with fse.ShardSink(tmp_path / "v", heads["quality"].encoder, shard_rows=8) as sink:
        got = list(
            heads.predict_batches(sr.batches_of(texts, 10), return_vectors=True, vector_sink=sink)
        )
        assert sink.flush() == 30
    assert got[-1].durable_rows <= 30
    assert sink.durable_rows == 30
    assert set(got[0].outputs) == {"quality", "toxic", "topic"}
    assert sr.read_vectors(tmp_path / "v")[0].shape == (30, 6)


# --- the minimum ----------------------------------------------------------------------


class _Counts:
    """A sink that reports scripted durable rows and counts its publishes."""

    def __init__(self, *rows):
        self.script, self.publishes = list(rows), 0
        self.durable_rows = None

    def publish(self):
        self.publishes += 1
        self.durable_rows = self.script.pop(0)


def _run_of(*sinks):
    run = SinkRun.__new__(SinkRun)
    run.sinks, run._last = list(sinks), [None] * len(sinks)
    return run


def test_the_durable_rows_are_the_minimum_and_both_sinks_publish_every_time():
    first, second = _Counts(None, None), _Counts(3, 5)
    run = _run_of(first, second)
    assert run.durable(6) is None
    assert run.durable(6) is None
    assert (first.publishes, second.publishes) == (2, 2)


def test_the_minimum_follows_the_sink_that_lags():
    run = _run_of(_Counts(2, 6, 10), _Counts(4, 4, 12))
    assert [run.durable(n) for n in (6, 8, 12)] == [2, 4, 10]


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        (-1, "must be between"),
        (9, "must be between"),
        (True, "must be an int"),
        (1.5, "must be an int"),
    ],
)
def test_every_count_is_checked_on_its_own(rows, match):
    for position in (0, 1):
        counts = [_Counts(2), _Counts(2)]
        counts[position] = _Counts(rows)
        with pytest.raises((ValueError, TypeError), match=match):
            _run_of(*counts).durable(8)


def test_a_count_that_falls_is_refused_even_above_the_minimum():
    run = _run_of(_Counts(1, 1), _Counts(8, 6))
    assert run.durable(8) == 1
    with pytest.raises(ValueError, match="between 8"):
        run.durable(8)


def test_the_flush_goes_to_the_vector_sink_first_and_to_both():
    order = []
    first, second = _Counts(), _Counts()
    first.flush = lambda: order.append("vectors")
    second.flush = lambda: order.append("scores")
    _run_of(first, second).flush()
    assert order == ["vectors", "scores"]


@pytest.mark.parametrize(("vector_rows", "score_rows"), [(7, 64), (64, 7)])
def test_the_limit_changes_with_the_grid_of_the_sinks(tmp_path, vector_rows, score_rows):
    heads, texts = sh.group(), sh.texts(130)
    vectors = fse.ShardSink(tmp_path / "v", heads["quality"].encoder, shard_rows=vector_rows)
    scores = fse.ScoreSink(tmp_path / "s", heads, shard_rows=score_rows)
    seen = []
    with vectors, scores:
        for batch in heads.predict_batches(
            sr.batches_of(texts, 13),
            return_vectors=True,
            vector_sink=vectors,
            score_sink=scores,
            workers=1,
        ):
            vectors.publish(), scores.publish()
            assert batch.durable_rows <= min(vectors.durable_rows, scores.durable_rows)
            seen.append(batch.durable_rows)
    assert seen == sorted(seen)
    assert min(vectors.durable_rows, scores.durable_rows) == 130


def test_a_sink_without_durability_confirms_nothing(tmp_path):
    heads, texts = sh.group(), sh.texts(30)
    got, *_ = sr.run(tmp_path, heads, texts, durability="none")
    assert [b.durable_rows for b in got] == [0, 0]


# --- the document IDs ------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["list", "int64", "arrow"])
def test_both_stores_hold_the_ids_of_the_input(tmp_path, kind):
    heads, texts = sh.group(), sh.texts(30)
    ids = np.arange(30, dtype=np.int64) * 5 + 3
    given = {"list": ids.tolist(), "int64": ids}.get(kind)
    if kind == "arrow":
        pa = pytest.importorskip("pyarrow")
        given = pa.array(ids)
    *_, vector_ids, _, score_ids = sr.run(tmp_path, heads, texts, doc_ids=True, ids=given)
    assert vector_ids.tolist() == ids.tolist()
    assert score_ids.tolist() == ids.tolist()


def test_the_two_sinks_get_the_same_id_array_and_one_fetch_per_task(tmp_path, monkeypatch):
    from fse._text._ids import IdSource

    calls, seen = [], []
    original = IdSource.block

    def counting(self, begin, end):
        calls.append((begin, end))
        return original(self, begin, end)

    monkeypatch.setattr(IdSource, "block", counting)
    heads, texts = sh.group(), sh.texts(40)
    vectors, scores = sr.sinks(tmp_path, heads, doc_ids=True)
    for sink in (vectors, scores):
        commit = sink.commit
        sink.commit = lambda first, rows, *, doc_ids=None, commit=commit: (
            seen.append((first, id(doc_ids), doc_ids.flags.writeable)),
            commit(first, rows, doc_ids=doc_ids),
        )[1]
    with vectors, scores:
        list(
            heads.predict_batches(
                sr.batches_of(texts, 17, np.arange(40, dtype=np.int64)),
                return_vectors=True,
                vector_sink=vectors,
                score_sink=scores,
                doc_ids=True,
                workers=2,
            )
        )
    by_row = {}
    for first, identity, writeable in seen:
        by_row.setdefault(first, set()).add(identity)
        assert not writeable
    assert all(len(identities) == 1 for identities in by_row.values())
    assert len(seen) == 2 * len(by_row)
    assert len(calls) == len(by_row)  # one fetch per task, not one per sink


def test_a_vector_sink_alone_checks_the_number_of_ids(tmp_path):
    heads, texts = sh.group(), sh.texts(20)
    with (
        fse.ShardSink(tmp_path / "v", heads["quality"].encoder, doc_ids=True, shard_rows=8) as sink,
        pytest.raises(ValueError, match="doc_ids has 19 rows for 20 texts"),
    ):
        list(
            heads.predict_batches(
                [(texts, np.arange(19))], return_vectors=True, vector_sink=sink, doc_ids=True
            )
        )


def test_a_vector_sink_alone_stores_the_ids_and_releases_every_block(tmp_path):
    heads, texts = sh.group(), sh.texts(40)
    ids = np.arange(40, dtype=np.int64) * 2 + 5
    with fse.ShardSink(
        tmp_path / "v", heads["quality"].encoder, doc_ids=True, shard_rows=8
    ) as sink:
        run = heads.predict_batches(
            sr.batches_of(texts, 13, ids),
            return_vectors=True,
            vector_sink=sink,
            doc_ids=True,
            workers=2,
        )
        list(run)
    vectors, stored = sr.read_vectors(tmp_path / "v")
    assert stored.tolist() == ids.tolist()
    assert vectors.shape == (40, 6)
