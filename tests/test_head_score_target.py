"""A group of heads into a score sink: the bits, the stored outputs, the batches, the
document IDs and the durable rows."""

import numpy as np
import pytest

import fse
from support import score_heads as sh
from support import scores as sc

F32 = np.float32


def bits(x):
    return np.ascontiguousarray(x, dtype=F32).view(np.uint32)


def _write(path, heads, texts, *, store=None, batch=15, workers=2, output=None, **kwargs):
    batches = [texts[i : i + batch] for i in range(0, len(texts), batch)]
    with fse.ScoreSink(path, heads, store=store, shard_rows=16) as sink:
        got = list(
            heads.predict_batches(
                batches, score_sink=sink, output=output, workers=workers, **kwargs
            )
        )
    return sink, got


def _reference(heads, texts, outputs, **kwargs):
    return heads.predict(texts, output=outputs, workers=3, **kwargs)


def test_the_stored_values_and_logits_are_the_bits_of_the_same_run_without_a_sink(tmp_path):
    heads, texts = sh.group(), sh.texts(40)
    _write(tmp_path / "s", heads, texts)
    _, scores, _ = sc.read_back(tmp_path / "s")
    ref = _reference(heads, texts, STORED)
    np.testing.assert_array_equal(
        bits(scores[:, 0:1]), bits(np.asarray(ref["quality"]).reshape(40, 1))
    )
    np.testing.assert_array_equal(bits(scores[:, 1:4]), bits(ref["toxic"]))
    np.testing.assert_array_equal(bits(scores[:, 4:7]), bits(ref["topic"]))


def test_stored_probabilities_are_those_of_the_run_without_a_sink(tmp_path):
    heads, texts = sh.group(), sh.texts(40)
    _write(tmp_path / "s", heads, texts, store={"toxic": "proba", "topic": "proba"})
    _, scores, _ = sc.read_back(tmp_path / "s")
    ref = _reference(heads, texts, {"toxic": "proba", "topic": "proba"})
    np.testing.assert_array_equal(bits(scores[:, 1:4]), bits(ref["toxic"]))
    np.testing.assert_array_equal(bits(scores[:, 4:7]), bits(ref["topic"]))


STORED = {"quality": "values", "toxic": "decision", "topic": "decision"}


@pytest.mark.parametrize("output", [None, STORED, {"toxic": "proba", "topic": "labels"}])
def test_what_a_batch_returns_is_independent_of_what_is_stored(tmp_path, output):
    heads, texts = sh.group(), sh.texts(40)
    _, got = _write(tmp_path / "s", heads, texts, output=output)
    ref = _reference(heads, texts, output)
    for name in heads:
        joined = np.concatenate([b.outputs[name] for b in got])
        if joined.dtype.kind == "f":
            np.testing.assert_array_equal(bits(joined), bits(ref[name]))
        else:
            np.testing.assert_array_equal(joined, ref[name])
    _, scores, _ = sc.read_back(tmp_path / "s")  # still the decisions and values
    stored = _reference(heads, texts, STORED)
    np.testing.assert_array_equal(bits(scores[:, 4:7]), bits(stored["topic"]))


def test_storing_what_is_returned_adds_no_second_output():
    from fse._head._score_target import ScoreRun

    with fse.ScoreSink("unused", sh.group()) as sink:  # nothing is created by the constructor
        same = ScoreRun(sink, ("logits", "logits", "logits"))
        assert same.extras == [None, None, None]
        assert all(tee is not None for tee in same.tees)
        other = ScoreRun(sink, ("labels", "proba", "logits"))
        assert [e[0] for e in other.extras[:2]] == ["logits", "logits"]
        assert other.tees[:2] == [None, None]
        assert other.tees[2] is not None


@pytest.mark.parametrize("vectors", [False, True])
def test_each_path_is_stored_and_matches_a_run_without_a_sink(tmp_path, vectors):
    heads, texts = sh.group(), sh.texts(30)
    _, got = _write(tmp_path / "s", heads, texts, return_vectors=vectors)
    info, scores, _ = sc.read_back(tmp_path / "s")
    paths = {h.name: h.path for h in info.heads}
    assert paths == dict(heads.paths(return_vectors=vectors))
    ref = _reference(heads, texts, STORED)
    if vectors:  # the heads on raw means ran unfolded: the bits differ from the folded run
        again = heads.predict_batches([texts], return_vectors=True, output=STORED, workers=3)
        (batch,) = list(again)
        np.testing.assert_array_equal(bits(scores[:, 4:7]), bits(batch.outputs["topic"]))
        assert all(b.vectors is not None for b in got)
    else:
        np.testing.assert_array_equal(bits(scores[:, 4:7]), bits(ref["topic"]))


def test_a_run_over_a_second_path_is_refused_before_the_first_text(tmp_path):
    heads = sh.group()
    _write(tmp_path / "s", heads, sh.texts(10))

    def explode():
        yield sh.texts(3)
        raise AssertionError("a text was read")

    with (
        fse.ScoreSink(tmp_path / "s", heads) as sink,
        pytest.raises(ValueError, match="'unfolded'"),
    ):
        heads.predict_batches(explode(), score_sink=sink, return_vectors=True)


def test_a_standalone_group_has_the_path_standalone(tmp_path):
    enc = sh.encoder()
    sh.classifier(enc, 3, seed=2).save(tmp_path / "a.safetensors", standalone=True)
    loaded = fse.Classifier.load(tmp_path / "a.safetensors")
    heads = fse.Heads(topic=loaded)
    _write(tmp_path / "s", heads, sh.texts(12))
    info, _, _ = sc.read_back(tmp_path / "s")
    assert [h.path for h in info.heads] == ["standalone"]


def test_a_known_stream_and_a_stream_of_unknown_length_store_the_same_rows(tmp_path):
    heads, texts = sh.group(), sh.texts(37)
    _write(tmp_path / "a", heads, texts, batch=37, workers=1)
    _write(tmp_path / "b", heads, texts, batch=5, workers=3)
    np.testing.assert_array_equal(
        bits(sc.read_back(tmp_path / "a")[1]), bits(sc.read_back(tmp_path / "b")[1])
    )


def test_an_empty_batch_stores_nothing_and_keeps_the_rows_straight(tmp_path):
    heads, texts = sh.group(), sh.texts(6)
    with fse.ScoreSink(tmp_path / "s", heads) as sink:
        got = list(heads.predict_batches([[], texts, []], score_sink=sink))
    assert [(b.first_row, b.n_rows) for b in got] == [(0, 0), (0, 6), (6, 0)]
    assert sc.read_back(tmp_path / "s")[1].shape == (6, 7)


def test_document_ids_follow_the_texts(tmp_path):
    heads, texts = sh.group(), sh.texts(30)
    ids = np.arange(30, dtype=np.int64) * 3 + 7
    batches = [(texts[:11], ids[:11]), (texts[11:], list(ids[11:]))]
    with fse.ScoreSink(tmp_path / "s", heads, doc_ids=True, shard_rows=16) as sink:
        list(heads.predict_batches(batches, score_sink=sink, doc_ids=True, workers=2))
    _, _, stored = sc.read_back(tmp_path / "s")
    assert stored.tolist() == ids.tolist()


def test_the_id_mode_must_match_the_sink_before_a_text_is_read(tmp_path):
    heads = sh.group()
    with (
        fse.ScoreSink(tmp_path / "a", heads, doc_ids=True) as sink,
        pytest.raises(ValueError, match="doc_ids=True is needed exactly"),
    ):
        heads.predict_batches(iter(()), score_sink=sink)
    with (
        fse.ScoreSink(tmp_path / "b", heads) as sink,
        pytest.raises(ValueError, match="doc_ids=True is needed exactly"),
    ):
        heads.predict_batches(iter(()), score_sink=sink, doc_ids=True)
    with pytest.raises(ValueError, match="needs a score_sink"):
        heads.predict_batches(iter(()), doc_ids=True)


def test_wrong_document_ids_are_refused(tmp_path):
    heads, texts = sh.group(), sh.texts(6)
    with (
        fse.ScoreSink(tmp_path / "a", heads, doc_ids=True) as sink,
        pytest.raises(TypeError, match="batch 0 must be a"),
    ):
        list(heads.predict_batches([texts], score_sink=sink, doc_ids=True))
    with (
        fse.ScoreSink(tmp_path / "b", heads, doc_ids=True) as sink,
        pytest.raises(ValueError, match="doc_ids has 5 rows for 6 texts"),
    ):
        list(heads.predict_batches([(texts, np.arange(5))], score_sink=sink, doc_ids=True))


def test_a_sink_takes_one_run(tmp_path):
    heads = sh.group()
    with fse.ScoreSink(tmp_path / "s", heads) as sink:
        list(heads.predict_batches([sh.texts(3)], score_sink=sink))
        with pytest.raises(RuntimeError, match="takes one run"):
            heads.predict_batches([sh.texts(3)], score_sink=sink)


def test_missing_document_ids_are_refused_before_the_text_is_processed(tmp_path):
    heads = sh.group()
    with (
        fse.ScoreSink(tmp_path / "s", heads, doc_ids=True) as sink,
        pytest.raises(TypeError, match="doc_ids, not None"),
    ):
        list(heads.predict_batches([([], None)], score_sink=sink, doc_ids=True))


def test_the_ids_of_a_task_are_fetched_once_and_shared_by_its_users():
    from fse._head._score_target import TaskIds

    ids = TaskIds(2)
    texts = ["a b", "c d", "e f", "g h"]
    with ids.batch(texts, 10, [5, 6, 7, 8]):
        first = ids.block(10, 2)
        assert ids.block(10, 2) is first
        assert first.tolist() == [5, 6] and not first.flags.writeable
        second = ids.block(12, 2)
        assert second.tolist() == [7, 8]
        ids.release(10)
        assert ids.block(10, 2) is first  # one user is still to commit
        ids.release(10)
        ids.release(12)
        ids.release(12)
        ids.release(12)  # a surplus release is harmless
        assert not ids._open


def test_a_run_without_ids_has_none():
    from fse._head._score_target import TaskIds

    ids = TaskIds(1)
    with ids.batch(["a"], 0, None):
        assert ids.block(0, 1) is None
    ids.release(0)


def test_the_ids_of_a_batch_are_checked_against_its_texts():
    from fse._head._score_target import TaskIds

    ids = TaskIds(1)
    with (
        pytest.raises(ValueError, match="doc_ids has 3 rows for 2 texts"),
        ids.batch(["a", "b"], 0, [1, 2, 3]),
    ):
        pass
    with (
        pytest.raises(ValueError, match="more rows than the text stream"),
        ids.batch(["a", "b"], 0, [1, 2]),
    ):
        ids.block(0, 1)  # the task of the second text is never opened


def test_a_score_run_fetches_one_id_block_per_task(tmp_path, monkeypatch):
    from fse._text._ids import IdSource

    calls = []
    original = IdSource.block

    def counting(self, begin, end):
        calls.append((begin, end))
        return original(self, begin, end)

    monkeypatch.setattr(IdSource, "block", counting)
    heads, texts = sh.group(), sh.texts(30)
    ids = np.arange(30, dtype=np.int64)
    with fse.ScoreSink(tmp_path / "s", heads, doc_ids=True, shard_rows=16) as sink:
        list(heads.predict_batches([(texts, ids)], score_sink=sink, doc_ids=True, workers=2))
    assert len(calls) == len(set(calls)) >= 1  # no task asks twice, three heads or not
    assert sorted(calls)[0][0] == 0 and max(end for _, end in calls) == 30
