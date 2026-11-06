"""``ScoreSink(resume=True)``: what the constructor does and does not do, how it attaches to a
source, and the sync of the store directory before it counts a shard as kept."""

from pathlib import Path

import numpy as np
import pytest

import fse
from fse import ScoreSink
from fse._container import PublishedNotDurableError, _npy_inplace
from fse._store import _resume_format as sync_module
from support import score_heads as sh
from support.resume import canonical, drop_tail
from support.resume_heads import HEADS, SHARD_ROWS, run, score_sink, scores_of, sentences

SENTENCES = sentences(320)


def test_the_constructor_makes_nothing_and_a_sink_never_attached_leaves_nothing(tmp_path):
    with score_sink(tmp_path / "s") as sink:
        assert sink.resume is True
        assert sink.first_row == 0
        assert sink.published_rows == sink.durable_rows == 0
    assert not (tmp_path / "s").exists()


def test_a_sink_without_resume_has_resume_false(tmp_path):
    with ScoreSink(tmp_path / "s", HEADS) as sink:
        assert sink.resume is False


def test_a_resume_sink_takes_no_rows_before_a_source(tmp_path):
    with (
        score_sink(tmp_path / "s") as sink,
        pytest.raises(RuntimeError, match="takes rows only from predict_batches"),
    ):
        sink.target(0, 4)


def test_the_constructor_checks_an_existing_store_before_a_source_comes(tmp_path):
    run(tmp_path, SENTENCES, vectors=False)
    with score_sink(tmp_path / "s", doc_ids=False) as sink:
        assert sink.first_row == len(SENTENCES)  # the rows it will keep, before they are verified
    with pytest.raises(ValueError, match="doc_ids"):
        score_sink(tmp_path / "s", doc_ids=True)
    other = sh.group()
    with pytest.raises(ValueError, match="holds"):
        score_sink(tmp_path / "s", heads=fse.Heads(topic=other["topic"]))


def test_the_thresholds_must_be_those_of_the_store(tmp_path):
    run(tmp_path, SENTENCES, vectors=False)
    enc = HEADS["quality"].encoder
    toxic = sh.classifier(
        enc, 3, link="multilabel", labels=["a", "b", "c"], thresholds=[0.1, 0.5, 0.7]
    )
    changed = fse.Heads(quality=HEADS["quality"], toxic=toxic, topic=HEADS["topic"])
    with pytest.raises(ValueError, match="thresholds"):
        score_sink(tmp_path / "s", heads=changed)


def test_resume_under_full_refuses_a_tagged_prefix(tmp_path):
    run(tmp_path, SENTENCES, vectors=False, durability="none")
    with pytest.raises(ValueError, match="durability='none'"):
        score_sink(tmp_path / "s", durability="full")


def test_resume_under_none_verifies_tagged_shards(tmp_path):
    run(tmp_path, SENTENCES, vectors=False, durability="none")
    shard = next((tmp_path / "s").glob(f"shard-*{SHARD_ROWS}"))
    data = bytearray((shard / "scores.npy").read_bytes())
    data[-1] ^= 0xFF
    (shard / "scores.npy").write_bytes(bytes(data))
    with pytest.raises(fse.CorruptArtifactError):
        run(tmp_path, SENTENCES, vectors=False, durability="none")


def test_the_store_that_resumed_opens_and_verifies(tmp_path):
    run(tmp_path, SENTENCES, vectors=False)
    drop_tail(tmp_path / "s", 2)
    run(tmp_path, SENTENCES, vectors=False)
    with fse.ScoreStore.open(tmp_path / "s") as store:
        assert len(store) == len(SENTENCES)
        store.verify()


def test_a_complete_store_is_left_alone(tmp_path):
    run(tmp_path, SENTENCES, vectors=False)
    before = canonical(tmp_path / "s")
    run(tmp_path, SENTENCES, vectors=False)
    assert canonical(tmp_path / "s") == before


# --- the sync of the store directory when a resume attaches ------------------------------


class Events:
    """The syncs of the store directory and the shards made, in order."""

    def __init__(self, monkeypatch):
        self.log = []
        real_sync, real_writer = sync_module._sync_directory, ScoreSink._new_writer

        def sync(path, **kwargs):
            self.log.append(f"sync {Path(path).name}")
            real_sync(path, **kwargs)

        def writer(sink, row):
            self.log.append("writer")
            return real_writer(sink, row)

        monkeypatch.setattr(sync_module, "_sync_directory", sync)
        monkeypatch.setattr(ScoreSink, "_new_writer", writer)


def test_a_resume_under_full_syncs_the_store_once_before_it_reserves(tmp_path, monkeypatch):
    run(tmp_path, SENTENCES, vectors=False)
    drop_tail(tmp_path / "s", 3)
    events = Events(monkeypatch)
    run(tmp_path, SENTENCES, vectors=False)
    assert events.log[:3] == ["sync s", f"sync {tmp_path.name}", "writer"]  # the store, its parent
    assert events.log.count("sync s") == 1


def test_a_resume_under_none_does_not_sync(tmp_path, monkeypatch):
    run(tmp_path, SENTENCES, vectors=False, durability="none")
    drop_tail(tmp_path / "s", 3)
    events = Events(monkeypatch)
    run(tmp_path, SENTENCES, vectors=False, durability="none")
    assert "sync s" not in events.log


def test_a_failed_sync_stops_the_resume_before_any_reservation(tmp_path, monkeypatch):
    run(tmp_path, SENTENCES, vectors=False)
    drop_tail(tmp_path / "s", 3)
    before = canonical(tmp_path / "s")
    events = Events(monkeypatch)

    def failing(*_args, **_kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(sync_module, "_sync_directory", failing)
    with pytest.raises(PublishedNotDurableError, match="not confirmed durable"):
        run(tmp_path, SENTENCES, vectors=False)
    assert "writer" not in events.log
    assert canonical(tmp_path / "s") == before


def test_a_rename_that_was_visible_but_not_synced_is_confirmed_by_the_resume(tmp_path, monkeypatch):
    """The power-loss model: the process ended after the rename of a shard and before the sync
    of its directory. The next run attaches, syncs the directory and only then counts the shard
    as kept."""
    store = tmp_path / "s"
    real = _npy_inplace._sync_directory

    def failing_once(path, **kwargs):
        if Path(path) == store and not hasattr(failing_once, "failed"):
            failing_once.failed = True
            raise OSError(5, "Input/output error")
        real(path, **kwargs)

    monkeypatch.setattr(_npy_inplace, "_sync_directory", failing_once)
    with pytest.raises(PublishedNotDurableError):
        run(tmp_path, SENTENCES, vectors=False)
    monkeypatch.undo()
    assert (store / "shard-0000000000000000").is_dir()  # visible, untagged, not confirmed
    events = Events(monkeypatch)
    run(tmp_path, SENTENCES, vectors=False)
    assert events.log[0] == "sync s"
    scores, _ = scores_of(store)
    assert len(scores) == len(SENTENCES)
    np.testing.assert_array_equal(
        scores.view(np.uint32), _whole_scores(tmp_path / "whole").view(np.uint32)
    )


def _whole_scores(path):
    run(path, SENTENCES, vectors=False)
    return scores_of(path / "s")[0]


def test_a_resume_syncs_the_parent_of_a_store_an_earlier_run_made(tmp_path, monkeypatch):
    run(tmp_path, SENTENCES, vectors=False)
    drop_tail(tmp_path / "s", 3)
    seen = []
    real = sync_module._sync_directory

    def failing_parent(path, **kwargs):
        seen.append(Path(path))
        if Path(path) == tmp_path:
            raise OSError(5, "Input/output error")
        real(path, **kwargs)

    monkeypatch.setattr(sync_module, "_sync_directory", failing_parent)
    with pytest.raises(PublishedNotDurableError, match="not confirmed durable"):
        run(tmp_path, SENTENCES, vectors=False)
    assert seen == [tmp_path / "s", tmp_path]
