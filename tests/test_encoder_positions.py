"""Resuming a stream with encode_batches(positions=True): arguments, modes, anchors, the rows
of a resumed run and what each batch reports."""

import numpy as np
import pytest

from fse import ShardSink
from fse._store._layout import read_store
from fse._store._position_format import MODE
from support import positions as ps
from support.resume import ENCODER, sentences, write

SENTENCES = sentences(300)


def _run(path, batches, **kwargs):
    with ps.sink(path) as target:
        return list(ENCODER.encode_batches(batches, sink=target, positions=True, **kwargs))


# --- arguments: raised when the method is called ---------------------------------------


@pytest.mark.parametrize("positions", [1, "yes", None])
def test_positions_is_a_bool(positions):
    with pytest.raises(TypeError, match="positions must be a bool"):
        ENCODER.encode_batches([], positions=positions)


def test_positions_need_a_sink():
    with pytest.raises(ValueError, match="positions=True needs a sink made with resume=True"):
        ENCODER.encode_batches([], positions=True)


def test_positions_need_a_resumable_sink(tmp_path):
    with (
        ShardSink(tmp_path / "s", ENCODER) as target,
        pytest.raises(ValueError, match="positions=True needs a sink made with resume=True"),
    ):
        ENCODER.encode_batches([], sink=target, positions=True)
    assert read_store(tmp_path / "s").resume is None


def test_a_resumable_sink_takes_positions_only(tmp_path):
    with ps.sink(tmp_path / "s") as target, pytest.raises(ValueError, match="resume=True"):
        ENCODER.encode_batches([], sink=target)
    assert not (tmp_path / "s").exists()


def test_a_new_store_is_made_with_the_first_record(tmp_path):
    with ps.sink(tmp_path / "s") as target:
        batches = ENCODER.encode_batches(iter([]), sink=target, positions=True)
        assert not (tmp_path / "s").exists()
        assert target.source_position is None
        assert list(batches) == []
    assert not (tmp_path / "s").exists()


# --- batches ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item", [SENTENCES[:3], (SENTENCES[:3],), (SENTENCES[:3], "o0", "x"), [SENTENCES[:3], "o0"]]
)
def test_a_batch_is_a_pair_of_texts_and_position(tmp_path, item):
    with pytest.raises(TypeError, match=r"batch 0 must be a \(texts, position\) tuple"):
        _run(tmp_path / "s", [item])
    assert not (tmp_path / "s").exists()


def test_a_batch_with_ids_is_a_triple(tmp_path):
    with ps.sink(tmp_path / "s", doc_ids=True) as target:
        batches = ENCODER.encode_batches(
            [(SENTENCES[:3], "o0")], sink=target, positions=True, doc_ids=True
        )
        with pytest.raises(TypeError, match=r"batch 0 must be a \(texts, doc_ids, position\)"):
            list(batches)
    with ps.sink(tmp_path / "t", doc_ids=True) as target:
        batches = ENCODER.encode_batches(
            [(SENTENCES[:3], None, "o0")], sink=target, positions=True, doc_ids=True
        )
        with pytest.raises(TypeError, match="must give its doc_ids"):
            list(batches)


@pytest.mark.parametrize(
    ("position", "error", "message"),
    [
        (5, TypeError, "a position is a str"),
        (b"o0", TypeError, "a position is a str"),
        ("", ValueError, "1 to 65536 bytes"),
        ("a" * 65537, ValueError, "1 to 65536 bytes"),
        ("a\ud800", ValueError, "UTF-8"),
    ],
)
def test_a_position_is_checked_before_a_store_is_made(tmp_path, position, error, message):
    with pytest.raises(error, match=f"batch 0.*{message}"):
        _run(tmp_path / "s", [(SENTENCES[:3], position)])
    assert not (tmp_path / "s").exists()


def test_a_position_of_64_kib_is_stored_whole(tmp_path):
    for name, position in (("a", "a" * 65536), ("b", "é" * 32768)):  # ASCII; two bytes each
        _run(tmp_path / name, [(SENTENCES[:100], position)])
        assert ps.attribute(tmp_path / name, "source_position") == (position, 100)
        rows = ps.anchors(tmp_path / name)
        assert [a[2] for a in rows] == [position, position]


# --- modes -------------------------------------------------------------------------------


def test_the_store_is_of_the_mode_source_position(tmp_path):
    _run(tmp_path / "s", ps.stream(SENTENCES, 50))
    info = read_store(tmp_path / "s")
    assert info.resume.mode == MODE
    assert (info.resume.position, info.resume.row) == ("o0", 0)
    assert info.version == 2


def test_a_finite_source_does_not_resume_a_stream(tmp_path):
    _run(tmp_path / "s", ps.stream(SENTENCES, 50))
    with ps.sink(tmp_path / "s") as target:
        for call in (
            lambda: ENCODER.encode(SENTENCES, sink=target),
            lambda: ENCODER.encode(SENTENCES, sink=target, shard=(0, 1)),
        ):
            with pytest.raises(ValueError, match="mode source_position@1"):
                call()


def test_a_stream_does_not_resume_a_finite_source(tmp_path):
    write(tmp_path / "s", SENTENCES)
    with (
        ps.sink(tmp_path / "s") as target,
        pytest.raises(ValueError, match="mode source_rows@1"),
    ):
        ENCODER.encode_batches([(SENTENCES, "o0")], sink=target, positions=True)
    assert read_store(tmp_path / "s").resume.mode == "source_rows@1"


def test_a_position_store_is_opened_with_resume_only(tmp_path):
    _run(tmp_path / "s", ps.stream(SENTENCES, 50))
    with pytest.raises(ValueError, match=r"is resumable.*open it with resume=True"):
        ShardSink(tmp_path / "s", ENCODER)


def test_a_store_that_does_not_resume_is_refused(tmp_path):
    with ShardSink(tmp_path / "s", ENCODER) as target:
        ENCODER.encode(SENTENCES[:10], sink=target)
    with pytest.raises(ValueError, match="not a resumable store"):
        ps.sink(tmp_path / "s")


def test_a_sink_takes_one_run(tmp_path):
    with ps.sink(tmp_path / "s") as target:
        list(ENCODER.encode_batches(ps.stream(SENTENCES, 50), sink=target, positions=True))
        with pytest.raises(RuntimeError, match="took a source already"):
            ENCODER.encode_batches([], sink=target, positions=True)


# --- the anchors ---------------------------------------------------------------------------


def test_a_shard_takes_the_position_of_the_batch_it_ends_in(tmp_path):
    ps.write(tmp_path / "s", ps.stream(SENTENCES, 50))
    # shards of 64 rows end at 64, 128, 192, 256, 300; their last rows lie in the batches of
    # the records 50, 100, 150, 250 and 250
    assert ps.anchors(tmp_path / "s") == [
        (0, 64, "o50", 50),
        (64, 64, "o100", 100),
        (128, 64, "o150", 150),
        (192, 64, "o250", 250),
        (256, 44, "o250", 250),
    ]


def test_the_root_holds_the_position_of_the_first_record(tmp_path):
    ps.write(tmp_path / "s", ps.stream(SENTENCES, 70, start=0))
    assert read_store(tmp_path / "s").resume.position == "o0"
    ps.write(tmp_path / "t", [([], "o0"), ([], "o0"), (SENTENCES[:70], "o90")])
    assert read_store(tmp_path / "t").resume.position == "o90"  # empty batches change nothing


def test_empty_batches_change_nothing(tmp_path):
    batches = [(SENTENCES[:50], "o0"), ([], "o50"), (SENTENCES[50:100], "o50")]
    results = ps.write(tmp_path / "s", batches)
    assert [(r.first_row, r.n_rows) for r in results] == [(0, 50), (50, 0), (50, 50)]
    assert np.array_equal(ps.vectors(tmp_path / "s"), ENCODER.encode(SENTENCES[:100]))


def test_a_failed_batch_is_not_the_anchor_of_a_shard_it_filled_nothing_of(tmp_path):
    # The first batch commits 5 rows; the second fails before it commits any. The shard that
    # closes at 5 rows takes the first batch (the batch that failed starts at row 5, not before).
    bad = ["a" * 5, "\ud800"]
    with ps.sink(tmp_path / "s") as target:
        batches = ENCODER.encode_batches(
            [(SENTENCES[:5], "o0"), (bad, "o5")], sink=target, positions=True
        )
        next(batches)
        with pytest.raises(UnicodeEncodeError):
            next(batches)
    assert ps.anchors(tmp_path / "s") == [(0, 5, "o0", 0)]
    assert ps.attribute(tmp_path / "s", "source_position") == ("o0", 5)


def test_the_position_moves_forward_with_the_published_shards(tmp_path):
    with ps.sink(tmp_path / "s") as target:
        assert target.source_position is None
        batches = ENCODER.encode_batches(ps.stream(SENTENCES, 50), sink=target, positions=True)
        next(batches)
        assert target.source_position == ("o0", 0)  # the root: nothing published yet
        next(batches)  # 100 rows: the shard of 64 rows ends in the batch of the record 50
        assert target.source_position == ("o50", 14)
        list(batches)
        assert target.source_position == ("o250", 50)
    assert ps.attribute(tmp_path / "s", "source_position") == ("o250", 50)
