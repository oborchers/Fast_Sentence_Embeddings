"""The resume mode source_position@1: positions, anchors, the keys and the two modes."""

import json
from dataclasses import replace

import pytest

import fse
from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._store._layout import StoreInfo, create_store, open_shard, read_store, shard_name
from fse._store._position_format import (
    MAX_POSITION_BYTES,
    MODE,
    MODES,
    POSITION_KEYS,
    SourcePosition,
    check_position,
    parse,
    parse_resume,
    resume_keys,
    same_resume,
    skip_of,
)
from fse._store._resume_format import MODE as SOURCE_ROWS
from fse._store._resume_format import SOURCE_KEYS, SourceRows, execution_fingerprint
from support import scores as score_support
from support.container import fake_fingerprint
from support.store import add_shard, change_metadata

SHARED = {"fse.kind": "sentence_vectors", "fse.dtype": "float32"}
FILE = "store.json"


def _metadata(position="o5", row="0"):
    return {
        "fse.execution_fingerprint": execution_fingerprint(SHARED),
        "fse.source_position": position,
        "fse.source_position_row": row,
    }


def test_the_names_are_frozen():
    assert MODE == "source_position@1"
    assert POSITION_KEYS == (
        "fse.execution_fingerprint",
        "fse.source_position",
        "fse.source_position_row",
    )
    assert MODES == ("none", "source_rows@1", "source_position@1")
    assert MAX_POSITION_BYTES == 65536


@pytest.mark.parametrize("position", ["o5", "x", "ä" * 100, "a" * 65536, "{'topic': 'a'}"])
def test_a_position_is_any_text_of_up_to_64_kib(position):
    assert check_position(position) == position


@pytest.mark.parametrize("position", [b"o5", 5, None, ["o5"]])
def test_a_position_must_be_a_str(position):
    with pytest.raises(TypeError, match="a position is a str"):
        check_position(position)


@pytest.mark.parametrize("position", ["", "a" * 65537, "ä" * 32769])
def test_a_position_is_between_1_and_65536_bytes(position):
    with pytest.raises(ValueError, match="1 to 65536 bytes"):
        check_position(position)


def test_a_position_must_encode_as_utf8():
    with pytest.raises(ValueError, match="UTF-8"):
        check_position("a\ud800")


def test_the_root_keys_hold_the_execution_the_position_and_row_zero():
    anchor = SourcePosition("o5", 0)
    assert anchor.metadata(SHARED) == _metadata()
    assert anchor.anchor_metadata() == {
        "fse.source_position": "o5",
        "fse.source_position_row": "0",
    }
    assert anchor.mode == MODE


def test_parse_reads_an_anchor_and_checks_the_execution():
    assert parse(_metadata("o9", "9"), SHARED, FILE, root=False) == SourcePosition("o9", 9)
    with pytest.raises(CorruptArtifactError, match="execution fingerprint does not match"):
        parse(_metadata(), {**SHARED, "fse.dtype": "int8"}, FILE, root=False)


@pytest.mark.parametrize("row", ["01", "-1", "1.0", "", " 1", 3])
def test_the_row_is_a_canonical_decimal(row):
    with pytest.raises(CorruptArtifactError, match="canonical decimal"):
        parse(_metadata(row=row), SHARED, FILE, root=False)


@pytest.mark.parametrize("position", ["", "a" * 65537, 5])
def test_a_stored_position_is_checked(position):
    with pytest.raises(CorruptArtifactError, match=r"fse\.source_position is invalid"):
        parse(_metadata(position=position), SHARED, FILE, root=False)


def test_the_execution_fingerprint_is_checked():
    metadata = _metadata()
    metadata["fse.execution_fingerprint"] = "nope"
    with pytest.raises(CorruptArtifactError, match="execution fingerprint is invalid"):
        parse(metadata, SHARED, FILE, root=False)


def test_the_anchor_of_a_root_is_at_row_zero():
    with pytest.raises(CorruptArtifactError, match="root is at row 0"):
        parse(_metadata(row="4"), SHARED, FILE, root=True)
    assert parse(_metadata(), SHARED, FILE, root=True).row == 0


def test_the_keys_follow_the_mode():
    assert resume_keys(SOURCE_ROWS) == SOURCE_KEYS
    assert resume_keys(MODE) == POSITION_KEYS
    assert resume_keys("none") == ()
    assert resume_keys(None) == ()
    assert parse_resume("none", {}, SHARED, FILE) is None
    assert parse_resume(MODE, _metadata(), SHARED, FILE) == SourcePosition("o5", 0)


def test_a_shard_describes_the_mode_of_its_store():
    rows = SourceRows("a" * 32, "strings", 10, (0, 1), 0)
    assert same_resume(SourcePosition("o0", 0), SourcePosition("o9", 9))  # an anchor of its own
    assert not same_resume(SourcePosition("o0", 0), rows)
    assert not same_resume(rows, SourcePosition("o0", 0))
    assert same_resume(rows, rows)
    assert same_resume(None, None)
    assert not same_resume(None, SourcePosition("o0", 0))


def test_the_skip_is_the_rows_after_the_anchor():
    assert skip_of(SourcePosition("o80", 80), 100) == ("o80", 20)
    assert skip_of(SourcePosition("o0", 0), 0) == ("o0", 0)


# --- stores of the mode ---------------------------------------------------------------------

ROOT = SourcePosition("o0", 0)
INFO = StoreInfo(fake_fingerprint("a"), "float32", 3, version=2, resume=ROOT)


def _store(path, anchors=((0, "o0"), (4, "o3")), sizes=(4, 4), info=INFO):
    create_store(path, info)
    first = 0
    for rows, (row, position) in zip(sizes, anchors, strict=True):
        add_shard(path, info, first, rows)
        change_metadata(path / shard_name(first), SourcePosition(position, row).anchor_metadata())
        first += rows


def test_a_store_reads_back_with_the_anchor_of_each_shard(tmp_path):
    _store(tmp_path / "s")
    info = read_store(tmp_path / "s")
    assert info == INFO
    assert info.metadata()["fse.resume"] == MODE
    first, _ = open_shard(tmp_path / "s", info, 0)
    second, _ = open_shard(tmp_path / "s", info, 4)
    assert first.anchor == SourcePosition("o0", 0)
    assert second.anchor == SourcePosition("o3", 4)
    with fse.VectorStore.open(tmp_path / "s") as store:
        store.verify()
        assert store.source_position == ("o3", 4)


def test_the_anchor_lies_before_the_end_of_its_shard(tmp_path):
    _store(tmp_path / "s", anchors=((0, "o0"), (8, "o9")))  # row 8 is the end of the second
    with pytest.raises(CorruptArtifactError, match="anchor row lies beyond"):
        fse.VectorStore.open(tmp_path / "s")


def test_the_anchor_rows_do_not_decrease(tmp_path):
    _store(tmp_path / "s", anchors=((3, "o3"), (2, "o2")))
    with pytest.raises(CorruptArtifactError, match="anchor row 2 lies before the anchor row 3"):
        fse.VectorStore.open(tmp_path / "s")
    _store(tmp_path / "t", anchors=((3, "o3"), (3, "o3")))  # equal is allowed
    with fse.VectorStore.open(tmp_path / "t") as store:
        store.verify()


def test_a_root_whose_anchor_is_not_at_row_zero_is_corruption(tmp_path):
    create_store(tmp_path / "s", replace(INFO, resume=SourcePosition("o0", 3)))
    with pytest.raises(CorruptArtifactError, match="root is at row 0"):
        read_store(tmp_path / "s")


def test_a_shard_of_another_mode_is_corruption(tmp_path):
    _store(tmp_path / "s", sizes=(4,), anchors=((0, "o0"),))
    other = replace(INFO, resume=None, version=2)
    change_metadata(tmp_path / "s" / shard_name(0), other.metadata())
    with pytest.raises(CorruptArtifactError):
        open_shard(tmp_path / "s", INFO, 0)


def test_a_shard_with_an_edited_execution_is_corruption(tmp_path):
    _store(tmp_path / "s", sizes=(4,), anchors=((0, "o0"),))
    change_metadata(
        tmp_path / "s" / shard_name(0), {"fse.execution_fingerprint": fake_fingerprint("d")}
    )
    with pytest.raises(CorruptArtifactError, match="execution fingerprint"):
        open_shard(tmp_path / "s", INFO, 0)


def test_a_root_may_hold_a_position_of_64_kib(tmp_path):
    big = replace(INFO, resume=SourcePosition("é" * 32768, 0))  # é each in the JSON
    create_store(tmp_path / "s", big)
    assert (tmp_path / "s" / "store.json").stat().st_size > 1 << 16
    assert read_store(tmp_path / "s").resume.position == "é" * 32768
    plain = StoreInfo(fake_fingerprint("a"), "float32", 3, version=2)
    create_store(tmp_path / "t", plain)
    text = (tmp_path / "t" / "store.json").read_text()
    root = json.loads(text)
    root["metadata"]["fse.dtype"] = "float32"
    root["metadata"]["x" * 70000] = "y"
    (tmp_path / "t" / "store.json").write_text(json.dumps(root, sort_keys=True, indent=1))
    with pytest.raises(CorruptArtifactError, match="larger than 65536 bytes"):
        read_store(tmp_path / "t")


def _edit_root(path, mode):
    root = json.loads((path / "store.json").read_text())
    root["metadata"]["fse.resume"] = mode
    (path / "store.json").write_text(json.dumps(root, sort_keys=True, indent=1) + "\n")


def test_a_mode_of_the_future_is_unsupported(tmp_path):
    _store(tmp_path / "s")
    _edit_root(tmp_path / "s", "source_position@2")
    with pytest.raises(UnsupportedArtifactError, match=r"unsupported fse\.resume"):
        read_store(tmp_path / "s")


@pytest.mark.parametrize("where", ["shard", "orphan"])
def test_repair_stops_before_any_change_at_a_future_mode(tmp_path, where):
    _store(tmp_path / "s", sizes=(4, 4, 4), anchors=((0, "o0"), (1, "o1"), (2, "o2")))
    last = shard_name(8)
    if where == "orphan":
        (tmp_path / "s" / shard_name(8)).rename(tmp_path / "s" / shard_name(40))
        last = shard_name(40)
    change_metadata(tmp_path / "s" / last, {"fse.resume": "source_position@2"})
    (tmp_path / "s" / shard_name(4) / "manifest.json").write_bytes(b"{")
    before = {p.name for p in (tmp_path / "s").iterdir()}
    with pytest.raises(UnsupportedArtifactError, match=r"unsupported fse\.resume"):
        fse.repair_store(tmp_path / "s")
    assert {p.name for p in (tmp_path / "s").iterdir()} == before


def test_repair_keeps_the_good_prefix_and_the_root_anchor(tmp_path):
    _store(tmp_path / "s", sizes=(4, 4, 4), anchors=((0, "o0"), (1, "o1"), (2, "o2")))
    (tmp_path / "s" / shard_name(8) / "manifest.json").write_bytes(b"{")
    assert fse.repair_store(tmp_path / "s") == 8
    assert read_store(tmp_path / "s") == INFO
    with fse.VectorStore.open(tmp_path / "s") as store:
        assert store.source_position == ("o1", 7)


def test_a_score_store_keeps_the_thresholds_and_the_anchors(tmp_path):
    info = replace(score_support.default_info(), resume=ROOT)
    score_support.make_scores(tmp_path / "s", info, sizes=(4, 4))
    assert read_store(tmp_path / "s") == info
    assert info.metadata()["fse.resume"] == MODE
    change_metadata(tmp_path / "s" / shard_name(4), SourcePosition("o2", 2).anchor_metadata())
    with fse.ScoreStore.open(tmp_path / "s") as store:
        store.verify()
        assert store.source_position == ("o2", 6)
    change_metadata(tmp_path / "s" / shard_name(0), SourcePosition("o3", 3).anchor_metadata())
    with pytest.raises(CorruptArtifactError, match="anchor row 2 lies before the anchor row 3"):
        fse.ScoreStore.open(tmp_path / "s")
