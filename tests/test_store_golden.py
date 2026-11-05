"""The bytes, fingerprints, log events and error classes of the vector store, pinned.

``tests/test_data/v2/store_golden.json`` was written once, by
``tests/test_data/v2/make_store_golden.py``, on the code before the shared shard base
(DR-162, Scores und Vorhersagen dauerhaft speichern). Every later build must write the same
bytes, and refuse the same stores with the same classes. The goldens are never rewritten.
"""

import json
import re
from pathlib import Path

import pytest

import fse
from fse import ShardSink
from fse._container import CorruptArtifactError
from fse._container import _npy_inplace as inplace
from fse._contract import UnsupportedArtifactError
from fse._store._layout import shard_name
from support.store import ENCODER, add_shard, change_metadata, make_store
from support.store_golden import observe, scenarios

GOLDEN = json.loads((Path(__file__).parent / "test_data/v2/store_golden.json").read_text("utf-8"))


def test_the_golden_file_covers_every_scenario():
    assert sorted(GOLDEN) == sorted(scenarios())


@pytest.mark.parametrize("name", sorted(scenarios()))
def test_a_store_has_the_pinned_bytes(tmp_path, name):
    got = observe(tmp_path, scenarios()[name])
    want = GOLDEN[name]
    assert got["entries"] == want["entries"]
    if got["files"] != want["files"]:
        pytest.fail(_differences(tmp_path / "store", got["files"], want["files"]))
    assert got["fingerprints"] == want["fingerprints"]
    assert got["events"] == json.loads(json.dumps(want["events"]))  # tuples as lists


def _differences(store, got, want):
    """The files whose bytes differ, with their size and first and last bytes."""
    lines = []
    for key in sorted(got):
        if got[key] != want.get(key):
            data = (store / key).read_bytes()
            lines.append(f"{key}: {len(data)} bytes {data[:150]!r} ... {data[-40:]!r}")
    return "\n".join(lines[:4])


# --- errors: the same class and message, in the same order of checks --------------------


def _store_json(path, **change):
    file = path / "store.json"
    store = json.loads(file.read_text("utf-8"))
    for key, value in change.items():
        if key == "metadata":
            store["metadata"].update(value)
        else:
            store[key] = value
    file.write_text(json.dumps(store, sort_keys=True, indent=1) + "\n", encoding="utf-8")


def test_a_foreign_kind_with_the_vector_keys_is_unsupported(tmp_path):
    make_store(tmp_path / "s", sizes=(4,))
    _store_json(tmp_path / "s", metadata={"fse.kind": "prediction_scores"})
    with pytest.raises(UnsupportedArtifactError) as raised:
        fse.VectorStore.open(tmp_path / "s")
    assert str(raised.value).endswith(
        "holds 'prediction_scores' format '1' transform '2'; this fse reads "
        "sentence_vectors format 1, transform 2"
    )


def test_a_foreign_kind_with_its_own_keys_is_corrupt(tmp_path):
    make_store(tmp_path / "s", sizes=(4,))
    _store_json(tmp_path / "s", metadata={"fse.kind": "prediction_scores", "fse.score_heads": "[]"})
    with pytest.raises(CorruptArtifactError) as raised:
        fse.VectorStore.open(tmp_path / "s")
    assert re.search(
        r"the metadata must hold exactly fse\.kind, fse\.format_version", str(raised.value)
    )


def test_a_foreign_store_version_is_unsupported_before_the_metadata(tmp_path):
    make_store(tmp_path / "s", sizes=(4,))
    _store_json(tmp_path / "s", store_version=2, metadata={"fse.kind": "unknown", "x": "y"})
    with pytest.raises(CorruptArtifactError):
        fse.VectorStore.open(tmp_path / "s")  # version 2 reads as the document-ID mode
    _store_json(tmp_path / "s", store_version=3)
    with pytest.raises(UnsupportedArtifactError, match="version 3"):
        fse.VectorStore.open(tmp_path / "s")


def test_an_unsynced_prefix_refuses_a_full_append(tmp_path):
    path = tmp_path / "s"
    make_store(path, sizes=(4, 4), tagged=(4,))
    with pytest.raises(ValueError, match="durability='none' from row 4") as raised:
        ShardSink(path, ENCODER, shard_rows=4)
    assert str(raised.value) == (
        f"the store {path} holds rows written with durability='none' from row 4; "
        "rows appended with durability='full' cannot be durable while rows before them "
        "are not; append with durability='none' or write a new store"
    )


def test_a_gap_and_an_orphan_are_corrupt_with_the_repair_hint(tmp_path):
    path = tmp_path / "s"
    info = make_store(path, sizes=(4, 4))
    add_shard(path, info, 12, 2)  # a gap from row 8
    with pytest.raises(CorruptArtifactError) as gap:
        fse.VectorStore.open(path)
    assert str(gap.value) == (
        f"{path / shard_name(12)}, first row 12: a shard follows a gap from row 8; "
        "run fse.repair_store(path) while no writer or reader runs"
    )
    info = make_store(tmp_path / "o", sizes=(4, 4))
    add_shard(tmp_path / "o", info, 2, 4)  # overlaps the chain
    with pytest.raises(CorruptArtifactError) as overlap:
        fse.VectorStore.open(tmp_path / "o")
    assert str(overlap.value).endswith(
        "first row 2: a shard overlaps the published chain; "
        "run fse.repair_store(path) while no writer or reader runs"
    )


def test_a_damaged_shard_names_its_row_and_repair_quarantines_the_tail(tmp_path):
    path = tmp_path / "s"
    make_store(path, sizes=(4, 4, 2))
    change_metadata(path / shard_name(4), {"fse.rows": "5"})
    with pytest.raises(CorruptArtifactError, match=r"first row 4: .*run fse\.repair_store"):
        fse.VectorStore.open(path)
    assert fse.repair_store(path) == 4
    (broken,) = list(path.glob(".broken-*"))
    assert sorted(p.name for p in broken.iterdir()) == [shard_name(4), shard_name(8)]
    assert sorted(p.name for p in path.iterdir() if not p.name.startswith(".")) == [
        shard_name(0),
        "store.json",
    ]


def test_an_unsupported_later_shard_leaves_the_store_untouched(tmp_path):
    path = tmp_path / "s"
    make_store(path, sizes=(4, 4, 2))
    change_metadata(path / shard_name(4), {"fse.rows": "5"})
    change_metadata(path / shard_name(8), {"fse.format_version": "9"})
    before = sorted(p.name for p in path.iterdir())
    with pytest.raises(UnsupportedArtifactError, match="unsupported"):
        fse.repair_store(path)
    assert sorted(p.name for p in path.iterdir()) == before


def test_a_tagged_shard_is_hashed_on_open_and_a_changed_value_is_corrupt(tmp_path):
    path = tmp_path / "s"
    make_store(path, sizes=(4, 4), tagged=(4,))
    with fse.VectorStore.open(path) as store:
        assert len(store) == 8
    file = path / shard_name(4) / "vectors.npy"
    data = bytearray(file.read_bytes())
    data[-1] ^= 0x01
    file.write_bytes(bytes(data))
    with pytest.raises(CorruptArtifactError, match=r"first row 4: .*checksum of vectors"):
        fse.VectorStore.open(path)


def test_a_full_disk_while_writing_publishes_the_committed_rows(tmp_path, monkeypatch):
    path = tmp_path / "s"
    sink = ShardSink(path, ENCODER, shard_rows=8)
    target = sink.target(0, 8)
    for array in target:
        array[...] = 1
    real = inplace._write_some

    def full(*_args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(inplace, "_write_some", full)

    def commit_and_close():
        try:
            sink.commit(0, target)  # the failure may already be reported here
        finally:
            sink.close()

    with pytest.raises(OSError, match="No space left on device") as raised:
        commit_and_close()
    assert raised.value.errno == 28
    monkeypatch.setattr(inplace, "_write_some", real)
    assert sink.published_rows == sink.durable_rows == 0
    assert sorted(p.name for p in path.iterdir()) == ["store.json"]


def test_a_second_writer_that_was_first_to_a_row_gives_file_exists(tmp_path):
    path = tmp_path / "s"
    first = ShardSink(path, ENCODER, shard_rows=4)
    second = ShardSink(path, ENCODER, shard_rows=4)  # both start at row 0
    target = first.target(0, 4)
    for array in target:
        array[...] = 1
    first.commit(0, target)
    first.flush()  # the first writer publishes row 0 before the second has a row
    target = second.target(0, 4)
    for array in target:
        array[...] = 1
    second.commit(0, target)
    with pytest.raises(FileExistsError, match="another writer appended to"):
        second.flush()
    second.close()
    first.close()
