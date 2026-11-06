"""The resume mode source_rows@1: the six keys of a root and of every shard, the parser and
what repair does with a mode this fse does not know."""

import json
from dataclasses import replace

import pytest

import fse
from fse._container import CorruptArtifactError, NpyContainer
from fse._contract import UnsupportedArtifactError
from fse._store._layout import StoreInfo, create_store, open_shard, read_store, shard_name
from fse._store._resume_format import (
    EXECUTION_KEY,
    SOURCE_KEYS,
    SourceRows,
    execution_fingerprint,
    parse,
)
from support import scores as score_support
from support.container import fake_fingerprint
from support.store import add_shard, change_metadata

SOURCE = fake_fingerprint("b")
IDENTITY = SourceRows(SOURCE, "strings", 100_000, (1, 3), 32_768)
INFO = StoreInfo(fake_fingerprint("a"), "float32", 3, version=2, resume=IDENTITY)
FILE = "store.json"


def _store(path, sizes=(4, 4)):
    create_store(path, INFO)
    first = 0
    for rows in sizes:
        add_shard(path, INFO, first, rows)
        first += rows


def test_the_root_and_every_shard_hold_the_same_six_values():
    root = INFO.metadata()
    assert root["fse.resume"] == "source_rows@1"
    assert set(SOURCE_KEYS) <= set(root)
    assert root["fse.source_shard"] == "[1,3]"
    assert root["fse.source_first_row"] == "32768"
    assert root["fse.source_rows"] == "100000"
    assert root["fse.source_form"] == "strings"


def test_the_execution_fingerprint_binds_the_root_without_the_source():
    root = INFO.metadata()
    assert root[EXECUTION_KEY] == execution_fingerprint(root)
    other = replace(INFO, resume=replace(IDENTITY, fingerprint=fake_fingerprint("c"))).metadata()
    assert other[EXECUTION_KEY] == root[EXECUTION_KEY]  # the source is not the execution
    assert replace(INFO, dtype="float16").metadata()[EXECUTION_KEY] != root[EXECUTION_KEY]
    assert replace(INFO, has_doc_ids=True).metadata()[EXECUTION_KEY] != root[EXECUTION_KEY]


def test_a_resumable_store_reads_back(tmp_path):
    _store(tmp_path / "s")
    assert read_store(tmp_path / "s") == INFO
    shard, _ = open_shard(tmp_path / "s", INFO, 0)
    assert shard.rows == 4


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"fse.source_form": "words"}, "source fingerprint, form or execution"),
        ({"fse.source_fingerprint": "x"}, "source fingerprint, form or execution"),
        ({"fse.source_rows": "01"}, "fse.source_first_row and fse.source_rows are invalid"),
        ({"fse.source_shard": "[1, 3]"}, r"must be the JSON list \[i,n\]"),
        ({"fse.source_shard": "[3,3]"}, r"must be the JSON list \[i,n\]"),
        ({"fse.source_shard": "[0,0]"}, r"must be the JSON list \[i,n\]"),
        ({"fse.source_first_row": "0"}, "not that of the partition"),
        ({EXECUTION_KEY: fake_fingerprint("d")}, "execution fingerprint does not match"),
        ({"fse.dtype": "float16"}, "execution fingerprint does not match"),
    ],
)
def test_a_damaged_value_is_corruption(changes, match):
    metadata = INFO.metadata() | changes
    with pytest.raises(CorruptArtifactError, match=match):
        fse._store._vectors_format.VECTORS.parse_root(2, metadata, FILE)


def test_a_root_without_its_keys_is_corruption():
    metadata = INFO.metadata()
    del metadata["fse.source_form"]
    with pytest.raises(CorruptArtifactError, match="must hold exactly"):
        fse._store._vectors_format.VECTORS.parse_root(2, metadata, FILE)
    with pytest.raises(CorruptArtifactError, match="must hold exactly"):
        fse._store._vectors_format.VECTORS.parse_root(
            2, INFO.metadata() | {"fse.source_extra": "1"}, FILE
        )


def test_a_store_without_the_mode_has_no_source_keys():
    plain = replace(INFO, resume=None)
    assert plain.metadata()["fse.resume"] == "none"
    assert not set(SOURCE_KEYS) & set(plain.metadata())
    with pytest.raises(CorruptArtifactError, match="must hold exactly"):
        fse._store._vectors_format.VECTORS.parse_root(
            2, {**plain.metadata(), "fse.source_rows": "1"}, FILE
        )


def test_a_shard_beyond_the_source_is_corruption(tmp_path):
    tiny = replace(INFO, resume=SourceRows(SOURCE, "strings", 5, (0, 1), 0))
    create_store(tmp_path / "s", tiny)
    add_shard(tmp_path / "s", tiny, 0, 5)
    open_shard(tmp_path / "s", tiny, 0)
    add_shard(tmp_path / "s", tiny, 5, 1)
    with pytest.raises(CorruptArtifactError, match="rows beyond the source"):
        open_shard(tmp_path / "s", tiny, 5)


def test_a_shard_of_another_source_is_corruption(tmp_path):
    _store(tmp_path / "s", sizes=(4,))
    other = replace(INFO, resume=replace(IDENTITY, fingerprint=fake_fingerprint("c")))
    shard = tmp_path / "s" / shard_name(0)
    change_metadata(shard, other.metadata())
    with pytest.raises(CorruptArtifactError, match="different sources"):
        open_shard(tmp_path / "s", INFO, 0)


def test_a_shard_with_an_edited_execution_is_corruption(tmp_path):
    _store(tmp_path / "s", sizes=(4,))
    change_metadata(tmp_path / "s" / shard_name(0), {EXECUTION_KEY: fake_fingerprint("d")})
    with pytest.raises(CorruptArtifactError, match="execution fingerprint"):
        open_shard(tmp_path / "s", INFO, 0)


def _edit_root(path, mode):
    root = json.loads((path / FILE).read_text())
    root["metadata"]["fse.resume"] = mode
    (path / FILE).write_text(json.dumps(root, sort_keys=True, indent=1) + "\n")


def test_an_unknown_mode_of_the_root_is_unsupported(tmp_path):
    _store(tmp_path / "s")
    _edit_root(tmp_path / "s", "source_rows@2")
    with pytest.raises(UnsupportedArtifactError, match=r"unsupported fse\.resume"):
        read_store(tmp_path / "s")
    _edit_root(tmp_path / "s", "source rows")
    with pytest.raises(CorruptArtifactError, match="invalid resume identifier"):
        read_store(tmp_path / "s")


def _snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()} | {
        str(p.relative_to(path)): None for p in path.rglob("*") if p.is_dir()
    }


@pytest.mark.parametrize("where", ["shard", "orphan"])
def test_repair_stops_before_any_change_at_a_future_mode(tmp_path, where):
    _store(tmp_path / "s", sizes=(4, 4, 4))
    last = shard_name(8)
    if where == "orphan":
        (tmp_path / "s" / shard_name(8)).rename(tmp_path / "s" / shard_name(40))
        last = shard_name(40)
    change_metadata(tmp_path / "s" / last, {"fse.resume": "source_rows@2"})
    (tmp_path / "s" / shard_name(4) / "manifest.json").write_bytes(b"{")  # damage before it
    before = _snapshot(tmp_path / "s")
    with pytest.raises(UnsupportedArtifactError, match=r"unsupported fse\.resume"):
        fse.repair_store(tmp_path / "s")
    assert _snapshot(tmp_path / "s") == before


def test_repair_keeps_the_good_prefix_of_a_resumable_store(tmp_path):
    _store(tmp_path / "s", sizes=(4, 4, 4))
    (tmp_path / "s" / shard_name(8) / "manifest.json").write_bytes(b"{")
    assert fse.repair_store(tmp_path / "s") == 8
    assert read_store(tmp_path / "s") == INFO


def test_the_identity_of_a_score_store_round_trips(tmp_path):
    info = replace(score_support.default_info(), resume=IDENTITY)
    score_support.make_scores(tmp_path / "s", info, sizes=(4, 4))
    assert read_store(tmp_path / "s") == info
    metadata = info.metadata()
    assert metadata["fse.resume"] == "source_rows@1"
    assert set(SOURCE_KEYS) <= set(metadata)
    shared = {k: v for k, v in metadata.items() if k not in SOURCE_KEYS}
    assert parse(metadata, shared, FILE) == IDENTITY
    with fse.ScoreStore.open(tmp_path / "s") as store:
        store.verify()


def test_a_score_shard_beyond_the_source_is_corruption(tmp_path):
    info = replace(score_support.default_info(), resume=SourceRows(SOURCE, "lines", 6, (0, 1), 0))
    score_support.make_scores(tmp_path / "s", info, sizes=(4, 4))
    with pytest.raises(CorruptArtifactError, match="rows beyond the source"):
        fse.ScoreStore.open(tmp_path / "s")


def test_an_unknown_mode_of_a_score_store_is_unsupported(tmp_path):
    info = replace(score_support.default_info(), resume=IDENTITY)
    score_support.make_scores(tmp_path / "s", info, sizes=(4,))
    _edit_root(tmp_path / "s", "source_rows@2")
    with pytest.raises(UnsupportedArtifactError, match=r"unsupported fse\.resume"):
        read_store(tmp_path / "s")


def test_a_manifest_of_a_resumable_shard_names_the_mode(tmp_path):
    _store(tmp_path / "s", sizes=(4,))
    _, metadata = NpyContainer().open(tmp_path / "s" / shard_name(0), mmap=False)
    assert metadata["fse.resume"] == "source_rows@1"
    assert metadata["fse.source_shard"] == "[1,3]"
