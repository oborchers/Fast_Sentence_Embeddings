"""Resuming a run into a store: arguments, modes, the identity of a source and the rows."""

import numpy as np
import pytest

import fse
from fse import ShardSink, _reference
from fse._contract import UnsupportedArtifactError
from fse._store._layout import read_store, shard_name
from fse._store._resume_format import MODE
from fse._text import StringSource
from support.encoder import finished
from support.logs import captured
from support.resume import ENCODER, SHARD_ROWS, canonical, drop_tail, rows_of, sentences, write

SENTENCES = sentences(300)


class Poisoned(list):
    """A list that fails when it is read: a check of arguments must not touch it."""

    def __iter__(self):
        raise AssertionError("the source was read")

    def __getitem__(self, _key):
        raise AssertionError("the source was read")

    def __len__(self):
        raise AssertionError("the source was read")


def _sink(path, **kwargs):
    return ShardSink(path, ENCODER, shard_rows=SHARD_ROWS, resume=True, **kwargs)


# --- arguments: raised before the source is read -----------------------------------------


@pytest.mark.parametrize("shard", [(0,), [0, 1], (True, 2), (0.0, 2), "01", (0, 1, 2), 3])
def test_a_shard_must_be_a_pair_of_ints(tmp_path, shard):
    with _sink(tmp_path / "s") as sink, pytest.raises(TypeError, match="shard must be a tuple"):
        ENCODER.encode(Poisoned(), sink=sink, shard=shard)
    assert not (tmp_path / "s").exists()


@pytest.mark.parametrize("shard", [(0, 0), (2, 2), (-1, 2), (0, -1)])
def test_a_shard_must_name_a_part(tmp_path, shard):
    with _sink(tmp_path / "s") as sink, pytest.raises(ValueError, match="shard must be"):
        ENCODER.encode(Poisoned(), sink=sink, shard=shard)


def test_a_shard_needs_a_resumable_sink(tmp_path):
    with pytest.raises(ValueError, match="shard= needs a sink made with resume=True"):
        ENCODER.encode(Poisoned(), shard=(0, 1))
    with (
        ShardSink(tmp_path / "s", ENCODER) as sink,
        pytest.raises(ValueError, match="shard= needs a sink made with resume=True"),
    ):
        ENCODER.encode(Poisoned(), sink=sink, shard=(0, 1))
    assert read_store(tmp_path / "s").resume is None


def test_a_sink_without_the_contract_is_refused_with_a_shard():
    class Foreign:
        def target(self, first_row, n_rows): ...

        def commit(self, first_row, rows): ...

    with pytest.raises(ValueError, match="shard= needs a sink made with resume=True"):
        ENCODER.encode(Poisoned(), sink=Foreign(), shard=(0, 1))


@pytest.mark.parametrize("kind", ["stream", "generator", "arrow"])
def test_a_source_that_cannot_be_read_again_is_refused(tmp_path, kind):
    if kind == "arrow":
        pa = pytest.importorskip("pyarrow")
        texts = pa.array(SENTENCES)
    else:
        texts = iter(SENTENCES)
    with (
        _sink(tmp_path / "s") as sink,
        pytest.raises(TypeError, match=r"cannot be read again|stream"),
    ):
        ENCODER.encode(texts, sink=sink)
    assert not (tmp_path / "s").exists()
    if kind != "arrow":
        assert next(texts) == SENTENCES[0]  # not one sentence was taken


def test_a_str_is_no_source(tmp_path):
    with _sink(tmp_path / "s") as sink, pytest.raises(TypeError, match="not one str"):
        ENCODER.encode("one sentence", sink=sink)


def test_other_entries_refuse_a_resumable_sink(tmp_path):
    with _sink(tmp_path / "s") as sink:
        with pytest.raises(ValueError, match=r"resume=True takes the rows of Encoder\.encode"):
            list(ENCODER.encode_batches([SENTENCES], sink=sink))
        with pytest.raises(ValueError, match=r"it takes rows from Encoder\.encode"):
            sink.target(0, 1)
        with pytest.raises(ValueError, match=r"it takes rows from Encoder\.encode"):
            sink._check_fresh()
    assert not (tmp_path / "s").exists()


def test_a_sink_takes_one_source(tmp_path):
    with _sink(tmp_path / "s") as sink:
        ENCODER.encode(SENTENCES[:100], sink=sink)
        with pytest.raises(RuntimeError, match="took a source already"):
            ENCODER.encode(SENTENCES[:100], sink=sink)


# --- modes --------------------------------------------------------------------------------


def test_a_resumable_store_is_opened_with_resume(tmp_path):
    write(tmp_path / "s", SENTENCES)
    with pytest.raises(
        ValueError, match=r"is resumable \(mode source_rows@1\); open it with resume=True"
    ):
        ShardSink(tmp_path / "s", ENCODER)


@pytest.mark.parametrize("ids", [False, True])
def test_a_plain_store_is_no_resumable_store(tmp_path, ids):
    with ShardSink(tmp_path / "s", ENCODER, doc_ids=ids) as sink:
        ENCODER.encode(SENTENCES[:10], sink=sink, doc_ids=list(range(10)) if ids else None)
    with pytest.raises(ValueError, match="is not a resumable store; write a new store"):
        _sink(tmp_path / "s", doc_ids=ids)


def test_a_new_resumable_store_is_made_when_a_source_comes(tmp_path):
    with _sink(tmp_path / "s") as sink:
        assert not (tmp_path / "s").exists()
        assert sink.first_row == 0
        ENCODER.encode(SENTENCES[:100], sink=sink)
    assert read_store(tmp_path / "s").resume is not None


def test_a_mode_of_the_future_is_unsupported(tmp_path):
    write(tmp_path / "s", SENTENCES)
    text = (tmp_path / "s" / "store.json").read_text()
    (tmp_path / "s" / "store.json").write_text(text.replace("source_rows@1", "source_position@1"))
    with pytest.raises(UnsupportedArtifactError, match=r"unsupported fse\.resume"):
        _sink(tmp_path / "s")


def test_the_mode_name_is_frozen():
    assert MODE == "source_rows@1"


# --- the rows -------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_a_resumed_run_gives_the_rows_of_one_run(tmp_path, dtype):
    whole = write(tmp_path / "whole", SENTENCES, dtype=dtype)
    write(tmp_path / "part", SENTENCES, dtype=dtype)
    drop_tail(tmp_path / "part", 2)
    again = write(tmp_path / "part", SENTENCES, dtype=dtype)
    assert again.first_row == 2 * SHARD_ROWS  # the shards kept
    assert again.published_rows == len(SENTENCES) - 2 * SHARD_ROWS
    assert canonical(tmp_path / "part") == canonical(tmp_path / "whole")
    assert whole.published_rows == len(SENTENCES)
    codes, _ = rows_of(tmp_path / "part")
    want = _reference.quantize_rows(ENCODER.encode(SENTENCES, workers=1), dtype)[0]
    np.testing.assert_array_equal(codes.view(np.uint8), want.view(np.uint8))


def test_a_complete_store_is_left_alone(tmp_path):
    write(tmp_path / "s", SENTENCES)
    before = canonical(tmp_path / "s")
    sink = write(tmp_path / "s", SENTENCES)
    assert (sink.first_row, sink.published_rows) == (len(SENTENCES), 0)
    assert canonical(tmp_path / "s") == before


def test_the_sources_of_a_collection_give_one_store(tmp_path):
    write(tmp_path / "list", SENTENCES)
    for name, source in (
        ("tuple", tuple(SENTENCES)),
        ("array", np.array(SENTENCES, dtype=object)),
        ("source", StringSource(SENTENCES)),
    ):
        write(tmp_path / name, source)
        assert canonical(tmp_path / name) == canonical(tmp_path / "list")


def test_a_file_gives_the_rows_of_its_lines(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("\n".join(SENTENCES) + "\n", encoding="utf-8")
    write(tmp_path / "file", path)
    codes, _ = rows_of(tmp_path / "file")
    want = _reference.quantize_rows(ENCODER.encode(SENTENCES, workers=1), "float32")[0]
    np.testing.assert_array_equal(codes.view(np.uint8), want.view(np.uint8))
    assert read_store(tmp_path / "file").resume.form == "lines"


def test_token_lists_are_resumable_for_the_external_tokenizer(tmp_path):
    encoder = finished(tokenizer="external")
    tokens = [s.split() for s in SENTENCES]
    write(tmp_path / "s", tokens, encoder=encoder)
    with ShardSink(tmp_path / "s", encoder, shard_rows=SHARD_ROWS, resume=True) as sink:
        encoder.encode(tokens, sink=sink)
        assert sink.first_row == len(SENTENCES)
    assert read_store(tmp_path / "s").resume.form == "tokens"
    with pytest.raises(TypeError, match="external tokenizer"):
        write(tmp_path / "t", tokens)  # the encoder tokenizes with fse_words@1


def test_an_empty_source_makes_a_store_without_rows(tmp_path):
    sink = write(tmp_path / "s", [])
    assert sink.published_rows == 0
    assert read_store(tmp_path / "s").resume.rows == 0
    write(tmp_path / "s", [])
    with pytest.raises(ValueError, match="another source or partition"):
        write(tmp_path / "s", ["a"])


def test_the_parts_of_a_source_give_the_rows_of_the_whole(tmp_path):
    rows = 40_000
    source = sentences(rows)
    parts = 3
    codes = []
    for i in range(parts):
        sink = write(tmp_path / f"p{i}", source, shard=(i, parts), rows=16_384)
        codes.append(rows_of(tmp_path / f"p{i}")[0])
        assert read_store(tmp_path / f"p{i}").resume.shard == (i, parts)
        assert sink.published_rows == len(codes[-1])
    assert [len(c) for c in codes] == [16_384, 16_384, 7_232]
    want = ENCODER.encode(source, workers=2)
    np.testing.assert_array_equal(np.concatenate(codes).view(np.uint32), want.view(np.uint32))


def test_a_part_can_be_empty(tmp_path):
    sink = write(tmp_path / "s", SENTENCES, shard=(0, 5))
    assert sink.published_rows == 0
    sink = write(tmp_path / "t", SENTENCES, shard=(4, 5))
    assert sink.published_rows == len(SENTENCES)


def test_a_part_of_a_file_resumes(tmp_path):
    rows = 40_000
    source = sentences(rows)
    path = tmp_path / "t.txt"
    path.write_text("\n".join(source) + "\n", encoding="utf-8")
    write(tmp_path / "whole", path, shard=(1, 3), rows=4096)
    write(tmp_path / "again", path, shard=(1, 3), rows=4096)
    drop_tail(tmp_path / "again", 2, rows=4096)
    write(tmp_path / "again", path, shard=(1, 3), rows=4096)
    assert canonical(tmp_path / "whole") == canonical(tmp_path / "again")
    codes, _ = rows_of(tmp_path / "whole")
    want = ENCODER.encode(source[16_384:32_768], workers=2)
    np.testing.assert_array_equal(codes.view(np.uint32), want.view(np.uint32))


def test_document_ids_follow_the_resumed_rows(tmp_path):
    ids = np.arange(1000, 1000 + len(SENTENCES), dtype=np.int64)
    write(tmp_path / "whole", SENTENCES, doc_ids=True, ids=ids)
    write(tmp_path / "part", SENTENCES, doc_ids=True, ids=ids)
    drop_tail(tmp_path / "part", 2)
    write(tmp_path / "part", SENTENCES, doc_ids=True, ids=list(ids))
    assert canonical(tmp_path / "part") == canonical(tmp_path / "whole")
    with fse.VectorStore.open(tmp_path / "part") as store:
        np.testing.assert_array_equal(store.get_doc_ids(), ids)
        store.verify()
    with pytest.raises(ValueError, match="another source or partition"):
        write(tmp_path / "part", SENTENCES, doc_ids=True, ids=ids + 1)


def test_the_ids_must_come_with_a_sink_that_stores_them(tmp_path):
    with (
        _sink(tmp_path / "s", doc_ids=True) as sink,
        pytest.raises(ValueError, match="doc_ids must be supplied"),
    ):
        ENCODER.encode(SENTENCES, sink=sink)
    with (
        _sink(tmp_path / "t") as sink,
        pytest.raises(ValueError, match="doc_ids must be supplied exactly"),
    ):
        ENCODER.encode(SENTENCES, sink=sink, doc_ids=list(range(len(SENTENCES))))


def test_a_resumed_store_opens_and_verifies(tmp_path):
    write(tmp_path / "s", SENTENCES, durability="none")
    with fse.VectorStore.open(tmp_path / "s") as store:
        assert len(store) == len(SENTENCES)
        store.verify()
    assert fse.repair_store(tmp_path / "s") == len(SENTENCES)
    assert (tmp_path / "s" / shard_name(0)).is_dir()


def test_resume_under_none_verifies_tagged_shards(tmp_path):
    write(tmp_path / "s", SENTENCES, durability="none")
    shard = tmp_path / "s" / shard_name(SHARD_ROWS)
    data = bytearray((shard / "vectors.npy").read_bytes())
    data[-1] ^= 0xFF
    (shard / "vectors.npy").write_bytes(bytes(data))
    with pytest.raises(fse.CorruptArtifactError):
        write(tmp_path / "s", SENTENCES, durability="none")


def test_resume_under_full_refuses_a_tagged_prefix(tmp_path):
    write(tmp_path / "s", SENTENCES, durability="none")
    with pytest.raises(ValueError, match="durability='none'"):
        _sink(tmp_path / "s", durability="full")


# --- logging --------------------------------------------------------------------------------


def test_the_records_of_a_resumed_run_name_what_was_kept(tmp_path):
    write(tmp_path / "s", SENTENCES)
    with captured() as records:
        write(tmp_path / "s", SENTENCES)
        with ShardSink(tmp_path / "plain", ENCODER) as sink:
            ENCODER.encode(SENTENCES[:10], sink=sink)
    opened, plain = [r for r in records if r.fse_event == "store.sink_opened"]
    assert opened.fse_resume == MODE
    assert opened.fse_kept_rows == opened.fse_rows == len(SENTENCES)
    assert opened.fse_source_rows == len(SENTENCES)
    assert opened.fse_source_bytes > 0
    assert opened.fse_source_seconds >= 0
    assert opened.fse_source_fingerprint.startswith("xxh3-128:")
    assert not hasattr(plain, "fse_resume")
    closed_resumed, closed_plain = [r for r in records if r.fse_event == "store.sink_closed"]
    assert closed_resumed.fse_skipped_rows == len(SENTENCES)
    assert not hasattr(closed_plain, "fse_skipped_rows")
    assert all("/" not in r.getMessage() for r in records if r.fse_event.startswith("store.sink"))
