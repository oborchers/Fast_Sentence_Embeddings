"""``ShardSink(sign_bits=True)``: the bits are written with the shards, published after the
rows of the shard are counted, and equal those of ``fse.add_sign_bits``."""

import numpy as np
import pytest
from pydantic import ValidationError

import fse
from fse import _reference_bits
from fse._container import NpyContainer, PublishedNotDurableError
from fse._store import ShardSink, _bits_write
from fse._store._bits_format import bits_name
from fse._store._layout import open_shard, read_store, shard_name, shards
from support.bits import stored_bits, tree
from support.encoder import finished
from support.logs import captured
from support.runtime import small_tasks, texts

ENCODER = finished()  # 6 dimensions, one byte of bits per row
SENTENCES = texts(40)


def write(path, *, n=40, rows=8, **kwargs):
    with ShardSink(path, ENCODER, shard_rows=rows, **kwargs) as sink:
        ENCODER.encode(SENTENCES[:n], workers=1, sink=sink)
    return sink


def entries(path):
    """Every canonical entry of the store (shards, bits, root), by name, with its bytes."""
    return {
        entry.name: tree(entry) if entry.is_dir() else entry.read_bytes()
        for entry in sorted(path.iterdir())
        if not entry.name.startswith(".")
    }


def bits_entries(path):
    return {name: value for name, value in entries(path).items() if name.startswith("sign_bits-")}


def shard_entries(path):
    return {name: value for name, value in entries(path).items() if name.startswith("shard-")}


def test_the_default_writes_no_bits_and_the_shards_are_those_with_bits(tmp_path):
    plain, signed = tmp_path / "plain", tmp_path / "signed"
    write(plain)
    write(signed, sign_bits=True)
    assert not bits_entries(plain)
    assert len(bits_entries(signed)) == 5
    assert shard_entries(plain) == shard_entries(signed)  # the shards are unchanged by it
    assert (plain / "store.json").read_bytes() == (signed / "store.json").read_bytes()


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("workers", [1, 3])
def test_the_bits_equal_add_sign_bits_and_the_reference(tmp_path, monkeypatch, dtype, workers):
    small_tasks(monkeypatch)  # many tasks, none inline
    plain, signed = tmp_path / "plain", tmp_path / "signed"
    for path, options in ((plain, {}), (signed, {"sign_bits": True})):
        with ShardSink(path, ENCODER, dtype=dtype, shard_rows=16, **options) as sink:
            ENCODER.encode(texts(100), workers=workers, sink=sink)
    assert fse.add_sign_bits(plain) == 7
    assert len(bits_entries(signed)) == 7
    assert bits_entries(plain) == bits_entries(signed)  # byte for byte, manifests included
    with fse.VectorStore.open(signed) as store:
        store.verify()
        found = stored_bits(store)
        for shard, tensors in store._shards:
            want = _reference_bits.sign_bits(tensors["vectors"])
            assert np.array_equal(found[shard.first_row], want)
        assert sum(len(bits) for bits in found.values()) == 100  # the short last shard too


def test_a_sealed_shard_holds_the_bits_of_its_rows_only(tmp_path):
    path = tmp_path / "store"
    with ShardSink(path, ENCODER, shard_rows=16, sign_bits=True) as sink:
        target = sink.target(0, 5)
        for array in target:
            array[...] = 0.5
        sink.commit(0, target)
        sink.flush()
    with fse.VectorStore.open(path) as store:
        ((shard, _),) = store._shards
        assert shard.rows == 5
        assert stored_bits(store)[0].shape == (5, 1)


def test_durability_none_tags_the_bits_and_the_directory_is_not_synced(tmp_path, monkeypatch):
    synced = []
    real = _bits_write.publish_directory
    monkeypatch.setattr(
        _bits_write,
        "publish_directory",
        lambda tmp, path, *, sync_parent=True: (
            synced.append(sync_parent),
            real(tmp, path, sync_parent=sync_parent),
        )[1],
    )
    path = tmp_path / "store"
    write(path, n=16, durability="none", sign_bits=True)
    assert synced == [False, False]
    assert NpyContainer().metadata(path / bits_name(0))["fse.durability"] == "none"
    with fse.VectorStore.open(path) as store:
        assert all(bits is not None for bits in stored_bits(store).values())


def test_full_durability_syncs_the_store_for_the_bits(tmp_path, monkeypatch):
    synced = []
    real = _bits_write.publish_directory
    monkeypatch.setattr(
        _bits_write,
        "publish_directory",
        lambda tmp, path, *, sync_parent=True: (
            synced.append(sync_parent),
            real(tmp, path, sync_parent=sync_parent),
        )[1],
    )
    path = tmp_path / "store"
    write(path, n=16, sign_bits=True)
    assert synced == [True, True]
    assert "fse.durability" not in NpyContainer().metadata(path / bits_name(0))


def test_the_switch_is_a_bool(tmp_path):
    with pytest.raises(ValidationError):
        ShardSink(tmp_path / "store", ENCODER, sign_bits="yes")


def test_it_logs_the_switch_and_the_bits_published(tmp_path):
    with captured() as records:
        write(tmp_path / "signed", sign_bits=True)
        write(tmp_path / "plain")
    opened = [r for r in records if r.fse_event == "store.sink_opened"]
    closed = [r for r in records if r.fse_event == "store.sink_closed"]
    assert [hasattr(r, "fse_sign_bits") for r in opened] == [True, False]
    assert closed[0].fse_sign_bits_shards == 5
    assert not hasattr(closed[1], "fse_sign_bits_shards")


def test_a_store_written_with_bits_opens_without_the_warning(tmp_path):
    path = tmp_path / "store"
    write(path, sign_bits=True)
    with captured() as records, fse.VectorStore.open(path) as store:
        assert len(store) == 40
    assert not [r for r in records if r.fse_event == "store.sign_bits_missing"]


# --- the bounds of the failures: the rows of the shard are never counted wrongly ----------------


def fail_before_the_rename(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_bits_write, "publish_directory", refuse)


def run_until_closed(path, **kwargs):
    """The sink after its close, the error that the close raised."""
    sink = ShardSink(path, ENCODER, shard_rows=8, sign_bits=True, **kwargs)
    try:
        with sink:
            ENCODER.encode(SENTENCES, workers=1, sink=sink)
    except Exception as error:
        return sink, error
    raise AssertionError("the sink raised nothing")


def test_a_failure_before_the_bits_are_published_counts_the_shard_and_stops_the_sink(
    tmp_path, monkeypatch
):
    fail_before_the_rename(monkeypatch)
    sink, error = run_until_closed(tmp_path / "store")
    assert isinstance(error, OSError)
    assert "No space left" in str(error)
    assert sink.published_rows == sink.durable_rows == 8  # the shard was published and synced
    path = tmp_path / "store"
    assert sorted(p.name for p in path.iterdir() if p.name.startswith(".")) == []
    assert not list(path.glob("sign_bits-*"))
    monkeypatch.undo()
    with fse.VectorStore.open(path) as store:  # a shard without bits is valid
        assert len(store) == 8
        store.verify()
    assert fse.add_sign_bits(path) == 1


def test_a_failure_after_the_rename_keeps_the_bits_and_reports_them(tmp_path, monkeypatch):
    real = _bits_write.publish_directory

    def renamed_but_not_synced(tmp, path, **_kwargs):
        real(tmp, path, sync_parent=False)
        raise PublishedNotDurableError(f"{path} is published, but syncing failed")

    monkeypatch.setattr(_bits_write, "publish_directory", renamed_but_not_synced)
    sink, error = run_until_closed(tmp_path / "store")
    assert isinstance(error, PublishedNotDurableError)
    assert "sign_bits-" in str(error)
    assert sink.published_rows == sink.durable_rows == 8  # the shard itself is durable
    path = tmp_path / "store"
    monkeypatch.undo()
    assert not [p for p in path.iterdir() if p.name.startswith(".")]
    with fse.VectorStore.open(path) as store:
        store.verify()
        assert stored_bits(store)[0] is not None


def test_a_failure_while_the_bits_are_made_drops_the_shard_but_not_the_ones_before(
    tmp_path, monkeypatch
):
    from fse._store import _bits_writer

    real = _bits_writer.BitsColumn.commit
    seen = []

    def commit(column, begin, codes):
        seen.append(column.first_row)
        if column.first_row == 8:
            raise MemoryError("no memory for the bits")
        real(column, begin, codes)

    monkeypatch.setattr(_bits_writer.BitsColumn, "commit", commit)
    sink, error = run_until_closed(tmp_path / "store")
    assert isinstance(error, MemoryError)
    assert sink.published_rows == 8  # the first shard only; the second never got published
    path = tmp_path / "store"
    assert sorted(p.name for p in path.iterdir() if p.name != "store.json") == [
        shard_name(0),
        bits_name(0),
    ]
    assert seen[:2] == [0, 8]


def test_a_second_writer_that_published_valid_bits_first_is_not_an_error(tmp_path, monkeypatch):
    path = tmp_path / "store"
    real = _bits_write.publish_directory
    raced = []

    def after_the_other(tmp, target, *, sync_parent=True):
        if not raced:
            raced.append(0)
            raced[0] = fse.add_sign_bits(path)  # the shard is published: it writes the bits
        real(tmp, target, sync_parent=sync_parent)

    monkeypatch.setattr(_bits_write, "publish_directory", after_the_other)
    sink = write(path, n=8, sign_bits=True)
    assert raced == [1]
    assert sink.published_rows == sink.durable_rows == 8
    assert sink._extras_published == 0  # the other writer's artifact stays
    assert not [p for p in path.iterdir() if p.name.startswith(".")]
    with fse.VectorStore.open(path) as store:
        store.verify()


def test_a_second_writer_with_a_damaged_artifact_is_an_error(tmp_path, monkeypatch):
    path = tmp_path / "store"
    real = _bits_write.publish_directory

    def after_a_bad_one(tmp, target, *, sync_parent=True):
        (path / bits_name(0)).mkdir()
        (path / bits_name(0) / "manifest.json").write_text("{}")
        real(tmp, target, sync_parent=sync_parent)

    monkeypatch.setattr(_bits_write, "publish_directory", after_a_bad_one)
    sink = ShardSink(path, ENCODER, shard_rows=8, sign_bits=True)
    with pytest.raises(fse.CorruptArtifactError), sink:
        ENCODER.encode(SENTENCES[:8], workers=1, sink=sink)
    assert sink.published_rows == 8  # the rows of the shard are counted all the same
    assert not [p for p in path.iterdir() if p.name.startswith(".")]


def test_an_interrupt_between_the_shard_and_its_bits_leaves_a_valid_store(tmp_path, monkeypatch):
    from fse._store import _bits_writer

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(_bits_writer.BitsColumn, "publish", interrupt)
    path = tmp_path / "store"
    sink = ShardSink(path, ENCODER, shard_rows=8, sign_bits=True)
    with pytest.raises(KeyboardInterrupt), sink:
        ENCODER.encode(SENTENCES[:16], workers=1, sink=sink)
    monkeypatch.undo()
    assert sink.published_rows == 8
    assert not list(path.glob("sign_bits-*"))
    assert not [p for p in path.iterdir() if p.name.startswith(".")]
    with fse.VectorStore.open(path) as store:
        store.verify()
    info = read_store(path)
    assert len(shards(path, info)) == 1


# --- sign bits of the rows read back by the store ------------------------------------------------


def test_the_bits_of_a_run_in_many_workers_are_those_of_one(tmp_path, monkeypatch):
    small_tasks(monkeypatch)
    one, many = tmp_path / "one", tmp_path / "many"
    for path, workers in ((one, 1), (many, 4)):
        with ShardSink(path, ENCODER, shard_rows=7, sign_bits=True) as sink:
            ENCODER.encode(texts(60), workers=workers, sink=sink)
    assert len(bits_entries(one)) == 9
    assert bits_entries(one) == bits_entries(many)


def test_a_failure_in_the_middle_of_writing_the_bits_leaves_nothing_but_the_shard(
    tmp_path, monkeypatch
):
    def fail(*_args, **_kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(_bits_write.NpyContainer, "_write_tensor", fail)
    sink, error = run_until_closed(tmp_path / "store")
    assert isinstance(error, OSError)
    assert sink.published_rows == sink.durable_rows == 8
    path = tmp_path / "store"
    assert sorted(p.name for p in path.iterdir()) == [shard_name(0), "store.json"]


def test_a_failed_allocation_of_the_bits_leaves_no_temporary_shard(tmp_path, monkeypatch):
    from fse._store import _bits_writer

    def fail(*_args, **_kwargs):
        raise MemoryError("no memory for the bits")

    monkeypatch.setattr(_bits_writer.BitsColumn, "__init__", fail)
    path = tmp_path / "store"
    sink = ShardSink(path, ENCODER, shard_rows=8, sign_bits=True)
    with pytest.raises(MemoryError), sink:
        sink.target(0, 4)
    assert not [p for p in path.iterdir() if p.name.startswith(".")]


def test_a_concurrent_artifact_with_other_bits_is_an_error_though_it_is_untagged(
    tmp_path, monkeypatch
):
    path = tmp_path / "store"
    real = _bits_write.publish_directory
    done = []

    def other_bits_first(tmp, target, **kwargs):
        if not done:
            done.append(1)
            info = read_store(path)
            shard, tensors = open_shard(path, info, 0)
            _bits_write.publish_bits(
                path,
                info,
                shard,
                ~_reference_bits.sign_bits(tensors["vectors"]),
                synced=True,
                by="x",
            )
        real(tmp, target, **kwargs)

    monkeypatch.setattr(_bits_write, "publish_directory", other_bits_first)
    sink = ShardSink(path, ENCODER, shard_rows=8, sign_bits=True)
    with pytest.raises(fse.CorruptArtifactError, match="not the bits"), sink:
        ENCODER.encode(SENTENCES[:8], workers=1, sink=sink)
    assert sink.published_rows == 8
