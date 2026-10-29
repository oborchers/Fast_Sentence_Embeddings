"""The syncs of a shard's publication: two full syncs on macOS, the same fsyncs elsewhere."""

import os
import types

import pytest

from fse._container import _common, _npy, _npy_inplace
from fse._store import ShardSink
from fse._store._layout import StoreInfo, create_store, shard_name, shards
from fse._store._shard import ShardWriter
from support.container import fake_fingerprint
from support.encoder import unfolded
from support.runtime import texts

INFO = StoreInfo(fake_fingerprint("c"), "int8", 4)  # int8: vectors, scales and inv_norms


class Recorder:
    """Records every fsync, F_FULLFSYNC, manifest replace and shard rename, in order, by the
    inode of the synced file or directory; ``fail`` raises an OSError at one of them."""

    def __init__(self, monkeypatch, *, full: bool):
        self.events: list[tuple[str, int | None]] = []
        self.fail: tuple[str, int] | None = None  # (kind, n): the n-th event of that kind
        real_fsync, real_replace = os.fsync, os.replace
        real_rename = _npy_inplace._rename_exclusive

        def fsync(fd):
            self._record("fsync", os.fstat(fd).st_ino)
            real_fsync(fd)

        def fcntl(fd, command):
            assert command == "F_FULLFSYNC"
            self._record("full", os.fstat(fd).st_ino)
            real_fsync(fd)

        def replace(src, dst):
            self._record("replace", None)
            real_replace(src, dst)

        def rename(tmp, path):
            self._record("rename", None)
            real_rename(tmp, path)

        fake = types.SimpleNamespace(fcntl=fcntl)
        if full:
            fake.F_FULLFSYNC = "F_FULLFSYNC"
        monkeypatch.setattr(_common, "_fcntl", fake)
        monkeypatch.setattr(os, "fsync", fsync)
        monkeypatch.setattr(_npy.os, "replace", replace)
        monkeypatch.setattr(_npy_inplace, "_rename_exclusive", rename)

    def _record(self, kind, inode):
        self.events.append((kind, inode))
        if self.fail == (kind, sum(k == kind for k, _ in self.events)):
            raise OSError(5, "Input/output error")

    def named(self, store, first_row=0):
        """The events, with the inodes named by the files of the published shard."""
        shard = store / shard_name(first_row)
        names = {store.stat().st_ino: "store", shard.stat().st_ino: "shard"}
        names |= {p.stat().st_ino: p.name for p in shard.iterdir()}
        return [kind if inode is None else f"{kind} {names[inode]}" for kind, inode in self.events]


@pytest.fixture
def store(tmp_path):
    create_store(tmp_path / "s", INFO)  # its syncs are not the shard's
    return tmp_path / "s"


def _writer(store, first_row=0, rows=3):
    writer = ShardWriter(store, INFO, first_row, 8, "test")
    codes, scales, norms = writer.reserve(rows)
    codes[...], scales[...], norms[...] = 1, 0.5, 0.25
    writer.commit(0, codes)
    writer.write(0, codes)
    writer.seal()
    return writer


def _sequence(final):
    """The calls of one publication; ``final`` is the kind of the two that make it durable."""
    return [
        "fsync vectors.npy",
        "fsync inv_norms.npy",
        "fsync scales.npy",
        "fsync manifest.json",
        "fsync shard",  # the entries of the files and the manifest temp reach the drive
        "replace",
        f"{final} shard",  # the barrier: files, manifest and entries durable
        "rename",
        f"{final} store",  # the rename durable; only then durable_rows grows
    ]


def test_a_shard_makes_two_full_syncs_on_macos(store, monkeypatch):
    recorder = Recorder(monkeypatch, full=True)
    assert _writer(store).finish() == 3
    assert recorder.named(store) == _sequence("full")
    assert sum(kind == "full" for kind, _ in recorder.events) == 2  # six before


def test_elsewhere_a_shard_makes_the_fsyncs_it_made_before(store, monkeypatch):
    recorder = Recorder(monkeypatch, full=False)
    assert _writer(store).finish() == 3
    assert recorder.named(store) == _sequence("fsync")


def test_a_single_file_is_still_made_durable_on_its_own(tmp_path, monkeypatch):
    recorder = Recorder(monkeypatch, full=True)
    with (tmp_path / "f").open("wb") as f:
        _npy_inplace.sync_file(f.fileno())
    _common._sync_directory(tmp_path)
    _common._sync_directory(tmp_path, full=False)
    file, directory = (tmp_path / "f").stat().st_ino, tmp_path.stat().st_ino
    assert recorder.events == [("full", file), ("full", directory), ("fsync", directory)]


def test_windows_syncs_no_directory(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(_common.os, "open", lambda *args: opened.append(args))
    monkeypatch.setattr(_common.os, "name", "nt")
    _common._sync_directory(tmp_path)
    _common._sync_directory(tmp_path, full=False)
    monkeypatch.undo()
    assert opened == []


@pytest.mark.parametrize(
    "fail",
    [("fsync", 1), ("fsync", 2), ("fsync", 3), ("fsync", 4), ("fsync", 5), ("full", 1)],
    ids=["vectors", "inv_norms", "scales", "manifest", "entries", "barrier"],
)
def test_a_failed_sync_before_the_rename_publishes_nothing(store, monkeypatch, fail):
    recorder = Recorder(monkeypatch, full=True)
    recorder.fail = fail
    writer = _writer(store)
    with pytest.raises(OSError, match="Input/output error"):
        writer.finish()
    writer.abort()
    assert "rename" not in [kind for kind, _ in recorder.events]
    assert not shards(store, INFO)
    assert sorted(p.name for p in store.iterdir()) == ["store.json"]


def test_a_failed_sync_of_the_store_leaves_the_shard_visible(store, monkeypatch):
    recorder = Recorder(monkeypatch, full=True)
    recorder.fail = ("full", 2)
    writer = _writer(store)
    with pytest.raises(OSError, match="Input/output error"):
        writer.finish()
    writer.abort()
    ((shard, _),) = shards(store, INFO)
    assert shard.rows == 3


ENCODER = unfolded()


@pytest.mark.parametrize(
    ("fail", "visible"),
    [(("full", 3), [0]), (("full", 4), [0, 8])],
    ids=["barrier", "store"],
)
def test_durable_rows_count_only_the_shards_whose_syncs_succeeded(
    tmp_path, monkeypatch, fail, visible
):
    sink = ShardSink(tmp_path / "s", ENCODER, dtype="int8", shard_rows=8)
    recorder = Recorder(monkeypatch, full=True)
    recorder.fail = fail  # the second shard: its barrier, or the store sync after its rename
    with pytest.raises(OSError, match="Input/output error"):
        ENCODER.encode(texts(20), sink=sink, workers=1)
    assert sink.durable_rows == 8
    info = StoreInfo(ENCODER.fingerprint, "int8", ENCODER.output_dim)
    assert [shard.first_row for shard, _ in shards(tmp_path / "s", info)] == visible
    sink.close()  # the error was raised; the rows after it are dropped
    assert sink.durable_rows == 8
