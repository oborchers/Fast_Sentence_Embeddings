"""Repair sync ordering and the two independent guards against tail relinking."""

import os
import shutil
from pathlib import Path

import pytest

import fse
from fse._container import _npy_inplace
from fse._store import _recovery, _repair
from fse._store._layout import shard_name
from support.durability import PowerLoss, materialize
from support.repair import after_crash, damaged_store
from support.store import append_rows

POSIX = pytest.mark.skipif(os.name == "nt", reason="the POSIX persistence models")


def test_every_move_syncs_destination_and_source_before_the_next_move(tmp_path, monkeypatch):
    path = tmp_path / "store"
    damaged_store(path)
    events = []
    real_mkdir, real_rename = Path.mkdir, _repair._rename_exclusive

    def mkdir(directory, *args, **kwargs):
        real_mkdir(directory, *args, **kwargs)
        events.append(("mkdir", "broken"))

    def rename(source, destination):
        real_rename(source, destination)
        events.append(("rename", source.name))

    def sync(directory, *, full=True):
        assert full
        events.append(("sync", "store" if directory == path else "broken"))

    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(_repair, "_rename_exclusive", rename)
    monkeypatch.setattr(_repair, "_sync_directory", sync)
    assert fse.repair_store(path) == 4
    assert events == [
        ("mkdir", "broken"),
        ("sync", "store"),
        *(
            event
            for row in (12, 8, 4)
            for event in (("rename", shard_name(row)), ("sync", "broken"), ("sync", "store"))
        ),
    ]


def _record(path, monkeypatch, platform, *, intermediate=True):
    damaged_store(path)
    with monkeypatch.context() as patch:
        trace = PowerLoss(path, patch, platform, include_hidden=True)
        renamed = []
        real_rename = _npy_inplace._rename_exclusive  # the model's mutation wrapper
        real_sync = _repair._sync_directory

        def rename(source, destination):
            real_rename(source, destination)
            renamed.append(source.name)

        def sync(directory, *, full=True):
            if intermediate or len(renamed) in (0, 3):
                real_sync(directory, full=full)

        patch.setattr(_repair, "_rename_exclusive", rename)
        patch.setattr(_repair, "_sync_directory", sync)
        assert fse.repair_store(path) == 4
        trace.capture()
    return trace


@POSIX
@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_power_loss_at_every_repair_event_allows_safe_rerun_and_append(
    tmp_path, monkeypatch, platform
):
    trace = _record(tmp_path / "store", monkeypatch, platform)
    count = 0
    for state, _ in trace.states():
        image = tmp_path / "image"
        materialize(image, state)
        try:
            after_crash(image)
        finally:
            shutil.rmtree(image)
        count += 1
    assert count >= 4


@POSIX
@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_gap_guard_alone_keeps_every_unsynced_repair_crash_safe(tmp_path, monkeypatch, platform):
    trace = _record(tmp_path / "store", monkeypatch, platform, intermediate=False)
    count = 0
    for state, _ in trace.states():
        image = tmp_path / "image"
        materialize(image, state)
        try:
            after_crash(image)
        finally:
            shutil.rmtree(image)
        count += 1
    assert count >= 8


@POSIX
@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_combined_mutant_relinks_old_rows_after_a_new_append(tmp_path, monkeypatch, platform):
    trace = _record(tmp_path / "store", monkeypatch, platform, intermediate=False)
    monkeypatch.setattr(_recovery, "check_gaps", lambda *_args: None)
    for state, _ in trace.states():
        image = tmp_path / "image"
        materialize(image, state)
        try:
            try:
                with fse.VectorStore.open(image) as store:
                    if len(store) != 4:
                        continue
                assert append_rows(image) == 4
                with fse.VectorStore.open(image) as store:
                    if len(store) > 8:
                        assert (store.get_vectors(start=4, stop=8) == 99).all()
                        assert (store.get_vectors(start=8) != 99).all()
                        return
            except fse.CorruptArtifactError:
                pass
        finally:
            shutil.rmtree(image)
    pytest.fail("the combined mutation failed to expose old-row relinking")
