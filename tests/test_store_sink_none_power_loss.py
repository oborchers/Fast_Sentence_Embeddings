"""Unsynced rows are either a verified prefix or diagnosed corruption after power loss."""

import os
import shutil

import pytest

import fse
from fse._store import _recovery, _repair
from support.durability import materialize
from support.store import append_rows
from support.store_power_loss import check_prefix, codes, record_none, recover_and_append

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the POSIX persistence models")


@pytest.mark.parametrize("platform", ["linux", "darwin"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_none_power_loss_is_detected_and_repair_allows_a_new_append(
    tmp_path, monkeypatch, platform, dtype
):
    trace, expected = record_none(tmp_path / "store", monkeypatch, platform, dtype)
    outcomes = set()
    count = 0
    # Replay tests data and recovery on real files. C separately models repair syncs;
    # no physical disk flush is needed between these already-materialized crash images.
    monkeypatch.setattr(_repair, "_sync_directory", lambda *_args, **_kwargs: None)
    for state, confirmed in trace.states():
        assert confirmed == 0
        image = tmp_path / "image"
        materialize(image, state)
        try:
            outcomes.add(check_prefix(image, expected))
            recover_and_append(image, expected)
        finally:
            shutil.rmtree(image)
        count += 1
    assert {None, 0, 4, 8, 10} <= outcomes
    assert count >= 8


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_removing_the_none_tag_serves_lost_vector_data_silently(tmp_path, monkeypatch, platform):
    trace, expected = record_none(tmp_path / "store", monkeypatch, platform, tag=False)
    for state, _ in trace.states():
        image = tmp_path / "image"
        materialize(image, state)
        try:
            try:
                with fse.VectorStore.open(image) as store:
                    if codes(store, expected).tobytes() != expected[: len(store)].tobytes():
                        return  # successful open with wrong bytes, never just a structural error
            except fse.CorruptArtifactError:
                pass
        finally:
            shutil.rmtree(image)
    pytest.fail("removing the tag did not expose silently lost vector data")


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_removing_the_gap_guard_relinks_a_surviving_tail_after_append(
    tmp_path, monkeypatch, platform
):
    trace, expected = record_none(tmp_path / "store", monkeypatch, platform)
    monkeypatch.setattr(_recovery, "check_gaps", lambda *_args: None)
    for state, _ in trace.states():
        image = tmp_path / "image"
        materialize(image, state)
        try:
            try:
                with fse.VectorStore.open(image) as store:
                    if len(store) != 4:
                        continue
                assert append_rows(image, durability="none") == 4
                with fse.VectorStore.open(image) as store:
                    if len(store) == 10:
                        found = codes(store, expected)
                        assert (found[4:8] == 99).all()
                        assert found[8:].tobytes() == expected[8:].tobytes()
                        return
            except fse.CorruptArtifactError:
                pass
        finally:
            shutil.rmtree(image)
    pytest.fail("removing the gap guard did not expose old-tail relinking")
