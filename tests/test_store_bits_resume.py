"""A resumed ``ShardSink(sign_bits=True)`` writes the bits that the published prefix lacks;
an appending one does not. The stores of any stop equal the store of one run (gate A)."""

import shutil

import pytest

import fse
from fse import ShardSink
from fse._store import _bits_write
from fse._store._bits_format import bits_name
from support.bits import tree
from support.logs import captured
from support.positions import stream
from support.positions import write as write_stream
from support.resume import ENCODER, SHARD_ROWS, canonical, sentences, write

SOURCE = sentences(300)


def with_bits(path):
    """Shards, bits and root, as bytes, by name."""
    found = canonical(path)
    for entry in sorted(path.iterdir()):
        if entry.name.startswith("sign_bits-"):
            found.update({f"{entry.name}/{k}": v for k, v in tree(entry).items() if v is not None})
    return found


@pytest.fixture
def whole(tmp_path):
    write(tmp_path / "whole", SOURCE, sign_bits=True)
    return with_bits(tmp_path / "whole")


def test_a_run_with_bits_has_the_bits_of_every_shard(whole):
    assert (
        sum(name.startswith("sign_bits-") and name.endswith("manifest.json") for name in whole) == 5
    )


def stopped_after(path, shards, *, bits=True, **kwargs):
    """The store of one run of ``SOURCE`` as a run that stopped after ``shards`` shards leaves
    it (the shards after them are gone, and so are their bits)."""
    write(path, SOURCE, sign_bits=bits, **kwargs)
    for entry in sorted(p for p in path.iterdir() if p.name[:6] in ("shard-", "sign_b"))[::1]:
        number = int(entry.name.rsplit("-", 1)[1])
        if number >= shards * SHARD_ROWS:
            shutil.rmtree(entry)


@pytest.mark.parametrize("lost", [[2], [1, 2], [0, 2], [0, 1, 2]])
def test_a_stop_between_the_shards_and_their_bits_continues_with_the_bytes_of_one_run(
    tmp_path, whole, lost
):
    path = tmp_path / "store"
    stopped_after(path, 3)
    for number in lost:
        shutil.rmtree(path / bits_name(number * SHARD_ROWS))
    assert write(path, SOURCE, sign_bits=True)._kept == 3 * SHARD_ROWS
    assert with_bits(path) == whole


def test_a_store_written_without_bits_gets_those_of_the_prefix_when_a_resume_asks(tmp_path, whole):
    path = tmp_path / "store"
    stopped_after(path, 2, bits=False)
    assert not list(path.glob("sign_bits-*"))
    with captured() as records:
        write(path, SOURCE, sign_bits=True)
    (added,) = [r for r in records if r.fse_event == "store.sign_bits_added"]
    assert (added.fse_shards, added.fse_rows) == (2, 2 * SHARD_ROWS)
    assert with_bits(path) == whole


def test_a_resume_without_the_switch_does_not_write_the_bits_of_the_prefix(tmp_path):
    path = tmp_path / "store"
    stopped_after(path, 2, bits=False)
    write(path, SOURCE)
    assert not list(path.glob("sign_bits-*"))


def test_a_sink_that_appends_leaves_the_prefix_to_add_sign_bits(tmp_path):
    path = tmp_path / "store"
    with ShardSink(path, ENCODER, shard_rows=SHARD_ROWS) as sink:
        ENCODER.encode(SOURCE[: 2 * SHARD_ROWS], workers=1, sink=sink)
    with ShardSink(path, ENCODER, shard_rows=SHARD_ROWS, sign_bits=True) as sink:
        ENCODER.encode(SOURCE[2 * SHARD_ROWS : 3 * SHARD_ROWS], workers=1, sink=sink)
    assert sorted(p.name for p in path.glob("sign_bits-*")) == [bits_name(2 * SHARD_ROWS)]
    assert fse.add_sign_bits(path) == 2


def test_the_backfill_is_checked_so_a_changed_shard_publishes_nothing(tmp_path):
    path = tmp_path / "store"
    stopped_after(path, 2, bits=False)
    file = path / "shard-0000000000000000" / "vectors.npy"
    data = bytearray(file.read_bytes())
    data[-1] ^= 0x40  # an untagged shard is not hashed when it is opened
    file.write_bytes(bytes(data))
    with pytest.raises(fse.CorruptArtifactError, match="do not match their checksum"):
        write(path, SOURCE, sign_bits=True)
    assert not list(path.glob("*sign_bits-*"))


def test_the_backfill_follows_the_durability_of_the_sink(tmp_path, monkeypatch):
    synced = []
    real = _bits_write.publish_directory

    def spy(tmp, target, *, sync_parent=True):
        synced.append(sync_parent)
        real(tmp, target, sync_parent=sync_parent)

    monkeypatch.setattr(_bits_write, "publish_directory", spy)
    path = tmp_path / "store"
    stopped_after(path, 1, bits=False, durability="none")
    synced.clear()
    write(path, SOURCE, sign_bits=True, durability="none")
    assert synced  # the backfill and the new shards alike
    assert not any(synced)


# --- a stream with positions ---------------------------------------------------------------------


def test_a_stream_resumes_with_the_bits_of_the_prefix(tmp_path):
    states = {}
    for name, bits in (("signed", True), ("plain", False)):
        path = tmp_path / name
        write_stream(path, stream(SOURCE, 50), stop=4)
        assert not list(path.glob("sign_bits-*"))
        write_stream(path, stream(SOURCE, 50), sign_bits=bits)
        states[name] = path
    assert len(list(states["signed"].glob("sign_bits-*"))) == len(
        list(states["signed"].glob("shard-*"))
    )
    assert fse.add_sign_bits(states["plain"]) == len(list(states["plain"].glob("shard-*")))
    assert with_bits(states["signed"]) == with_bits(states["plain"])
