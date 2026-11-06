"""What a sink of the mode source_position@1 knows of its stream."""

from fse._store._position_format import SourcePosition
from fse._store._stream_state import StreamState

ROOT = SourcePosition("o0", 0)


def _state(rows=100, row=80):
    return StreamState(
        ROOT, SourcePosition("o80", row), rows, ((64, ROOT), (100, SourcePosition("o80", row)))
    )


def test_the_position_is_the_last_anchor_and_the_rows_since():
    state = _state()
    assert state.position() == ("o80", 20)
    assert state.skip == 20  # the run drops what the store holds after the anchor


def test_a_published_shard_moves_the_position_forward():
    state = _state()
    state.published(SourcePosition("o120", 120), 164)
    assert state.position() == ("o120", 44)
    assert state.skip == 20  # the run started from the old anchor


def test_a_store_without_shards_starts_at_its_root():
    state = StreamState(ROOT, ROOT, 0, ())
    assert state.position() == ("o0", 0)
    assert state.root == ROOT
