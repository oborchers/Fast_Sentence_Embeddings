"""Views over the parts of one source (the mode ``source_rows@1``): the parts are read in the
order of the source, and a position is a row of the source."""

import shutil

import numpy as np
import pytest

from fse import ShardSink, VectorStore
from support.resume import ENCODER, drop_tail, write

ROWS = 49_252  # four blocks of 16,384 rows: three parts of 16,384, 16,384 and 16,484 rows
SOURCE = [f"w{i % 200} w{(i * 7) % 200}" for i in range(ROWS)]
FIRST = (0, 16_384, 32_768)
QUERIES = np.random.default_rng(2).normal(size=(2, ENCODER.output_dim)).astype(np.float32)


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    root = tmp_path_factory.mktemp("parts")
    write(root / "whole", SOURCE, rows=8192)
    for k in range(3):
        write(root / f"part{k}", SOURCE, shard=(k, 3), rows=4096)
    return root


def bits(array):
    return np.asarray(array).view(np.uint8).tobytes()


def test_the_parts_are_read_in_the_order_of_the_source(stores):
    paths = [stores / "part2", stores / "part0", stores / "part1"]
    with VectorStore.open(stores / "whole") as one, VectorStore.open_view(paths) as view:
        assert view.paths == (stores / "part0", stores / "part1", stores / "part2")
        assert len(view) == len(one) == ROWS
        got, want = view.search(QUERIES, 12, mode="exact"), one.search(QUERIES, 12, mode="exact")
        assert bits(got.ids) == bits(want.ids)
        assert bits(got.scores) == bits(want.scores)
        assert bits(view.get_vectors(start=16_000, stop=17_000)) == bits(
            one.get_vectors(start=16_000, stop=17_000)
        )


def test_the_source_of_a_view_of_all_parts(stores):
    with VectorStore.open(stores / "whole") as one:
        whole = dict(one.source)
    paths = [stores / f"part{k}" for k in range(3)]
    with VectorStore.open_view(paths) as view:
        assert dict(view.source) == {
            "mode": "source_rows@1",
            "fingerprint": whole["fingerprint"],
            "form": "strings",
            "first_row": 0,
            "rows": ROWS,
            "shard": [0, 1],
        }
        assert whole["shard"] == [0, 1]
        assert [dict(m["source"])["shard"] for m in view.members] == [[0, 3], [1, 3], [2, 3]]
        assert [dict(m["source"])["first_row"] for m in view.members] == list(FIRST)
        assert [m["first_row"] for m in view.members] == list(FIRST)
        member, row = view.locate([0, 16_383, 16_384, 32_768, ROWS - 1])
        assert member.tolist() == [0, 0, 1, 2, 2]
        assert row.tolist() == [0, 16_383, 0, 0, 16_483]


def test_a_view_of_all_parts_is_the_source_only_when_all_are_there(stores):
    parts = [stores / f"part{k}" for k in range(3)]
    with VectorStore.open_view(parts) as view:
        assert view.source["shard"] == [0, 1]
    with VectorStore.open_view(parts[:2]) as view:
        assert view.source is None


def test_a_view_of_one_whole_store_is_that_store(stores):
    with VectorStore.open_view([stores / "whole"]) as view:
        assert view.source["shard"] == [0, 1]
        assert len(view) == ROWS


def test_a_view_of_some_parts_is_a_snapshot_of_their_concatenation(stores):
    with VectorStore.open(stores / "part0") as p0, VectorStore.open(stores / "part2") as p2:
        want = np.concatenate([p0.get_vectors(), p2.get_vectors()])
    with VectorStore.open_view([stores / "part2", stores / "part0"]) as view:
        assert view.paths == (stores / "part0", stores / "part2")
        assert view.source is None
        assert len(view) == 16_384 + 16_484
        assert bits(view.get_vectors()) == bits(want)
        assert [dict(m["source"])["shard"] for m in view.members] == [[0, 3], [2, 3]]
        assert [m["rows"] for m in view.members] == [16_384, 16_484]
        assert view.locate([16_384])[0].tolist() == [1]
        with pytest.raises(ValueError, match="does not refresh"):
            view.refresh()
    with VectorStore.open_view([stores / "part1"]) as single:
        assert single.source is None


def test_an_unfinished_part_is_read_as_far_as_it_is_written(stores, tmp_path):
    unfinished = tmp_path / "unfinished"
    shutil.copytree(stores / "part2", unfinished)
    kept = drop_tail(unfinished, 2, rows=4096)
    parts = [stores / "part0", stores / "part1", unfinished]
    with VectorStore.open_view(parts) as view:
        assert view.source is None  # not all parts are finished: positions are not source rows
        assert len(view) == 2 * 16_384 + kept
        assert [m["rows"] for m in view.members] == [16_384, 16_384, kept]


def test_no_part_twice(stores, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(stores / "part1", copy)
    with pytest.raises(ValueError, match="part 1 of 3 is given twice"):
        VectorStore.open_view([stores / "part0", stores / "part1", copy])


def test_parts_of_another_source_or_cut_are_refused(stores, tmp_path):
    parts = [stores / f"part{k}" for k in range(3)]
    other = tmp_path / "other"
    write(other, SOURCE[:-1], shard=(2, 3), rows=4096)
    with pytest.raises(ValueError, match="another"):
        VectorStore.open_view([parts[0], parts[1], other])
    halves = tmp_path / "halves"
    write(halves, SOURCE, shard=(1, 2), rows=4096)
    with pytest.raises(ValueError, match="part of 2 parts"):
        VectorStore.open_view([parts[0], halves])


def test_stores_with_and_without_a_source_are_not_mixed(stores, tmp_path):
    plain = tmp_path / "plain"
    with ShardSink(plain, ENCODER, shard_rows=3) as sink:
        ENCODER.encode(["w1 w2"] * 5, sink=sink, workers=1)
    for paths in ([plain, stores / "part1"], [stores / "part1", plain]):
        with pytest.raises(ValueError, match="has no source"):
            VectorStore.open_view(paths)
    with VectorStore.open_view([plain]) as view:
        assert view.source is None
        assert dict(view.members[0]) == {
            "path": plain,
            "first_row": 0,
            "rows": 5,
            "source": None,
        }
